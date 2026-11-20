#!/usr/bin/env python3
"""
inference_engine.py
Lõi Suy luận Tổng quát NIDS (Multi-Model NIDS Inference Engine).
Hỗ trợ kiến trúc 2-Tier Hybrid NIDS, Auto-Discovery Model Registry,
thread-safe lazy caching, startup threshold index cache (`models_index.json`),
suy luận luồng đơn lẻ thời gian thực, suy luận hàng loạt (batch)
và trích xuất mẫu kiểm thử thực tế từ tập Held-out Test.
"""

from __future__ import annotations

import glob
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import joblib
import numpy as np
import pandas as pd

# Thiết lập đường dẫn dự án
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Danh sách 10 đặc trưng NetFlow chuẩn theo NetFlow_v1
NETFLOW_FEATURES: List[str] = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
]

CANONICAL_ATTACKS: List[str] = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]
FAMILY_DISPLAY_ORDER: List[str] = ["PAPER", "RF", "GB"]

RISK_ZONE_ATTACK = "ATTACK (Chặn ngay - Tầng 1)"
RISK_ZONE_SUSPICIOUS = "SUSPICIOUS / NEAR-MISS (Chuyển DPI Tầng 2)"
RISK_ZONE_BENIGN = "BENIGN (Cho phép qua)"


class ModelMetricsDict(TypedDict, total=False):
    val_f1: Optional[float]
    test_f1: Optional[float]
    test_precision: Optional[float]
    test_recall: Optional[float]
    test_accuracy: Optional[float]
    train_time_sec: Optional[float]


class ModelRegistryEntry(TypedDict):
    model_key: str
    model_family: str
    attack: str
    condition: str
    condition_label: str
    model_path: Path
    selector_path: Optional[Path]
    metadata_path: Path
    selected_features: List[str]
    k_features: int
    default_f1_threshold: float
    thresholds: Dict[str, float]
    metrics: ModelMetricsDict
    raw_metadata: Dict[str, Any]


class VerdictDict(TypedDict):
    status: str
    tier: str
    title: str
    message: str
    badge_color: str
    max_probability: float
    primary_attack: str
    triggered_by: str
    near_miss_low: float
    threshold_used: float


class SingleFlowPrediction(TypedDict):
    verdict: VerdictDict
    details: pd.DataFrame


class BatchMetricsDict(TypedDict):
    TP: int
    FP: int
    FN: int
    TN: int
    Accuracy: float
    Precision: float
    Recall: float
    F1: float
    Evaluated_Samples: int


class BatchSummaryDict(TypedDict, total=False):
    model_key: str
    model_used: str
    target_attack: str
    threshold_used: float
    near_miss_low: float
    threshold_mode: str
    total_flows: int
    benign_count: int
    suspicious_count: int
    attack_count: int
    missing_features: List[str]
    missing_model_features: List[str]
    coerced_columns: List[str]
    batch_inference_time_sec: float
    throughput_flows_per_sec: float
    cross_attack_total: int
    cross_attack_detected: int
    metrics: BatchMetricsDict


def get_near_miss_bounds(thresh: float) -> Tuple[float, float]:
    """
    Tính cặp cận vùng Near-Miss (near_low, thresh) đảm bảo 3 vùng rủi ro
    [0, near_low), [near_low, thresh), [thresh, 1] luôn tách rời nhau
    kể cả khi ngưỡng hiệu chuẩn < 0.10.
    """
    t = float(thresh)
    near_low = round(min(0.10, 0.5 * t), 4) if t > 0.0 else 0.0
    return (near_low, t)


def build_consensus_matrix(details_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Tổng hợp bảng chi tiết đa mô hình thành ma trận đồng thuận 5×3
    (5 loại tấn công × PAPER / RF / GB) theo từng điều kiện huấn luyện.
    """
    if details_df is None or details_df.empty:
        return {}

    matrices: Dict[str, pd.DataFrame] = {}
    conditions = list(dict.fromkeys(details_df["Condition"].tolist()))

    for cond_label in conditions:
        cond_df = details_df[details_df["Condition"] == cond_label]
        rows: List[Dict[str, Any]] = []

        for attack in CANONICAL_ATTACKS:
            att_df = cond_df[cond_df["Attack"].str.lower() == attack.lower()]
            row: Dict[str, Any] = {"Attack": attack}
            attack_votes = 0
            suspicious_votes = 0
            available_families = 0
            total_latency = 0.0

            for fam in FAMILY_DISPLAY_ORDER:
                fam_df = att_df[att_df["Model_Family"].str.upper() == fam]
                col_prefix = "Ours-HistGB" if fam == "GB" else ("Ours-RF" if fam == "RF" else "PAPER")
                if fam_df.empty:
                    row[col_prefix] = "N/A"
                    row[f"{col_prefix}_Status"] = "N/A"
                    continue

                rec = fam_df.iloc[0]
                prob = float(rec["Probability"])
                thresh = float(rec["Threshold_Used"])
                margin = prob - thresh
                status = str(rec["Status_Code"])
                latency = float(rec["Latency_ms"])

                available_families += 1
                total_latency += latency
                if status == "ATTACK":
                    attack_votes += 1
                    icon = "🔴"
                elif status == "SUSPICIOUS":
                    suspicious_votes += 1
                    icon = "🟡"
                else:
                    icon = "🟢"

                row[col_prefix] = f"{icon} {prob * 100:.1f}% ({margin * 100:+.1f}%)"
                row[f"{col_prefix}_Status"] = status

            consensus_str = f"{attack_votes}/{available_families}" if available_families > 0 else "0/0"
            if suspicious_votes > 0:
                consensus_str += f" (+{suspicious_votes} DPI)"

            row["Consensus"] = consensus_str
            row["Attack_Votes"] = attack_votes
            row["Suspicious_Votes"] = suspicious_votes
            row["Total_Latency_ms"] = round(total_latency, 2)
            rows.append(row)

        matrices[cond_label] = pd.DataFrame(rows)

    return matrices


class NIDSInferenceEngine:
    """
    Lõi Suy luận NIDS hướng đối tượng, hỗ trợ:
    1. Tự động khám phá mô hình trong results/*/models/ (Auto-Discovery Model Registry).
    2. Cache ngưỡng hiệu chuẩn từ `.npy` theo mtime trong `models_index.json`.
    3. Thread-safe lazy caching cho mô hình `.joblib` và dữ liệu `heldout_test.parquet`.
    4. Dự đoán luồng đơn lẻ phân 3 vùng rủi ro tách rời (2-Tier Hybrid NIDS).
    5. Dự đoán hàng loạt (Batch Inference) với chế độ strict tránh fallback sai mô hình.
    """

    def __init__(
        self,
        models_base_dir: Optional[Union[str, Path]] = None,
        lazy_load: bool = True,
        index_cache_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.project_root: Path = PROJECT_ROOT
        self.results_dir: Path = RESULTS_DIR
        self.models_base_dir: Path = Path(models_base_dir) if models_base_dir else self.results_dir
        self.lazy_load: bool = lazy_load
        self._lock: threading.RLock = threading.RLock()

        if index_cache_path is not None:
            self.index_cache_path: Optional[Path] = Path(index_cache_path)
        elif self.models_base_dir.resolve() == self.results_dir.resolve():
            self.index_cache_path = CURRENT_DIR / "models_index.json"
        else:
            # Khi test với models_base_dir tùy biến, không đọc/ghi đè cache mặc định của web_app
            self.index_cache_path = None

        self.models_registry: Dict[str, ModelRegistryEntry] = {}
        self._loaded_models: Dict[str, Any] = {}
        self.recall_thresholds_df: Optional[pd.DataFrame] = None
        self.cost_thresholds_df: Optional[pd.DataFrame] = None
        self._heldout_test_df: Optional[pd.DataFrame] = None

        self._load_specialized_thresholds()
        self._auto_discover_models()

        if not self.lazy_load:
            self.load_all_models()

    @staticmethod
    def _format_model_input(model: Any, df_sub: pd.DataFrame) -> Any:
        """
        Chuẩn hóa đầu vào khớp với trạng thái feature_names_in_ của mô hình scikit-learn:
        - Nếu model được fit trên DataFrame (có feature_names_in_), giữ nguyên DataFrame.
        - Nếu model được fit trên NumPy array (không có feature_names_in_), chuyển sang ndarray.
        """
        if hasattr(model, "feature_names_in_"):
            return df_sub
        return df_sub.to_numpy()

    @staticmethod
    def _coerce_numeric_features(
        df: pd.DataFrame,
        columns: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Ép kiểu số an toàn cho các cột NetFlow đầu vào (xử lý CSV có chuỗi/NaN/Inf).
        Trả về bản sao DataFrame các cột đã ép kiểu và danh sách cột có giá trị bị ép/điền 0.
        """
        coerced_cols: List[str] = []
        X_sub = df[columns].copy()
        for col in columns:
            raw_series = X_sub[col]
            numeric_series = pd.to_numeric(raw_series, errors="coerce")
            invalid_mask = numeric_series.isna() | ~np.isfinite(numeric_series.to_numpy(dtype=float, na_value=np.nan))
            if bool(invalid_mask.any()):
                coerced_cols.append(col)
                numeric_series = numeric_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_sub[col] = numeric_series.astype(float)
        return X_sub, coerced_cols

    @staticmethod
    def _calibrate_thresholds_from_val_arrays(
        y_val: np.ndarray,
        p_val: np.ndarray,
        t_f1_opt: float,
    ) -> Dict[str, float]:
        """
        Hiệu chuẩn bộ ngưỡng Recall-Targeted (95%, 98%, 99%) và Cost-Sensitive (c=10, c=20)
        trực tiếp từ mảng xác suất Validation của chính mô hình đó (vectorized qua searchsorted).
        Đảm bảo khớp 100% thuật toán trong src/advanced_evaluation.py.
        """
        calibrated: Dict[str, float] = {}
        pos_sorted = np.sort(p_val[y_val == 1])
        neg_sorted = np.sort(p_val[y_val == 0])
        n_pos = len(pos_sorted)
        n_neg = len(neg_sorted)
        if n_pos == 0 or n_neg == 0:
            return calibrated

        # 1. Recall-Targeted: quét từ 0.99 -> 0.001 (990 mốc), chặn trên bởi t_f1_opt
        t_rec_scan = np.round(np.linspace(0.99, 0.001, 990), 4)
        tp_counts = n_pos - np.searchsorted(pos_sorted, t_rec_scan, side="left")
        recalls = tp_counts / n_pos
        for tr, label in [(0.95, "Recall >= 95%"), (0.98, "Recall >= 98%"), (0.99, "Recall >= 99%")]:
            hit_idx = np.where(recalls >= tr)[0]
            t_found = float(t_rec_scan[hit_idx[0]]) if len(hit_idx) > 0 else 0.001
            calibrated[label] = round(min(float(t_f1_opt), t_found), 4)

        # 2. Cost-Sensitive: quét từ 0.005 -> 0.995 (199 mốc), tối thiểu hóa c * FN + FP
        t_cost_scan = np.round(np.linspace(0.005, 0.995, 199), 3)
        fn_counts = np.searchsorted(pos_sorted, t_cost_scan, side="left")
        fp_counts = n_neg - np.searchsorted(neg_sorted, t_cost_scan, side="left")
        for c_val, label in [(10, "Cost-Sensitive (c=10)"), (20, "Cost-Sensitive (c=20)")]:
            costs = c_val * fn_counts + fp_counts
            best_idx = int(np.argmin(costs))
            calibrated[label] = round(float(t_cost_scan[best_idx]), 4)

        return calibrated

    def _load_specialized_thresholds(self) -> None:
        """Nạp các bảng ngưỡng tối ưu từ results/improved/ nếu tồn tại."""
        self.recall_thresholds_df = None
        self.cost_thresholds_df = None

        # Chỉ đọc CSV của results/improved khi models_base_dir trùng results_dir (hoặc subfolder tồn tại)
        improved_dir = self.models_base_dir / "improved"
        if not improved_dir.exists() and self.models_base_dir.resolve() == self.results_dir.resolve():
            improved_dir = self.results_dir / "improved"

        recall_path = improved_dir / "recall_targeted_thresholds.csv"
        if recall_path.exists():
            try:
                self.recall_thresholds_df = pd.read_csv(recall_path)
            except Exception as exc:
                print(f"[!] Cảnh báo: Không thể đọc {recall_path}: {exc}")

        cost_path = improved_dir / "cost_sensitive_evaluation.csv"
        if cost_path.exists():
            try:
                self.cost_thresholds_df = pd.read_csv(cost_path)
            except Exception as exc:
                print(f"[!] Cảnh báo: Không thể đọc {cost_path}: {exc}")

    def _read_index_cache(self) -> Dict[str, Any]:
        """Đọc cache ngưỡng `models_index.json` an toàn (fallback rỗng nếu thiếu hoặc lỗi)."""
        if self.index_cache_path is None or not self.index_cache_path.exists():
            return {"schema_version": 1, "entries": {}}
        try:
            with open(self.index_cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict) and isinstance(payload.get("entries"), dict):
                return payload
        except Exception:
            pass
        return {"schema_version": 1, "entries": {}}

    def _write_index_cache(self, entries: Dict[str, Any]) -> None:
        """Lưu cache ngưỡng `models_index.json` an toàn (không làm crash startup nếu lỗi I/O)."""
        if self.index_cache_path is None:
            return
        try:
            self.index_cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "entries": entries,
            }
            tmp_path = self.index_cache_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            tmp_path.replace(self.index_cache_path)
        except Exception:
            pass

    @staticmethod
    def _safe_mtime_ns(path: Path) -> Optional[int]:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def _auto_discover_models(self) -> None:
        """
        Tự động quét tất cả các thư mục con trong models_base_dir/*/models/
        để tìm mọi cặp file *_metadata.json và *_best_model.joblib.
        """
        with self._lock:
            discovered: Dict[str, ModelRegistryEntry] = {}

            search_pattern = str(self.models_base_dir / "*" / "models")
            model_dirs = [Path(p) for p in glob.glob(search_pattern)]
            if (self.models_base_dir / "models").exists():
                model_dirs.append(self.models_base_dir / "models")

            cache_data = self._read_index_cache()
            cached_entries: Dict[str, Any] = cache_data.get("entries", {})
            updated_cache_entries: Dict[str, Any] = {}
            cache_dirty = False

            for mdir in sorted(set(model_dirs), key=lambda p: str(p)):
                if not mdir.is_dir():
                    continue

                meta_files = sorted(mdir.glob("*_metadata.json"), key=lambda p: p.name)
                for meta_path in meta_files:
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                    except Exception as exc:
                        print(f"[!] Bỏ qua file metadata hỏng {meta_path}: {exc}")
                        continue

                    prefix = meta_path.stem.replace("_metadata", "")
                    model_path = mdir / f"{prefix}_best_model.joblib"
                    if not model_path.exists():
                        cand = sorted(mdir.glob(f"{prefix}*.joblib"), key=lambda p: p.name)
                        cand = [c for c in cand if "selector" not in c.name]
                        if cand:
                            model_path = cand[0]
                        else:
                            continue

                    selector_path = mdir / f"{prefix}_best_selector.joblib"

                    if prefix.startswith("rf_"):
                        model_family = "RF"
                    elif prefix.startswith("gb_"):
                        model_family = "GB"
                    else:
                        parts = prefix.split("_")
                        model_family = parts[0].upper()

                    raw_attack = str(meta.get("attack", ""))
                    if not raw_attack:
                        for att in CANONICAL_ATTACKS:
                            if att.lower() in prefix.lower():
                                raw_attack = att
                                break
                    attack = raw_attack.capitalize()
                    for canonical in CANONICAL_ATTACKS:
                        if canonical.lower() == raw_attack.lower():
                            attack = canonical
                            break

                    condition = str(meta.get("condition", ""))
                    if not condition:
                        if "nosmote" in prefix.lower():
                            condition = "nosmote"
                        elif "smote" in prefix.lower():
                            condition = "smote"
                        else:
                            condition = "nosmote"
                    cond_label = "Without SMOTE" if condition == "nosmote" else "With SMOTE"

                    raw_feats = meta.get("selected_features", "")
                    if isinstance(raw_feats, str) and raw_feats:
                        selected_features = [f.strip() for f in raw_feats.split(";") if f.strip()]
                    elif isinstance(raw_feats, list):
                        selected_features = [str(f).strip() for f in raw_feats if str(f).strip()]
                    else:
                        selected_features = list(NETFLOW_FEATURES)

                    default_thresh = float(meta.get("val_threshold", 0.50))
                    thresholds_dict: Dict[str, float] = {
                        "F1-Optimal": round(default_thresh, 4),
                        "Standard (0.50)": 0.50,
                    }

                    csv_matched = False
                    if condition == "nosmote":
                        if self.recall_thresholds_df is not None:
                            rc_mask = (
                                (self.recall_thresholds_df["Attack"].str.lower() == attack.lower())
                                & (self.recall_thresholds_df["Model"].str.upper() == model_family)
                            )
                            if "Condition" in self.recall_thresholds_df.columns:
                                rc_mask &= (
                                    self.recall_thresholds_df["Condition"].str.lower() == cond_label.lower()
                                )
                            sub_rc = self.recall_thresholds_df[rc_mask]
                            for _, r_row in sub_rc.iterrows():
                                csv_matched = True
                                mode_name = str(r_row["Mode"]).strip()
                                if mode_name in ["Recall>=95%", "Recall >= 95%"]:
                                    thresholds_dict["Recall >= 95%"] = round(float(r_row["Threshold"]), 4)
                                elif mode_name in ["Recall>=98%", "Recall >= 98%"]:
                                    thresholds_dict["Recall >= 98%"] = round(float(r_row["Threshold"]), 4)
                                elif mode_name in ["Recall>=99%", "Recall >= 99%"]:
                                    thresholds_dict["Recall >= 99%"] = round(float(r_row["Threshold"]), 4)

                        if self.cost_thresholds_df is not None:
                            cost_mask = (
                                (self.cost_thresholds_df["Attack"].str.lower() == attack.lower())
                                & (self.cost_thresholds_df["Model"].str.upper() == model_family)
                            )
                            if "Condition" in self.cost_thresholds_df.columns:
                                cost_mask &= (
                                    self.cost_thresholds_df["Condition"].str.lower() == cond_label.lower()
                                )
                            sub_cost = self.cost_thresholds_df[cost_mask]
                            for _, c_row in sub_cost.iterrows():
                                csv_matched = True
                                ratio_c = int(c_row["Cost_Ratio_c"])
                                if ratio_c == 10:
                                    thresholds_dict["Cost-Sensitive (c=10)"] = round(
                                        float(c_row["Optimal_Threshold"]), 4
                                    )
                                elif ratio_c == 20:
                                    thresholds_dict["Cost-Sensitive (c=20)"] = round(
                                        float(c_row["Optimal_Threshold"]), 4
                                    )

                    model_key = f"{model_family.lower()}_{attack.lower()}_{condition.lower()}"

                    if not csv_matched:
                        y_val_file = mdir.parent / f"{prefix}_y_val.npy"
                        p_val_file = mdir.parent / f"{prefix}_best_proba_val.npy"
                        if y_val_file.exists() and p_val_file.exists():
                            meta_mtime = self._safe_mtime_ns(meta_path)
                            yv_mtime = self._safe_mtime_ns(y_val_file)
                            pv_mtime = self._safe_mtime_ns(p_val_file)
                            cached = cached_entries.get(model_key)
                            if (
                                isinstance(cached, dict)
                                and cached.get("meta_mtime_ns") == meta_mtime
                                and cached.get("y_val_mtime_ns") == yv_mtime
                                and cached.get("p_val_mtime_ns") == pv_mtime
                                and float(cached.get("default_f1_threshold", -1.0)) == round(default_thresh, 4)
                                and isinstance(cached.get("calibrated_thresholds"), dict)
                            ):
                                cal_map = {
                                    str(k): float(v) for k, v in cached["calibrated_thresholds"].items()
                                }
                                thresholds_dict.update(cal_map)
                                updated_cache_entries[model_key] = cached
                            else:
                                try:
                                    yv_arr = np.load(y_val_file)
                                    pv_arr = np.load(p_val_file)
                                    cal_map = self._calibrate_thresholds_from_val_arrays(
                                        yv_arr, pv_arr, default_thresh
                                    )
                                    thresholds_dict.update(cal_map)
                                    updated_cache_entries[model_key] = {
                                        "meta_mtime_ns": meta_mtime,
                                        "y_val_mtime_ns": yv_mtime,
                                        "p_val_mtime_ns": pv_mtime,
                                        "default_f1_threshold": round(default_thresh, 4),
                                        "calibrated_thresholds": cal_map,
                                    }
                                    cache_dirty = True
                                except Exception as exc:
                                    print(f"[!] Cảnh báo: Không thể hiệu chuẩn ngưỡng từ Validation cho {prefix}: {exc}")

                    discovered[model_key] = {
                        "model_key": model_key,
                        "model_family": model_family,
                        "attack": attack,
                        "condition": condition,
                        "condition_label": cond_label,
                        "model_path": model_path,
                        "selector_path": selector_path if selector_path.exists() else None,
                        "metadata_path": meta_path,
                        "selected_features": selected_features,
                        "k_features": len(selected_features),
                        "default_f1_threshold": default_thresh,
                        "thresholds": thresholds_dict,
                        "metrics": {
                            "val_f1": meta.get("val_f1"),
                            "test_f1": meta.get("test_f1"),
                            "test_precision": meta.get("test_precision"),
                            "test_recall": meta.get("test_recall"),
                            "test_accuracy": meta.get("test_accuracy"),
                            "train_time_sec": meta.get("train_time_sec"),
                        },
                        "raw_metadata": meta,
                    }

            self.models_registry = dict(sorted(discovered.items()))

            if self.index_cache_path is not None and (
                cache_dirty or (updated_cache_entries and not self.index_cache_path.exists())
            ):
                self._write_index_cache(updated_cache_entries)

    def get_available_model_families(self) -> List[str]:
        """Trả về danh sách các họ mô hình hiện có trong registry theo thứ tự ưu tiên."""
        families = {info["model_family"] for info in self.models_registry.values()}
        preferred = ["GB", "RF", "PAPER"]
        ordered = [f for f in preferred if f in families]
        extras = sorted(f for f in families if f not in preferred)
        return ordered + extras

    def get_available_conditions(self) -> List[Tuple[str, str]]:
        """Trả về danh sách cặp `(condition_key, condition_label)` hiện có trong registry."""
        seen: Dict[str, str] = {}
        for info in self.models_registry.values():
            seen[info["condition"]] = info["condition_label"]
        order = ["nosmote", "smote"]
        result: List[Tuple[str, str]] = [(c, seen[c]) for c in order if c in seen]
        for c in sorted(seen):
            if c not in order:
                result.append((c, seen[c]))
        return result

    def get_model(self, model_key: str) -> Any:
        """Nạp model từ cache hoặc từ file joblib (thread-safe lazy loading)."""
        with self._lock:
            if model_key in self._loaded_models:
                return self._loaded_models[model_key]

            if model_key not in self.models_registry:
                raise KeyError(f"Mô hình '{model_key}' không có trong registry!")

            info = self.models_registry[model_key]
            model_path = info["model_path"]
            model = joblib.load(model_path)
            self._loaded_models[model_key] = model
            return model

    def load_all_models(self) -> None:
        """Nạp toàn bộ các mô hình trong registry vào RAM trước."""
        with self._lock:
            for key in list(self.models_registry.keys()):
                self.get_model(key)

    def reload_registry(self) -> None:
        """Xóa sạch cache mô hình trong RAM và nạp mới toàn bộ ngưỡng + mô hình từ đĩa."""
        with self._lock:
            self._loaded_models.clear()
            self._heldout_test_df = None
            self._load_specialized_thresholds()
            self._auto_discover_models()
            if not self.lazy_load:
                self.load_all_models()

    def get_registered_models_summary(self) -> pd.DataFrame:
        """Trả về bảng tổng kết tất cả các mô hình có trong hệ thống."""
        records: List[Dict[str, Any]] = []
        for key, info in sorted(self.models_registry.items()):
            m = info["metrics"]
            records.append({
                "Model_Key": key,
                "Family": info["model_family"],
                "Attack": info["attack"],
                "Condition": info["condition_label"],
                "Features_Count": info["k_features"],
                "Val_Threshold": info["default_f1_threshold"],
                "Test_F1": m.get("test_f1"),
                "Test_Recall": m.get("test_recall"),
                "Test_Precision": m.get("test_precision"),
                "Train_Time_s": m.get("train_time_sec"),
            })
        return pd.DataFrame(records)

    def predict_single_flow(
        self,
        flow_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        threshold_mode: str = "F1-Optimal",
        condition_filter: str = "all",
        attack_filter: str = "all",
        custom_threshold: Optional[float] = None,
    ) -> SingleFlowPrediction:
        """
        Dự đoán cho 1 luồng mạng đơn lẻ qua các mô hình NIDS phù hợp.
        Phân loại trạng thái 3 vùng tách rời nhau theo `get_near_miss_bounds(thresh)`:
          - proba >= thresh: 'ATTACK (Chặn ngay - Tầng 1)'
          - near_low <= proba < thresh: 'SUSPICIOUS / NEAR-MISS (Chuyển DPI Tầng 2)'
          - proba < near_low: 'BENIGN (Cho phép qua)'
        """
        if not self.models_registry:
            raise ValueError("Model registry đang trống, không có mô hình nào để suy luận!")

        if isinstance(flow_data, dict):
            flow_df = pd.DataFrame([flow_data])
        elif isinstance(flow_data, pd.Series):
            flow_df = pd.DataFrame([flow_data.to_dict()])
        elif isinstance(flow_data, pd.DataFrame):
            if flow_data.empty:
                raise ValueError("flow_data DataFrame rỗng!")
            flow_df = flow_data.iloc[[0]].copy()
        else:
            raise TypeError("flow_data phải là dict, Series hoặc DataFrame!")

        for col in NETFLOW_FEATURES:
            if col not in flow_df.columns:
                flow_df[col] = 0

        matched_keys: List[str] = []
        for key, info in sorted(self.models_registry.items()):
            if condition_filter != "all":
                cond = condition_filter.lower()
                if cond in ["nosmote", "without smote"] and info["condition"] != "nosmote":
                    continue
                if cond in ["smote", "with smote"] and info["condition"] != "smote":
                    continue

            if attack_filter != "all":
                if info["attack"].lower() != attack_filter.lower():
                    continue

            matched_keys.append(key)

        if not matched_keys:
            raise ValueError(
                f"Không tìm thấy mô hình nào khớp với bộ lọc (condition={condition_filter}, attack={attack_filter})!"
            )

        results_list: List[Dict[str, Any]] = []
        for key in matched_keys:
            info = self.models_registry[key]
            model = self.get_model(key)
            feats = info["selected_features"]

            for feat in feats:
                if feat not in flow_df.columns:
                    flow_df[feat] = 0

            X_sub, _ = self._coerce_numeric_features(flow_df, feats)

            if custom_threshold is not None:
                raw_thresh = float(custom_threshold)
            elif threshold_mode in info["thresholds"]:
                raw_thresh = float(info["thresholds"][threshold_mode])
            else:
                raw_thresh = float(info["default_f1_threshold"])

            near_low, thresh = get_near_miss_bounds(raw_thresh)

            t_start = time.perf_counter()
            X_input = self._format_model_input(model, X_sub)
            probas = model.predict_proba(X_input)[0]
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            attack_prob = float(probas[1]) if len(probas) > 1 else float(probas[0])

            if attack_prob >= thresh:
                risk_zone = RISK_ZONE_ATTACK
                pred_label = 1
                status_code = "ATTACK"
            elif attack_prob >= near_low:
                risk_zone = RISK_ZONE_SUSPICIOUS
                pred_label = 0
                status_code = "SUSPICIOUS"
            else:
                risk_zone = RISK_ZONE_BENIGN
                pred_label = 0
                status_code = "BENIGN"

            results_list.append({
                "Model_Key": key,
                "Attack": info["attack"],
                "Model_Family": info["model_family"],
                "Condition": info["condition_label"],
                "Probability": round(attack_prob, 4),
                "Threshold_Used": round(thresh, 4),
                "Near_Miss_Low": round(near_low, 4),
                "Threshold_Mode": threshold_mode if custom_threshold is None else f"Custom ({thresh:.2f})",
                "Prediction_Label": pred_label,
                "Risk_Zone": risk_zone,
                "Status_Code": status_code,
                "Latency_ms": round(latency_ms, 3),
                "Selected_Features_Count": len(feats),
                "Features_Used": ";".join(feats),
            })

        results_df = pd.DataFrame(results_list)

        attack_records = [r for r in results_list if r["Status_Code"] == "ATTACK"]
        suspicious_records = [r for r in results_list if r["Status_Code"] == "SUSPICIOUS"]

        if attack_records:
            top_rec = max(attack_records, key=lambda x: x["Probability"])
            verdict: VerdictDict = {
                "status": "ATTACK",
                "tier": "Tier 1: Chặn Cứng (Inline Firewall Prevention)",
                "title": f"PHÁT HIỆN TẤN CÔNG {top_rec['Attack'].upper()}",
                "message": (
                    f"Mô hình {top_rec['Model_Family']} ({top_rec['Condition']}) phát hiện tấn công "
                    f"{top_rec['Attack']} với xác suất {top_rec['Probability']*100:.1f}% "
                    f"(vượt ngưỡng phát hiện {top_rec['Threshold_Used']*100:.1f}%). "
                    f"Hệ thống Tầng 1 kích hoạt ngắt kết nối (TCP Reset / Drop Packet) tức thì!"
                ),
                "badge_color": "danger",
                "max_probability": float(top_rec["Probability"]),
                "primary_attack": str(top_rec["Attack"]),
                "triggered_by": f"{top_rec['Model_Family']} ({top_rec['Condition']})",
                "near_miss_low": float(top_rec["Near_Miss_Low"]),
                "threshold_used": float(top_rec["Threshold_Used"]),
            }
        elif suspicious_records:
            top_rec = max(suspicious_records, key=lambda x: x["Probability"])
            verdict = {
                "status": "SUSPICIOUS",
                "tier": "Tier 2: Kiểm Tra Sâu DPI (Deep Packet Inspection)",
                "title": f"CẢNH BÁO LƯU LƯỢNG NGHI VẤN ({top_rec['Attack'].upper()})",
                "message": (
                    f"Lưu lượng có xác suất tấn công {top_rec['Probability']*100:.1f}%, "
                    f"rơi vào vùng Near-Miss [{top_rec['Near_Miss_Low']*100:.1f}% - {top_rec['Threshold_Used']*100:.1f}%) "
                    f"của mô hình {top_rec['Model_Family']}. "
                    f"Không đủ để chặn cứng ở Tầng 1, tự động chuyển tiếp sang Tầng 2 để bóc tách Payload DPI!"
                ),
                "badge_color": "warning",
                "max_probability": float(top_rec["Probability"]),
                "primary_attack": str(top_rec["Attack"]),
                "triggered_by": f"{top_rec['Model_Family']} ({top_rec['Condition']})",
                "near_miss_low": float(top_rec["Near_Miss_Low"]),
                "threshold_used": float(top_rec["Threshold_Used"]),
            }
        else:
            top_rec = max(results_list, key=lambda x: x["Probability"])
            max_prob = float(top_rec["Probability"])
            near_low_ref = float(top_rec["Near_Miss_Low"])
            verdict = {
                "status": "BENIGN",
                "tier": "Fast Path: Cho phép thông qua (Normal Forwarding)",
                "title": "LƯU LƯỢNG AN TOÀN (BENIGN)",
                "message": (
                    f"Tất cả các mô hình xác nhận lưu lượng an toàn bình thường "
                    f"(xác suất rủi ro cao nhất đạt {max_prob*100:.2f}% < ngưỡng nghi vấn {near_low_ref*100:.1f}%). "
                    f"Lưu lượng được cho phép đi qua Fast Path với độ trễ cực thấp."
                ),
                "badge_color": "success",
                "max_probability": max_prob,
                "primary_attack": "Benign",
                "triggered_by": "None",
                "near_miss_low": near_low_ref,
                "threshold_used": float(top_rec["Threshold_Used"]),
            }

        return {
            "verdict": verdict,
            "details": results_df,
        }

    def predict_batch_flows(
        self,
        df: pd.DataFrame,
        attack_name: str = "DoS",
        model_family: str = "RF",
        condition: str = "nosmote",
        threshold_mode: str = "F1-Optimal",
        custom_threshold: Optional[float] = None,
        strict: bool = True,
    ) -> Tuple[pd.DataFrame, BatchSummaryDict]:
        """
        Dự đoán hàng loạt (Batch Inference) trên một DataFrame nhiều dòng.
        Khi `strict=True` (mặc định), báo lỗi `KeyError`/`ValueError` tường minh
        nếu registry rỗng hoặc không tồn tại đúng cặp `(model_family, attack_name, condition)`.
        """
        if not self.models_registry:
            raise ValueError("Model registry đang trống, không có mô hình nào để suy luận!")

        if df is None or len(df) == 0:
            raise ValueError("DataFrame đầu vào rỗng (0 dòng), không thể suy luận hàng loạt!")

        attack_clean = (attack_name or "").strip()
        family_clean = (model_family or "").strip()
        condition_clean = (condition or "").strip()
        if not attack_clean or not family_clean or not condition_clean:
            raise ValueError("attack_name, model_family và condition không được để trống!")

        key = f"{family_clean.lower()}_{attack_clean.lower()}_{condition_clean.lower()}"
        if key not in self.models_registry:
            if strict:
                raise KeyError(
                    f"Không tìm thấy mô hình '{key}' trong registry "
                    f"(family={family_clean}, attack={attack_clean}, condition={condition_clean})!"
                )
            matched_key: Optional[str] = None
            for k, v in sorted(self.models_registry.items()):
                if v["attack"].lower() == attack_clean.lower() and v["condition"].lower() == condition_clean.lower():
                    matched_key = k
                    break
            if matched_key is None:
                for k, v in sorted(self.models_registry.items()):
                    if v["attack"].lower() == attack_clean.lower():
                        matched_key = k
                        break
            if matched_key is None:
                raise KeyError(f"Không có mô hình nào khớp bài toán tấn công '{attack_clean}' trong registry!")
            key = matched_key

        info = self.models_registry[key]
        model = self.get_model(key)
        feats = info["selected_features"]

        if custom_threshold is not None:
            raw_thresh = float(custom_threshold)
        elif threshold_mode in info["thresholds"]:
            raw_thresh = float(info["thresholds"][threshold_mode])
        else:
            raw_thresh = float(info["default_f1_threshold"])

        near_low, thresh = get_near_miss_bounds(raw_thresh)

        res_df = df.copy()
        missing_features = [col for col in NETFLOW_FEATURES if col not in res_df.columns]
        missing_model_features = [col for col in feats if col not in res_df.columns]
        for col in missing_model_features:
            res_df[col] = 0.0

        X_sub, coerced_columns = self._coerce_numeric_features(res_df, feats)

        t0 = time.perf_counter()
        X_input = self._format_model_input(model, X_sub)
        raw_probas = model.predict_proba(X_input)
        probas = raw_probas[:, 1] if raw_probas.shape[1] > 1 else raw_probas[:, 0]
        batch_duration_sec = time.perf_counter() - t0

        res_df["Risk_Score"] = np.round(probas, 4)
        res_df["Predicted_Label"] = (probas >= thresh).astype(int)

        attack_mask = probas >= thresh
        suspicious_mask = (probas >= near_low) & (probas < thresh)
        benign_mask = probas < near_low

        conditions = [attack_mask, suspicious_mask, benign_mask]
        choices = [RISK_ZONE_ATTACK, RISK_ZONE_SUSPICIOUS, RISK_ZONE_BENIGN]
        res_df["Risk_Zone"] = np.select(conditions, choices, default=RISK_ZONE_BENIGN)

        total_flows = len(res_df)
        benign_count = int(np.sum(benign_mask))
        suspicious_count = int(np.sum(suspicious_mask))
        attack_count = int(np.sum(attack_mask))

        summary: BatchSummaryDict = {
            "model_key": key,
            "model_used": f"{info['model_family']} - {info['attack']} ({info['condition_label']})",
            "target_attack": info["attack"],
            "threshold_used": round(thresh, 4),
            "near_miss_low": round(near_low, 4),
            "threshold_mode": threshold_mode if custom_threshold is None else f"Custom ({thresh:.2f})",
            "total_flows": total_flows,
            "benign_count": benign_count,
            "suspicious_count": suspicious_count,
            "attack_count": attack_count,
            "missing_features": missing_features,
            "missing_model_features": missing_model_features,
            "coerced_columns": coerced_columns,
            "batch_inference_time_sec": round(batch_duration_sec, 4),
            "throughput_flows_per_sec": round(total_flows / max(batch_duration_sec, 1e-6), 1),
        }

        if "Attack" in res_df.columns:
            attack_series = res_df["Attack"].fillna("Benign").astype(str)
            other_mask = ~attack_series.str.lower().isin(["benign", info["attack"].lower()])
            if other_mask.any():
                cross_total = int(other_mask.sum())
                cross_detected = int((res_df.loc[other_mask, "Predicted_Label"] == 1).sum())
                summary["cross_attack_total"] = cross_total
                summary["cross_attack_detected"] = cross_detected

            eval_mask = attack_series.str.lower().isin(["benign", info["attack"].lower()])
            y_true = (attack_series.loc[eval_mask].str.lower() == info["attack"].lower()).astype(int).to_numpy()
            y_pred = res_df.loc[eval_mask, "Predicted_Label"].to_numpy()
            total_eval = len(y_true)

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            acc = (tp + tn) / total_eval if total_eval > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            summary["metrics"] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "Evaluated_Samples": total_eval,
            }
        elif "Label" in res_df.columns:
            y_true = pd.to_numeric(res_df["Label"], errors="coerce").fillna(0).astype(int).to_numpy()
            y_pred = res_df["Predicted_Label"].to_numpy()

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            acc = (tp + tn) / total_flows if total_flows > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            summary["metrics"] = {
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "Evaluated_Samples": total_flows,
            }

        return res_df, summary

    def _get_heldout_test_df(self) -> pd.DataFrame:
        """Đọc và cache tập heldout_test.parquet (thread-safe, chỉ nạp cột cần thiết)."""
        with self._lock:
            if self._heldout_test_df is None:
                test_path = self.project_root / "data" / "processed" / "heldout_test.parquet"
                if not test_path.exists():
                    raise FileNotFoundError(f"Không tìm thấy tập dữ liệu {test_path}!")

                target_cols = NETFLOW_FEATURES + ["Attack"]
                try:
                    import pyarrow.parquet as pq

                    available_cols = set(pq.read_schema(test_path).names)
                    read_cols = [c for c in target_cols if c in available_cols]
                    if not read_cols:
                        read_cols = None
                except Exception:
                    read_cols = target_cols

                self._heldout_test_df = pd.read_parquet(test_path, columns=read_cols)
            return self._heldout_test_df

    def sample_batch_test_flows(
        self,
        attack_name: str = "DoS",
        total_samples: int = 500,
        attack_ratio: float = 0.10,
        include_other_attacks: bool = False,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """
        Rút mẫu kiểm thử Batch phân tầng từ heldout_test.parquet.
        """
        df_test = self._get_heldout_test_df()
        if df_test.empty:
            raise ValueError("Tập heldout_test.parquet rỗng!")

        if not include_other_attacks:
            attack_pool = df_test[df_test["Attack"].astype(str).str.lower() == attack_name.lower()]
            benign_pool = df_test[df_test["Attack"].astype(str).str.lower() == "benign"]

            if attack_pool.empty:
                raise ValueError(f"Không có mẫu nào thuộc loại tấn công '{attack_name}' trong tập Test!")

            n_attack = min(int(total_samples * attack_ratio), len(attack_pool))
            n_benign_target = max(0, total_samples - n_attack)
            if n_benign_target > 0 and benign_pool.empty:
                raise ValueError("Không có mẫu Benign nào trong tập Test để pha trộn!")
            n_benign = min(n_benign_target, len(benign_pool))

            parts: List[pd.DataFrame] = []
            if n_attack > 0:
                parts.append(attack_pool.sample(n=n_attack, random_state=random_seed))
            if n_benign > 0:
                parts.append(benign_pool.sample(n=n_benign, random_state=random_seed))
            if not parts:
                raise ValueError("Số lượng mẫu rút ra bằng 0!")

            batch_sample = pd.concat(parts).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            return batch_sample
        else:
            return df_test.sample(n=min(total_samples, len(df_test)), random_state=random_seed).reset_index(drop=True)

    def sample_real_test_flow(
        self,
        sample_type: str = "benign",
        attack_name: str = "DoS",
        random_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Rút 1 dòng thực tế từ heldout_test.parquet thuộc 1 trong các kịch bản:
          - 'benign', 'attack_typical', 'attack_stealth_nearmiss', 'random'.
        """
        df_test = self._get_heldout_test_df()
        if df_test.empty:
            raise ValueError("Tập heldout_test.parquet rỗng!")

        seed_val = random_seed if random_seed is not None else int(time.time_ns() % (2**31 - 1))
        rng = np.random.RandomState(seed_val)

        if sample_type == "random":
            sample_type = str(rng.choice(["benign", "attack_typical", "attack_stealth_nearmiss"]))

        if sample_type == "benign":
            benign_pool = df_test[df_test["Attack"].astype(str).str.lower() == "benign"]
            if benign_pool.empty:
                raise ValueError("Không có mẫu Benign nào trong tập Test!")
            idx = rng.choice(benign_pool.index)
            row = benign_pool.loc[idx]
            flow_dict = row[NETFLOW_FEATURES].to_dict()
            flow_dict["True_Attack"] = "Benign"
            flow_dict["Sample_Scenario"] = "Lưu lượng Sạch Thông thường (Benign Flow)"
            return flow_dict

        attack_pool = df_test[df_test["Attack"].astype(str).str.lower() == attack_name.lower()]
        if attack_pool.empty:
            raise ValueError(f"Không có mẫu nào thuộc loại tấn công '{attack_name}' trong tập Test!")

        if not self.models_registry:
            raise ValueError("Model registry đang trống, không thể phân loại mẫu tấn công!")

        rf_key = f"rf_{attack_name.lower()}_nosmote"
        if rf_key not in self.models_registry:
            found_key: Optional[str] = None
            for k, v in sorted(self.models_registry.items()):
                if v["attack"].lower() == attack_name.lower() and v["condition"] == "nosmote":
                    found_key = k
                    break
            if found_key is None:
                for k, v in sorted(self.models_registry.items()):
                    if v["attack"].lower() == attack_name.lower():
                        found_key = k
                        break
            if found_key is None:
                raise KeyError(f"Không tìm thấy mô hình nào cho '{attack_name}' trong registry!")
            rf_key = found_key

        model = self.get_model(rf_key)
        feats = self.models_registry[rf_key]["selected_features"]
        f1_opt_thresh = self.models_registry[rf_key]["default_f1_threshold"]

        if len(attack_pool) > 1000:
            sample_candidates = attack_pool.sample(n=1000, random_state=rng).copy()
        else:
            sample_candidates = attack_pool.copy()

        X_cand, _ = self._coerce_numeric_features(sample_candidates, feats)
        probas = model.predict_proba(self._format_model_input(model, X_cand))[:, 1]
        sample_candidates["prob"] = probas

        if sample_type == "attack_typical":
            high_prob = sample_candidates[sample_candidates["prob"] >= 0.80]
            if not high_prob.empty:
                chosen = high_prob.loc[rng.choice(high_prob.index)]
            else:
                top_candidates = sample_candidates.sort_values(by="prob", ascending=False).head(10)
                chosen = top_candidates.loc[rng.choice(top_candidates.index)]

            flow_dict = chosen[NETFLOW_FEATURES].to_dict()
            flow_dict["True_Attack"] = str(chosen["Attack"])
            flow_dict["Sample_Scenario"] = f"Tấn công Điển hình ({attack_name} - Proba: {chosen['prob']*100:.1f}%)"
            return flow_dict

        elif sample_type == "attack_stealth_nearmiss":
            cost_thresh = self.models_registry[rf_key]["thresholds"].get("Cost-Sensitive (c=10)", 0.40)

            gb_key = f"gb_{attack_name.lower()}_nosmote"
            if gb_key in self.models_registry:
                gb_model = self.get_model(gb_key)
                gb_feats = self.models_registry[gb_key]["selected_features"]
                gb_f1_th = self.models_registry[gb_key]["default_f1_threshold"]
                X_gb, _ = self._coerce_numeric_features(sample_candidates, gb_feats)
                sample_candidates["prob_gb"] = gb_model.predict_proba(
                    self._format_model_input(gb_model, X_gb)
                )[:, 1]

                both_near_miss = sample_candidates[
                    (sample_candidates["prob"] >= cost_thresh)
                    & (sample_candidates["prob"] < f1_opt_thresh)
                    & (sample_candidates["prob_gb"] < gb_f1_th)
                ]
                if not both_near_miss.empty:
                    chosen = both_near_miss.loc[rng.choice(both_near_miss.index)]
                else:
                    ideal_near_miss = sample_candidates[
                        (sample_candidates["prob"] >= cost_thresh) & (sample_candidates["prob"] < f1_opt_thresh)
                    ]
                    if not ideal_near_miss.empty:
                        chosen = ideal_near_miss.loc[rng.choice(ideal_near_miss.index)]
                    else:
                        below_thresh = sample_candidates[sample_candidates["prob"] < f1_opt_thresh]
                        if not below_thresh.empty:
                            top_below = below_thresh.sort_values(by="prob", ascending=False).head(10)
                            chosen = top_below.loc[rng.choice(top_below.index)]
                        else:
                            low_cand = sample_candidates.sort_values(by="prob", ascending=True).head(10)
                            chosen = low_cand.loc[rng.choice(low_cand.index)]
            else:
                ideal_near_miss = sample_candidates[
                    (sample_candidates["prob"] >= cost_thresh) & (sample_candidates["prob"] < f1_opt_thresh)
                ]
                if not ideal_near_miss.empty:
                    chosen = ideal_near_miss.loc[rng.choice(ideal_near_miss.index)]
                else:
                    below_thresh = sample_candidates[sample_candidates["prob"] < f1_opt_thresh]
                    if not below_thresh.empty:
                        top_below = below_thresh.sort_values(by="prob", ascending=False).head(10)
                        chosen = top_below.loc[rng.choice(top_below.index)]
                    else:
                        low_cand = sample_candidates.sort_values(by="prob", ascending=True).head(10)
                        chosen = low_cand.loc[rng.choice(low_cand.index)]

            flow_dict = chosen[NETFLOW_FEATURES].to_dict()
            flow_dict["True_Attack"] = str(chosen["Attack"])
            flow_dict["Sample_Scenario"] = (
                f"Tấn công Tinh vi Near-Miss ({attack_name} - Proba: {chosen['prob']*100:.1f}%, "
                f"bị lọt ở F1-Opt {f1_opt_thresh:.2f} nhưng bắt dính ở Cost-Sensitive {cost_thresh:.2f})"
            )
            return flow_dict

        else:
            raise ValueError(
                f"sample_type '{sample_type}' không hợp lệ! (chọn: 'benign', 'attack_typical', 'attack_stealth_nearmiss', 'random')"
            )


if __name__ == "__main__":
    print("=" * 80)
    print(" BẮT ĐẦU KIỂM THỬ LÕI SUY LUẬN NIDS (NIDSInferenceEngine)")
    print("=" * 80)

    print("\n[BƯỚC 1] Khởi tạo NIDSInferenceEngine và tự động quét mô hình...")
    engine = NIDSInferenceEngine(lazy_load=True)
    summary_df = engine.get_registered_models_summary()
    print(f"[+] Tổng số mô hình tìm thấy trong Registry: {len(summary_df)}")
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("[BƯỚC 2] Rút 3 mẫu thực tế từ heldout_test.parquet và chạy dự đoán luồng đơn lẻ:")
    print("=" * 80)

    for stype, att in [("benign", "DoS"), ("attack_typical", "DoS"), ("attack_stealth_nearmiss", "DoS")]:
        print(f"\n---> Kịch bản Mẫu: '{stype}' (Attack: {att})")
        sample_flow = engine.sample_real_test_flow(sample_type=stype, attack_name=att, random_seed=42)
        print(
            f"[*] Metadata mẫu: Scenario = '{sample_flow.get('Sample_Scenario')}', True_Attack = {sample_flow.get('True_Attack')}"
        )

        for t_mode in ["F1-Optimal", "Cost-Sensitive (c=10)"]:
            pred = engine.predict_single_flow(
                flow_data=sample_flow,
                threshold_mode=t_mode,
                condition_filter="nosmote",
                attack_filter=att,
            )
            verdict = pred["verdict"]
            det = pred["details"]
            print(f"  [Ngưỡng: {t_mode:<22}] -> KẾT LUẬN: {verdict['status']:<10} | {verdict['tier']}")
            for _, r in det.iterrows():
                print(
                    f"      + {r['Model_Family']} ({r['Condition']}): Xác suất={r['Probability']*100:5.1f}% "
                    f"| Ngưỡng={r['Threshold_Used']:.3f} | NearMissLow={r['Near_Miss_Low']:.4f} | Vùng: {r['Risk_Zone']}"
                )

    print("\n" + "=" * 80)
    print("[BƯỚC 3] Chạy Dự đoán Hàng loạt (predict_batch_flows) trên 500 mẫu test chuẩn One-vs-Benign:")
    print("=" * 80)
    sample_batch = engine.sample_batch_test_flows(
        attack_name="DoS",
        total_samples=500,
        attack_ratio=0.10,
        include_other_attacks=False,
        random_seed=42,
    )

    batch_res, batch_summary = engine.predict_batch_flows(
        df=sample_batch,
        attack_name="DoS",
        model_family="RF",
        condition="nosmote",
        threshold_mode="F1-Optimal",
    )

    print(f"[+] Mô hình sử dụng: {batch_summary['model_used']}")
    print(f"[+] Ngưỡng áp dụng: {batch_summary['threshold_used']} ({batch_summary['threshold_mode']})")
    print(f"[+] Near-Miss bounds: [{batch_summary['near_miss_low']}, {batch_summary['threshold_used']})")
    print(f"[+] Tổng số luồng xử lý: {batch_summary['total_flows']} luồng")
    print(f"    - Benign (<{batch_summary['near_miss_low']*100:.1f}%): {batch_summary['benign_count']} luồng")
    print(f"    - Suspicious:       {batch_summary['suspicious_count']} luồng")
    print(f"    - Attack (>=th):    {batch_summary['attack_count']} luồng")
    print(
        f"[+] Thời gian suy luận: {batch_summary['batch_inference_time_sec']}s "
        f"(Throughput: {batch_summary['throughput_flows_per_sec']:,} luồng/giây)"
    )

    if "metrics" in batch_summary:
        m = batch_summary["metrics"]
        print(
            f"[+] Đánh giá thực nghiệm: Accuracy={m['Accuracy']*100:.2f}%, Precision={m['Precision']*100:.2f}%, "
            f"Recall={m['Recall']*100:.2f}%, F1={m['F1']*100:.2f}% (TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']})"
        )

    print("\n" + "=" * 80)
    print(" HOÀN TẤT KIỂM THỬ LÕI SUY LUẬN THÀNH CÔNG 100%!")
    print("=" * 80)
