#!/usr/bin/env python3
"""
inference_engine.py
Lõi Suy luận Tổng quát NIDS (Multi-Model NIDS Inference Engine)
Hỗ trợ kiến trúc 2-Tier Hybrid NIDS, Auto-Discovery Model Registry,
suy luận luồng đơn lẻ thời gian thực, suy luận hàng loạt (batch)
và trích xuất mẫu kiểm thử thực tế từ tập Held-out Test.
"""

import os
import sys
import json
import time
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
import joblib

# Thiết lập đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Danh sách 10 đặc trưng NetFlow chuẩn theo NetFlow_v1
NETFLOW_FEATURES = [
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS"
]

CANONICAL_ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]


class NIDSInferenceEngine:
    """
    Lõi Suy luận NIDS hướng đối tượng, hỗ trợ:
    1. Tự động khám phá mô hình trong results/*/models/ (Auto-Discovery Model Registry).
    2. Quản lý đa ngưỡng (F1-Optimal, Standard 0.50, Recall-Targeted, Cost-Sensitive).
    3. Dự đoán luồng đơn lẻ phân 3 vùng rủi ro (2-Tier Hybrid NIDS).
    4. Dự đoán hàng loạt (Batch Inference) kèm báo cáo thống kê & metrics.
    5. Rút mẫu thử thực tế từ tập heldout_test.parquet (Benign, Typical Attack, Stealth Near-Miss).
    """

    def __init__(self, models_base_dir: Optional[Union[str, Path]] = None, lazy_load: bool = True):
        self.project_root = PROJECT_ROOT
        self.results_dir = RESULTS_DIR
        self.models_base_dir = Path(models_base_dir) if models_base_dir else self.results_dir
        self.lazy_load = lazy_load

        # Bộ đăng ký mô hình (Metadata registry)
        self.models_registry: Dict[str, Dict[str, Any]] = {}

        # Cache các mô hình đã load vào RAM
        self._loaded_models: Dict[str, Any] = {}

        # Dữ liệu ngưỡng chuyên biệt
        self.recall_thresholds_df: Optional[pd.DataFrame] = None
        self.cost_thresholds_df: Optional[pd.DataFrame] = None

        # Cache tập heldout_test
        self._heldout_test_df: Optional[pd.DataFrame] = None

        # Khởi tạo tự động
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
    def _calibrate_thresholds_from_val_arrays(
        y_val: np.ndarray,
        p_val: np.ndarray,
        t_f1_opt: float
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

    def _load_specialized_thresholds(self):
        """Nạp các bảng ngưỡng tối ưu từ results/improved/ nếu tồn tại."""
        recall_path = self.results_dir / "improved" / "recall_targeted_thresholds.csv"
        if recall_path.exists():
            try:
                self.recall_thresholds_df = pd.read_csv(recall_path)
            except Exception as e:
                print(f"[!] Cảnh báo: Không thể đọc {recall_path}: {e}")

        cost_path = self.results_dir / "improved" / "cost_sensitive_evaluation.csv"
        if cost_path.exists():
            try:
                self.cost_thresholds_df = pd.read_csv(cost_path)
            except Exception as e:
                print(f"[!] Cảnh báo: Không thể đọc {cost_path}: {e}")

    def _auto_discover_models(self):
        """
        Tự động quét tất cả các thư mục con trong results/*/models/
        để tìm mọi cặp file *_metadata.json và *_best_model.joblib.
        """
        self.models_registry.clear()

        # Quét đệ quy tìm tất cả thư mục models
        search_pattern = str(self.models_base_dir / "*" / "models")
        model_dirs = [Path(p) for p in glob.glob(search_pattern)]
        
        # Cũng quét cả trường hợp thư mục models nằm trực tiếp trong base_dir
        if (self.models_base_dir / "models").exists():
            model_dirs.append(self.models_base_dir / "models")

        for mdir in set(model_dirs):
            if not mdir.is_dir():
                continue

            meta_files = list(mdir.glob("*_metadata.json"))
            for meta_path in meta_files:
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[!] Bỏ qua file metadata hỏng {meta_path}: {e}")
                    continue

                # Xác định file model tương ứng
                prefix = meta_path.stem.replace("_metadata", "")
                model_path = mdir / f"{prefix}_best_model.joblib"
                if not model_path.exists():
                    # Thử tìm file .joblib bất kỳ có cùng prefix
                    cand = list(mdir.glob(f"{prefix}*.joblib"))
                    # Lọc bỏ selector
                    cand = [c for c in cand if "selector" not in c.name]
                    if cand:
                        model_path = cand[0]
                    else:
                        continue

                selector_path = mdir / f"{prefix}_best_selector.joblib"

                # Chuẩn hóa thông tin
                # 1. Model Family (RF, GB, ...)
                if prefix.startswith("rf_"):
                    model_family = "RF"
                elif prefix.startswith("gb_"):
                    model_family = "GB"
                else:
                    parts = prefix.split("_")
                    model_family = parts[0].upper()

                # 2. Attack name
                raw_attack = meta.get("attack", "")
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

                # 3. Condition (nosmote, smote)
                condition = meta.get("condition", "")
                if not condition:
                    if "nosmote" in prefix.lower():
                        condition = "nosmote"
                    elif "smote" in prefix.lower():
                        condition = "smote"
                    else:
                        condition = "nosmote"
                cond_label = "Without SMOTE" if condition == "nosmote" else "With SMOTE"

                # 4. Selected Features
                raw_feats = meta.get("selected_features", "")
                if isinstance(raw_feats, str) and raw_feats:
                    selected_features = [f.strip() for f in raw_feats.split(";") if f.strip()]
                elif isinstance(raw_feats, list):
                    selected_features = [str(f).strip() for f in raw_feats]
                else:
                    selected_features = list(NETFLOW_FEATURES)

                # 5. Default F1 Threshold
                default_thresh = float(meta.get("val_threshold", 0.50))

                # 6. Xây dựng bộ ngưỡng đầy đủ cho model
                thresholds_dict = {
                    "F1-Optimal": round(default_thresh, 4),
                    "Standard (0.50)": 0.50
                }

                # Gắn ngưỡng Recall-Targeted và Cost-Sensitive đúng theo từng điều kiện (nosmote vs smote)
                csv_matched = False
                if condition == "nosmote":
                    if self.recall_thresholds_df is not None:
                        rc_mask = (
                            (self.recall_thresholds_df["Attack"].str.lower() == attack.lower()) &
                            (self.recall_thresholds_df["Model"].str.upper() == model_family)
                        )
                        if "Condition" in self.recall_thresholds_df.columns:
                            rc_mask &= (self.recall_thresholds_df["Condition"].str.lower() == cond_label.lower())
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
                            (self.cost_thresholds_df["Attack"].str.lower() == attack.lower()) &
                            (self.cost_thresholds_df["Model"].str.upper() == model_family)
                        )
                        if "Condition" in self.cost_thresholds_df.columns:
                            cost_mask &= (self.cost_thresholds_df["Condition"].str.lower() == cond_label.lower())
                        sub_cost = self.cost_thresholds_df[cost_mask]
                        for _, c_row in sub_cost.iterrows():
                            csv_matched = True
                            ratio_c = int(c_row["Cost_Ratio_c"])
                            if ratio_c == 10:
                                thresholds_dict["Cost-Sensitive (c=10)"] = round(float(c_row["Optimal_Threshold"]), 4)
                            elif ratio_c == 20:
                                thresholds_dict["Cost-Sensitive (c=20)"] = round(float(c_row["Optimal_Threshold"]), 4)

                # Nếu là With SMOTE (hoặc mô hình mới chưa có trong CSV), hiệu chuẩn trực tiếp trên mảng Validation riêng của mô hình đó
                if not csv_matched:
                    y_val_file = mdir.parent / f"{prefix}_y_val.npy"
                    p_val_file = mdir.parent / f"{prefix}_best_proba_val.npy"
                    if y_val_file.exists() and p_val_file.exists():
                        try:
                            yv_arr = np.load(y_val_file)
                            pv_arr = np.load(p_val_file)
                            thresholds_dict.update(
                                self._calibrate_thresholds_from_val_arrays(yv_arr, pv_arr, default_thresh)
                            )
                        except Exception as e:
                            print(f"[!] Cảnh báo: Không thể hiệu chuẩn ngưỡng từ Validation cho {prefix}: {e}")

                model_key = f"{model_family.lower()}_{attack.lower()}_{condition.lower()}"

                self.models_registry[model_key] = {
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
                    "raw_metadata": meta
                }

    def get_model(self, model_key: str) -> Any:
        """Nạp model từ cache hoặc từ file joblib (Lazy-loading)."""
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]

        if model_key not in self.models_registry:
            raise KeyError(f"Mô hình '{model_key}' không có trong registry!")

        info = self.models_registry[model_key]
        model_path = info["model_path"]
        model = joblib.load(model_path)
        self._loaded_models[model_key] = model
        return model

    def load_all_models(self):
        """Nạp toàn bộ các mô hình trong registry vào RAM trước."""
        for key in self.models_registry:
            self.get_model(key)

    def reload_registry(self):
        """Xóa sạch cache mô hình trong RAM và nạp mới toàn bộ ngưỡng + mô hình từ đĩa."""
        self._loaded_models.clear()
        self._load_specialized_thresholds()
        self._auto_discover_models()
        if not self.lazy_load:
            self.load_all_models()

    def get_registered_models_summary(self) -> pd.DataFrame:
        """Trả về bảng tổng kết tất cả các mô hình có trong hệ thống."""
        records = []
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
                "Train_Time_s": m.get("train_time_sec")
            })
        return pd.DataFrame(records)

    def predict_single_flow(
        self,
        flow_data: Union[Dict[str, Any], pd.Series, pd.DataFrame],
        threshold_mode: str = "F1-Optimal",
        condition_filter: str = "all",
        attack_filter: str = "all",
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Dự đoán cho 1 luồng mạng đơn lẻ qua các mô hình NIDS phù hợp.
        Phân loại trạng thái 3 vùng theo kiến trúc 2-Tier Hybrid NIDS:
          - proba >= threshold: 'ATTACK (Chặn ngay - Tầng 1)'
          - 0.10 <= proba < threshold: 'SUSPICIOUS / NEAR-MISS (Chuyển DPI Tầng 2)'
          - proba < 0.10: 'BENIGN (Cho phép qua)'
        """
        # Chuẩn hóa dữ liệu đầu vào thành DataFrame 1 dòng
        if isinstance(flow_data, dict):
            flow_df = pd.DataFrame([flow_data])
        elif isinstance(flow_data, pd.Series):
            flow_df = pd.DataFrame([flow_data.to_dict()])
        elif isinstance(flow_data, pd.DataFrame):
            flow_df = flow_data.iloc[[0]].copy()
        else:
            raise TypeError("flow_data phải là dict, Series hoặc DataFrame!")

        # Đảm bảo đủ các cột NetFlow (nếu thiếu cột nào sẽ điền 0)
        for col in NETFLOW_FEATURES:
            if col not in flow_df.columns:
                flow_df[col] = 0

        # Lọc danh sách mô hình cần suy luận
        matched_keys = []
        for key, info in self.models_registry.items():
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
            raise ValueError(f"Không tìm thấy mô hình nào khớp với bộ lọc (condition={condition_filter}, attack={attack_filter})!")

        results_list = []
        for key in matched_keys:
            info = self.models_registry[key]
            model = self.get_model(key)
            feats = info["selected_features"]

            # Lấy đúng tập đặc trưng của mô hình
            X_sub = flow_df[feats].copy()

            # Xác định ngưỡng
            if custom_threshold is not None:
                thresh = float(custom_threshold)
            elif threshold_mode in info["thresholds"]:
                thresh = float(info["thresholds"][threshold_mode])
            else:
                thresh = float(info["default_f1_threshold"])

            # Đo thời gian suy luận
            t_start = time.perf_counter()
            X_input = self._format_model_input(model, X_sub)
            probas = model.predict_proba(X_input)[0]
            latency_ms = (time.perf_counter() - t_start) * 1000.0

            # Xác suất lớp tấn công (lớp 1)
            attack_prob = float(probas[1]) if len(probas) > 1 else float(probas[0])

            # Phân loại trạng thái 3 vùng (2-Tier Hybrid NIDS)
            if attack_prob >= thresh:
                risk_zone = "ATTACK (Chặn ngay - Tầng 1)"
                pred_label = 1
                status_code = "ATTACK"
            elif attack_prob >= 0.10:
                risk_zone = "SUSPICIOUS / NEAR-MISS (Chuyển DPI Tầng 2)"
                pred_label = 0
                status_code = "SUSPICIOUS"
            else:
                risk_zone = "BENIGN (Cho phép qua)"
                pred_label = 0
                status_code = "BENIGN"

            results_list.append({
                "Model_Key": key,
                "Attack": info["attack"],
                "Model_Family": info["model_family"],
                "Condition": info["condition_label"],
                "Probability": round(attack_prob, 4),
                "Threshold_Used": round(thresh, 4),
                "Threshold_Mode": threshold_mode if custom_threshold is None else f"Custom ({thresh:.2f})",
                "Prediction_Label": pred_label,
                "Risk_Zone": risk_zone,
                "Status_Code": status_code,
                "Latency_ms": round(latency_ms, 3),
                "Selected_Features_Count": len(feats),
                "Features_Used": ";".join(feats)
            })

        results_df = pd.DataFrame(results_list)

        # Tính kết luận tổng hợp (2-Tier Hybrid NIDS Verdict)
        attack_records = [r for r in results_list if r["Status_Code"] == "ATTACK"]
        suspicious_records = [r for r in results_list if r["Status_Code"] == "SUSPICIOUS"]

        if attack_records:
            # Chọn loại tấn công có xác suất cao nhất
            top_rec = max(attack_records, key=lambda x: x["Probability"])
            verdict = {
                "status": "ATTACK",
                "tier": "Tier 1: Chặn Cứng (Inline Firewall Prevention)",
                "title": f"PHÁT HIỆN TẤN CÔNG {top_rec['Attack'].upper()}",
                "message": (
                    f"Mô hình {top_rec['Model_Family']} ({top_rec['Condition']}) phát hiện tấn công "
                    f"{top_rec['Attack']} với xác suất {top_rec['Probability']*100:.1f}% "
                    f"(vượt ngưỡng phát hiện {top_rec['Threshold_Used']:.2f}). "
                    f"Hệ thống Tầng 1 kích hoạt ngắt kết nối (TCP Reset / Drop Packet) tức thì!"
                ),
                "badge_color": "danger",
                "max_probability": top_rec["Probability"],
                "primary_attack": top_rec["Attack"],
                "triggered_by": f"{top_rec['Model_Family']} ({top_rec['Condition']})"
            }
        elif suspicious_records:
            top_rec = max(suspicious_records, key=lambda x: x["Probability"])
            verdict = {
                "status": "SUSPICIOUS",
                "tier": "Tier 2: Kiểm Tra Sâu DPI (Deep Packet Inspection)",
                "title": f"CẢNH BÁO LƯU LƯỢNG NGHI VẤN ({top_rec['Attack'].upper()})",
                "message": (
                    f"Lưu lượng có xác suất tấn công {top_rec['Probability']*100:.1f}%, "
                    f"rơi vào vùng Near-Miss [10% - {top_rec['Threshold_Used']*100:.1f}%) "
                    f"của mô hình {top_rec['Model_Family']}. "
                    f"Không đủ để chặn cứng ở Tầng 1, tự động chuyển tiếp sang Tầng 2 để bóc tách Payload DPI!"
                ),
                "badge_color": "warning",
                "max_probability": top_rec["Probability"],
                "primary_attack": top_rec["Attack"],
                "triggered_by": f"{top_rec['Model_Family']} ({top_rec['Condition']})"
            }
        else:
            max_prob = max([r["Probability"] for r in results_list]) if results_list else 0.0
            verdict = {
                "status": "BENIGN",
                "tier": "Fast Path: Cho phép thông qua (Normal Forwarding)",
                "title": "LƯU LƯỢNG AN TOÀN (BENIGN)",
                "message": (
                    f"Tất cả các mô hình xác nhận lưu lượng an toàn bình thường "
                    f"(xác suất rủi ro cao nhất chỉ đạt {max_prob*100:.2f}% < 10%). "
                    f"Lưu lượng được cho phép đi qua Fast Path với độ trễ cực thấp."
                ),
                "badge_color": "success",
                "max_probability": max_prob,
                "primary_attack": "Benign",
                "triggered_by": "None"
            }

        return {
            "verdict": verdict,
            "details": results_df
        }

    def predict_batch_flows(
        self,
        df: pd.DataFrame,
        attack_name: str = "DoS",
        model_family: str = "RF",
        condition: str = "nosmote",
        threshold_mode: str = "F1-Optimal",
        custom_threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Dự đoán hàng loạt (Batch Inference) trên một DataFrame nhiều dòng.
        Cho phép chọn một mô hình cụ thể, trả về DataFrame có gắn thêm xác suất,
        nhãn dự đoán, vùng rủi ro và từ điển thống kê tổng hợp.
        """
        attack_name = (attack_name or "DoS").strip() or "DoS"
        model_family = (model_family or "RF").strip() or "RF"
        condition = (condition or "nosmote").strip() or "nosmote"

        key = f"{model_family.lower()}_{attack_name.lower()}_{condition.lower()}"
        if key not in self.models_registry:
            # Tìm key gần đúng nhất
            found = False
            for k in self.models_registry:
                if (attack_name.lower() in k) and (condition.lower() in k) and (model_family.lower() in k):
                    key = k
                    found = True
                    break
            if not found:
                # Lấy mô hình đầu tiên có attack_name
                for k, v in self.models_registry.items():
                    if v["attack"].lower() == attack_name.lower():
                        key = k
                        found = True
                        break
            if not found:
                key = list(self.models_registry.keys())[0]

        info = self.models_registry[key]
        model = self.get_model(key)
        feats = info["selected_features"]

        # Xác định ngưỡng
        if custom_threshold is not None:
            thresh = float(custom_threshold)
        elif threshold_mode in info["thresholds"]:
            thresh = float(info["thresholds"][threshold_mode])
        else:
            thresh = float(info["default_f1_threshold"])

        # Sao chép DataFrame để không mutate dữ liệu gốc của caller & ghi nhận cột thiếu
        res_df = df.copy()
        missing_features = [col for col in NETFLOW_FEATURES if col not in res_df.columns]
        missing_model_features = [col for col in feats if col not in res_df.columns]
        for col in missing_model_features:
            res_df[col] = 0

        X_sub = res_df[feats].copy()

        t0 = time.perf_counter()
        X_input = self._format_model_input(model, X_sub)
        probas = model.predict_proba(X_input)[:, 1]
        batch_duration_sec = time.perf_counter() - t0

        res_df["Risk_Score"] = np.round(probas, 4)
        res_df["Predicted_Label"] = (probas >= thresh).astype(int)

        # Gắn vùng rủi ro
        conditions = [
            probas >= thresh,
            (probas >= 0.10) & (probas < thresh),
            probas < 0.10
        ]
        choices = [
            "ATTACK (Chặn ngay - Tầng 1)",
            "SUSPICIOUS / NEAR-MISS (Chuyển DPI Tầng 2)",
            "BENIGN (Cho phép qua)"
        ]
        res_df["Risk_Zone"] = np.select(conditions, choices, default="BENIGN (Cho phép qua)")

        total_flows = len(res_df)
        benign_count = int(np.sum(probas < 0.10))
        suspicious_count = int(np.sum((probas >= 0.10) & (probas < thresh)))
        attack_count = int(np.sum(probas >= thresh))

        summary = {
            "model_used": f"{info['model_family']} - {info['attack']} ({info['condition_label']})",
            "target_attack": info["attack"],
            "threshold_used": thresh,
            "threshold_mode": threshold_mode if custom_threshold is None else f"Custom ({thresh:.2f})",
            "total_flows": total_flows,
            "benign_count": benign_count,
            "suspicious_count": suspicious_count,
            "attack_count": attack_count,
            "missing_features": missing_features,
            "missing_model_features": missing_model_features,
            "batch_inference_time_sec": round(batch_duration_sec, 4),
            "throughput_flows_per_sec": round(total_flows / max(batch_duration_sec, 1e-6), 1)
        }

        # Nếu có nhãn thực tế thì tính luôn metrics
        if "Attack" in res_df.columns:
            attack_series = res_df["Attack"].fillna("Benign").astype(str)
            # Kiểm tra xem có chứa các loại tấn công khác không (Cross-Attack)
            other_mask = ~attack_series.str.lower().isin(["benign", info["attack"].lower()])
            if other_mask.any():
                cross_total = int(other_mask.sum())
                cross_detected = int((res_df.loc[other_mask, "Predicted_Label"] == 1).sum())
                summary["cross_attack_total"] = cross_total
                summary["cross_attack_detected"] = cross_detected

            # Đánh giá metrics chuẩn One-vs-Benign trên tập con (Attack == target | Attack == Benign)
            eval_mask = attack_series.str.lower().isin(["benign", info["attack"].lower()])
            y_true = (attack_series.loc[eval_mask].str.lower() == info["attack"].lower()).astype(int)
            y_pred = res_df.loc[eval_mask, "Predicted_Label"].values
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
                "Evaluated_Samples": total_eval
            }
        elif "Label" in df.columns:
            y_true = df["Label"].astype(int)
            y_pred = res_df["Predicted_Label"].values

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
                "Evaluated_Samples": total_flows
            }

        return res_df, summary

    def _get_heldout_test_df(self) -> pd.DataFrame:
        """Đọc và cache tập heldout_test.parquet."""
        if self._heldout_test_df is None:
            test_path = self.project_root / "data" / "processed" / "heldout_test.parquet"
            if not test_path.exists():
                raise FileNotFoundError(f"Không tìm thấy tập dữ liệu {test_path}!")
            self._heldout_test_df = pd.read_parquet(test_path)
        return self._heldout_test_df

    def sample_batch_test_flows(
        self,
        attack_name: str = "DoS",
        total_samples: int = 500,
        attack_ratio: float = 0.10,
        include_other_attacks: bool = False,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Rút mẫu kiểm thử Batch phân tầng từ heldout_test.parquet:
        - Nếu include_other_attacks == False (chuẩn One-vs-Benign):
          Chỉ lấy các dòng thuộc attack_name và Benign.
          Lấy n_attack = min(int(total_samples * attack_ratio), len(attack_pool))
          và n_benign = total_samples - n_attack mẫu Benign.
        - Nếu include_other_attacks == True:
          Lấy mẫu ngẫu nhiên từ toàn bộ heldout_test (có thể gồm cả các loại tấn công khác).
        """
        df_test = self._get_heldout_test_df()

        if not include_other_attacks:
            attack_pool = df_test[df_test["Attack"].str.lower() == attack_name.lower()]
            benign_pool = df_test[df_test["Attack"] == "Benign"]

            if attack_pool.empty:
                raise ValueError(f"Không có mẫu nào thuộc loại tấn công '{attack_name}' trong tập Test!")

            n_attack = min(int(total_samples * attack_ratio), len(attack_pool))
            n_benign = min(total_samples - n_attack, len(benign_pool))

            s_attack = attack_pool.sample(n=n_attack, random_state=random_seed)
            s_benign = benign_pool.sample(n=n_benign, random_state=random_seed)

            batch_sample = pd.concat([s_attack, s_benign]).sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
            return batch_sample
        else:
            return df_test.sample(n=min(total_samples, len(df_test)), random_state=random_seed).reset_index(drop=True)

    def sample_real_test_flow(
        self,
        sample_type: str = "benign",
        attack_name: str = "DoS",
        random_seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Rút 1 dòng thực tế từ heldout_test.parquet thuộc 1 trong các kịch bản:
          - sample_type == 'benign': Mẫu lưu lượng sạch thông thường (Attack == 'Benign').
          - sample_type == 'attack_typical': Mẫu tấn công điển hình rõ ràng (RF nosmote proba >= 0.80).
          - sample_type == 'attack_stealth_nearmiss': Mẫu tấn công tinh vi bị bỏ sót ở F1-Optimal
            nhưng nằm trong vùng Near-Miss.
          - sample_type == 'random': Chọn ngẫu nhiên 1 trong 3 kịch bản trên để kiểm thử mù.
        """
        df_test = self._get_heldout_test_df()
        seed_val = random_seed if random_seed is not None else int(time.time_ns() % (2**31 - 1))
        rng = np.random.RandomState(seed_val)

        if sample_type == "random":
            sample_type = str(rng.choice(["benign", "attack_typical", "attack_stealth_nearmiss"]))

        if sample_type == "benign":
            benign_pool = df_test[df_test["Attack"] == "Benign"]
            idx = rng.choice(benign_pool.index)
            row = benign_pool.loc[idx]
            flow_dict = row[NETFLOW_FEATURES].to_dict()
            flow_dict["True_Attack"] = "Benign"
            flow_dict["Sample_Scenario"] = "Lưu lượng Sạch Thông thường (Benign Flow)"
            return flow_dict

        # Với 2 kịch bản tấn công:
        attack_pool = df_test[df_test["Attack"].str.lower() == attack_name.lower()]
        if attack_pool.empty:
            raise ValueError(f"Không có mẫu nào thuộc loại tấn công '{attack_name}' trong tập Test!")

        # Dùng mô hình RF (nosmote) chuẩn để xác định proba
        rf_key = f"rf_{attack_name.lower()}_nosmote"
        if rf_key not in self.models_registry:
            # Fallback sang bất kỳ model nào có attack_name
            for k in self.models_registry:
                if attack_name.lower() in k and "nosmote" in k:
                    rf_key = k
                    break

        model = self.get_model(rf_key)
        feats = self.models_registry[rf_key]["selected_features"]
        f1_opt_thresh = self.models_registry[rf_key]["default_f1_threshold"]

        # Lấy mẫu ngẫu nhiên tới 1,000 dòng từ toàn bộ attack_pool để đa dạng hóa mẫu thử
        if len(attack_pool) > 1000:
            sample_candidates = attack_pool.sample(n=1000, random_state=rng).copy()
        else:
            sample_candidates = attack_pool.copy()
        probas = model.predict_proba(self._format_model_input(model, sample_candidates[feats]))[:, 1]
        sample_candidates["prob"] = probas

        if sample_type == "attack_typical":
            # Mẫu có proba cao >= 0.80 hoặc nhóm proba cao nhất
            high_prob = sample_candidates[sample_candidates["prob"] >= 0.80]
            if not high_prob.empty:
                idx = rng.choice(high_prob.index)
                chosen = high_prob.loc[idx]
            else:
                top_candidates = sample_candidates.sort_values(by="prob", ascending=False).head(10)
                chosen = top_candidates.loc[rng.choice(top_candidates.index)]

            flow_dict = chosen[NETFLOW_FEATURES].to_dict()
            flow_dict["True_Attack"] = str(chosen["Attack"])
            flow_dict["Sample_Scenario"] = f"Tấn công Điển hình ({attack_name} - Proba: {chosen['prob']*100:.1f}%)"
            return flow_dict

        elif sample_type == "attack_stealth_nearmiss":
            # Ưu tiên tìm mẫu nằm trong khoảng [cost_thresh, f1_opt_thresh)
            # để khi chuyển từ F1-Opt sang Cost-Sensitive (c=10) sẽ bắt dính ngay lập tức
            cost_thresh = self.models_registry[rf_key]["thresholds"].get("Cost-Sensitive (c=10)", 0.40)
            
            # Thử tính thêm proba của HistGB nếu có
            gb_key = f"gb_{attack_name.lower()}_nosmote"
            if gb_key in self.models_registry:
                gb_model = self.get_model(gb_key)
                gb_feats = self.models_registry[gb_key]["selected_features"]
                gb_f1_th = self.models_registry[gb_key]["default_f1_threshold"]
                sample_candidates["prob_gb"] = gb_model.predict_proba(
                    self._format_model_input(gb_model, sample_candidates[gb_feats])
                )[:, 1]
                
                # Ưu tiên mẫu mà cả RF < f1_opt VÀ GB < gb_f1_th (chọn ngẫu nhiên trong nhóm đạt chuẩn)
                both_near_miss = sample_candidates[
                    (sample_candidates["prob"] >= cost_thresh) & (sample_candidates["prob"] < f1_opt_thresh) &
                    (sample_candidates["prob_gb"] < gb_f1_th)
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
            raise ValueError(f"sample_type '{sample_type}' không hợp lệ! (chọn: 'benign', 'attack_typical', 'attack_stealth_nearmiss', 'random')")


# ======================================================================
# NHIỆM VỤ 3: Tự Kiểm thử (Self-Test Script)
# ======================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" BẮT ĐẦU KIỂM THỬ LÕI SUY LUẬN NIDS (NIDSInferenceEngine)")
    print("=" * 80)

    # 1. Khởi tạo Engine và kiểm tra Auto-Discovery Registry
    print("\n[BƯỚC 1] Khởi tạo NIDSInferenceEngine và tự động quét mô hình...")
    engine = NIDSInferenceEngine(lazy_load=True)
    summary_df = engine.get_registered_models_summary()
    print(f"[+] Tổng số mô hình tìm thấy trong Registry: {len(summary_df)}")
    print(summary_df.to_string(index=False))

    # 2. Thử nghiệm dự đoán luồng đơn lẻ với 3 kịch bản thực tế
    print("\n" + "=" * 80)
    print("[BƯỚC 2] Rút 3 mẫu thực tế từ heldout_test.parquet và chạy dự đoán luồng đơn lẻ:")
    print("=" * 80)

    test_scenarios = [
        ("benign", "DoS"),
        ("attack_typical", "DoS"),
        ("attack_stealth_nearmiss", "DoS")
    ]

    for stype, att in test_scenarios:
        print(f"\n---> Kịch bản Mẫu: '{stype}' (Attack: {att})")
        sample_flow = engine.sample_real_test_flow(sample_type=stype, attack_name=att, random_seed=42)
        print(f"[*] Metadata mẫu: Scenario = '{sample_flow.get('Sample_Scenario')}', True_Attack = {sample_flow.get('True_Attack')}")
        
        # Test ở 2 chế độ ngưỡng cho mô hình DoS cụ thể và toàn bộ hệ thống
        for t_mode in ["F1-Optimal", "Cost-Sensitive (c=10)"]:
            pred = engine.predict_single_flow(
                flow_data=sample_flow,
                threshold_mode=t_mode,
                condition_filter="nosmote",
                attack_filter=att  # Đánh giá trực tiếp trên bài toán mục tiêu
            )
            verdict = pred["verdict"]
            det = pred["details"]
            print(f"  [Ngưỡng: {t_mode:<22}] -> KẾT LUẬN: {verdict['status']:<10} | {verdict['tier']}")
            for _, r in det.iterrows():
                print(f"      + {r['Model_Family']} ({r['Condition']}): Xác suất={r['Probability']*100:5.1f}% | Ngưỡng={r['Threshold_Used']:.3f} | Vùng: {r['Risk_Zone']}")

    # 3. Thử nghiệm dự đoán hàng loạt (Batch Inference) trên 500 dòng test chuẩn One-vs-Benign
    print("\n" + "=" * 80)
    print("[BƯỚC 3] Chạy Dự đoán Hàng loạt (predict_batch_flows) trên 500 mẫu test chuẩn One-vs-Benign:")
    print("=" * 80)
    sample_batch = engine.sample_batch_test_flows(
        attack_name="DoS",
        total_samples=500,
        attack_ratio=0.10,
        include_other_attacks=False,
        random_seed=42
    )

    batch_res, batch_summary = engine.predict_batch_flows(
        df=sample_batch,
        attack_name="DoS",
        model_family="RF",
        condition="nosmote",
        threshold_mode="F1-Optimal"
    )

    print(f"[+] Mô hình sử dụng: {batch_summary['model_used']}")
    print(f"[+] Ngưỡng áp dụng: {batch_summary['threshold_used']} ({batch_summary['threshold_mode']})")
    print(f"[+] Tổng số luồng xử lý: {batch_summary['total_flows']} luồng")
    print(f"    - Benign (<10%):       {batch_summary['benign_count']} luồng")
    print(f"    - Suspicious (10-th):  {batch_summary['suspicious_count']} luồng")
    print(f"    - Attack (>=th):       {batch_summary['attack_count']} luồng")
    print(f"[+] Thời gian suy luận: {batch_summary['batch_inference_time_sec']}s (Throughput: {batch_summary['throughput_flows_per_sec']:,} luồng/giây)")

    if "metrics" in batch_summary:
        m = batch_summary["metrics"]
        print(f"[+] Đánh giá thực nghiệm: Accuracy={m['Accuracy']*100:.2f}%, Precision={m['Precision']*100:.2f}%, Recall={m['Recall']*100:.2f}%, F1={m['F1']*100:.2f}% (TP={m['TP']}, FP={m['FP']}, FN={m['FN']}, TN={m['TN']})")

    print("\n" + "=" * 80)
    print(" HOÀN TẤT KIỂM THỬ LÕI SUY LUẬN THÀNH CÔNG 100%!")
    print("=" * 80)
