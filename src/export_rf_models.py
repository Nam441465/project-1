#!/usr/bin/env python3
"""
export_rf_models.py
Xuất bổ sung đầy đủ bộ Model Random Forest (.joblib + selector.joblib + metadata.json)
vào results/baseline/models/ mà không cần chạy lại RandomizedSearchCV, kèm kiểm tra
đối khớp đặc trưng và metric parity trên tập Validation / Held-out Test.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
BASELINE_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = BASELINE_DIR / "models"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_selection import apply_smote, fit_feature_selector, transform_features  # noqa: E402
from pipeline_utils import (  # noqa: E402
    TARGET_ATTACKS,
    build_model_metadata,
    evaluate_binary_predictions,
    json_serializer,
    normalize_nullable_param,
    select_best_row,
    trim_memory,
)
from preprocessor import get_binary_data, normalize_attack_name  # noqa: E402

CONDITIONS = ["nosmote", "smote"]


def _verify_split_parity(
    rf: RandomForestClassifier,
    selector,
    attack_name: str,
    split: str,
    threshold: float,
    expected_f1: float,
    tol: float = 5e-4,
) -> dict:
    """Load a single evaluation split, verify F1 parity against CSV, and free memory."""
    X_split, y_split = get_binary_data(attack_name, split=split, drop_ip=True)
    X_split_k = transform_features(selector, X_split)
    del X_split
    t0 = time.perf_counter()
    proba = rf.predict_proba(X_split_k)[:, 1]
    eval_time = time.perf_counter() - t0
    del X_split_k
    metrics = evaluate_binary_predictions(y_split, proba, threshold)
    del y_split, proba
    trim_memory()

    if abs(metrics["f1"] - float(expected_f1)) > tol:
        raise ValueError(
            f"Metric parity mismatch for {attack_name} [{split}]: "
            f"recomputed F1={metrics['f1']:.6f} vs CSV={expected_f1:.6f} (tol={tol})."
        )
    metrics["eval_time_sec"] = eval_time
    return metrics


def export_missing_rf_models(
    force: bool = False,
    attacks: list[str] | None = None,
    n_jobs: int = 4,
    random_state: int = 10,
    smote_target: int = 56000,
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_attacks = [normalize_attack_name(a) for a in (attacks or TARGET_ATTACKS)]

    print("=" * 75)
    print(" BẮT ĐẦU XUẤT BỔ SUNG MÔ HÌNH RANDOM FOREST VÀO results/baseline/models/")
    print("=" * 75)

    summary_csv = BASELINE_DIR / "summary_rf_comparison.csv"
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else None

    start_total = time.perf_counter()
    count_saved = 0
    count_skipped = 0

    for att in target_attacks:
        for cond in CONDITIONS:
            prefix = f"rf_{att.lower()}_{cond}"
            model_path = MODELS_DIR / f"{prefix}_best_model.joblib"
            selector_path = MODELS_DIR / f"{prefix}_best_selector.joblib"
            meta_path = MODELS_DIR / f"{prefix}_metadata.json"

            if (not force) and model_path.exists() and selector_path.exists() and meta_path.exists():
                print(f"[>] ĐÃ TỒN TẠI: {prefix} -> Bỏ qua.")
                count_skipped += 1
                continue

            metrics_file = BASELINE_DIR / f"{prefix}_metrics.csv"
            if not metrics_file.exists():
                print(f"[!] LỖI: Không tìm thấy file metric {metrics_file}! Bỏ qua.")
                continue

            metrics_df = pd.read_csv(metrics_file)
            if metrics_df.empty:
                print(f"[!] LỖI: File metric {metrics_file} rỗng! Bỏ qua.")
                continue

            cond_name = "Without SMOTE" if cond == "nosmote" else "With SMOTE"
            best_row = None
            if summary_df is not None:
                sub_sum = summary_df[(summary_df["Attack"] == att) & (summary_df["Condition"] == cond_name)]
                if not sub_sum.empty:
                    best_k_target = int(sub_sum["Best_k"].iloc[0])
                    match_rows = metrics_df[metrics_df["k"].astype(int) == best_k_target]
                    if not match_rows.empty:
                        best_row = match_rows.iloc[0]
            if best_row is None:
                best_row = select_best_row(metrics_df)

            k = int(best_row["k"])
            n_estimators = int(best_row["n_estimators"])
            min_samples_leaf = int(best_row["min_samples_leaf"])
            max_features = normalize_nullable_param(best_row.get("max_features"))
            class_weight = normalize_nullable_param(best_row.get("class_weight"))

            raw_thresh = best_row.get("raw_val_threshold", best_row["val_threshold"])
            if pd.isna(raw_thresh):
                raw_thresh = best_row["val_threshold"]
            val_threshold = float(raw_thresh)

            print(
                f"\n[*] Đang xử lý: {prefix} (k={k}, n_est={n_estimators}, min_leaf={min_samples_leaf}, "
                f"max_feat={max_features}, class_wt={class_weight}, n_jobs={n_jobs})..."
            )

            # Chỉ nạp split="train" để giảm đỉnh RAM
            X_train, y_train = get_binary_data(att, split="train", drop_ip=True)
            if cond == "smote":
                X_train, y_train = apply_smote(
                    X_train, y_train, target_samples=smote_target, random_state=random_state
                )

            selector, selected_cols, _ = fit_feature_selector(
                X_train, y_train, k=k, random_state=random_state
            )
            expected_cols = [c for c in str(best_row["selected_features"]).split(";") if c]
            assert selected_cols == expected_cols, (
                f"Selected features mismatch for {prefix} (k={k}): "
                f"got {selected_cols}, expected {expected_cols}"
            )

            X_train_k = transform_features(selector, X_train)
            del X_train
            trim_memory()

            print(f"[*] Huấn luyện Random Forest với {X_train_k.shape[0]:,} mẫu và {k} đặc trưng (n_jobs={n_jobs})...")
            t_fit = time.perf_counter()
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            rf.fit(X_train_k, y_train)
            fit_duration = time.perf_counter() - t_fit
            del X_train_k, y_train
            trim_memory()
            print(f"[+] Huấn luyện xong trong {fit_duration:.2f}s!")

            # Kiểm tra parity trên Validation & Held-out Test
            val_metrics = _verify_split_parity(
                rf, selector, att, split="val", threshold=val_threshold, expected_f1=float(best_row["val_f1"])
            )
            test_metrics = _verify_split_parity(
                rf, selector, att, split="heldout_test", threshold=val_threshold, expected_f1=float(best_row["test_f1"])
            )
            print(
                f"[+] Parity OK: Val F1={val_metrics['f1']:.4f} (CSV={float(best_row['val_f1']):.4f}) | "
                f"Test F1={test_metrics['f1']:.4f} (CSV={float(best_row['test_f1']):.4f})"
            )

            joblib.dump(rf, model_path, compress=3)
            joblib.dump(selector, selector_path)

            meta_clean = build_model_metadata(
                best_row,
                extra={
                    "attack": att,
                    "condition": cond,
                    "k": k,
                    "selected_features": ";".join(selected_cols),
                    "n_estimators": n_estimators,
                    "min_samples_leaf": min_samples_leaf,
                    "max_features": max_features,
                    "class_weight": class_weight,
                    "train_time_sec": float(round(fit_duration, 2)),
                    "test_time_sec": float(round(test_metrics["eval_time_sec"], 2)),
                },
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_clean, f, indent=4, default=json_serializer, allow_nan=False)

            del rf, selector
            trim_memory()
            print(f"[+] ĐÃ XUẤT THÀNH CÔNG: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
            count_saved += 1

    print("\n" + "=" * 75)
    print(f" HOÀN TẤT: Đã xuất {count_saved} mô hình mới, {count_skipped} mô hình đã có sẵn.")
    print(f" Tổng thời gian: {time.perf_counter() - start_total:.2f}s")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất bổ sung đầy đủ bộ Model Random Forest (.joblib + metadata)")
    parser.add_argument("--force", action="store_true", help="Ghi đè xuất lại toàn bộ mô hình kể cả khi đã tồn tại")
    parser.add_argument("--attacks", nargs="+", default=None, help="Danh sách loại tấn công cần xuất")
    parser.add_argument("--n_jobs", type=int, default=4, help="Số luồng huấn luyện RandomForest (mặc định: 4)")
    parser.add_argument("--random_state", type=int, default=10, help="Random seed (mặc định: 10)")
    parser.add_argument("--smote_target", type=int, default=56000, help="Mục tiêu SMOTE (mặc định: 56000)")
    args = parser.parse_args()

    export_missing_rf_models(
        force=args.force,
        attacks=args.attacks,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        smote_target=args.smote_target,
    )
