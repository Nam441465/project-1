#!/usr/bin/env python3
"""
export_rf_models.py
Xuất bổ sung đầy đủ bộ Model Random Forest (.joblib + selector.joblib + metadata.json) 
vào results/baseline/models/ mà không cần chạy lại RandomizedSearchCV.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Đường dẫn dự án
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
BASELINE_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = BASELINE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(SRC_DIR))
import argparse
from preprocessor import get_all_splits_for_attack
from feature_selection import apply_smote, fit_feature_selector, transform_features

TARGET_ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]
CONDITIONS = ["nosmote", "smote"]

def default_serializer(obj):
    """Serialize numpy và các kiểu số sang chuẩn JSON native."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return str(obj)

def export_missing_rf_models(force: bool = False):
    print("=" * 75)
    print(" BẮT ĐẦU XUẤT BỔ SUNG MÔ HÌNH RANDOM FOREST VÀO results/baseline/models/")
    print("=" * 75)
    
    summary_csv = BASELINE_DIR / "summary_rf_comparison.csv"
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
    else:
        summary_df = None
        
    start_total = time.perf_counter()
    count_saved = 0
    count_skipped = 0
    
    for att in TARGET_ATTACKS:
        for cond in CONDITIONS:
            prefix = f"rf_{att.lower()}_{cond}"
            model_path = MODELS_DIR / f"{prefix}_best_model.joblib"
            selector_path = MODELS_DIR / f"{prefix}_best_selector.joblib"
            meta_path = MODELS_DIR / f"{prefix}_metadata.json"
            
            # 1. Kiểm tra nếu đã đủ 3 file và không bật --force thì bỏ qua
            if (not force) and model_path.exists() and selector_path.exists() and meta_path.exists():
                print(f"[>] ĐÃ TỒN TẠI: {prefix} -> Bỏ qua.")
                count_skipped += 1
                continue
                
            metrics_file = BASELINE_DIR / f"{prefix}_metrics.csv"
            if not metrics_file.exists():
                print(f"[!] LỖI: Không tìm thấy file metric {metrics_file}! Bỏ qua.")
                continue
                
            metrics_df = pd.read_csv(metrics_file)
            
            # Tìm dòng cấu hình tối ưu đã lưu
            cond_name = "Without SMOTE" if cond == "nosmote" else "With SMOTE"
            if summary_df is not None:
                sub_sum = summary_df[(summary_df["Attack"] == att) & (summary_df["Condition"] == cond_name)]
                if not sub_sum.empty:
                    best_k_target = int(sub_sum["Best_k"].iloc[0])
                    match_rows = metrics_df[metrics_df["k"] == best_k_target]
                    if not match_rows.empty:
                        best_row = match_rows.iloc[0]
                    else:
                        best_row = metrics_df.loc[metrics_df["val_f1"].idxmax()]
                else:
                    best_row = metrics_df.loc[metrics_df["val_f1"].idxmax()]
            else:
                best_row = metrics_df.loc[metrics_df["val_f1"].idxmax()]
                
            k = int(best_row["k"])
            n_estimators = int(best_row["n_estimators"])
            min_samples_leaf = int(best_row["min_samples_leaf"])
            
            raw_mf = best_row["max_features"]
            max_features = None if (pd.isna(raw_mf) or str(raw_mf).lower() in ["none", "nan"]) else str(raw_mf)
            
            raw_cw = best_row["class_weight"]
            class_weight = None if (pd.isna(raw_cw) or str(raw_cw).lower() in ["none", "nan"]) else str(raw_cw)
            
            print(f"\n[*] Đang xử lý: {prefix} (k={k}, n_est={n_estimators}, min_leaf={min_samples_leaf}, max_feat={max_features}, class_wt={class_weight})...")
            t0 = time.perf_counter()
            
            # Nạp X_train, y_train từ get_all_splits_for_attack
            X_train, y_train, _, _, _, _ = get_all_splits_for_attack(att, drop_ip=True)
            
            # Áp dụng SMOTE nếu cond == 'smote'
            if cond == "smote":
                X_train, y_train = apply_smote(X_train, y_train, target_samples=56000, random_state=10)
                
            # Chọn k đặc trưng bằng Mutual Information (giữ nguyên DataFrame có tên cột)
            selector, selected_cols, _ = fit_feature_selector(X_train, y_train, k=k, random_state=10)
            X_train_k = transform_features(selector, X_train)
            
            # Huấn luyện Random Forest
            print(f"[*] Huấn luyện Random Forest với {X_train_k.shape[0]:,} mẫu và {k} đặc trưng (n_jobs=4)...")
            t_fit = time.perf_counter()
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight=class_weight,
                random_state=10,
                n_jobs=4
            )
            rf.fit(X_train_k, y_train)
            fit_duration = time.perf_counter() - t_fit
            print(f"[+] Huấn luyện xong trong {fit_duration:.2f}s!")
            
            # Lưu model (dùng compress=3 để giảm kích thước và nạp nhanh)
            joblib.dump(rf, model_path, compress=3)
            joblib.dump(selector, selector_path)
            
            meta_dict = best_row.to_dict()
            meta_dict["attack"] = att
            meta_dict["condition"] = cond
            meta_dict["k"] = k
            meta_dict["selected_features"] = ";".join(selected_cols)
            meta_dict["n_estimators"] = n_estimators
            meta_dict["min_samples_leaf"] = min_samples_leaf
            meta_dict["max_features"] = max_features
            meta_dict["class_weight"] = class_weight
            
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=4, default=default_serializer)
                
            print(f"[+] ĐÃ XUẤT THÀNH CÔNG: {model_path} ({model_path.stat().st_size / (1024*1024):.2f} MB)")
            count_saved += 1
            
    print("\n" + "=" * 75)
    print(f" HOÀN TẤT: Đã xuất {count_saved} mô hình mới, {count_skipped} mô hình đã có sẵn.")
    print(f" Tổng thời gian: {time.perf_counter() - start_total:.2f}s")
    print("=" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Xuất bổ sung đầy đủ bộ Model Random Forest (.joblib + metadata)")
    parser.add_argument("--force", action="store_true", help="Ghi đè xuất lại toàn bộ mô hình kể cả khi đã tồn tại")
    args = parser.parse_args()
    export_missing_rf_models(force=args.force)
