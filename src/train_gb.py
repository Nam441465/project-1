import os
import sys
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Đường dẫn src
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))
from preprocessor import get_all_splits_for_attack, normalize_attack_name
from feature_selection import apply_smote, fit_feature_selector, transform_features

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "improved"
MODELS_DIR = RESULTS_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]

def train_and_evaluate_gb(
    attack_name: str,
    use_smote: bool = False,
    smote_target: int = 56000,
    k_values: list = None,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 10,
    search_sample_limit: int = 150000,
    prevent_cv_leakage: bool = True,
    n_jobs: int = 2
):
    """
    Huấn luyện và đánh giá mô hình Gradient Boosting (HistGradientBoostingClassifier):
    - Tối ưu cho tập dữ liệu lớn (>1 triệu dòng).
    - n_iter=10, cv=3, scoring='f1', random_state=10, n_jobs=2.
    - Chống rò rỉ dữ liệu trong CV (prevent_cv_leakage=True): dùng ImbPipeline để SMOTE chỉ chạy trên train fold của CV.
    - Quét ngưỡng F1 trên Validation set mở rộng từ 0.01 -> 0.99 (step 0.01).
    - Lưu trữ mô hình và selector tối ưu ra đĩa (joblib).
    - Đánh giá trên Held-out Test set độc lập.
    """
    if k_values is None:
        k_values = [3, 5, 7, 10]
        
    norm_attack = normalize_attack_name(attack_name)
    condition = "smote" if use_smote else "nosmote"
    print("\n" + "=" * 75)
    print(f" KHỞI CHẠY HUẤN LUYỆN GRADIENT BOOSTING - {norm_attack.upper()} [{condition.upper()}]")
    print("=" * 75)
    
    # 1. Nạp dữ liệu 3 phân vùng
    print(f"[*] Nạp dữ liệu Train / Val / Held-out Test cho {norm_attack}...")
    t0 = time.perf_counter()
    X_train, y_train, X_val, y_val, X_test, y_test = get_all_splits_for_attack(norm_attack)
    print(f"[+] Đã nạp dữ liệu ({time.perf_counter()-t0:.2f}s):")
    print(f"    - Train:    {len(X_train):,} mẫu (Benign: {np.bincount(y_train)[0]:,}, Attack: {np.bincount(y_train)[1]:,})")
    print(f"    - Val:      {len(X_val):,} mẫu (Benign: {np.bincount(y_val)[0]:,}, Attack: {np.bincount(y_val)[1]:,})")
    print(f"    - Held-out: {len(X_test):,} mẫu (Benign: {np.bincount(y_test)[0]:,}, Attack: {np.bincount(y_test)[1]:,})")
    
    # Lưu bản sao dữ liệu gốc trước SMOTE để dùng cho CV chống rò rỉ
    X_train_raw = X_train.copy()
    y_train_raw = y_train.copy()
    
    # 2. Áp dụng SMOTE trên tập Train (nếu chọn) trước khi feature selection
    if use_smote:
        print(f"\n[*] Bước cân bằng lớp: Áp dụng SMOTE trên tập Train (mục tiêu {smote_target:,} mẫu)...")
        t_sm = time.perf_counter()
        X_train, y_train = apply_smote(X_train, y_train, target_samples=smote_target, random_state=random_state)
        print(f"[+] SMOTE hoàn thành trong {time.perf_counter()-t_sm:.2f}s.")
        
    # Chuẩn bị file lưu kết quả CSV
    csv_file = RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_metrics.csv"
    results_list = []
    
    best_val_f1 = -1.0
    best_k = None
    best_proba_val = None
    best_proba_test = None
    best_model = None
    best_selector = None
    best_model_info = {}
    
    # Không gian siêu tham số cho HistGradientBoostingClassifier
    param_dist = {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [20, 50, 100],
        "class_weight": [None, "balanced"]
    }
    
    # 3. Quét qua từng giá trị k
    for k in k_values:
        print(f"\n>>> Đang xử lý k = {k} đặc trưng...")
        t_k_start = time.perf_counter()
        
        # 3.1. Chọn đặc trưng bằng Mutual Information
        selector, selected_cols, _ = fit_feature_selector(
            X_train, y_train, k=k, random_state=random_state, max_samples_for_mi=200000
        )
        print(f"    - Đặc trưng được chọn: {selected_cols}")
        
        X_train_k = transform_features(selector, X_train)
        X_val_k = transform_features(selector, X_val)
        X_test_k = transform_features(selector, X_test)
        
        # 3.2. Tìm kiếm siêu tham số bằng RandomizedSearchCV
        if use_smote and prevent_cv_leakage:
            X_train_raw_k = transform_features(selector, X_train_raw)
            if len(X_train_raw_k) > search_sample_limit:
                minority_idx = np.where(y_train_raw == 1)[0]
                majority_idx = np.where(y_train_raw == 0)[0]
                n_min = len(minority_idx)
                n_maj_needed = max(1, search_sample_limit - n_min)
                rng = np.random.RandomState(random_state)
                sampled_maj = rng.choice(majority_idx, size=min(n_maj_needed, len(majority_idx)), replace=False)
                combined_idx = np.concatenate([minority_idx, sampled_maj])
                rng.shuffle(combined_idx)
                X_search = X_train_raw_k.iloc[combined_idx]
                y_search = y_train_raw[combined_idx]
            else:
                X_search = X_train_raw_k
                y_search = y_train_raw
                
            n_min_in_search = np.sum(y_search == 1)
            min_samples_per_fold = (cv - 1) * n_min_in_search // cv
            smote_k_neighbors = min(5, max(1, min_samples_per_fold - 1))
            
            if smote_target == "auto":
                cv_sampling_strategy = "auto"
            else:
                target_ratio = float(smote_target) / np.sum(y_train_raw == 0)
                n_maj_fold = int(np.sum(y_search == 0) * (cv - 1) / cv)
                n_min_fold_actual = int(np.ceil(np.sum(y_search == 1) * (cv - 1) / cv))
                n_min_fold_target = max(n_min_fold_actual + 10, int(round(n_maj_fold * target_ratio)))
                cv_sampling_strategy = {1: min(n_maj_fold, n_min_fold_target)}

            print(f"    - Khởi chạy RandomizedSearchCV với ImbPipeline (chống rò rỉ CV, mẫu={len(X_search):,}, SMOTE strategy={cv_sampling_strategy}, k_neighbors={smote_k_neighbors})...")
            pipe = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=cv_sampling_strategy, k_neighbors=smote_k_neighbors, random_state=random_state)),
                ("gb", HistGradientBoostingClassifier(random_state=random_state, early_stopping=False))
            ])
            param_dist_pipe = {f"gb__{p}": v for p, v in param_dist.items()}
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist_pipe,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=n_jobs,
                refit=False
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = {p.replace("gb__", ""): v for p, v in search.best_params_.items()}
        else:
            if len(X_train_k) > search_sample_limit:
                minority_idx = np.where(y_train == 1)[0]
                majority_idx = np.where(y_train == 0)[0]
                n_min = len(minority_idx)
                n_maj_needed = max(1, search_sample_limit - n_min)
                rng = np.random.RandomState(random_state)
                sampled_maj = rng.choice(majority_idx, size=min(n_maj_needed, len(majority_idx)), replace=False)
                combined_idx = np.concatenate([minority_idx, sampled_maj])
                rng.shuffle(combined_idx)
                X_search = X_train_k.iloc[combined_idx]
                y_search = y_train[combined_idx]
            else:
                X_search = X_train_k
                y_search = y_train
                
            print(f"    - Khởi chạy RandomizedSearchCV (n_iter={n_iter}, cv={cv}, mẫu tìm kiếm={len(X_search):,}, n_jobs={n_jobs})...")
            gb_base = HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)
            search = RandomizedSearchCV(
                gb_base,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=n_jobs,
                refit=False
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = search.best_params_

        print(f"    - Tham số tốt nhất ({search_time:.1f}s): {best_params}")
        
        # 3.3. Huấn luyện mô hình tối ưu
        if not use_smote:
            minority_idx = np.where(y_train == 1)[0]
            majority_idx = np.where(y_train == 0)[0]
            n_maj_fit = min(150000, len(majority_idx))
            rng = np.random.RandomState(random_state)
            sampled_maj_fit = rng.choice(majority_idx, size=n_maj_fit, replace=False)
            combined_fit_idx = np.concatenate([minority_idx, sampled_maj_fit])
            rng.shuffle(combined_fit_idx)
            X_train_fit = X_train_k.iloc[combined_fit_idx]
            y_train_fit = y_train[combined_fit_idx]
            print(f"    - Huấn luyện HistGradientBoosting (nosmote) trên {len(X_train_fit):,} mẫu (100% Attack: {len(minority_idx):,}, Benign: {n_maj_fit:,}, early_stopping=False)...")
        else:
            X_train_fit = X_train_k
            y_train_fit = y_train
            print(f"    - Huấn luyện HistGradientBoosting (smote) trên toàn bộ {len(X_train_fit):,} mẫu (early_stopping=False)...")

        t_fit = time.perf_counter()
        curr_gb = HistGradientBoostingClassifier(**best_params, random_state=random_state, early_stopping=False)
        curr_gb.fit(X_train_fit, y_train_fit)
        train_time = time.perf_counter() - t_fit
        print(f"    - Huấn luyện xong trong {train_time:.2f}s.")
        
        # 3.4. Dò ngưỡng trên tập Validation (Threshold Tuning: 0.01 -> 0.99)
        val_proba = curr_gb.predict_proba(X_val_k)[:, 1]
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
        
        val_best_thresh = 0.5
        val_best_f1 = 0.0
        
        for t in thresholds:
            val_pred = (val_proba >= t).astype(int)
            f1_t = f1_score(y_val, val_pred, zero_division=0)
            if f1_t > val_best_f1:
                val_best_f1 = f1_t
                val_best_thresh = round(t, 2)
                
        print(f"    - Ngưỡng tối ưu trên Validation: t={val_best_thresh} (Val F1={val_best_f1:.4f})")
        
        # 3.5. Đánh giá trên Held-out Test với ngưỡng đã chọn
        t_test = time.perf_counter()
        test_proba = curr_gb.predict_proba(X_test_k)[:, 1]
        test_pred = (test_proba >= val_best_thresh).astype(int)
        test_time = time.perf_counter() - t_test
        
        acc = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)
        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"    -> HELD-OUT METRICS: F1={f1:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Acc={acc:.4f}")
        print(f"       Confusion Matrix: TP={tp:,}, FP={fp:,}, FN={fn:,}, TN={tn:,}")
        
        record = {
            "attack": norm_attack,
            "condition": condition,
            "k": int(k),
            "val_threshold": float(val_best_thresh),
            "val_f1": float(round(val_best_f1, 4)),
            "test_accuracy": float(round(acc, 4)),
            "test_precision": float(round(prec, 4)),
            "test_recall": float(round(rec, 4)),
            "test_f1": float(round(f1, 4)),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "max_iter": int(best_params["max_iter"]) if best_params.get("max_iter") is not None else None,
            "learning_rate": float(best_params["learning_rate"]) if best_params.get("learning_rate") is not None else None,
            "max_depth": int(best_params["max_depth"]) if best_params.get("max_depth") is not None else None,
            "min_samples_leaf": int(best_params["min_samples_leaf"]) if best_params.get("min_samples_leaf") is not None else None,
            "class_weight": str(best_params.get("class_weight")),
            "selected_features": ";".join(selected_cols),
            "train_time_sec": float(round(train_time, 2)),
            "test_time_sec": float(round(test_time, 2)),
            "total_k_time_sec": float(round(time.perf_counter() - t_k_start, 2))
        }
        results_list.append(record)
        
        # Cập nhật kết quả tốt nhất theo F1 Validation
        if val_best_f1 > best_val_f1:
            best_val_f1 = val_best_f1
            best_k = k
            best_proba_val = val_proba
            best_proba_test = test_proba
            best_model = curr_gb
            best_selector = selector
            best_model_info = record
            
        # Ghi ngay vào file CSV
        pd.DataFrame(results_list).to_csv(csv_file, index=False)
        
    # 4. Lưu xác suất dự đoán của cấu hình tốt nhất
    print(f"\n[+] Lưu xác suất dự đoán của k={best_k} (Val F1 cao nhất={best_val_f1:.4f})...")
    np.save(RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_best_proba_val.npy", best_proba_val)
    np.save(RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_best_proba_test.npy", best_proba_test)
    np.save(RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_y_val.npy", y_val)
    np.save(RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_y_test.npy", y_test)
    
    # 5. Lưu trữ mô hình tối ưu ra đĩa
    if best_model is not None:
        model_save_path = MODELS_DIR / f"gb_{norm_attack.lower()}_{condition}_best_model.joblib"
        selector_save_path = MODELS_DIR / f"gb_{norm_attack.lower()}_{condition}_best_selector.joblib"
        meta_save_path = MODELS_DIR / f"gb_{norm_attack.lower()}_{condition}_metadata.json"
        
        joblib.dump(best_model, model_save_path)
        joblib.dump(best_selector, selector_save_path)
        
        def default_serializer(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return str(obj)

        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(best_model_info, f, indent=4, default=default_serializer)
        print(f"[+] Đã lưu mô hình tốt nhất vào: {model_save_path}")
        print(f"[+] Đã lưu selector đặc trưng vào: {selector_save_path}")
    
    print("\n" + "=" * 75)
    print(f" TỔNG KẾT GRADIENT BOOSTING: {norm_attack} [{condition.upper()}] (BEST k={best_k})")
    print("=" * 75)
    df_res = pd.DataFrame(results_list)
    cols_display = ["k", "val_threshold", "test_precision", "test_recall", "test_f1", "FP", "FN", "train_time_sec"]
    print(df_res[cols_display].to_string(index=False))
    print(f"\n[+] Kết quả đã lưu tại: {csv_file}")
    return df_res

def generate_comparison_tables():
    """Tạo bảng summary_gb_comparison.csv và rf_vs_gb_comparison.csv từ kết quả thực nghiệm."""
    summary_gb_file = RESULTS_DIR / "summary_gb_comparison.csv"
    rf_baseline_summary_file = BASE_DIR / "results" / "baseline" / "summary_rf_comparison.csv"
    rf_vs_gb_file = RESULTS_DIR / "rf_vs_gb_comparison.csv"
    
    gb_summary_records = []
    
    conditions = [("nosmote", "Without SMOTE"), ("smote", "With SMOTE")]
    
    for att in TARGET_ATTACKS:
        for cond_key, cond_name in conditions:
            fpath = RESULTS_DIR / f"gb_{att.lower()}_{cond_key}_metrics.csv"
            if fpath.exists():
                df = pd.read_csv(fpath)
                best_row = df.loc[df["val_f1"].idxmax()]
                gb_summary_records.append({
                    "Attack": att,
                    "Condition": cond_name,
                    "Best_k": int(best_row["k"]),
                    "Threshold": float(best_row["val_threshold"]),
                    "Precision": float(best_row["test_precision"]),
                    "Recall": float(best_row["test_recall"]),
                    "F1_Score": float(best_row["test_f1"]),
                    "FP": int(best_row["FP"]),
                    "FN": int(best_row["FN"]),
                    "Train_Time_Sec": float(best_row["train_time_sec"])
                })
                
    if gb_summary_records:
        gb_summary_df = pd.DataFrame(gb_summary_records)
        gb_summary_df.to_csv(summary_gb_file, index=False)
        print(f"\n[+] Đã tạo bảng tổng kết GB: {summary_gb_file}")
        print(gb_summary_df.to_string(index=False))
        
        # Tạo bảng đối sánh trực tiếp RF vs GB
        if rf_baseline_summary_file.exists():
            rf_summary_df = pd.read_csv(rf_baseline_summary_file)
            
            # Đọc train_time_sec của RF từ các file metrics chi tiết
            rf_train_times = {}
            for att in TARGET_ATTACKS:
                for cond_key, cond_name in conditions:
                    rf_fpath = BASE_DIR / "results" / "baseline" / f"rf_{att.lower()}_{cond_key}_metrics.csv"
                    if rf_fpath.exists():
                        df_rf_detail = pd.read_csv(rf_fpath)
                        best_rf_detail = df_rf_detail.loc[df_rf_detail["val_f1"].idxmax()]
                        rf_train_times[(att, cond_name)] = float(best_rf_detail["train_time_sec"])
                    else:
                        rf_train_times[(att, cond_name)] = np.nan
                        
            comparison_rows = []
            for _, gb_row in gb_summary_df.iterrows():
                att = gb_row["Attack"]
                cond = gb_row["Condition"]
                rf_match = rf_summary_df[(rf_summary_df["Attack"] == att) & (rf_summary_df["Condition"] == cond)]
                if not rf_match.empty:
                    rf_row = rf_match.iloc[0]
                    rf_time = rf_train_times.get((att, cond), np.nan)
                    gb_time = float(gb_row["Train_Time_Sec"])
                    
                    delta_f1 = round(float(gb_row["F1_Score"]) - float(rf_row["F1_Score"]), 4)
                    speedup = round(rf_time / gb_time, 2) if (gb_time > 0 and not np.isnan(rf_time)) else np.nan
                    
                    comparison_rows.append({
                        "Attack": att,
                        "Condition": cond,
                        "RF_Best_k": int(rf_row["Best_k"]),
                        "RF_Threshold": float(rf_row["Threshold"]),
                        "RF_Precision": float(rf_row["Precision"]),
                        "RF_Recall": float(rf_row["Recall"]),
                        "RF_F1_Score": float(rf_row["F1_Score"]),
                        "RF_FP": int(rf_row["FP"]),
                        "RF_FN": int(rf_row["FN"]),
                        "RF_Train_Time_Sec": float(rf_time),
                        "GB_Best_k": int(gb_row["Best_k"]),
                        "GB_Threshold": float(gb_row["Threshold"]),
                        "GB_Precision": float(gb_row["Precision"]),
                        "GB_Recall": float(gb_row["Recall"]),
                        "GB_F1_Score": float(gb_row["F1_Score"]),
                        "GB_FP": int(gb_row["FP"]),
                        "GB_FN": int(gb_row["FN"]),
                        "GB_Train_Time_Sec": gb_time,
                        "Delta_F1": delta_f1,
                        "Speedup_Factor": speedup
                    })
                    
            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                comp_df.to_csv(rf_vs_gb_file, index=False)
                print(f"\n[+] Đã tạo bảng đối sánh RF vs GB: {rf_vs_gb_file}")
                disp_cols = ["Attack", "Condition", "RF_F1_Score", "GB_F1_Score", "Delta_F1", "RF_Train_Time_Sec", "GB_Train_Time_Sec", "Speedup_Factor"]
                print(comp_df[disp_cols].to_string(index=False))

def run_all_gb_pipeline(
    attacks=None,
    run_nosmote=True,
    run_smote=True,
    k_values=None,
    n_iter=10,
    cv=3,
    sample_limit=150000,
    smote_target=56000,
    prevent_cv_leakage=True,
    n_jobs=2
):
    """Điều phối toàn bộ chuỗi thực nghiệm Gradient Boosting cho các loại tấn công."""
    if attacks is None:
        attacks = TARGET_ATTACKS
    if k_values is None:
        k_values = [3, 5, 7, 10]
        
    start_total = time.perf_counter()
    print("=" * 80)
    print(" TIẾN TRÌNH THỰC NGHIỆM GRADIENT BOOSTING (HISTGRADIENTBOOSTING) - GIAI ĐOẠN 2")
    print("=" * 80)
    print(f" Danh sách tấn công: {attacks}")
    print(f" Giá trị k:         {k_values}")
    print(f" Chạy Không SMOTE:  {run_nosmote}")
    print(f" Chạy Có SMOTE:     {run_smote} (Target = {smote_target:,})")
    print(f" Siêu tham số:      n_iter={n_iter}, cv={cv}, sample_limit={sample_limit:,}, n_jobs={n_jobs}")
    print(f" Chống rò rỉ CV:    {prevent_cv_leakage}")
    print("=" * 80)
    
    for att in attacks:
        if run_nosmote:
            csv_path = RESULTS_DIR / f"gb_{att.lower()}_nosmote_metrics.csv"
            model_path = MODELS_DIR / f"gb_{att.lower()}_nosmote_best_model.joblib"
            if csv_path.exists() and model_path.exists() and len(pd.read_csv(csv_path)) >= len(k_values):
                print(f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {att} [NOSMOTE], bỏ qua huấn luyện lại.")
            else:
                print(f"\n>>> [1/2] BẮT ĐẦU CHẠY: {att} - KHÔNG SMOTE...")
                train_and_evaluate_gb(
                    attack_name=att,
                    use_smote=False,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs
                )
            generate_comparison_tables()
            
        if run_smote:
            csv_path = RESULTS_DIR / f"gb_{att.lower()}_smote_metrics.csv"
            model_path = MODELS_DIR / f"gb_{att.lower()}_smote_best_model.joblib"
            if csv_path.exists() and model_path.exists() and len(pd.read_csv(csv_path)) >= len(k_values):
                print(f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {att} [SMOTE], bỏ qua huấn luyện lại.")
            else:
                print(f"\n>>> [2/2] BẮT ĐẦU CHẠY: {att} - CÓ SMOTE (Target = {smote_target:,})...")
                train_and_evaluate_gb(
                    attack_name=att,
                    use_smote=True,
                    smote_target=smote_target,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs
                )
            generate_comparison_tables()
            
    print(f"\n[+] Tổng thời gian thực nghiệm toàn bộ: {(time.perf_counter() - start_total)/60:.2f} phút.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Gradient Boosting (HistGradientBoostingClassifier)")
    parser.add_argument("--attack", type=str, default=None, help="Tên loại tấn công (mặc định None)")
    parser.add_argument("--all", action="store_true", help="Chạy tất cả 5 loại tấn công")
    parser.add_argument("--only_nosmote", action="store_true", help="Chỉ chạy Không SMOTE")
    parser.add_argument("--only_smote", action="store_true", help="Chỉ chạy Có SMOTE")
    parser.add_argument("--k_values", nargs="+", type=int, default=[3, 5, 7, 10])
    parser.add_argument("--n_iter", type=int, default=10, help="Số lần ngẫu nhiên của RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--sample_limit", type=int, default=150000, help="Giới hạn mẫu tìm kiếm siêu tham số")
    parser.add_argument("--smote_target", type=int, default=56000, help="Mục tiêu số mẫu thiểu số sau SMOTE")
    parser.add_argument("--n_jobs", type=int, default=2, help="Số tiến trình song song cho RandomizedSearchCV")
    parser.add_argument("--no_leakage_fix", action="store_true", help="Tắt cơ chế chống rò rỉ dữ liệu trong CV")
    args = parser.parse_args()
    
    if args.all or (args.attack is None):
        target_attacks = TARGET_ATTACKS
    else:
        target_attacks = [args.attack]
        
    run_no = not args.only_smote
    run_sm = not args.only_nosmote
    
    run_all_gb_pipeline(
        attacks=target_attacks,
        run_nosmote=run_no,
        run_smote=run_sm,
        k_values=args.k_values,
        n_iter=args.n_iter,
        cv=args.cv,
        sample_limit=args.sample_limit,
        smote_target=args.smote_target,
        prevent_cv_leakage=not args.no_leakage_fix,
        n_jobs=args.n_jobs
    )
