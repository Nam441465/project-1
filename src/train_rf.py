import os
import sys
import time
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from pathlib import Path

# Đường dẫn src
SRC_DIR = Path(__file__).resolve().parent
sys.path.append(str(SRC_DIR))
from preprocessor import get_all_splits_for_attack, normalize_attack_name
from feature_selection import apply_smote, fit_feature_selector, transform_features

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = RESULTS_DIR / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_and_evaluate_rf(
    attack_name: str,
    use_smote: bool = False,
    smote_target: int = 56000,
    k_values: list = None,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 10,
    search_sample_limit: int = 150000,
    prevent_cv_leakage: bool = True
):
    """
    Huấn luyện và đánh giá mô hình Random Forest:
    - n_iter=10, cv=3, scoring='f1', random_state=10.
    - Chống rò rỉ dữ liệu trong CV (prevent_cv_leakage=True): dùng ImbPipeline để SMOTE chỉ chạy trên train fold của CV.
    - Quét ngưỡng F1 trên Validation set mở rộng từ 0.01 -> 0.99 (step 0.01).
    - Lưu trữ mô hình và selector tối ưu ra đĩa (joblib).
    - Đánh giá trên Held-out Test set.
    """
    if k_values is None:
        # Tập NetFlow có 10 đặc trưng lưu lượng
        k_values = [3, 5, 7, 10]
        
    norm_attack = normalize_attack_name(attack_name)
    condition = "smote" if use_smote else "nosmote"
    print("\n" + "=" * 70)
    print(f" KHỞI CHẠY HUẤN LUYỆN RANDOM FOREST - {norm_attack.upper()} [{condition.upper()}]")
    print("=" * 70)
    
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
    csv_file = os.path.join(RESULTS_DIR, f"rf_{norm_attack.lower()}_{condition}_metrics.csv")
    results_list = []
    
    best_val_f1 = -1.0
    best_k = None
    best_proba_val = None
    best_proba_test = None
    best_model = None
    best_selector = None
    best_model_info = {}
    
    # 3. Quét qua từng giá trị k
    for k in k_values:
        print(f"\n>>> Đang xử lý k = {k} đặc trưng...")
        t_k_start = time.perf_counter()
        
        # 3.1. Chọn đặc trưng bằng Mutual Information (hỗ trợ discrete_features)
        selector, selected_cols, _ = fit_feature_selector(
            X_train, y_train, k=k, random_state=random_state, max_samples_for_mi=200000
        )
        print(f"    - Đặc trưng được chọn: {selected_cols}")
        
        X_train_k = transform_features(selector, X_train)
        X_val_k = transform_features(selector, X_val)
        X_test_k = transform_features(selector, X_test)
        
        # 3.2. Tìm kiếm siêu tham số bằng RandomizedSearchCV
        param_dist = {
            "n_estimators": [100, 200, 300, 400],
            "min_samples_leaf": [1, 2, 3],
            "max_features": ["sqrt", None],
            "class_weight": [None, "balanced"]
        }
        
        # Chống rò rỉ dữ liệu trong CV: Nếu dùng SMOTE và bật prevent_cv_leakage,
        # ta tìm kiếm trên dữ liệu gốc chưa SMOTE và đóng gói SMOTE vào ImbPipeline
        if use_smote and prevent_cv_leakage:
            X_train_raw_k = transform_features(selector, X_train_raw)
            if len(X_train_raw_k) > search_sample_limit:
                # Giữ lại toàn bộ mẫu tấn công (minority) và chỉ lấy mẫu ngẫu nhiên từ lớp đa số (majority)
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
                
            # Đảm bảo k_neighbors của SMOTE phù hợp với số mẫu thiểu số thực tế trong mỗi fold CV
            n_min_in_search = np.sum(y_search == 1)
            min_samples_per_fold = (cv - 1) * n_min_in_search // cv
            smote_k_neighbors = min(5, max(1, min_samples_per_fold - 1))
            
            # Tính cv_smote_target tỷ lệ thuận với kích thước mẫu tìm kiếm và số fold CV
            if smote_target == "auto":
                cv_sampling_strategy = "auto"
            else:
                target_ratio = float(smote_target) / np.sum(y_train_raw == 0)
                n_maj_fold = int(np.sum(y_search == 0) * (cv - 1) / cv)
                n_min_fold_actual = int(np.ceil(np.sum(y_search == 1) * (cv - 1) / cv))
                n_min_fold_target = max(n_min_fold_actual, int(round(n_maj_fold * target_ratio)))
                cv_sampling_strategy = {1: min(n_maj_fold, n_min_fold_target)}

            print(f"    - Khởi chạy RandomizedSearchCV với ImbPipeline (chống rò rỉ CV, mẫu={len(X_search):,}, SMOTE strategy={cv_sampling_strategy}, k_neighbors={smote_k_neighbors})...")
            pipe = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=cv_sampling_strategy, k_neighbors=smote_k_neighbors, random_state=random_state)),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=1))
            ])
            param_dist_pipe = {f"rf__{p}": v for p, v in param_dist.items()}
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist_pipe,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=-1
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = {p.replace("rf__", ""): v for p, v in search.best_params_.items()}
        else:
            # Nếu tập train quá lớn (> 150k), lấy mẫu đại diện để tìm tham số tối ưu nhanh chóng
            if len(X_train_k) > search_sample_limit:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, train_size=search_sample_limit, random_state=random_state)
                idx, _ = next(sss.split(X_train_k, y_train))
                X_search = X_train_k.iloc[idx]
                y_search = y_train[idx]
            else:
                X_search = X_train_k
                y_search = y_train
                
            print(f"    - Khởi chạy RandomizedSearchCV (n_iter={n_iter}, cv={cv}, mẫu tìm kiếm={len(X_search):,})...")
            rf_base = RandomForestClassifier(random_state=random_state, n_jobs=1)
            search = RandomizedSearchCV(
                rf_base,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=-1
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = search.best_params_

        print(f"    - Tham số tốt nhất ({search_time:.1f}s): {best_params}")
        
        # 3.3. Huấn luyện mô hình tối ưu trên toàn bộ X_train_k
        print(f"    - Huấn luyện Random Forest tối ưu trên toàn bộ {len(X_train_k):,} mẫu...")
        t_fit = time.perf_counter()
        curr_rf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=-1)
        curr_rf.fit(X_train_k, y_train)
        train_time = time.perf_counter() - t_fit
        print(f"    - Huấn luyện xong trong {train_time:.1f}s.")
        
        # 3.4. Dò ngưỡng trên tập Validation (Threshold Tuning: mở rộng dải 0.01 -> 0.99)
        val_proba = curr_rf.predict_proba(X_val_k)[:, 1]
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
        test_proba = curr_rf.predict_proba(X_test_k)[:, 1]
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
            "n_estimators": int(best_params["n_estimators"]) if best_params.get("n_estimators") is not None else None,
            "min_samples_leaf": int(best_params["min_samples_leaf"]) if best_params.get("min_samples_leaf") is not None else None,
            "max_features": str(best_params.get("max_features")),
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
            best_model = curr_rf
            best_selector = selector
            best_model_info = record
            
        # Ghi ngay một dòng vào file CSV để không bị mất kết quả nếu có gián đoạn
        pd.DataFrame(results_list).to_csv(csv_file, index=False)
        
    # 4. Lưu xác suất dự đoán của cấu hình tốt nhất
    print(f"\n[+] Lưu xác suất dự đoán của k={best_k} (Val F1 cao nhất={best_val_f1:.4f})...")
    np.save(os.path.join(RESULTS_DIR, f"rf_{norm_attack.lower()}_{condition}_best_proba_val.npy"), best_proba_val)
    np.save(os.path.join(RESULTS_DIR, f"rf_{norm_attack.lower()}_{condition}_best_proba_test.npy"), best_proba_test)
    np.save(os.path.join(RESULTS_DIR, f"rf_{norm_attack.lower()}_{condition}_y_val.npy"), y_val)
    np.save(os.path.join(RESULTS_DIR, f"rf_{norm_attack.lower()}_{condition}_y_test.npy"), y_test)
    
    # 5. Lưu trữ mô hình tối ưu ra đĩa (Model Persistence)
    if best_model is not None:
        model_save_path = os.path.join(MODELS_DIR, f"rf_{norm_attack.lower()}_{condition}_best_model.joblib")
        selector_save_path = os.path.join(MODELS_DIR, f"rf_{norm_attack.lower()}_{condition}_best_selector.joblib")
        meta_save_path = os.path.join(MODELS_DIR, f"rf_{norm_attack.lower()}_{condition}_metadata.json")
        
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
    
    print("\n" + "=" * 70)
    print(f" TỔNG KẾT RANDOM FOREST: {norm_attack} [{condition.upper()}] (BEST k={best_k})")
    print("=" * 70)
    df_res = pd.DataFrame(results_list)
    cols_display = ["k", "val_threshold", "test_precision", "test_recall", "test_f1", "FP", "FN", "n_estimators", "max_features"]
    print(df_res[cols_display].to_string(index=False))
    print(f"\n[+] Kết quả đã lưu tại: {csv_file}")
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Random Forest chuẩn theo bài báo MDPI 2026")
    parser.add_argument("--attack", type=str, default="DoS", help="Tên loại tấn công: DoS, Exploits, Backdoor, Analysis, Reconnaissance")
    parser.add_argument("--smote", action="store_true", help="Có áp dụng SMOTE hay không")
    parser.add_argument("--smote_target", type=int, default=56000, help="Mục tiêu số mẫu thiểu số sau SMOTE (mặc định 56000)")
    parser.add_argument("--n_iter", type=int, default=10, help="Số lần ngẫu nhiên của RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--sample_limit", type=int, default=150000, help="Giới hạn mẫu khi chạy RandomizedSearchCV để tối ưu tốc độ")
    parser.add_argument("--no_leakage_fix", action="store_true", help="Tắt cơ chế chống rò rỉ dữ liệu trong CV (chạy theo pipeline cũ)")
    args = parser.parse_args()
    
    train_and_evaluate_rf(
        attack_name=args.attack,
        use_smote=args.smote,
        smote_target=args.smote_target,
        n_iter=args.n_iter,
        cv=args.cv,
        search_sample_limit=args.sample_limit,
        prevent_cv_leakage=not args.no_leakage_fix
    )
