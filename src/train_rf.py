import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

# Đường dẫn src
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from feature_selection import (  # noqa: E402
    apply_smote,
    build_selector_from_scores,
    compute_mi_scores,
    transform_features,
)
from pipeline_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    build_model_metadata,
    compute_cv_smote_strategy,
    evaluate_binary_predictions,
    json_serializer,
    normalize_nullable_param,
    subsample_minority_preserving,
    trim_memory,
    vectorized_threshold_sweep,
)
from preprocessor import get_all_splits_for_attack, normalize_attack_name  # noqa: E402

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "baseline"
MODELS_DIR = RESULTS_DIR / "models"


def train_and_evaluate_rf(
    attack_name: str,
    use_smote: bool = False,
    smote_target: int = 56000,
    k_values: list | None = None,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 10,
    search_sample_limit: int = 150000,
    prevent_cv_leakage: bool = True,
    n_jobs: int = 4,
    cv_jobs: int = 1,
):
    """
    Huấn luyện và đánh giá mô hình Random Forest:
    - n_iter=10, cv=3, scoring='f1', random_state=10.
    - Chống rò rỉ dữ liệu trong CV (prevent_cv_leakage=True): fit MI CV trên X_train_raw
      và đóng gói SMOTE vào ImbPipeline để SMOTE chỉ chạy trên train fold của CV.
    - Tính MI một lần duy nhất cho mỗi điều kiện trước khi quét các mốc k.
    - Quét ngưỡng F1 vector hóa trên Validation set từ 0.01 -> 0.99 (step 0.01).
    - Lưu trữ mô hình, selector tối ưu và metadata chuẩn hóa ra đĩa.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    if not k_values:
        raise ValueError("k_values must not be empty.")

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

    # Chỉ giữ bản sao X_train_raw & tính MI không rò rỉ khi SMOTE + prevent_cv_leakage
    if use_smote and prevent_cv_leakage:
        X_train_raw = X_train.copy()
        y_train_raw = y_train.copy()
        raw_scorer, raw_feature_names, raw_scores, _ = compute_mi_scores(
            X_train_raw, y_train_raw, random_state=random_state, max_samples_for_mi=200000
        )
    else:
        X_train_raw = None
        y_train_raw = None
        raw_scorer = raw_feature_names = raw_scores = None

    # 2. Áp dụng SMOTE trên tập Train (nếu chọn)
    if use_smote:
        print(f"\n[*] Bước cân bằng lớp: Áp dụng SMOTE trên tập Train (mục tiêu {smote_target:,} mẫu)...")
        t_sm = time.perf_counter()
        X_train, y_train = apply_smote(X_train, y_train, target_samples=smote_target, random_state=random_state)
        print(f"[+] SMOTE hoàn thành trong {time.perf_counter()-t_sm:.2f}s.")

    # Tính Mutual Information 1 lần duy nhất cho toàn bộ dải k
    scorer, feature_names, scores, _ = compute_mi_scores(
        X_train, y_train, random_state=random_state, max_samples_for_mi=200000
    )

    csv_file = RESULTS_DIR / f"rf_{norm_attack.lower()}_{condition}_metrics.csv"
    results_list = []

    best_val_f1 = -1.0
    best_k = None
    best_proba_val = None
    best_proba_test = None
    best_model = None
    best_selector = None
    best_model_info = {}

    rf_inner_jobs = max(1, int(n_jobs)) if int(n_jobs) > 0 else min(4, os.cpu_count() or 4)
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)

    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", None],
        "class_weight": [None, "balanced"],
    }

    # 3. Quét qua từng giá trị k
    for k in k_values:
        k = int(k)
        print(f"\n>>> Đang xử lý k = {k} đặc trưng...")
        t_k_start = time.perf_counter()

        # 3.1. Tạo SelectKBest từ điểm MI đã tính sẵn
        selector, selected_cols, _ = build_selector_from_scores(
            feature_names, scores, k=k, scorer=scorer
        )
        print(f"    - Đặc trưng được chọn: {selected_cols}")

        X_train_k = transform_features(selector, X_train)
        X_val_k = transform_features(selector, X_val)
        X_test_k = transform_features(selector, X_test)

        # 3.2. Tìm kiếm siêu tham số bằng RandomizedSearchCV (refit=False)
        if use_smote and prevent_cv_leakage:
            cv_selector, _, _ = build_selector_from_scores(
                raw_feature_names, raw_scores, k=k, scorer=raw_scorer
            )
            X_train_raw_k = transform_features(cv_selector, X_train_raw)
            X_search, y_search = subsample_minority_preserving(
                X_train_raw_k, y_train_raw, search_sample_limit, random_state=random_state
            )
            del X_train_raw_k

            cv_sampling_strategy, smote_k_neighbors = compute_cv_smote_strategy(
                y_train_raw, y_search, smote_target, cv=cv
            )
            print(
                f"    - Khởi chạy RandomizedSearchCV với ImbPipeline (chống rò rỉ CV, mẫu={len(X_search):,}, "
                f"SMOTE strategy={cv_sampling_strategy}, k_neighbors={smote_k_neighbors}, cv_jobs={cv_jobs}, inner_jobs={rf_inner_jobs})..."
            )
            pipe = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=cv_sampling_strategy, k_neighbors=smote_k_neighbors, random_state=random_state)),
                ("rf", RandomForestClassifier(random_state=random_state, n_jobs=rf_inner_jobs)),
            ])
            param_dist_pipe = {f"rf__{p}": v for p, v in param_dist.items()}
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_dist_pipe,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=cv_jobs,
                refit=False,
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = {p.replace("rf__", ""): v for p, v in search.best_params_.items()}
        else:
            X_search, y_search = subsample_minority_preserving(
                X_train_k, y_train, search_sample_limit, random_state=random_state
            )
            print(
                f"    - Khởi chạy RandomizedSearchCV (n_iter={n_iter}, cv={cv}, mẫu tìm kiếm={len(X_search):,}, "
                f"cv_jobs={cv_jobs}, inner_jobs={rf_inner_jobs})..."
            )
            rf_base = RandomForestClassifier(random_state=random_state, n_jobs=rf_inner_jobs)
            search = RandomizedSearchCV(
                rf_base,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=cv_jobs,
                refit=False,
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = search.best_params_

        del search, X_search, y_search
        trim_memory()
        print(f"    - Tham số tốt nhất ({search_time:.1f}s): {best_params}")

        # 3.3. Huấn luyện mô hình tối ưu trên toàn bộ X_train_k
        print(f"    - Huấn luyện Random Forest tối ưu trên toàn bộ {len(X_train_k):,} mẫu (n_jobs={n_jobs})...")
        t_fit = time.perf_counter()
        curr_rf = RandomForestClassifier(**best_params, random_state=random_state, n_jobs=n_jobs)
        curr_rf.fit(X_train_k, y_train)
        train_time = time.perf_counter() - t_fit
        del X_train_k
        trim_memory()
        print(f"    - Huấn luyện xong trong {train_time:.1f}s.")

        # 3.4. Dò ngưỡng vector hóa trên Validation (0.01 -> 0.99)
        val_proba = curr_rf.predict_proba(X_val_k)[:, 1]
        val_best_thresh, val_best_f1 = vectorized_threshold_sweep(
            y_val, val_proba, thresholds, default_threshold=0.5, initial_best_f1=0.0
        )
        print(f"    - Ngưỡng tối ưu trên Validation: t={val_best_thresh:.2f} (Val F1={val_best_f1:.4f})")

        # 3.5. Đánh giá trên Held-out Test với ngưỡng đã chọn
        t_test = time.perf_counter()
        test_proba = curr_rf.predict_proba(X_test_k)[:, 1]
        test_time = time.perf_counter() - t_test
        test_metrics = evaluate_binary_predictions(y_test, test_proba, val_best_thresh)

        print(
            f"    -> HELD-OUT METRICS: F1={test_metrics['f1']:.4f} | Prec={test_metrics['precision']:.4f} "
            f"| Rec={test_metrics['recall']:.4f} | Acc={test_metrics['accuracy']:.4f}"
        )
        print(
            f"       Confusion Matrix: TP={test_metrics['TP']:,}, FP={test_metrics['FP']:,}, "
            f"FN={test_metrics['FN']:,}, TN={test_metrics['TN']:,}"
        )

        record = {
            "attack": norm_attack,
            "condition": condition,
            "k": int(k),
            "val_threshold": float(round(val_best_thresh, 2)),
            "val_f1": float(round(val_best_f1, 4)),
            "raw_val_threshold": float(val_best_thresh),
            "raw_val_f1": float(val_best_f1),
            "test_accuracy": float(round(test_metrics["accuracy"], 4)),
            "test_precision": float(round(test_metrics["precision"], 4)),
            "test_recall": float(round(test_metrics["recall"], 4)),
            "test_f1": float(round(test_metrics["f1"], 4)),
            "TP": int(test_metrics["TP"]),
            "FP": int(test_metrics["FP"]),
            "FN": int(test_metrics["FN"]),
            "TN": int(test_metrics["TN"]),
            "n_estimators": int(best_params["n_estimators"]) if best_params.get("n_estimators") is not None else None,
            "min_samples_leaf": int(best_params["min_samples_leaf"]) if best_params.get("min_samples_leaf") is not None else None,
            "max_features": normalize_nullable_param(best_params.get("max_features")),
            "class_weight": normalize_nullable_param(best_params.get("class_weight")),
            "selected_features": ";".join(selected_cols),
            "train_time_sec": float(round(train_time, 2)),
            "test_time_sec": float(round(test_time, 2)),
            "total_k_time_sec": float(round(time.perf_counter() - t_k_start, 2)),
        }
        results_list.append(record)

        if val_best_f1 > best_val_f1 or (
            val_best_f1 == best_val_f1 and (best_k is None or k < int(best_k))
        ):
            best_val_f1 = val_best_f1
            best_k = k
            best_proba_val = val_proba
            best_proba_test = test_proba
            best_model = curr_rf
            best_selector = selector
            best_model_info = record

        del X_val_k, X_test_k
        trim_memory()
        pd.DataFrame(results_list).to_csv(csv_file, index=False)

    # 4. Lưu xác suất dự đoán của cấu hình tốt nhất
    print(f"\n[+] Lưu xác suất dự đoán của k={best_k} (Val F1 cao nhất={best_val_f1:.4f})...")
    np.save(RESULTS_DIR / f"rf_{norm_attack.lower()}_{condition}_best_proba_val.npy", best_proba_val)
    np.save(RESULTS_DIR / f"rf_{norm_attack.lower()}_{condition}_best_proba_test.npy", best_proba_test)
    np.save(RESULTS_DIR / f"rf_{norm_attack.lower()}_{condition}_y_val.npy", y_val)
    np.save(RESULTS_DIR / f"rf_{norm_attack.lower()}_{condition}_y_test.npy", y_test)

    # 5. Lưu trữ mô hình tối ưu ra đĩa
    if best_model is not None:
        model_save_path = MODELS_DIR / f"rf_{norm_attack.lower()}_{condition}_best_model.joblib"
        selector_save_path = MODELS_DIR / f"rf_{norm_attack.lower()}_{condition}_best_selector.joblib"
        meta_save_path = MODELS_DIR / f"rf_{norm_attack.lower()}_{condition}_metadata.json"

        joblib.dump(best_model, model_save_path, compress=3)
        joblib.dump(best_selector, selector_save_path)

        meta_clean = build_model_metadata(best_model_info)
        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta_clean, f, indent=4, default=json_serializer, allow_nan=False)
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
    parser.add_argument("--k_values", nargs="+", type=int, default=list(DEFAULT_K_VALUES), help="Các mốc số lượng đặc trưng k")
    parser.add_argument("--n_iter", type=int, default=10, help="Số lần ngẫu nhiên của RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--sample_limit", type=int, default=150000, help="Giới hạn mẫu khi chạy RandomizedSearchCV")
    parser.add_argument("--random_state", type=int, default=10, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=4, help="Số luồng cho RandomForest cuối và inner CV (mặc định: 4)")
    parser.add_argument("--cv_jobs", type=int, default=1, help="Số tiến trình song song cho fold CV (mặc định: 1)")
    parser.add_argument("--no_leakage_fix", action="store_true", help="Tắt cơ chế chống rò rỉ dữ liệu trong CV")
    args = parser.parse_args()

    train_and_evaluate_rf(
        attack_name=args.attack,
        use_smote=args.smote,
        smote_target=args.smote_target,
        k_values=args.k_values,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        search_sample_limit=args.sample_limit,
        prevent_cv_leakage=not args.no_leakage_fix,
        n_jobs=args.n_jobs,
        cv_jobs=args.cv_jobs,
    )
