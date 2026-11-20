import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import HistGradientBoostingClassifier
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
    TARGET_ATTACKS,
    build_model_metadata,
    compute_cv_smote_strategy,
    evaluate_binary_predictions,
    is_experiment_complete,
    json_serializer,
    normalize_nullable_param,
    select_best_row,
    subsample_minority_preserving,
    trim_memory,
    vectorized_threshold_sweep,
)
from preprocessor import get_all_splits_for_attack, normalize_attack_name  # noqa: E402

BASE_DIR = SRC_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "improved"
MODELS_DIR = RESULTS_DIR / "models"

VALID_BALANCE_MODES = ("hybrid_downsample", "sample_weight", "full")


def _build_prior_sample_weights(
    y_subset: np.ndarray,
    total_majority_count: int,
    subset_majority_count: int,
) -> np.ndarray:
    """Re-weight downsampled majority rows back to the full training prior."""
    maj_weight = float(total_majority_count) / max(1, int(subset_majority_count))
    return np.where(np.asarray(y_subset) == 0, maj_weight, 1.0).astype(np.float64)


def train_and_evaluate_gb(
    attack_name: str,
    use_smote: bool = False,
    smote_target: int = 56000,
    k_values: list | None = None,
    n_iter: int = 10,
    cv: int = 3,
    random_state: int = 10,
    search_sample_limit: int = 150000,
    prevent_cv_leakage: bool = True,
    n_jobs: int = 2,
    balance_mode: str = "hybrid_downsample",
):
    """
    Huấn luyện và đánh giá mô hình Gradient Boosting (HistGradientBoostingClassifier):
    - Tính MI một lần duy nhất cho mỗi điều kiện trước khi quét qua các mốc k.
    - Chống rò rỉ dữ liệu trong CV (prevent_cv_leakage=True): fit MI CV trên X_train_raw
      và đóng gói SMOTE vào ImbPipeline để SMOTE chỉ chạy trên train fold của CV.
    - Hỗ trợ balance_mode cho nhánh nosmote:
      * 'hybrid_downsample' (mặc định, tương thích ngược): 100% Attack + min(150k, Benign)
      * 'sample_weight': gắn trọng số chuẩn hóa tiên nghiệm (N_maj_full / N_maj_sub) với class_weight=None
      * 'full': huấn luyện cuối trên 100% X_train_k với class_weight=None
    - Quét ngưỡng F1 vector hóa trên Validation set.
    - Lưu trữ mô hình, selector tối ưu và metadata chuẩn hóa ra đĩa.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    if not k_values:
        raise ValueError("k_values must not be empty.")
    if balance_mode not in VALID_BALANCE_MODES:
        raise ValueError(f"balance_mode '{balance_mode}' không hợp lệ. Chọn một trong: {VALID_BALANCE_MODES}")

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

    # Chỉ sao chép X_train_raw & tính MI CV khi SMOTE + prevent_cv_leakage
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

    # Tính MI một lần duy nhất trước khi lặp qua k_values
    scorer, feature_names, scores, _ = compute_mi_scores(
        X_train, y_train, random_state=random_state, max_samples_for_mi=200000
    )

    csv_file = RESULTS_DIR / f"gb_{norm_attack.lower()}_{condition}_metrics.csv"
    results_list = []

    best_val_f1 = -1.0
    best_k = None
    best_proba_val = None
    best_proba_test = None
    best_model = None
    best_selector = None
    best_model_info = {}

    use_calibrated_prior = (not use_smote) and (balance_mode in ("sample_weight", "full"))
    param_dist = {
        "max_iter": [100, 200, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, None],
        "min_samples_leaf": [20, 50, 100],
        "class_weight": [None] if use_calibrated_prior else [None, "balanced"],
    }

    if use_calibrated_prior:
        thresholds = np.round(
            np.concatenate([np.linspace(0.01, 0.99, 99), np.linspace(0.991, 0.999, 9)]),
            3,
        )
    else:
        thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)

    # 3. Quét qua từng giá trị k
    for k in k_values:
        k = int(k)
        print(f"\n>>> Đang xử lý k = {k} đặc trưng...")
        t_k_start = time.perf_counter()

        # 3.1. Tạo selector từ điểm MI đã tính sẵn
        selector, selected_cols, _ = build_selector_from_scores(
            feature_names, scores, k=k, scorer=scorer
        )
        print(f"    - Đặc trưng được chọn: {selected_cols}")

        X_train_k = transform_features(selector, X_train)
        X_val_k = transform_features(selector, X_val)
        X_test_k = transform_features(selector, X_test)

        # 3.2. Tìm kiếm siêu tham số bằng RandomizedSearchCV
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
                f"SMOTE strategy={cv_sampling_strategy}, k_neighbors={smote_k_neighbors})..."
            )
            pipe = ImbPipeline([
                ("smote", SMOTE(sampling_strategy=cv_sampling_strategy, k_neighbors=smote_k_neighbors, random_state=random_state)),
                ("gb", HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)),
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
                refit=False,
            )
            t_search = time.perf_counter()
            search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = {p.replace("gb__", ""): v for p, v in search.best_params_.items()}
        else:
            X_search, y_search = subsample_minority_preserving(
                X_train_k, y_train, search_sample_limit, random_state=random_state
            )
            sw_search = None
            if (not use_smote) and balance_mode == "sample_weight":
                total_maj = int(np.sum(y_train == 0))
                sub_maj = int(np.sum(y_search == 0))
                sw_search = _build_prior_sample_weights(y_search, total_maj, sub_maj)

            print(
                f"    - Khởi chạy RandomizedSearchCV (n_iter={n_iter}, cv={cv}, mẫu tìm kiếm={len(X_search):,}, "
                f"n_jobs={n_jobs}, balance_mode={balance_mode})..."
            )
            gb_base = HistGradientBoostingClassifier(random_state=random_state, early_stopping=False)
            search = RandomizedSearchCV(
                gb_base,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv,
                scoring="f1",
                random_state=random_state,
                n_jobs=n_jobs,
                refit=False,
            )
            t_search = time.perf_counter()
            if sw_search is not None:
                search.fit(X_search, y_search, sample_weight=sw_search)
            else:
                search.fit(X_search, y_search)
            search_time = time.perf_counter() - t_search
            best_params = search.best_params_

        del search, X_search, y_search
        trim_memory()
        print(f"    - Tham số tốt nhất ({search_time:.1f}s): {best_params}")

        # 3.3. Huấn luyện mô hình tối ưu
        sample_weight_fit = None
        if not use_smote:
            if balance_mode == "full":
                X_train_fit = X_train_k
                y_train_fit = y_train
                print(
                    f"    - Huấn luyện HistGradientBoosting (nosmote, full) trên 100% {len(X_train_fit):,} mẫu "
                    f"(early_stopping=False)..."
                )
            else:
                minority_idx = np.where(y_train == 1)[0]
                majority_idx = np.where(y_train == 0)[0]
                n_maj_fit = min(150000, len(majority_idx))
                rng = np.random.RandomState(random_state)
                sampled_maj_fit = rng.choice(majority_idx, size=n_maj_fit, replace=False)
                combined_fit_idx = np.concatenate([minority_idx, sampled_maj_fit])
                rng.shuffle(combined_fit_idx)
                X_train_fit = X_train_k.iloc[combined_fit_idx]
                y_train_fit = y_train[combined_fit_idx]
                if balance_mode == "sample_weight":
                    sample_weight_fit = _build_prior_sample_weights(
                        y_train_fit, len(majority_idx), n_maj_fit
                    )
                    print(
                        f"    - Huấn luyện HistGradientBoosting (nosmote, sample_weight={len(majority_idx)/max(1, n_maj_fit):.3f}) "
                        f"trên {len(X_train_fit):,} mẫu..."
                    )
                else:
                    print(
                        f"    - Huấn luyện HistGradientBoosting (nosmote) trên {len(X_train_fit):,} mẫu "
                        f"(100% Attack: {len(minority_idx):,}, Benign: {n_maj_fit:,}, early_stopping=False)..."
                    )
        else:
            X_train_fit = X_train_k
            y_train_fit = y_train
            print(f"    - Huấn luyện HistGradientBoosting (smote) trên toàn bộ {len(X_train_fit):,} mẫu (early_stopping=False)...")

        t_fit = time.perf_counter()
        curr_gb = HistGradientBoostingClassifier(**best_params, random_state=random_state, early_stopping=False)
        if sample_weight_fit is not None:
            curr_gb.fit(X_train_fit, y_train_fit, sample_weight=sample_weight_fit)
        else:
            curr_gb.fit(X_train_fit, y_train_fit)
        train_time = time.perf_counter() - t_fit
        del X_train_k, X_train_fit
        trim_memory()
        print(f"    - Huấn luyện xong trong {train_time:.2f}s.")

        # 3.4. Dò ngưỡng vector hóa trên Validation
        val_proba = curr_gb.predict_proba(X_val_k)[:, 1]
        val_best_thresh, val_best_f1 = vectorized_threshold_sweep(
            y_val, val_proba, thresholds, default_threshold=0.5, initial_best_f1=0.0
        )
        print(f"    - Ngưỡng tối ưu trên Validation: t={val_best_thresh} (Val F1={val_best_f1:.4f})")

        # 3.5. Đánh giá trên Held-out Test với ngưỡng đã chọn
        t_test = time.perf_counter()
        test_proba = curr_gb.predict_proba(X_test_k)[:, 1]
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
            "val_threshold": float(round(val_best_thresh, 3 if use_calibrated_prior else 2)),
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
            "max_iter": int(best_params["max_iter"]) if best_params.get("max_iter") is not None else None,
            "learning_rate": float(best_params["learning_rate"]) if best_params.get("learning_rate") is not None else None,
            "max_depth": int(best_params["max_depth"]) if best_params.get("max_depth") is not None else None,
            "min_samples_leaf": int(best_params["min_samples_leaf"]) if best_params.get("min_samples_leaf") is not None else None,
            "class_weight": normalize_nullable_param(best_params.get("class_weight")),
            "balance_mode": balance_mode if not use_smote else "smote",
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
            best_model = curr_gb
            best_selector = selector
            best_model_info = record

        del X_val_k, X_test_k
        trim_memory()
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

        meta_clean = build_model_metadata(best_model_info)
        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(meta_clean, f, indent=4, default=json_serializer, allow_nan=False)
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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
                if df.empty:
                    continue
                best_row = select_best_row(df)
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
                    "Train_Time_Sec": float(best_row["train_time_sec"]),
                })

    if gb_summary_records:
        gb_summary_df = pd.DataFrame(gb_summary_records)
        gb_summary_df.to_csv(summary_gb_file, index=False)
        print(f"\n[+] Đã tạo bảng tổng kết GB: {summary_gb_file}")
        print(gb_summary_df.to_string(index=False))

        if rf_baseline_summary_file.exists():
            rf_summary_df = pd.read_csv(rf_baseline_summary_file)
            rf_train_times = {}
            for att in TARGET_ATTACKS:
                for cond_key, cond_name in conditions:
                    rf_fpath = BASE_DIR / "results" / "baseline" / f"rf_{att.lower()}_{cond_key}_metrics.csv"
                    if rf_fpath.exists():
                        df_rf_detail = pd.read_csv(rf_fpath)
                        if not df_rf_detail.empty:
                            best_rf_detail = select_best_row(df_rf_detail)
                            rf_train_times[(att, cond_name)] = float(best_rf_detail["train_time_sec"])
                        else:
                            rf_train_times[(att, cond_name)] = np.nan
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
                        "Speedup_Factor": speedup,
                    })

            if comparison_rows:
                comp_df = pd.DataFrame(comparison_rows)
                comp_df.to_csv(rf_vs_gb_file, index=False)
                print(f"\n[+] Đã tạo bảng đối sánh RF vs GB: {rf_vs_gb_file}")
                disp_cols = [
                    "Attack", "Condition", "RF_F1_Score", "GB_F1_Score",
                    "Delta_F1", "RF_Train_Time_Sec", "GB_Train_Time_Sec", "Speedup_Factor",
                ]
                print(comp_df[disp_cols].to_string(index=False))


def run_all_gb_pipeline(
    attacks=None,
    run_nosmote=True,
    run_smote=True,
    k_values=None,
    n_iter=10,
    cv=3,
    random_state=10,
    sample_limit=150000,
    smote_target=56000,
    prevent_cv_leakage=True,
    n_jobs=2,
    force_retrain=False,
    balance_mode="hybrid_downsample",
):
    """Điều phối toàn bộ chuỗi thực nghiệm Gradient Boosting cho các loại tấn công."""
    if attacks is None:
        attacks = TARGET_ATTACKS
    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    if not k_values:
        raise ValueError("k_values must not be empty.")

    start_total = time.perf_counter()
    print("=" * 80)
    print(" TIẾN TRÌNH THỰC NGHIỆM GRADIENT BOOSTING (HISTGRADIENTBOOSTING) - GIAI ĐOẠN 2")
    print("=" * 80)
    print(f" Danh sách tấn công: {attacks}")
    print(f" Giá trị k:         {k_values}")
    print(f" Chạy Không SMOTE:  {run_nosmote} (balance_mode={balance_mode})")
    print(f" Chạy Có SMOTE:     {run_smote} (Target = {smote_target:,})")
    print(f" Siêu tham số:      n_iter={n_iter}, cv={cv}, sample_limit={sample_limit:,}, n_jobs={n_jobs}, seed={random_state}")
    print(f" Chống rò rỉ CV:    {prevent_cv_leakage}")
    print(f" Ép huấn luyện lại: {force_retrain}")
    print("=" * 80)

    any_executed = False
    for att in attacks:
        norm_att = normalize_attack_name(att)
        if run_nosmote:
            prefix_no = f"gb_{norm_att.lower()}_nosmote"
            if (not force_retrain) and is_experiment_complete(RESULTS_DIR, MODELS_DIR, prefix_no, k_values):
                print(f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {norm_att} [NOSMOTE], bỏ qua huấn luyện lại (dùng --force để chạy lại).")
            else:
                print(f"\n>>> [1/2] BẮT ĐẦU CHẠY: {norm_att} - KHÔNG SMOTE...")
                train_and_evaluate_gb(
                    attack_name=norm_att,
                    use_smote=False,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    random_state=random_state,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs,
                    balance_mode=balance_mode,
                )
                any_executed = True

        if run_smote:
            prefix_sm = f"gb_{norm_att.lower()}_smote"
            if (not force_retrain) and is_experiment_complete(RESULTS_DIR, MODELS_DIR, prefix_sm, k_values):
                print(f"\n[+] Đã tồn tại kết quả hoàn chỉnh cho {norm_att} [SMOTE], bỏ qua huấn luyện lại (dùng --force để chạy lại).")
            else:
                print(f"\n>>> [2/2] BẮT ĐẦU CHẠY: {norm_att} - CÓ SMOTE (Target = {smote_target:,})...")
                train_and_evaluate_gb(
                    attack_name=norm_att,
                    use_smote=True,
                    smote_target=smote_target,
                    k_values=k_values,
                    n_iter=n_iter,
                    cv=cv,
                    random_state=random_state,
                    search_sample_limit=sample_limit,
                    prevent_cv_leakage=prevent_cv_leakage,
                    n_jobs=n_jobs,
                    balance_mode=balance_mode,
                )
                any_executed = True

    summary_missing = not (
        (RESULTS_DIR / "summary_gb_comparison.csv").exists()
        and (RESULTS_DIR / "rf_vs_gb_comparison.csv").exists()
    )
    if any_executed or summary_missing:
        generate_comparison_tables()

    print(f"\n[+] Tổng thời gian thực nghiệm toàn bộ: {(time.perf_counter() - start_total)/60:.2f} phút.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Gradient Boosting (HistGradientBoostingClassifier)")
    parser.add_argument("--attack", type=str, default=None, help="Tên một loại tấn công (mặc định None)")
    parser.add_argument("--attacks", nargs="+", default=None, help="Danh sách các loại tấn công cần chạy")
    parser.add_argument("--all", action="store_true", help="Chạy tất cả 5 loại tấn công")
    parser.add_argument("--only_nosmote", action="store_true", help="Chỉ chạy Không SMOTE")
    parser.add_argument("--only_smote", action="store_true", help="Chỉ chạy Có SMOTE")
    parser.add_argument("--k_values", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--n_iter", type=int, default=10, help="Số lần ngẫu nhiên của RandomizedSearchCV")
    parser.add_argument("--cv", type=int, default=3, help="Số fold cross-validation")
    parser.add_argument("--random_state", type=int, default=10, help="Random seed (mặc định: 10)")
    parser.add_argument("--sample_limit", type=int, default=150000, help="Giới hạn mẫu tìm kiếm siêu tham số")
    parser.add_argument("--smote_target", type=int, default=56000, help="Mục tiêu số mẫu thiểu số sau SMOTE")
    parser.add_argument("--n_jobs", type=int, default=2, help="Số tiến trình song song cho RandomizedSearchCV")
    parser.add_argument("--no_leakage_fix", action="store_true", help="Tắt cơ chế chống rò rỉ dữ liệu trong CV")
    parser.add_argument("--force", action="store_true", help="Ép huấn luyện lại ngay cả khi đã có kết quả lưu sẵn")
    parser.add_argument(
        "--balance_mode",
        type=str,
        choices=list(VALID_BALANCE_MODES),
        default="hybrid_downsample",
        help="Chế độ xử lý mất cân bằng cho nhánh nosmote (mặc định: hybrid_downsample)",
    )
    args = parser.parse_args()

    if args.all:
        target_attacks = list(TARGET_ATTACKS)
    elif args.attacks is not None:
        target_attacks = args.attacks
    elif args.attack is not None:
        target_attacks = [args.attack]
    else:
        target_attacks = list(TARGET_ATTACKS)

    run_no = not args.only_smote
    run_sm = not args.only_nosmote

    run_all_gb_pipeline(
        attacks=target_attacks,
        run_nosmote=run_no,
        run_smote=run_sm,
        k_values=args.k_values,
        n_iter=args.n_iter,
        cv=args.cv,
        random_state=args.random_state,
        sample_limit=args.sample_limit,
        smote_target=args.smote_target,
        prevent_cv_leakage=not args.no_leakage_fix,
        n_jobs=args.n_jobs,
        force_retrain=args.force,
        balance_mode=args.balance_mode,
    )
