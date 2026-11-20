#!/usr/bin/env python3
"""
train_paper_rf.py
Nhánh thực nghiệm "Paper-Exact Baseline":
Tái lập nguyên bản 100% pipeline Random Forest của bài báo MDPI Information 2026
(Pospíchal et al., "Random Forest-Based Network Intrusion Detection with Feature
Selection and Class Balancing on UNSW-NB15 Traffic") trên bộ dữ liệu NF-UNSW-NB15
của dự án (cùng phân vùng train.parquet / val.parquet / heldout_test.parquet).

Đặc tả nguyên bản (KHÔNG thêm bất kỳ cải tiến toán học nào):
1. Khi use_smote=True: chạy SMOTE(sampling_strategy={1: 56000}, random_state=10)
   trực tiếp 1 lần trên toàn bộ (X_train, y_train) TRƯỚC SelectKBest và TRƯỚC
   RandomizedSearchCV (không dùng ImbPipeline, giữ nguyên thiết kế gốc).
2. Chọn đặc trưng bằng SelectKBest(functools.partial(mutual_info_classif, random_state=10), k=k)
   với mặc định discrete_features='auto' trên 100% (X_train, y_train).
   (Điểm MI thô được cache trong .k_cache/ giữa các subprocess k để tránh tính lại 4 lần).
3. RandomizedSearchCV (n_iter=10, cv=3, scoring='f1', random_state=10, refit=False)
   trực tiếp trên 100% (X_train_k, y_train) — không lấy mẫu con.
4. Huấn luyện lại RandomForestClassifier(**best_params, random_state=10, n_jobs=-1)
   trên toàn bộ (X_train_k, y_train) và đo train_time_sec.
5. Dò ngưỡng quyết định trên Validation theo đúng np.round(np.arange(0.05, 0.95, 0.01), 2)
   tối ưu F1 (vector hóa O(N log N)).
6. Đánh giá trên Held-out Test, lưu checkpoint sau từng mốc k, lưu xác suất (.npy)
   và mô hình (.joblib + selector + metadata.json) vào results/paper_original/.

Quản lý bộ nhớ an toàn trên WSL2 (không thay đổi toán học):
- Cô lập từng mốc k trong tiến trình con riêng biệt kèm MALLOC_ARENA_MAX=2 và malloc_trim(0)
  để giải phóng 100% bộ nhớ Cython/glibc sau mỗi mốc k, chống tràn RAM vmmemWSL.
"""

import argparse
import functools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import RandomizedSearchCV

PAPER_SRC_DIR = Path(__file__).resolve().parent
SRC_DIR = PAPER_SRC_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline_utils import (  # noqa: E402
    DEFAULT_K_VALUES,
    TARGET_ATTACKS,
    build_model_metadata,
    evaluate_binary_predictions,
    is_experiment_complete,
    json_serializer,
    normalize_nullable_param,
    select_best_row,
    trim_memory,
    vectorized_threshold_sweep,
)
from preprocessor import get_all_splits_for_attack, normalize_attack_name  # noqa: E402

PAPER_RESULTS_DIR = PROJECT_ROOT / "results" / "paper_original"
PAPER_MODELS_DIR = PAPER_RESULTS_DIR / "models"
K_CACHE_DIR = PAPER_RESULTS_DIR / ".k_cache"


def _save_mi_scores(attack_name: str, condition: str, feature_names: list, scores: np.ndarray):
    """Lưu bảng điểm Mutual Information nguyên bản (discrete_features='auto')."""
    mi_csv = PAPER_RESULTS_DIR / "paper_mi_scores_5attacks.csv"
    cond_label = "With SMOTE" if condition == "smote" else "Without SMOTE"
    if mi_csv.exists():
        df_old = pd.read_csv(mi_csv)
        if not df_old[(df_old["Attack"] == attack_name) & (df_old["Condition"] == cond_label)].empty:
            return
    else:
        df_old = None

    df_new = pd.DataFrame({
        "Attack": attack_name,
        "Condition": cond_label,
        "Feature": feature_names,
        "MI_Score": np.round(scores.astype(float), 6),
    })
    df_new["Rank"] = df_new["MI_Score"].rank(ascending=False, method="min").astype(int)
    df_new = df_new.sort_values("Rank").reset_index(drop=True)

    df_all = pd.concat([df_old, df_new], ignore_index=True) if df_old is not None else df_new
    df_all.to_csv(mi_csv, index=False)


def rebuild_summary_csv():
    """Tổng hợp bảng summary_paper_rf_comparison.csv từ các file metrics đã hoàn thành."""
    summary_rows = []
    for att in TARGET_ATTACKS:
        for cond_code, cond_label in [("nosmote", "Without SMOTE"), ("smote", "With SMOTE")]:
            prefix = f"paper_rf_{att.lower()}_{cond_code}"
            metrics_path = PAPER_RESULTS_DIR / f"{prefix}_metrics.csv"
            if not metrics_path.exists():
                continue
            df_m = pd.read_csv(metrics_path)
            if df_m.empty:
                continue
            best_row = select_best_row(df_m)
            summary_rows.append({
                "Attack": att,
                "Condition": cond_label,
                "Best_k": int(best_row["k"]),
                "Threshold": round(float(best_row["val_threshold"]), 2),
                "Val_F1": round(float(best_row["val_f1"]), 4),
                "CV_Best_F1": round(float(best_row["cv_best_f1"]), 4) if "cv_best_f1" in best_row else np.nan,
                "Accuracy": round(float(best_row["test_accuracy"]), 4),
                "Precision": round(float(best_row["test_precision"]), 4),
                "Recall": round(float(best_row["test_recall"]), 4),
                "F1_Score": round(float(best_row["test_f1"]), 4),
                "TP": int(best_row["TP"]),
                "FP": int(best_row["FP"]),
                "FN": int(best_row["FN"]),
                "TN": int(best_row["TN"]),
                "Search_Time_Sec": round(float(best_row["search_time_sec"]), 2),
                "Train_Time_Sec": round(float(best_row["train_time_sec"]), 2),
                "Test_Time_Sec": round(float(best_row["test_time_sec"]), 2),
                "Total_4k_Search_Time_Sec": round(float(df_m["search_time_sec"].sum()), 2),
                "Total_4k_Train_Time_Sec": round(float(df_m["train_time_sec"].sum()), 2),
            })

    if summary_rows:
        out_df = pd.DataFrame(summary_rows)
        out_path = PAPER_RESULTS_DIR / "summary_paper_rf_comparison.csv"
        out_df.to_csv(out_path, index=False)
        return out_df
    return None


def is_k_cached(prefix: str, k: int) -> bool:
    """Kiểm tra xem mốc k cụ thể đã có dòng metrics CSV và bundle cache chưa."""
    metrics_path = PAPER_RESULTS_DIR / f"{prefix}_metrics.csv"
    cache_bundle = K_CACHE_DIR / f"{prefix}_k{int(k)}.joblib"
    if not (metrics_path.exists() and cache_bundle.exists()):
        return False
    try:
        df_m = pd.read_csv(metrics_path)
        return int(k) in set(df_m["k"].astype(int).tolist())
    except Exception:
        return False


def _build_paper_selector_from_cache(
    mi_score_func,
    feature_names: list[str],
    scores: np.ndarray,
    k: int,
) -> SelectKBest:
    """Construct SelectKBest with exact paper score_func from cached raw MI scores."""
    selector = SelectKBest(score_func=mi_score_func, k=min(int(k), len(feature_names)))
    selector.scores_ = np.asarray(scores, dtype=float).copy()
    selector.pvalues_ = None
    selector.n_features_in_ = len(feature_names)
    selector.feature_names_in_ = np.asarray(feature_names, dtype=object)
    return selector


def _run_worker_single_k(
    attack_name: str,
    use_smote: bool,
    k: int,
    cv_jobs: int,
    rf_inner_jobs: int,
):
    """Tiến trình con thực thi độc lập 1 giá trị k và thoát ngay sau khi lưu checkpoint để giải phóng 100% RAM."""
    PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    K_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    norm_attack = normalize_attack_name(attack_name)
    condition = "smote" if use_smote else "nosmote"
    prefix = f"paper_rf_{norm_attack.lower()}_{condition}"
    metrics_csv = PAPER_RESULTS_DIR / f"{prefix}_metrics.csv"

    print(f"\n>>> [{norm_attack} | {condition.upper()}] Đang chạy Paper-Exact Pipeline với k = {k} đặc trưng...", flush=True)

    t_load0 = time.perf_counter()
    X_train, y_train, X_val, y_val, X_test, y_test = get_all_splits_for_attack(norm_attack, drop_ip=True)
    feature_names = X_train.columns.tolist()
    print(
        f"    [+] Nạp dữ liệu ({time.perf_counter() - t_load0:.2f}s): "
        f"Train={len(X_train):,} (Benign={np.bincount(y_train)[0]:,}, Attack={np.bincount(y_train)[1]:,}) | "
        f"Val={len(X_val):,} | Test={len(X_test):,}",
        flush=True,
    )

    # BƯỚC 1: SMOTE(sampling_strategy={1: 56000}, random_state=10) trước SelectKBest & CV
    smote_time_sec = 0.0
    if use_smote:
        t_sm0 = time.perf_counter()
        smote = SMOTE(sampling_strategy={1: 56000}, random_state=10)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        smote_time_sec = time.perf_counter() - t_sm0
        print(
            f"    [Bước 1] SMOTE ({smote_time_sec:.2f}s) -> Train mới={len(X_train):,} "
            f"(Lớp 0={np.bincount(y_train)[0]:,}, Lớp 1={np.bincount(y_train)[1]:,})",
            flush=True,
        )

    # BƯỚC 2: SelectKBest + mutual_info_classif chuẩn (discrete_features='auto'), cache raw MI score vector
    mi_score_func = functools.partial(mutual_info_classif, random_state=10)
    mi_cache_path = K_CACHE_DIR / f"{prefix}_mi_scores.joblib"
    selector = None
    mi_time_sec = 0.0

    if mi_cache_path.exists():
        try:
            cached_mi = joblib.load(mi_cache_path)
            if cached_mi.get("feature_names") == feature_names:
                t_mi0 = time.perf_counter()
                selector = _build_paper_selector_from_cache(
                    mi_score_func=mi_score_func,
                    feature_names=feature_names,
                    scores=cached_mi["scores"],
                    k=int(k),
                )
                X_train_k = selector.transform(X_train)
                X_val_k = selector.transform(X_val)
                X_test_k = selector.transform(X_test)
                mi_time_sec = float(cached_mi.get("mi_time_sec", time.perf_counter() - t_mi0))
        except Exception:
            selector = None

    if selector is None:
        t_mi0 = time.perf_counter()
        selector = SelectKBest(score_func=mi_score_func, k=min(int(k), len(feature_names)))
        X_train_k = selector.fit_transform(X_train, y_train)
        X_val_k = selector.transform(X_val)
        X_test_k = selector.transform(X_test)
        mi_time_sec = time.perf_counter() - t_mi0
        joblib.dump(
            {
                "feature_names": feature_names,
                "scores": np.asarray(selector.scores_, dtype=float),
                "mi_time_sec": float(mi_time_sec),
            },
            mi_cache_path,
        )

    selected_cols = [feature_names[idx] for idx in selector.get_support(indices=True)]
    print(f"    [Bước 2] SelectKBest (MI auto, {mi_time_sec:.2f}s) -> {selected_cols}", flush=True)
    _save_mi_scores(norm_attack, condition, feature_names, selector.scores_)

    del X_train, X_val, X_test
    trim_memory()

    # BƯỚC 3: RandomizedSearchCV trên 100% (X_train_k, y_train)
    param_dist = {
        "n_estimators": [100, 200, 300, 400],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["sqrt", None],
        "class_weight": [None, "balanced"],
    }
    print(
        f"    [Bước 3] RandomizedSearchCV (n_iter=10, cv=3, scoring='f1', 100% mẫu={len(X_train_k):,}, "
        f"cv_jobs={cv_jobs}, rf_inner_jobs={rf_inner_jobs})...",
        flush=True,
    )
    t_search0 = time.perf_counter()
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            random_state=10,
            max_depth=None,
            n_jobs=rf_inner_jobs,
        ),
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1",
        cv=3,
        random_state=10,
        n_jobs=cv_jobs,
        refit=False,
    )
    search.fit(X_train_k, y_train)
    search_time_sec = time.perf_counter() - t_search0
    best_params = search.best_params_
    cv_best_f1 = float(search.best_score_)
    print(
        f"    [Bước 3] Hoàn tất ({search_time_sec:.2f}s) | CV F1={cv_best_f1:.4f} | best_params={best_params}",
        flush=True,
    )
    del search
    trim_memory()

    # BƯỚC 4: Huấn luyện mô hình tốt nhất trên 100% (X_train_k, y_train)
    print(f"    [Bước 4] Fit RandomForestClassifier tốt nhất trên {len(X_train_k):,} mẫu...", flush=True)
    t_train0 = time.perf_counter()
    rf = RandomForestClassifier(**best_params, random_state=10, n_jobs=-1)
    rf.fit(X_train_k, y_train)
    train_time_sec = time.perf_counter() - t_train0
    print(f"    [Bước 4] Hoàn tất fit trong {train_time_sec:.2f}s.", flush=True)

    del X_train_k, y_train
    trim_memory()

    # BƯỚC 5: Dò ngưỡng vector hóa trên Validation theo lưới làm tròn chính xác 0.05..0.94
    val_proba = rf.predict_proba(X_val_k)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.95, 0.01), 2)
    best_threshold, val_best_f1 = vectorized_threshold_sweep(
        y_val, val_proba, thresholds, default_threshold=0.50, initial_best_f1=-1.0
    )

    print(
        f"    [Bước 5] Ngưỡng tối ưu Validation (0.05..0.94): t={best_threshold:.2f} (Val F1={val_best_f1:.4f})",
        flush=True,
    )

    # BƯỚC 6: Đánh giá trên Held-out Test & Ghi Checkpoint ngay sau mỗi k
    t_test0 = time.perf_counter()
    test_proba = rf.predict_proba(X_test_k)[:, 1]
    test_time_sec = time.perf_counter() - t_test0
    test_metrics = evaluate_binary_predictions(y_test, test_proba, best_threshold)

    print(
        f"    [Bước 6] HELD-OUT TEST: F1={test_metrics['f1']:.4f} | Prec={test_metrics['precision']:.4f} "
        f"| Rec={test_metrics['recall']:.4f} | Acc={test_metrics['accuracy']:.4f} "
        f"| TP={test_metrics['TP']:,}, FP={test_metrics['FP']:,}, FN={test_metrics['FN']:,}, TN={test_metrics['TN']:,}",
        flush=True,
    )

    record = {
        "attack": norm_attack,
        "condition": condition,
        "k": int(k),
        "val_threshold": float(round(best_threshold, 2)),
        "val_f1": float(round(val_best_f1, 4)),
        "raw_val_threshold": float(best_threshold),
        "raw_val_f1": float(val_best_f1),
        "cv_best_f1": float(round(cv_best_f1, 4)),
        "test_accuracy": float(round(test_metrics["accuracy"], 4)),
        "test_precision": float(round(test_metrics["precision"], 4)),
        "test_recall": float(round(test_metrics["recall"], 4)),
        "test_f1": float(round(test_metrics["f1"], 4)),
        "TP": int(test_metrics["TP"]),
        "FP": int(test_metrics["FP"]),
        "FN": int(test_metrics["FN"]),
        "TN": int(test_metrics["TN"]),
        "n_estimators": int(best_params["n_estimators"]),
        "min_samples_leaf": int(best_params["min_samples_leaf"]),
        "max_features": normalize_nullable_param(best_params.get("max_features")),
        "class_weight": normalize_nullable_param(best_params.get("class_weight")),
        "selected_features": ";".join(selected_cols),
        "mi_time_sec": float(round(mi_time_sec, 2)),
        "search_time_sec": float(round(search_time_sec, 2)),
        "train_time_sec": float(round(train_time_sec, 2)),
        "test_time_sec": float(round(test_time_sec, 2)),
    }

    joblib.dump(
        {
            "model": rf,
            "selector": selector,
            "val_proba": val_proba.astype(np.float64),
            "test_proba": test_proba.astype(np.float64),
            "raw_val_f1": float(val_best_f1),
            "raw_val_threshold": float(best_threshold),
            "smote_time_sec": float(smote_time_sec),
        },
        K_CACHE_DIR / f"{prefix}_k{int(k)}.joblib",
        compress=3,
    )

    if metrics_csv.exists():
        df_old = pd.read_csv(metrics_csv)
        df_old = df_old[df_old["k"].astype(int) != int(k)]
        df_save = pd.concat([df_old, pd.DataFrame([record])], ignore_index=True)
    else:
        df_save = pd.DataFrame([record])

    df_save = df_save.sort_values("k").reset_index(drop=True)
    df_save.to_csv(metrics_csv, index=False)


def _finalize_experiment(attack_name: str, use_smote: bool, k_values: list):
    """Chọn cấu hình best_k theo raw_val_f1 (hòa điểm chọn k nhỏ hơn) và lưu model + .npy chính thức."""
    norm_attack = normalize_attack_name(attack_name)
    condition = "smote" if use_smote else "nosmote"
    prefix = f"paper_rf_{norm_attack.lower()}_{condition}"
    metrics_csv = PAPER_RESULTS_DIR / f"{prefix}_metrics.csv"

    df_final = pd.read_csv(metrics_csv).sort_values("k").reset_index(drop=True)

    # Lấy chính xác raw_val_f1 từ các bundle cache để tránh hòa điểm do làm tròn 4 chữ số
    best_k = None
    best_raw_val_f1 = -1.0
    for _, row in df_final.iterrows():
        k_val = int(row["k"])
        b_path = K_CACHE_DIR / f"{prefix}_k{k_val}.joblib"
        b_meta = joblib.load(b_path)
        rv = float(b_meta.get("raw_val_f1", row.get("raw_val_f1", row["val_f1"])))
        if rv > best_raw_val_f1 or (rv == best_raw_val_f1 and (best_k is None or k_val < best_k)):
            best_raw_val_f1 = rv
            best_k = k_val
        del b_meta
        trim_memory()

    best_row = df_final[df_final["k"].astype(int) == best_k].iloc[0].to_dict()
    bundle = joblib.load(K_CACHE_DIR / f"{prefix}_k{best_k}.joblib")

    _, _, _, y_val, _, y_test = get_all_splits_for_attack(norm_attack, drop_ip=True)

    np.save(PAPER_RESULTS_DIR / f"{prefix}_best_proba_val.npy", bundle["val_proba"])
    np.save(PAPER_RESULTS_DIR / f"{prefix}_best_proba_test.npy", bundle["test_proba"])
    np.save(PAPER_RESULTS_DIR / f"{prefix}_y_val.npy", y_val)
    np.save(PAPER_RESULTS_DIR / f"{prefix}_y_test.npy", y_test)

    model_path = PAPER_MODELS_DIR / f"{prefix}_best_model.joblib"
    selector_path = PAPER_MODELS_DIR / f"{prefix}_best_selector.joblib"
    meta_path = PAPER_MODELS_DIR / f"{prefix}_metadata.json"

    joblib.dump(bundle["model"], model_path, compress=3)
    joblib.dump(bundle["selector"], selector_path)

    meta_clean = build_model_metadata(
        best_row,
        extra={
            "raw_val_f1": float(bundle.get("raw_val_f1", best_row.get("raw_val_f1", best_row["val_f1"]))),
            "val_threshold_exact": float(bundle.get("raw_val_threshold", best_row["val_threshold"])),
            "smote_time_sec": round(float(bundle.get("smote_time_sec", 0.0)), 2),
            "pipeline_variant": "Paper-Exact (Pospichal et al. MDPI Information 2026)",
        },
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_clean, f, indent=4, default=json_serializer, allow_nan=False)

    del bundle
    trim_memory()

    # Dọn dẹp cache tạm của bài toán này sau khi đã lưu chính thức thành công
    for k in k_values:
        tmp_f = K_CACHE_DIR / f"{prefix}_k{int(k)}.joblib"
        if tmp_f.exists():
            tmp_f.unlink()
    mi_cache_path = K_CACHE_DIR / f"{prefix}_mi_scores.joblib"
    if mi_cache_path.exists():
        mi_cache_path.unlink()
    try:
        if K_CACHE_DIR.exists() and not any(K_CACHE_DIR.iterdir()):
            K_CACHE_DIR.rmdir()
    except OSError:
        pass

    rebuild_summary_csv()
    print(
        f"\n[HOÀN TẤT {prefix}] Best k={best_k} | Val Threshold={best_row['val_threshold']} "
        f"| Val F1={best_row['val_f1']:.4f} | Test F1={best_row['test_f1']:.4f}",
        flush=True,
    )


def train_paper_exact_single(
    attack_name: str,
    use_smote: bool = False,
    k_values: list | None = None,
    cv_jobs: int = 1,
    rf_inner_jobs: int = 3,
    force: bool = False,
):
    """Điều phối chạy từng mốc k trong subprocess cô lập bộ nhớ, sau đó tổng hợp best_k."""
    PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    K_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if k_values is None:
        k_values = list(DEFAULT_K_VALUES)
    if not k_values:
        raise ValueError("k_values must not be empty.")

    norm_attack = normalize_attack_name(attack_name)
    condition = "smote" if use_smote else "nosmote"
    prefix = f"paper_rf_{norm_attack.lower()}_{condition}"
    metrics_csv = PAPER_RESULTS_DIR / f"{prefix}_metrics.csv"

    if (not force) and is_experiment_complete(PAPER_RESULTS_DIR, PAPER_MODELS_DIR, prefix, k_values):
        print(f"[>] BỎ QUA (Đã hoàn tất đủ {k_values} + model/proba): {prefix}", flush=True)
        return pd.read_csv(metrics_csv)

    if force:
        if metrics_csv.exists():
            metrics_csv.unlink()
        mi_cache_path = K_CACHE_DIR / f"{prefix}_mi_scores.joblib"
        if mi_cache_path.exists():
            mi_cache_path.unlink()
        for k in k_values:
            tmp_f = K_CACHE_DIR / f"{prefix}_k{int(k)}.joblib"
            if tmp_f.exists():
                tmp_f.unlink()

    print("\n" + "=" * 78)
    print(f" [PAPER-EXACT BASELINE] {norm_attack.upper()} | {condition.upper()} | cv_jobs={cv_jobs}, rf_inner_jobs={rf_inner_jobs}")
    print("=" * 78, flush=True)

    child_env = os.environ.copy()
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    # Giới hạn glibc arenas và hạ ngưỡng mmap để trả bộ nhớ ngay lập tức về cho OS sau mỗi cây RF
    child_env.setdefault("MALLOC_ARENA_MAX", "2")
    child_env.setdefault("MALLOC_MMAP_THRESHOLD_", "65536")
    child_env.setdefault("MALLOC_TRIM_THRESHOLD_", "65536")

    for k in k_values:
        k = int(k)
        if (not force) and is_k_cached(prefix, k):
            df_c = pd.read_csv(metrics_csv)
            r_c = df_c[df_c["k"].astype(int) == k].iloc[0]
            print(
                f"\n>>> [k = {k}] Đã có trong checkpoint (val_f1={r_c['val_f1']:.4f}, test_f1={r_c['test_f1']:.4f}) -> Bỏ qua.",
                flush=True,
            )
            continue

        cmd = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--worker-single-k",
            str(k),
            "--attacks",
            norm_attack,
            "--conditions",
            condition,
            "--cv-jobs",
            str(cv_jobs),
            "--rf-inner-jobs",
            str(rf_inner_jobs),
        ]
        subprocess.run(cmd, env=child_env, check=True)

    # Chốt best_k trong một subprocess sạch bộ nhớ
    cmd_fin = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--worker-finalize",
        "--attacks",
        norm_attack,
        "--conditions",
        condition,
        "--k-values",
        *[str(int(kv)) for kv in k_values],
    ]
    subprocess.run(cmd_fin, env=child_env, check=True)
    return pd.read_csv(metrics_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện nhánh thực nghiệm Paper-Exact Baseline (Pospíchal et al. 2026)"
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        default=list(TARGET_ATTACKS),
        help="Danh sách tấn công cần chạy (mặc định: cả 5 loại)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=["nosmote", "smote"],
        default=["nosmote", "smote"],
        help="Điều kiện cân bằng lớp cần chạy (mặc định: nosmote smote)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_K_VALUES),
        help="Dải số lượng đặc trưng k (mặc định: 3 5 7 10)",
    )
    parser.add_argument(
        "--cv-jobs",
        type=int,
        default=1,
        help="Số tiến trình song song cho các fold CV trong RandomizedSearchCV (mặc định: 1)",
    )
    parser.add_argument(
        "--rf-inner-jobs",
        type=int,
        default=3,
        help="Số luồng CPU cho mỗi RandomForestClassifier bên trong CV (mặc định: 3 để an toàn RAM trên máy 16GB)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Buộc chạy lại từ đầu kể cả khi kết quả của bài toán đó đã tồn tại",
    )
    parser.add_argument("--worker-single-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-finalize", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_single_k is not None:
        _run_worker_single_k(
            attack_name=args.attacks[0],
            use_smote=(args.conditions[0] == "smote"),
            k=args.worker_single_k,
            cv_jobs=args.cv_jobs,
            rf_inner_jobs=args.rf_inner_jobs,
        )
        return

    if args.worker_finalize:
        _finalize_experiment(
            attack_name=args.attacks[0],
            use_smote=(args.conditions[0] == "smote"),
            k_values=args.k_values,
        )
        return

    t_all0 = time.perf_counter()
    for att in args.attacks:
        for cond in args.conditions:
            train_paper_exact_single(
                attack_name=att,
                use_smote=(cond == "smote"),
                k_values=args.k_values,
                cv_jobs=args.cv_jobs,
                rf_inner_jobs=args.rf_inner_jobs,
                force=args.force,
            )

    summary_df = rebuild_summary_csv()
    print("\n" + "=" * 78)
    print(f" TOÀN BỘ CHUỖI PAPER-EXACT HOÀN TẤT TRONG {(time.perf_counter() - t_all0) / 60.0:.2f} PHÚT!")
    if summary_df is not None:
        print(summary_df.to_string(index=False))
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
