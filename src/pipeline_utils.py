"""Shared non-pickled helpers for NIDS training, evaluation, and export scripts."""

from __future__ import annotations

import ctypes
import gc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

TARGET_ATTACKS: list[str] = [
    "DoS",
    "Exploits",
    "Backdoor",
    "Analysis",
    "Reconnaissance",
]
DEFAULT_K_VALUES: list[int] = [3, 5, 7, 10]


def trim_memory() -> None:
    """Run garbage collection and ask glibc to release freed pages to the OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def vectorized_threshold_metrics(
    y_true: np.ndarray | Iterable[int],
    proba: np.ndarray | Iterable[float],
    thresholds: np.ndarray | Iterable[float],
) -> dict[str, np.ndarray]:
    """Compute TP, FP, FN, TN, F1, precision, recall, and accuracy in O(N log N).

    Predictions are positive whenever ``proba >= threshold`` for each threshold in
    ``thresholds``. Ordering of ``thresholds`` is preserved in all returned arrays.
    """
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    p_arr = np.asarray(proba, dtype=np.float64).ravel()
    t_arr = np.asarray(thresholds, dtype=np.float64).ravel()

    if y_arr.shape[0] != p_arr.shape[0]:
        raise ValueError(
            f"Length mismatch between y_true ({y_arr.shape[0]}) and proba ({p_arr.shape[0]})."
        )
    if t_arr.shape[0] == 0:
        raise ValueError("thresholds must not be empty.")

    n_samples = y_arr.shape[0]
    if n_samples == 0:
        zeros_i = np.zeros(t_arr.shape[0], dtype=np.int64)
        zeros_f = np.zeros(t_arr.shape[0], dtype=np.float64)
        return {
            "thresholds": t_arr,
            "tp": zeros_i,
            "fp": zeros_i.copy(),
            "fn": zeros_i.copy(),
            "tn": zeros_i.copy(),
            "f1": zeros_f,
            "precision": zeros_f.copy(),
            "recall": zeros_f.copy(),
            "accuracy": zeros_f.copy(),
        }

    order = np.argsort(-p_arr, kind="mergesort")
    p_sorted = p_arr[order]
    y_sorted = y_arr[order]

    # -p_sorted is non-decreasing; -p_sorted <= -t <=> p_sorted >= t
    neg_p_sorted = -p_sorted
    counts = np.searchsorted(neg_p_sorted, -t_arr, side="right")

    tp_cum = np.empty(n_samples + 1, dtype=np.int64)
    tp_cum[0] = 0
    np.cumsum(y_sorted, out=tp_cum[1:])

    tp = tp_cum[counts]
    fp = counts - tp
    total_pos = int(tp_cum[-1])
    fn = total_pos - tp
    tn = n_samples - total_pos - fp

    denom_f1 = 2 * tp + fp + fn
    f1 = np.where(denom_f1 > 0, (2.0 * tp) / np.where(denom_f1 > 0, denom_f1, 1), 0.0)
    precision = np.where(counts > 0, tp / np.where(counts > 0, counts, 1), 0.0)
    if total_pos > 0:
        recall = tp / float(total_pos)
    else:
        recall = np.zeros(t_arr.shape[0], dtype=np.float64)
    accuracy = (tp + tn) / float(n_samples)

    return {
        "thresholds": t_arr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def vectorized_threshold_sweep(
    y_true: np.ndarray | Iterable[int],
    proba: np.ndarray | Iterable[float],
    thresholds: np.ndarray | Iterable[float],
    default_threshold: float = 0.5,
    initial_best_f1: float = -1.0,
) -> tuple[float, float]:
    """Find the first threshold in ``thresholds`` maximizing binary F1 score."""
    metrics = vectorized_threshold_metrics(y_true, proba, thresholds)
    f1_scores = metrics["f1"]
    t_arr = metrics["thresholds"]

    best_idx = int(np.argmax(f1_scores))
    best_f1 = float(f1_scores[best_idx])
    if best_f1 > initial_best_f1:
        return float(t_arr[best_idx]), best_f1
    return float(default_threshold), float(initial_best_f1)


def evaluate_binary_predictions(
    y_true: np.ndarray | Iterable[int],
    proba: np.ndarray | Iterable[float],
    threshold: float,
) -> dict[str, float | int]:
    """Evaluate binary predictions at a fixed threshold with safe [0, 1] confusion matrix."""
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    p_arr = np.asarray(proba, dtype=np.float64).ravel()
    y_pred = (p_arr >= float(threshold)).astype(np.int64)

    acc = accuracy_score(y_arr, y_pred)
    prec = precision_score(y_arr, y_pred, zero_division=0)
    rec = recall_score(y_arr, y_pred, zero_division=0)
    f1 = f1_score(y_arr, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_arr, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def select_best_row(df: pd.DataFrame) -> pd.Series:
    """Deterministically select the best row by raw_val_f1 (desc) then k (asc)."""
    if df.empty:
        raise ValueError("Cannot select best row from an empty metrics DataFrame.")

    use_raw = "raw_val_f1" in df.columns and df["raw_val_f1"].notna().all()
    score_col = "raw_val_f1" if use_raw else "val_f1"
    work = df.copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce").fillna(-1.0)
    work["k"] = pd.to_numeric(work["k"], errors="coerce").fillna(10**9).astype(int)
    sorted_df = work.sort_values(by=[score_col, "k"], ascending=[False, True], kind="mergesort")
    return df.loc[sorted_df.index[0]]


def json_serializer(obj: Any) -> Any:
    """Serialize numpy scalars/arrays, Paths, and NA values into valid JSON primitives."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if np.isnan(val) else val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def normalize_nullable_param(value: Any) -> Any:
    """Normalize 'None'/'nan'/NaN parameter representations to Python None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in {"none", "nan", ""}:
        return None
    return value


def _sanitize_json_value(value: Any) -> Any:
    """Recursively convert numpy types and NaN/NA floats into strict JSON-safe primitives."""
    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json_value(v) for v in value]
    if isinstance(value, float):
        return None if np.isnan(value) else value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        val = float(value)
        return None if np.isnan(val) else val
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_json_value(v) for v in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def build_model_metadata(
    record: dict[str, Any] | pd.Series,
    extra: dict[str, Any] | None = None,
    nullable_keys: tuple[str, ...] = ("max_features", "class_weight", "max_depth"),
) -> dict[str, Any]:
    """Build a unified, strict-JSON-safe metadata dict with null instead of 'None'/NaN."""
    meta: dict[str, Any] = dict(record)
    if extra:
        meta.update(extra)

    for key in nullable_keys:
        if key in meta:
            meta[key] = normalize_nullable_param(meta[key])

    return _sanitize_json_value(meta)


def subsample_minority_preserving(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    max_samples: int,
    random_state: int = 10,
):
    """Keep all minority (y == 1) rows and sample majority (y == 0) up to max_samples."""
    y_arr = np.asarray(y)
    if len(y_arr) <= max_samples:
        return X, y_arr

    minority_idx = np.where(y_arr == 1)[0]
    majority_idx = np.where(y_arr == 0)[0]
    n_min = len(minority_idx)
    n_maj_needed = max(1, int(max_samples) - n_min)

    rng = np.random.RandomState(random_state)
    sampled_maj = rng.choice(
        majority_idx,
        size=min(n_maj_needed, len(majority_idx)),
        replace=False,
    )
    combined_idx = np.concatenate([minority_idx, sampled_maj])
    rng.shuffle(combined_idx)

    X_sub = X.iloc[combined_idx] if isinstance(X, pd.DataFrame) else X[combined_idx]
    y_sub = y_arr[combined_idx]
    return X_sub, y_sub


def compute_cv_smote_strategy(
    y_train_raw: np.ndarray,
    y_search: np.ndarray,
    smote_target: int | str,
    cv: int = 3,
) -> tuple[dict[int, int] | str, int]:
    """Compute fold-safe SMOTE sampling_strategy and k_neighbors for ImbPipeline CV."""
    n_min_in_search = int(np.sum(y_search == 1))
    min_samples_per_fold = (cv - 1) * n_min_in_search // cv
    smote_k_neighbors = min(5, max(1, min_samples_per_fold - 1))

    if smote_target == "auto":
        return "auto", smote_k_neighbors

    n_maj_raw = max(1, int(np.sum(y_train_raw == 0)))
    target_ratio = float(smote_target) / n_maj_raw
    n_maj_fold = int(np.sum(y_search == 0) * (cv - 1) / cv)
    n_min_fold_actual = int(np.ceil(np.sum(y_search == 1) * (cv - 1) / cv))
    # + 10 safe margin ensures SMOTE never receives target < n_min_in_fold on rounding edges
    n_min_fold_target = max(n_min_fold_actual + 10, int(round(n_maj_fold * target_ratio)))
    cv_sampling_strategy = {1: min(n_maj_fold, n_min_fold_target)}
    return cv_sampling_strategy, smote_k_neighbors


def is_experiment_complete(
    results_dir: Path,
    models_dir: Path,
    prefix: str,
    k_values: Iterable[int],
) -> bool:
    """Return True only when CSV covers all k_values and all model/proba/meta files exist."""
    metrics_path = results_dir / f"{prefix}_metrics.csv"
    required_files = [
        metrics_path,
        models_dir / f"{prefix}_best_model.joblib",
        models_dir / f"{prefix}_best_selector.joblib",
        models_dir / f"{prefix}_metadata.json",
        results_dir / f"{prefix}_best_proba_val.npy",
        results_dir / f"{prefix}_best_proba_test.npy",
        results_dir / f"{prefix}_y_val.npy",
        results_dir / f"{prefix}_y_test.npy",
    ]
    if not all(path.exists() for path in required_files):
        return False

    try:
        df_m = pd.read_csv(metrics_path)
        if df_m.empty or "k" not in df_m.columns:
            return False
        done_ks = set(df_m["k"].astype(int).tolist())
        return {int(k) for k in k_values}.issubset(done_ks)
    except Exception:
        return False
