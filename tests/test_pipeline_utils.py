import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from pipeline_utils import (
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
    vectorized_threshold_metrics,
    vectorized_threshold_sweep,
)


def test_constants():
    assert TARGET_ATTACKS == ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]
    assert DEFAULT_K_VALUES == [3, 5, 7, 10]


@pytest.mark.parametrize(
    "thresholds",
    [
        np.round(np.arange(0.01, 1.00, 0.01), 2),
        np.round(np.arange(0.05, 0.95, 0.01), 2),
        np.round(np.linspace(0.99, 0.01, 99), 2),
    ],
)
def test_vectorized_threshold_sweep_matches_bruteforce_f1(thresholds):
    rng = np.random.default_rng(123)
    y_true = rng.integers(0, 2, size=500)
    # Include deliberate probability ties around thresholds
    proba = np.round(rng.uniform(0.0, 1.0, size=500), 2)

    metrics = vectorized_threshold_metrics(y_true, proba, thresholds)
    best_t, best_f1 = vectorized_threshold_sweep(y_true, proba, thresholds)

    brute_f1s = []
    for idx, th in enumerate(thresholds):
        y_pred = (proba >= th).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        assert metrics["tp"][idx] == tp
        assert metrics["fp"][idx] == fp
        assert metrics["fn"][idx] == fn
        assert metrics["tn"][idx] == tn
        assert metrics["f1"][idx] == pytest.approx(f1, abs=1e-12)
        assert metrics["precision"][idx] == pytest.approx(prec, abs=1e-12)
        assert metrics["recall"][idx] == pytest.approx(rec, abs=1e-12)
        brute_f1s.append(f1)

    expected_idx = int(np.argmax(brute_f1s))
    assert best_t == pytest.approx(float(thresholds[expected_idx]), abs=1e-12)
    assert best_f1 == pytest.approx(float(brute_f1s[expected_idx]), abs=1e-12)


def test_vectorized_threshold_edge_cases():
    # Empty samples
    m_empty = vectorized_threshold_metrics([], [], [0.5, 0.8])
    assert list(m_empty["tp"]) == [0, 0]
    assert list(m_empty["f1"]) == [0.0, 0.0]

    # All-zero labels -> recall & f1 stay 0.0, fallback initial_best_f1 respected
    t_fb, f1_fb = vectorized_threshold_sweep(
        [0, 0, 0], [0.2, 0.5, 0.9], [0.3, 0.7], default_threshold=0.42, initial_best_f1=0.0
    )
    assert t_fb == 0.42
    assert f1_fb == 0.0

    with pytest.raises(ValueError, match="mismatch"):
        vectorized_threshold_metrics([0, 1], [0.5], [0.5])

    with pytest.raises(ValueError, match="empty"):
        vectorized_threshold_metrics([0, 1], [0.2, 0.8], [])


def test_evaluate_binary_predictions_all_zero_and_all_one():
    res_all_zero = evaluate_binary_predictions([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4], threshold=0.9)
    assert res_all_zero["TP"] == 0
    assert res_all_zero["FP"] == 0
    assert res_all_zero["FN"] == 0
    assert res_all_zero["TN"] == 4
    assert res_all_zero["accuracy"] == 1.0
    assert res_all_zero["f1"] == 0.0

    res_all_one = evaluate_binary_predictions([1, 1, 1], [0.8, 0.9, 0.95], threshold=0.5)
    assert res_all_one["TP"] == 3
    assert res_all_one["TN"] == 0
    assert res_all_one["f1"] == 1.0


def test_select_best_row_tie_breaking():
    df = pd.DataFrame(
        [
            {"k": 10, "val_f1": 0.8863, "raw_val_f1": 0.886301},
            {"k": 3, "val_f1": 0.8863, "raw_val_f1": 0.886349},
            {"k": 7, "val_f1": 0.8863, "raw_val_f1": 0.886349},
        ]
    )
    best = select_best_row(df)
    assert int(best["k"]) == 3

    with pytest.raises(ValueError):
        select_best_row(df.iloc[0:0])


def test_metadata_and_json_strict_serialization(tmp_path):
    assert normalize_nullable_param("None") is None
    assert normalize_nullable_param("nan") is None
    assert normalize_nullable_param(np.nan) is None
    assert normalize_nullable_param("sqrt") == "sqrt"

    assert json_serializer(np.int64(7)) == 7
    assert json_serializer(np.float64(np.nan)) is None
    assert json_serializer(tmp_path) == str(tmp_path)

    meta = build_model_metadata(
        {
            "attack": "Analysis",
            "k": np.int64(3),
            "max_features": np.nan,
            "class_weight": "None",
            "max_depth": None,
            "val_f1": np.float64(0.8863),
        },
        extra={"nan_metric": float("nan")},
    )
    dumped = json.dumps(meta, allow_nan=False)
    loaded = json.loads(dumped)
    assert loaded["max_features"] is None
    assert loaded["class_weight"] is None
    assert loaded["max_depth"] is None
    assert loaded["nan_metric"] is None
    assert loaded["k"] == 3


def test_subsample_minority_preserving_and_cv_smote():
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)})
    y = np.array([1] * 25 + [0] * 175)

    X_sub, y_sub = subsample_minority_preserving(X, y, max_samples=100, random_state=10)
    assert len(X_sub) == 100
    assert int(np.sum(y_sub == 1)) == 25
    assert int(np.sum(y_sub == 0)) == 75

    # No-op when already <= max_samples
    X_same, y_same = subsample_minority_preserving(X, y, max_samples=500, random_state=10)
    assert len(X_same) == 200
    assert np.array_equal(y_same, y)

    strat_auto, k_auto = compute_cv_smote_strategy(y, y_sub, smote_target="auto", cv=3)
    assert strat_auto == "auto"
    assert 1 <= k_auto <= 5

    strat_num, k_num = compute_cv_smote_strategy(y, y_sub, smote_target=100, cv=3)
    assert isinstance(strat_num, dict) and 1 in strat_num
    n_min_fold_actual = int(np.ceil(np.sum(y_sub == 1) * 2 / 3))
    assert strat_num[1] >= n_min_fold_actual + 10
    assert 1 <= k_num <= 5


def test_is_experiment_complete(tmp_path):
    res_dir = tmp_path / "results"
    mod_dir = res_dir / "models"
    mod_dir.mkdir(parents=True)
    prefix = "rf_dos_nosmote"

    assert not is_experiment_complete(res_dir, mod_dir, prefix, [3, 5, 7, 10])

    pd.DataFrame({"k": [3, 5, 7, 10], "val_f1": [0.8, 0.85, 0.86, 0.87]}).to_csv(
        res_dir / f"{prefix}_metrics.csv", index=False
    )
    for path in [
        mod_dir / f"{prefix}_best_model.joblib",
        mod_dir / f"{prefix}_best_selector.joblib",
        mod_dir / f"{prefix}_metadata.json",
        res_dir / f"{prefix}_best_proba_val.npy",
        res_dir / f"{prefix}_best_proba_test.npy",
        res_dir / f"{prefix}_y_val.npy",
        res_dir / f"{prefix}_y_test.npy",
    ]:
        path.write_bytes(b"stub")

    assert is_experiment_complete(res_dir, mod_dir, prefix, [3, 5, 7, 10])
    assert not is_experiment_complete(res_dir, mod_dir, prefix, [3, 5, 7, 10, 12])
