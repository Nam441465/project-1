import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from feature_selection import transform_features
from preprocessor import FEATURE_COLS, get_all_splits_for_attack, load_split


ATTACKS = ["dos", "exploits", "backdoor", "analysis", "reconnaissance"]
CONDITIONS = ["nosmote", "smote"]


@pytest.mark.slow
def test_real_parquet_splits_match_metadata(processed_dir: Path):
    meta_path = processed_dir / "split_metadata.json"
    summary_path = processed_dir / "split_summary.csv"
    assert meta_path.exists()
    assert summary_path.exists()

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("heldout_test")

    assert len(train_df) == meta["train_records"] == 1104497
    assert len(val_df) == meta["val_records"] == 155586
    assert len(test_df) == meta["heldout_test_records"] == 363035
    assert train_df.isnull().sum().sum() == 0
    assert val_df.isnull().sum().sum() == 0
    assert test_df.isnull().sum().sum() == 0

    for col in FEATURE_COLS + ["Attack", "Label"]:
        assert col in test_df.columns


@pytest.mark.slow
def test_get_all_splits_for_analysis():
    X_tr, y_tr, X_v, y_v, X_te, y_te = get_all_splits_for_attack("Analysis")
    assert len(X_tr) == 1055226 + 1358
    assert len(X_v) == 148645 + 191
    assert len(X_te) == 346841 + 446
    assert (np.bincount(y_tr) == [1055226, 1358]).all()
    assert (np.bincount(y_v) == [148645, 191]).all()
    assert (np.bincount(y_te) == [346841, 446]).all()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("subdir", "prefix"),
    [
        ("baseline", "rf"),
        ("paper_original", "paper_rf"),
    ],
)
def test_all_20_model_bundles_exist_and_metadata_valid(results_dir: Path, subdir: str, prefix: str):
    models_dir = results_dir / subdir / "models"
    assert models_dir.is_dir()

    for attack in ATTACKS:
        for cond in CONDITIONS:
            stem = f"{prefix}_{attack}_{cond}"
            model_path = models_dir / f"{stem}_best_model.joblib"
            selector_path = models_dir / f"{stem}_best_selector.joblib"
            meta_path = models_dir / f"{stem}_metadata.json"

            assert model_path.exists(), f"Missing {model_path}"
            assert selector_path.exists(), f"Missing {selector_path}"
            assert meta_path.exists(), f"Missing {meta_path}"

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            assert meta["attack"].lower() == attack
            assert meta["condition"] == cond
            assert 1 <= int(meta["k"]) <= len(FEATURE_COLS)
            assert 0.0 < float(meta["val_threshold"]) < 1.0
            assert 0.0 <= float(meta["test_f1"]) <= 1.0
            selected_feats = [f for f in meta["selected_features"].split(";") if f]
            assert len(selected_feats) == int(meta["k"])


@pytest.mark.slow
@pytest.mark.parametrize(
    ("subdir", "stem"),
    [
        ("baseline", "rf_analysis_nosmote"),
        ("paper_original", "paper_rf_analysis_nosmote"),
    ],
)
def test_sample_joblib_models_predict_valid_probabilities(results_dir: Path, subdir: str, stem: str):
    models_dir = results_dir / subdir / "models"
    model = joblib.load(models_dir / f"{stem}_best_model.joblib")
    selector = joblib.load(models_dir / f"{stem}_best_selector.joblib")

    test_df = load_split("heldout_test").iloc[:64]
    X_slice = test_df[FEATURE_COLS]
    X_sel = transform_features(selector, X_slice)

    proba = model.predict_proba(X_sel)
    assert proba.shape == (64, 2)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.slow
def test_summary_and_cost_csv_schemas(results_dir: Path):
    for rel_path in [
        "baseline/summary_rf_comparison.csv",
        "paper_original/summary_paper_rf_comparison.csv",
    ]:
        df = pd.read_csv(results_dir / rel_path)
        assert len(df) == 10
        assert set(["Attack", "Condition", "Best_k", "Threshold", "Precision", "Recall", "F1_Score", "FP", "FN"]).issubset(
            df.columns
        )
        assert df[["Best_k", "Threshold", "F1_Score"]].notna().all().all()

    cost_df = pd.read_csv(results_dir / "improved" / "cost_sensitive_evaluation.csv")
    assert len(cost_df) == 20
    assert set(cost_df["Cost_Ratio_c"].unique()) == {1, 5, 10, 20}
    assert set(cost_df["Model"].unique()) == {"RF"}
