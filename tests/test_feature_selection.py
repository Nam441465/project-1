import warnings
import joblib
import numpy as np
import pandas as pd
import pytest

from feature_selection import (
    DISCRETE_FEATURES,
    MutualInfoScorer,
    apply_smote,
    build_selector_from_scores,
    compute_mi_scores,
    fit_feature_selector,
    transform_features,
)


def test_discrete_features_constant():
    assert DISCRETE_FEATURES == {"PROTOCOL", "L7_PROTO", "TCP_FLAGS"}


def test_apply_smote_target_samples_and_skip(synthetic_binary_xy):
    X, y = synthetic_binary_xy
    # Minority is 20, majority is 80
    X_res, y_res = apply_smote(X, y, target_samples=50, random_state=10)
    counts = np.bincount(y_res)
    assert counts[0] == 80
    assert counts[1] == 50
    assert isinstance(X_res, pd.DataFrame)
    assert list(X_res.columns) == list(X.columns)

    # When minority >= target_samples, SMOTE should be skipped cleanly
    X_skip, y_skip = apply_smote(X, y, target_samples=15, random_state=10)
    assert len(X_skip) == len(X)
    assert np.array_equal(y_skip, y)


def test_apply_smote_auto_balances(synthetic_binary_xy):
    X, y = synthetic_binary_xy
    X_res, y_res = apply_smote(X, y, target_samples="auto", random_state=10)
    counts = np.bincount(y_res)
    assert counts[0] == counts[1] == 80


def test_fit_and_transform_features_preserves_l7_proto_decimals(synthetic_binary_xy):
    X, y = synthetic_binary_xy
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector, selected_feats, scores_df = fit_feature_selector(
            X, y, k=5, random_state=10, max_samples_for_mi=60
        )
    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert user_warnings == []

    assert len(selected_feats) == 5
    assert list(scores_df.columns) == ["Feature", "MI_Score"]
    assert (scores_df["MI_Score"] >= 0).all()
    assert scores_df["MI_Score"].is_monotonic_decreasing

    X_trans = transform_features(selector, X)
    assert X_trans.shape == (len(X), 5)
    assert list(X_trans.columns) == selected_feats
    assert X_trans.index.equals(X.index)

    # With k=10 (all features), L7_PROTO fractional values must remain unmodified
    selector_all, _, _ = fit_feature_selector(X, y, k=99, random_state=10, max_samples_for_mi=200)
    X_all = transform_features(selector_all, X)
    assert X_all.shape[1] == X.shape[1]
    assert (X_all["L7_PROTO"] == X["L7_PROTO"]).all()
    assert (X_all["L7_PROTO"] % 1 != 0).any()


def test_mutual_info_scorer_ndarray_and_joblib_roundtrip(synthetic_binary_xy, tmp_path):
    X, y = synthetic_binary_xy
    discrete_mask = [col in DISCRETE_FEATURES for col in X.columns]
    scorer = MutualInfoScorer(discrete_mask=discrete_mask, random_state=10, max_samples_for_mi=50)
    scores = scorer(X.to_numpy(), y)
    assert scores.shape == (X.shape[1],)
    assert np.all(scores >= 0)

    selector, _, _ = fit_feature_selector(X, y, k=4, random_state=10, max_samples_for_mi=200)
    out_path = tmp_path / "selector.joblib"
    joblib.dump(selector, out_path)
    loaded = joblib.load(out_path)
    assert np.array_equal(loaded.get_support(), selector.get_support())
    pd.testing.assert_frame_equal(transform_features(loaded, X), transform_features(selector, X))


def test_compute_mi_scores_and_build_selector_parity(synthetic_binary_xy):
    X, y = synthetic_binary_xy
    scorer, feat_names, scores, scores_df_once = compute_mi_scores(
        X, y, random_state=10, max_samples_for_mi=200
    )
    sel_a, feats_a, df_a = build_selector_from_scores(
        feature_names=feat_names, scores=scores, k=5, scorer=scorer, has_feature_names=True
    )
    sel_b, feats_b, df_b = fit_feature_selector(
        X, y, k=5, random_state=10, max_samples_for_mi=200
    )

    assert feats_a == feats_b
    assert np.array_equal(sel_a.get_support(), sel_b.get_support())
    pd.testing.assert_frame_equal(scores_df_once, df_a)
    pd.testing.assert_frame_equal(df_a, df_b)
