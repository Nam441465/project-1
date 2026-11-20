import numpy as np
import pandas as pd
import pytest

import preprocessor
from preprocessor import (
    ATTACK_MAP,
    FEATURE_COLS,
    filter_binary_attack,
    load_split,
    normalize_attack_name,
    prepare_features_and_labels,
)


def test_feature_cols_schema():
    assert len(FEATURE_COLS) == 10
    assert FEATURE_COLS == [
        "L4_SRC_PORT",
        "L4_DST_PORT",
        "PROTOCOL",
        "L7_PROTO",
        "IN_BYTES",
        "OUT_BYTES",
        "IN_PKTS",
        "OUT_PKTS",
        "TCP_FLAGS",
        "FLOW_DURATION_MILLISECONDS",
    ]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dos", "DoS"),
        (" DoS ", "DoS"),
        ("exploit", "Exploits"),
        ("exploits", "Exploits"),
        ("backdoor", "Backdoor"),
        ("analysis", "Analysis"),
        ("recon", "Reconnaissance"),
        ("reconnaissance", "Reconnaissance"),
        ("fuzzers", "Fuzzers"),
        ("Generic", "Generic"),
    ],
)
def test_normalize_attack_name_valid(raw: str, expected: str):
    assert normalize_attack_name(raw) == expected


def test_normalize_attack_name_invalid_raises():
    with pytest.raises(ValueError, match="không hợp lệ"):
        normalize_attack_name("unknown_zero_day")


def test_attack_map_covers_five_targets():
    assert set(ATTACK_MAP.values()) == {
        "DoS",
        "Exploits",
        "Backdoor",
        "Analysis",
        "Reconnaissance",
    }


def test_filter_binary_attack_isolates_benign_and_target(synthetic_flow_df: pd.DataFrame):
    orig_cols = list(synthetic_flow_df.columns)
    filtered = filter_binary_attack(synthetic_flow_df, "recon")

    # Input DataFrame must not be mutated
    assert list(synthetic_flow_df.columns) == orig_cols
    assert "target_label" not in synthetic_flow_df.columns

    # Only Benign and Reconnaissance should remain (other attacks such as DoS/Fuzzers dropped)
    assert set(filtered["Attack"].unique()) == {"Benign", "Reconnaissance"}
    assert (filtered.loc[filtered["Attack"] == "Benign", "target_label"] == 0).all()
    assert (filtered.loc[filtered["Attack"] == "Reconnaissance", "target_label"] == 1).all()
    assert len(filtered) == 80 + 20


def test_prepare_features_and_labels_drop_ip(synthetic_flow_df: pd.DataFrame):
    filtered = filter_binary_attack(synthetic_flow_df, "dos")
    X_no_ip, y = prepare_features_and_labels(filtered, drop_ip=True)
    assert list(X_no_ip.columns) == FEATURE_COLS
    assert X_no_ip.shape == (100, 10)
    assert y.dtype == np.int32
    assert set(np.unique(y)) == {0, 1}

    X_with_ip, _ = prepare_features_and_labels(filtered, drop_ip=False)
    assert "IPV4_SRC_ADDR" in X_with_ip.columns
    assert "IPV4_DST_ADDR" in X_with_ip.columns
    assert "target_label" not in X_with_ip.columns
    assert "Attack" not in X_with_ip.columns
    assert "Label" not in X_with_ip.columns


def test_prepare_features_and_labels_fallback_label_or_error(synthetic_flow_df: pd.DataFrame):
    X, y = prepare_features_and_labels(synthetic_flow_df, drop_ip=True)
    assert len(X) == len(y) == len(synthetic_flow_df)

    no_label_df = synthetic_flow_df.drop(columns=["Label"])
    with pytest.raises(ValueError, match="nhãn"):
        prepare_features_and_labels(no_label_df, drop_ip=True)


def test_load_split_validation(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="không hợp lệ"):
        load_split("invalid_split")

    monkeypatch.setattr(preprocessor, "PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        load_split("train")
