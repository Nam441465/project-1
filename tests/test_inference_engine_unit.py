import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from inference_engine import (
    NETFLOW_FEATURES,
    NIDSInferenceEngine,
    build_consensus_matrix,
    get_near_miss_bounds,
)
from protocol_decoder import (
    DEFAULT_BENIGN_FLOW,
    analyze_tier2_dpi_heuristics,
    decode_l4_protocol,
    decode_l7_protocol,
    decode_port,
    decode_tcp_flags,
)


def _create_mini_bundle(
    base_dir: Path,
    subdir: str,
    prefix: str,
    attack: str,
    condition: str,
    val_threshold: float,
    selected_features: list[str],
    fit_with_df: bool = True,
) -> None:
    res_dir = base_dir / subdir
    mod_dir = res_dir / "models"
    mod_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(10)
    X_benign = rng.uniform(0, 50, size=(30, len(selected_features)))
    X_attack = rng.uniform(500, 1000, size=(30, len(selected_features)))
    X_arr = np.vstack([X_benign, X_attack])
    y_arr = np.array([0] * 30 + [1] * 30, dtype=np.int32)

    X_fit = pd.DataFrame(X_arr, columns=selected_features) if fit_with_df else X_arr
    clf = RandomForestClassifier(n_estimators=5, random_state=10)
    clf.fit(X_fit, y_arr)

    joblib.dump(clf, mod_dir / f"{prefix}_best_model.joblib")
    meta = {
        "attack": attack,
        "condition": condition,
        "k": len(selected_features),
        "val_threshold": val_threshold,
        "val_f1": 0.95,
        "test_f1": 0.94,
        "test_precision": 0.93,
        "test_recall": 0.95,
        "test_accuracy": 0.96,
        "selected_features": ";".join(selected_features),
        "train_time_sec": 0.1,
    }
    (mod_dir / f"{prefix}_metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Save validation arrays for mtime-keyed threshold calibration cache
    p_val = clf.predict_proba(X_fit)[:, 1]
    np.save(res_dir / f"{prefix}_y_val.npy", y_arr)
    np.save(res_dir / f"{prefix}_best_proba_val.npy", p_val)


@pytest.fixture
def isolated_engine_dir(tmp_path: Path) -> Path:
    feats = ["L4_DST_PORT", "PROTOCOL", "IN_BYTES"]
    _create_mini_bundle(tmp_path, "baseline", "rf_dos_nosmote", "DoS", "nosmote", 0.60, feats, fit_with_df=False)
    _create_mini_bundle(tmp_path, "improved", "gb_dos_nosmote", "DoS", "nosmote", 0.80, feats, fit_with_df=True)
    _create_mini_bundle(tmp_path, "paper_original", "paper_rf_dos_nosmote", "DoS", "nosmote", 0.50, feats, fit_with_df=False)
    _create_mini_bundle(tmp_path, "baseline", "rf_analysis_nosmote", "Analysis", "nosmote", 0.035, feats, fit_with_df=False)
    return tmp_path


def test_get_near_miss_bounds_prevents_zone_collapse():
    assert get_near_miss_bounds(0.50) == (0.10, 0.50)
    assert get_near_miss_bounds(0.20) == (0.10, 0.20)
    # Threshold < 0.10 must shrink near_low to 0.5 * thresh so [near_low, thresh) remains non-empty
    near_low, th = get_near_miss_bounds(0.035)
    assert th == 0.035
    assert near_low == 0.0175
    assert near_low < th
    assert get_near_miss_bounds(0.0) == (0.0, 0.0)


def test_build_consensus_matrix_5x3():
    assert build_consensus_matrix(pd.DataFrame()) == {}

    rows = []
    for att in ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]:
        for fam, status, prob, th in [
            ("PAPER", "BENIGN", 0.05, 0.50),
            ("RF", "SUSPICIOUS", 0.30, 0.45),
            ("GB", "ATTACK", 0.90, 0.80),
        ]:
            rows.append(
                {
                    "Attack": att,
                    "Model_Family": fam,
                    "Condition": "Without SMOTE",
                    "Probability": prob,
                    "Threshold_Used": th,
                    "Status_Code": status,
                    "Latency_ms": 1.25,
                }
            )
    matrices = build_consensus_matrix(pd.DataFrame(rows))
    assert list(matrices.keys()) == ["Without SMOTE"]
    mat = matrices["Without SMOTE"]
    assert len(mat) == 5
    assert list(mat["Attack"]) == ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]
    assert (mat["Consensus"] == "1/3 (+1 DPI)").all()
    assert (mat["Total_Latency_ms"] == 3.75).all()


def test_protocol_decoder_nan_safe_and_dpi_heuristics():
    assert decode_port(80) == "80 (HTTP (Standard Web Server))"
    assert decode_port(float("nan")) == "N/A (Invalid Port)"
    assert decode_port("not_a_port") == "N/A (Invalid Port)"
    assert "Dynamic" in decode_port(60000)

    assert decode_l4_protocol(6)["name"] == "TCP"
    assert decode_l4_protocol(float("nan"))["name"] == "UNKNOWN"
    assert decode_l7_protocol(7.0)["name"] == "HTTP"
    assert decode_l7_protocol("bad")["name"] == "L7 Unknown"
    assert decode_tcp_flags(0) == ["NONE (0x00)"]
    assert decode_tcp_flags(0x12) == ["SYN", "ACK"]
    assert decode_tcp_flags(float("nan")) == ["INVALID"]

    # Default benign HTTP flow (cumulative FIN|SYN|PSH|ACK=27) should pass cleanly
    rep_ok = analyze_tier2_dpi_heuristics(DEFAULT_BENIGN_FLOW, {"status": "BENIGN"})
    assert rep_ok["recommendation"] == "CLEAR_PASS"
    assert rep_ok["findings"][0]["rule_id"] == "DPI-BASELINE-OK"

    # Malformed SYN+FIN without ACK on port 80 triggers high-severity escalation
    malformed = dict(DEFAULT_BENIGN_FLOW)
    malformed.update({"PROTOCOL": 6, "L4_DST_PORT": 80, "TCP_FLAGS": 0x03})
    rep_bad = analyze_tier2_dpi_heuristics(malformed, {"status": "SUSPICIOUS"})
    assert rep_bad["recommendation"] == "ESCALATE_BLOCK"
    rule_ids = {f["rule_id"] for f in rep_bad["findings"]}
    assert "DPI-TCP-SYNFIN" in rule_ids


def test_engine_discovery_cache_and_sorted_registry(isolated_engine_dir: Path, tmp_path: Path):
    cache_path = tmp_path / "cache" / "models_index.json"
    engine = NIDSInferenceEngine(
        models_base_dir=isolated_engine_dir,
        lazy_load=True,
        index_cache_path=cache_path,
    )

    keys = list(engine.models_registry.keys())
    assert keys == sorted(keys)
    assert len(keys) == 4
    assert engine.get_available_model_families() == ["GB", "RF", "PAPER"]
    assert engine.get_available_conditions() == [("nosmote", "Without SMOTE")]
    assert cache_path.exists()

    # Corrupt cache must fall back cleanly without crashing startup
    cache_path.write_text("{corrupt json", encoding="utf-8")
    engine2 = NIDSInferenceEngine(
        models_base_dir=isolated_engine_dir,
        lazy_load=True,
        index_cache_path=cache_path,
    )
    assert len(engine2.models_registry) == 4


def test_engine_predict_single_and_batch_strict_and_dirty_csv(isolated_engine_dir: Path):
    engine = NIDSInferenceEngine(models_base_dir=isolated_engine_dir, lazy_load=True)

    # Single flow: low values -> BENIGN; high values -> ATTACK; custom low thresh preserves Near-Miss
    low_flow = {col: 1 for col in NETFLOW_FEATURES}
    pred_low = engine.predict_single_flow(low_flow, attack_filter="Analysis")
    assert pred_low["verdict"]["status"] == "BENIGN"
    assert pred_low["verdict"]["near_miss_low"] == 0.0175
    assert pred_low["verdict"]["threshold_used"] == 0.035

    high_flow = {col: 800 for col in NETFLOW_FEATURES}
    pred_high = engine.predict_single_flow(high_flow, attack_filter="DoS")
    assert pred_high["verdict"]["status"] == "ATTACK"

    with pytest.raises(TypeError):
        engine.predict_single_flow(["not", "a", "dict"])

    # Strict mode raises KeyError when requested model family/attack/condition does not exist
    batch_df = pd.DataFrame(
        [
            {"L4_DST_PORT": "80", "PROTOCOL": "6", "IN_BYTES": "10", "Attack": "Benign"},
            {"L4_DST_PORT": "bad", "PROTOCOL": np.inf, "IN_BYTES": "900", "Attack": "Analysis"},
            {"L4_DST_PORT": 800, "PROTOCOL": 800, "IN_BYTES": 900, "Attack": "Fuzzers"},
        ]
    )
    with pytest.raises(KeyError):
        engine.predict_batch_flows(batch_df, attack_name="Analysis", model_family="GB", condition="nosmote", strict=True)

    # Dirty CSV coercion + mutually exclusive zone sum when threshold < 0.10
    res_df, summary = engine.predict_batch_flows(
        batch_df,
        attack_name="Analysis",
        model_family="RF",
        condition="nosmote",
        threshold_mode="F1-Optimal",
        strict=True,
    )
    assert set(summary["coerced_columns"]) >= {"L4_DST_PORT", "PROTOCOL"}
    assert summary["near_miss_low"] == 0.0175
    assert summary["threshold_used"] == 0.035
    assert summary["benign_count"] + summary["suspicious_count"] + summary["attack_count"] == 3
    assert summary["cross_attack_total"] == 1
    assert summary["metrics"]["Evaluated_Samples"] == 2
    assert len(res_df) == 3

    with pytest.raises(ValueError, match="rỗng"):
        engine.predict_batch_flows(batch_df.iloc[0:0], attack_name="Analysis", model_family="RF")


def test_engine_empty_registry_guards(tmp_path: Path):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()
    engine = NIDSInferenceEngine(models_base_dir=empty_dir, lazy_load=True)
    assert engine.models_registry == {}
    with pytest.raises(ValueError, match="trống"):
        engine.predict_single_flow(DEFAULT_BENIGN_FLOW)
    with pytest.raises(ValueError, match="trống"):
        engine.predict_batch_flows(pd.DataFrame([DEFAULT_BENIGN_FLOW]))
