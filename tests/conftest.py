from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
WEB_APP_DIR = PROJECT_ROOT / "web_app"

for _p in (str(SRC_DIR), str(WEB_APP_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessor import FEATURE_COLS


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return PROJECT_ROOT / "results"


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed"


@pytest.fixture
def synthetic_flow_df() -> pd.DataFrame:
    """Deterministic synthetic NetFlow v1 DataFrame with Benign, 5 target attacks, and Fuzzers."""
    rng = np.random.default_rng(42)
    rows = []
    attack_specs = [
        ("Benign", 0, 80),
        ("DoS", 1, 20),
        ("Exploits", 1, 20),
        ("Backdoor", 1, 15),
        ("Analysis", 1, 15),
        ("Reconnaissance", 1, 20),
        ("Fuzzers", 1, 10),
    ]
    l7_choices = [0.0, 7.12, 91.1, 126.0, 5.25]
    for attack_name, label, count in attack_specs:
        is_attack = label == 1
        for i in range(count):
            rows.append(
                {
                    "IPV4_SRC_ADDR": f"10.0.{label}.{i % 250}",
                    "IPV4_DST_ADDR": f"192.168.1.{(i % 50) + 1}",
                    "L4_SRC_PORT": int(rng.integers(1024, 65535)),
                    "L4_DST_PORT": int(80 if is_attack else rng.choice([53, 443, 8080])),
                    "PROTOCOL": int(rng.choice([6, 17])),
                    "L7_PROTO": float(l7_choices[i % len(l7_choices)]),
                    "IN_BYTES": int(rng.integers(400, 12000) if is_attack else rng.integers(60, 1500)),
                    "OUT_BYTES": int(rng.integers(0, 8000)),
                    "IN_PKTS": int(rng.integers(1, 50)),
                    "OUT_PKTS": int(rng.integers(0, 40)),
                    "TCP_FLAGS": int(rng.choice([2, 16, 18, 24, 27])),
                    "FLOW_DURATION_MILLISECONDS": int(rng.integers(1, 5000)),
                    "Label": label,
                    "Attack": attack_name,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_binary_xy(synthetic_flow_df: pd.DataFrame):
    """Binary (X, y) pair derived from Benign + DoS rows for fast feature-selection/SMOTE tests."""
    mask = synthetic_flow_df["Attack"].isin(["Benign", "DoS"])
    sub = synthetic_flow_df.loc[mask].copy()
    X = sub[FEATURE_COLS].reset_index(drop=True)
    y = (sub["Attack"] == "DoS").to_numpy(dtype=np.int32)
    return X, y
