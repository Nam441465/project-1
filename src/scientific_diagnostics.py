"""
Zero-Model Scientific Diagnostics for CIC-UNSW-NB15 NIDS.

Strictly read-only evaluation over existing `.npy`, `.joblib`, `.json`, `.csv`
artifacts and `data/processed/{train,heldout_test}.parquet`.
ZERO model fitting, training, or `.joblib` modification is performed.

Generates under `results/improved/`:
  - `calibration_diagnostics.csv`: Brier, LogLoss, ECE-15, MCE-15, PR-AUC for
    Ours Improved RF and Paper-Exact RF (`nosmote`).
  - `bootstrap_mcnemar_tests.csv`: Stratified Bootstrap 95% CIs (B=1000) and
    exact two-sided McNemar tests (`RF vs PaperRF`).
  - `mixed_flow_orchestration_eval.csv`: Any-of-5 specialist union evaluation
    across all 363,035 rows of `heldout_test.parquet` (Benign, 5 targets,
    and 4 unseen attack classes: Fuzzers, Generic, Shellcode, Worms).
  - `scientific_diagnostics_findings.md`: Full analytical markdown report.
And under `figures/`:
  - `fig14_calibration_and_bootstrap.png`
  - `fig15_mixed_flow_orchestration.png`
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

import feature_selection  # noqa: E402, F401 - needed to unpickle MutualInfoScorer
from feature_selection import transform_features  # noqa: E402
from pipeline_utils import (  # noqa: E402
    TARGET_ATTACKS,
    evaluate_binary_predictions,
    trim_memory,
    vectorized_threshold_metrics,
)
from preprocessor import FEATURE_COLS, load_split  # noqa: E402

BASELINE_DIR = PROJECT_DIR / "results" / "baseline"
IMPROVED_DIR = PROJECT_DIR / "results" / "improved"
PAPER_DIR = PROJECT_DIR / "results" / "paper_original"
FIGURES_DIR = PROJECT_DIR / "figures"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.titlesize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MODEL_SPECS = {
    "RF": {
        "dir": BASELINE_DIR,
        "prefix": "rf",
        "summary_csv": BASELINE_DIR / "summary_rf_comparison.csv",
    },
    "PaperRF": {
        "dir": PAPER_DIR,
        "prefix": "paper_rf",
        "summary_csv": PAPER_DIR / "summary_paper_rf_comparison.csv",
    },
}

UNSEEN_ATTACKS = ["Fuzzers", "Generic", "Shellcode", "Worms"]
COST_RATIO_C = 10
THRESHOLD_GRID = np.round(np.arange(0.005, 1.0, 0.005), 4)


def compute_ece_mce(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)

    n = len(y_true)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        err = abs(acc - conf)
        ece += (cnt / n) * err
        if err > mce:
            mce = err
    return float(ece), float(mce)


def _load_saved_arrays(model_key: str, attack: str) -> dict[str, np.ndarray]:
    """Load saved validation/test labels and probabilities for nosmote models."""
    spec = MODEL_SPECS[model_key]
    base = spec["dir"]
    p = spec["prefix"]
    a_low = attack.lower()
    return {
        "y_val": np.load(base / f"{p}_{a_low}_nosmote_y_val.npy"),
        "p_val": np.load(base / f"{p}_{a_low}_nosmote_best_proba_val.npy"),
        "y_test": np.load(base / f"{p}_{a_low}_nosmote_y_test.npy"),
        "p_test": np.load(base / f"{p}_{a_low}_nosmote_best_proba_test.npy"),
    }


def _load_thresholds() -> dict[str, dict[str, dict[str, float]]]:
    """Read F1-Optimal and Cost-Opt (c=10) thresholds from existing CSV outputs."""
    thresholds: dict[str, dict[str, dict[str, float]]] = {
        m: {"F1-Optimal": {}, "Cost-Opt c=10": {}} for m in MODEL_SPECS
    }

    for model_key, spec in MODEL_SPECS.items():
        sdf = pd.read_csv(spec["summary_csv"])
        sub = sdf[sdf["Condition"] == "Without SMOTE"]
        for _, row in sub.iterrows():
            thresholds[model_key]["F1-Optimal"][row["Attack"]] = float(row["Threshold"])

    cost_df = pd.read_csv(IMPROVED_DIR / "cost_sensitive_evaluation.csv")
    c10_df = cost_df[cost_df["Cost_Ratio_c"] == COST_RATIO_C]
    for model_key in ("RF",):
        sub = c10_df[c10_df["Model"] == model_key]
        for _, row in sub.iterrows():
            thresholds[model_key]["Cost-Opt c=10"][row["Attack"]] = float(
                row["Optimal_Threshold"]
            )

    paper_c10_df = pd.read_csv(PAPER_DIR / "soc_operational_comparison_c10.csv")
    for _, row in paper_c10_df.iterrows():
        thresholds["PaperRF"]["Cost-Opt c=10"][row["Attack"]] = float(
            row["PaperRF_CostOpt_Threshold"]
        )

    return thresholds


def _select_cost_opt_threshold(
    y_val: np.ndarray, p_val: np.ndarray, c_ratio: int = COST_RATIO_C
) -> float:
    """Find optimal cost-sensitive threshold on validation set over THRESHOLD_GRID."""
    sweep = vectorized_threshold_metrics(y_val, p_val, THRESHOLD_GRID)
    costs = sweep["fp"] + c_ratio * sweep["fn"]
    best_idx = int(np.argmin(costs))
    return float(sweep["thresholds"][best_idx])


def _select_f1_opt_threshold(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Find F1-optimal threshold on validation set over THRESHOLD_GRID."""
    sweep = vectorized_threshold_metrics(y_val, p_val, THRESHOLD_GRID)
    best_idx = int(np.argmax(sweep["f1"]))
    return float(sweep["thresholds"][best_idx])


def run_calibration_diagnostics(
    saved_thresholds: dict[str, dict[str, dict[str, float]]]
) -> pd.DataFrame:
    """Evaluate probability calibration metrics for Ours Improved RF vs Paper-Exact RF."""
    print("\n" + "=" * 72)
    print("  1. PROBABILITY CALIBRATION DIAGNOSTICS (OURS RF VS PAPER-EXACT RF)")
    print("=" * 72)

    rows = []
    for attack in TARGET_ATTACKS:
        arrays_by_model = {m: _load_saved_arrays(m, attack) for m in MODEL_SPECS}

        for name in ("PaperRF", "RF"):
            arr = arrays_by_model[name]
            y_val, p_val = arr["y_val"], arr["p_val"]
            y_test, p_test = arr["y_test"], arr["p_test"]
            t_f1 = saved_thresholds[name]["F1-Optimal"][attack]
            t_c10 = saved_thresholds[name]["Cost-Opt c=10"][attack]

            brier = float(brier_score_loss(y_test, p_test))
            ll = float(log_loss(y_test, np.clip(p_test, 1e-15, 1.0 - 1e-15), labels=[0, 1]))
            ece_15, mce_15 = compute_ece_mce(y_test, p_test, n_bins=15)
            pr_auc = float(average_precision_score(y_test, p_test))

            val_f1_res = evaluate_binary_predictions(y_val, p_val, t_f1)
            test_f1_res = evaluate_binary_predictions(y_test, p_test, t_f1)
            test_c10_res = evaluate_binary_predictions(y_test, p_test, t_c10)
            test_cost_c10 = int(test_c10_res["FP"]) + COST_RATIO_C * int(test_c10_res["FN"])

            rows.append(
                {
                    "Attack": attack,
                    "Model": name,
                    "Brier_Score": round(brier, 6),
                    "Log_Loss": round(ll, 6),
                    "ECE_15": round(ece_15, 6),
                    "MCE_15": round(mce_15, 6),
                    "PR_AUC": round(pr_auc, 6),
                    "Threshold_F1Opt": round(float(t_f1), 4),
                    "Val_F1_at_F1Opt": round(float(val_f1_res["f1"]), 4),
                    "Test_F1_at_F1Opt": round(float(test_f1_res["f1"]), 4),
                    "Test_Precision_at_F1Opt": round(float(test_f1_res["precision"]), 4),
                    "Test_Recall_at_F1Opt": round(float(test_f1_res["recall"]), 4),
                    "Threshold_Cost_c10": round(float(t_c10), 4),
                    "Test_Cost_c10": int(test_cost_c10),
                    "Test_FN_Cost_c10": int(test_c10_res["FN"]),
                    "Test_FP_Cost_c10": int(test_c10_res["FP"]),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = IMPROVED_DIR / "calibration_diagnostics.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df)} rows)")
    return df


def _stratified_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 10,
) -> dict[str, float]:
    """Compute exact stratified bootstrap 95% CIs for binary classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = n_pos - tp

    # Point estimates
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / n_pos if n_pos > 0 else 0.0
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    cost_c10 = fp + COST_RATIO_C * fn

    # Stratified resampling of positives and negatives is exact Binomial draw
    rng = np.random.default_rng(seed)
    tp_b = rng.binomial(n_pos, tp / n_pos if n_pos > 0 else 0.0, size=n_boot)
    fp_b = rng.binomial(n_neg, fp / n_neg if n_neg > 0 else 0.0, size=n_boot)
    fn_b = n_pos - tp_b

    denom_p = tp_b + fp_b
    prec_b = np.where(denom_p > 0, tp_b / np.maximum(denom_p, 1), 0.0)
    rec_b = tp_b / max(1, n_pos)
    denom_f1 = 2 * tp_b + fp_b + fn_b
    f1_b = np.where(denom_f1 > 0, (2 * tp_b) / np.maximum(denom_f1, 1), 0.0)
    cost_b = fp_b + COST_RATIO_C * fn_b

    return {
        "f1": float(f1),
        "f1_lo": float(np.percentile(f1_b, 2.5)),
        "f1_hi": float(np.percentile(f1_b, 97.5)),
        "prec": float(prec),
        "prec_lo": float(np.percentile(prec_b, 2.5)),
        "prec_hi": float(np.percentile(prec_b, 97.5)),
        "rec": float(rec),
        "rec_lo": float(np.percentile(rec_b, 2.5)),
        "rec_hi": float(np.percentile(rec_b, 97.5)),
        "cost": int(cost_c10),
        "cost_lo": float(np.percentile(cost_b, 2.5)),
        "cost_hi": float(np.percentile(cost_b, 97.5)),
    }


def _exact_mcnemar_test(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> tuple[int, int, float]:
    """Return discordant counts (A right/B wrong, A wrong/B right) and two-sided exact p."""
    corr_a = pred_a == y_true
    corr_b = pred_b == y_true
    b_cnt = int(np.sum(corr_a & (~corr_b)))
    c_cnt = int(np.sum((~corr_a) & corr_b))
    n_disc = b_cnt + c_cnt
    if n_disc == 0:
        return b_cnt, c_cnt, 1.0
    p_val = float(binomtest(min(b_cnt, c_cnt), n_disc, p=0.5, alternative="two-sided").pvalue)
    return b_cnt, c_cnt, p_val


def run_bootstrap_and_mcnemar(
    saved_thresholds: dict[str, dict[str, dict[str, float]]],
    n_boot: int = 1000,
) -> pd.DataFrame:
    """Compute stratified bootstrap 95% CIs and exact two-sided McNemar tests."""
    print("\n" + "=" * 72)
    print(f"  2. STRATIFIED BOOTSTRAP 95% CIs (B={n_boot}) & EXACT MCNEMAR TESTS")
    print("=" * 72)

    pairs = [("RF", "PaperRF")]
    operating_points = ["F1-Optimal", "Cost-Opt c=10"]
    n_tests_total = len(TARGET_ATTACKS) * len(operating_points) * len(pairs)

    rows = []
    for a_idx, attack in enumerate(TARGET_ATTACKS):
        arrays = {m: _load_saved_arrays(m, attack) for m in MODEL_SPECS}
        y_test = arrays["RF"]["y_test"]
        for m in ("PaperRF",):
            if not np.array_equal(y_test, arrays[m]["y_test"]):
                raise ValueError(f"y_test mismatch between RF and {m} for {attack}")

        for op_idx, op in enumerate(operating_points):
            preds = {}
            boots = {}
            for m_idx, m in enumerate(("RF", "PaperRF")):
                t = saved_thresholds[m][op][attack]
                pred = (arrays[m]["p_test"] >= t).astype(np.int8)
                preds[m] = pred
                seed = 10 + a_idx * 100 + op_idx * 10 + m_idx
                boots[m] = _stratified_bootstrap_ci(y_test, pred, n_boot=n_boot, seed=seed)

            for m_a, m_b in pairs:
                ba = boots[m_a]
                bb = boots[m_b]
                disc_a, disc_b, p_val = _exact_mcnemar_test(y_test, preds[m_a], preds[m_b])
                rows.append(
                    {
                        "Attack": attack,
                        "Operating_Point": op,
                        "Model_A": m_a,
                        "Model_B": m_b,
                        "Threshold_A": round(saved_thresholds[m_a][op][attack], 4),
                        "Threshold_B": round(saved_thresholds[m_b][op][attack], 4),
                        "Model_A_F1": round(ba["f1"], 4),
                        "Model_A_F1_CI_Low": round(ba["f1_lo"], 4),
                        "Model_A_F1_CI_High": round(ba["f1_hi"], 4),
                        "Model_A_Precision": round(ba["prec"], 4),
                        "Model_A_Precision_CI_Low": round(ba["prec_lo"], 4),
                        "Model_A_Precision_CI_High": round(ba["prec_hi"], 4),
                        "Model_A_Recall": round(ba["rec"], 4),
                        "Model_A_Recall_CI_Low": round(ba["rec_lo"], 4),
                        "Model_A_Recall_CI_High": round(ba["rec_hi"], 4),
                        "Model_A_Cost_c10": ba["cost"],
                        "Model_A_Cost_CI_Low": round(ba["cost_lo"], 1),
                        "Model_A_Cost_CI_High": round(ba["cost_hi"], 1),
                        "Model_B_F1": round(bb["f1"], 4),
                        "Model_B_F1_CI_Low": round(bb["f1_lo"], 4),
                        "Model_B_F1_CI_High": round(bb["f1_hi"], 4),
                        "Model_B_Precision": round(bb["prec"], 4),
                        "Model_B_Precision_CI_Low": round(bb["prec_lo"], 4),
                        "Model_B_Precision_CI_High": round(bb["prec_hi"], 4),
                        "Model_B_Recall": round(bb["rec"], 4),
                        "Model_B_Recall_CI_Low": round(bb["rec_lo"], 4),
                        "Model_B_Recall_CI_High": round(bb["rec_hi"], 4),
                        "Model_B_Cost_c10": bb["cost"],
                        "Model_B_Cost_CI_Low": round(bb["cost_lo"], 1),
                        "Model_B_Cost_CI_High": round(bb["cost_hi"], 1),
                        "Delta_F1_A_minus_B": round(ba["f1"] - bb["f1"], 4),
                        "Delta_Cost_c10_A_minus_B": int(ba["cost"] - bb["cost"]),
                        "Discordant_A_Right_B_Wrong": disc_a,
                        "Discordant_A_Wrong_B_Right": disc_b,
                        "Discordant_Total": disc_a + disc_b,
                        "McNemar_Exact_p": p_val,
                        "Significant_alpha_0_05": bool(p_val < 0.05),
                        "Significant_Bonferroni": bool(p_val < (0.05 / n_tests_total)),
                    }
                )

    df = pd.DataFrame(rows)
    out_csv = IMPROVED_DIR / "bootstrap_mcnemar_tests.csv"
    df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv} ({len(df)} rows)")
    return df


def run_mixed_flow_orchestration(
    saved_thresholds: dict[str, dict[str, dict[str, float]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate Any-of-5 specialist orchestration across all 363,035 heldout flows."""
    print("\n" + "=" * 72)
    print("  3. MIXED-FLOW 5-SPECIALIST ORCHESTRATION (ALL 363,035 HELDOUT FLOWS)")
    print("=" * 72)

    df_test = load_split("heldout_test", columns=FEATURE_COLS + ["Attack"])
    X_all = df_test[FEATURE_COLS]
    att_all = df_test["Attack"].to_numpy()
    n_total = len(df_test)
    print(f"  Loaded heldout_test.parquet: {n_total:,} rows | Classes: {df_test['Attack'].nunique()}")

    eval_rows = []
    summary_rows = []

    for model_key, spec in MODEL_SPECS.items():
        base_models = spec["dir"] / "models"
        prefix = spec["prefix"]
        proba_mat = np.zeros((n_total, len(TARGET_ATTACKS)), dtype=np.float64)

        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            for idx, attack in enumerate(TARGET_ATTACKS):
                a_low = attack.lower()
                sel = joblib.load(base_models / f"{prefix}_{a_low}_nosmote_best_selector.joblib")
                mod = joblib.load(base_models / f"{prefix}_{a_low}_nosmote_best_model.joblib")
                X_sel = transform_features(sel, X_all)
                X_in = X_sel if hasattr(mod, "feature_names_in_") else X_sel.to_numpy()
                proba_mat[:, idx] = mod.predict_proba(X_in)[:, 1]
                del sel, mod, X_sel, X_in
        infer_sec = time.perf_counter() - t0
        print(f"  {model_key:7s}: 5-specialist inference on {n_total:,} rows in {infer_sec:.2f}s")

        for op in ("F1-Optimal", "Cost-Opt c=10"):
            t_vec = np.array(
                [saved_thresholds[model_key][op][att] for att in TARGET_ATTACKS],
                dtype=np.float64,
            )
            trig_mat = proba_mat >= t_vec[None, :]
            any_alarm = trig_mat.any(axis=1)
            n_triggered = trig_mat.sum(axis=1)

            # Per-class and aggregate breakdown
            slices: list[tuple[str, str, np.ndarray, int | None]] = [
                ("Benign", "Benign", att_all == "Benign", None)
            ]
            for idx, att in enumerate(TARGET_ATTACKS):
                slices.append((att, "Target_5", att_all == att, idx))
            for att in UNSEEN_ATTACKS:
                slices.append((att, "Unseen_4", att_all == att, None))

            is_target5 = np.isin(att_all, TARGET_ATTACKS)
            is_unseen4 = np.isin(att_all, UNSEEN_ATTACKS)
            is_any_attack = att_all != "Benign"

            # Exact specialist match across Target-5: flow of attack k triggered specialist k
            target5_matched = np.zeros(n_total, dtype=bool)
            for idx, att in enumerate(TARGET_ATTACKS):
                target5_matched |= (att_all == att) & trig_mat[:, idx]

            slices.extend(
                [
                    ("TARGET_5_TOTAL", "Aggregate", is_target5, -1),
                    ("UNSEEN_4_TOTAL", "Aggregate", is_unseen4, None),
                    ("ALL_ATTACKS_TOTAL", "Aggregate", is_any_attack, -1),
                ]
            )

            for cls_name, cls_group, mask, spec_idx in slices:
                tot = int(mask.sum())
                flagged = int(any_alarm[mask].sum())
                rate = (flagged / tot * 100.0) if tot > 0 else 0.0

                if spec_idx is None:
                    matched = 0
                    matched_rate = 0.0
                elif spec_idx == -1:
                    matched = int(target5_matched[mask].sum())
                    matched_rate = (matched / tot * 100.0) if tot > 0 else 0.0
                else:
                    matched = int((mask & trig_mat[:, spec_idx]).sum())
                    matched_rate = (matched / tot * 100.0) if tot > 0 else 0.0

                overlap = int((mask & (n_triggered > 1)).sum())
                eval_rows.append(
                    {
                        "Model_Family": model_key,
                        "Operating_Point": op,
                        "Traffic_Class": cls_name,
                        "Class_Group": cls_group,
                        "Total_Flows": tot,
                        "Any5_Flagged_Flows": flagged,
                        "Any5_Flag_Rate_Pct": round(rate, 2),
                        "Matched_Specialist_Flows": matched,
                        "Matched_Specialist_Rate_Pct": round(matched_rate, 2),
                        "Multi_Specialist_Overlap_Flows": overlap,
                        "Inference_Time_Sec_363k": round(infer_sec, 2),
                    }
                )

            # Overall system metrics (Benign vs All Attacks, Benign vs Target-5)
            is_benign = att_all == "Benign"
            fp_benign = int((is_benign & any_alarm).sum())
            tn_benign = int((is_benign & (~any_alarm)).sum())
            tp_t5 = int((is_target5 & any_alarm).sum())
            fn_t5 = int((is_target5 & (~any_alarm)).sum())
            tp_all = int((is_any_attack & any_alarm).sum())
            fn_all = int((is_any_attack & (~any_alarm)).sum())

            fpr_benign = fp_benign / max(1, int(is_benign.sum())) * 100.0
            rec_t5 = tp_t5 / max(1, int(is_target5.sum())) * 100.0
            prec_t5 = tp_t5 / max(1, tp_t5 + fp_benign) * 100.0
            f1_t5 = (2 * tp_t5) / max(1, 2 * tp_t5 + fp_benign + fn_t5)

            rec_all = tp_all / max(1, int(is_any_attack.sum())) * 100.0
            prec_all = tp_all / max(1, tp_all + fp_benign) * 100.0
            f1_all = (2 * tp_all) / max(1, 2 * tp_all + fp_benign + fn_all)

            summary_rows.append(
                {
                    "Model_Family": model_key,
                    "Operating_Point": op,
                    "Benign_FP": fp_benign,
                    "Benign_TN": tn_benign,
                    "Benign_FPR_Pct": round(fpr_benign, 2),
                    "Target5_TP": tp_t5,
                    "Target5_FN": fn_t5,
                    "Target5_Recall_Pct": round(rec_t5, 2),
                    "Target5_Precision_Pct": round(prec_t5, 2),
                    "Target5_F1": round(f1_t5, 4),
                    "Target5_Cost_c10": fp_benign + COST_RATIO_C * fn_t5,
                    "Unseen4_Flagged": int((is_unseen4 & any_alarm).sum()),
                    "Unseen4_Recall_Pct": round(
                        int((is_unseen4 & any_alarm).sum())
                        / max(1, int(is_unseen4.sum()))
                        * 100.0,
                        2,
                    ),
                    "AllAttacks_TP": tp_all,
                    "AllAttacks_FN": fn_all,
                    "AllAttacks_Recall_Pct": round(rec_all, 2),
                    "AllAttacks_Precision_Pct": round(prec_all, 2),
                    "AllAttacks_F1": round(f1_all, 4),
                    "AllAttacks_Cost_c10": fp_benign + COST_RATIO_C * fn_all,
                    "Inference_Time_Sec_363k": round(infer_sec, 2),
                }
            )

        del proba_mat
        trim_memory()

    eval_df = pd.DataFrame(eval_rows)
    summary_df = pd.DataFrame(summary_rows)
    out_csv = IMPROVED_DIR / "mixed_flow_orchestration_eval.csv"
    eval_df.to_csv(out_csv, index=False)
    print(f"  Saved {out_csv} ({len(eval_df)} rows)")
    return eval_df, summary_df


def plot_fig14_calibration_and_bootstrap(
    calib_df: pd.DataFrame, boot_df: pd.DataFrame
) -> None:
    """Plot Fig 14: ECE calibration comparison + F1 95% CIs (Ours RF vs PaperRF)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))

    # Panel 1: ECE-15 across PaperRF and Ours RF
    ax = axes[0]
    x = np.arange(len(TARGET_ATTACKS))
    w = 0.34
    plot_models = [
        ("PaperRF", "Paper-Exact RF", "#8c96a5"),
        ("RF", "Ours Improved RF", "#2b5c8f"),
    ]
    for idx, (m_key, label, color) in enumerate(plot_models):
        vals = [
            float(
                calib_df[
                    (calib_df["Attack"] == att) & (calib_df["Model"] == m_key)
                ]["ECE_15"].iloc[0]
            )
            * 1000.0
            for att in TARGET_ATTACKS
        ]
        bars = ax.bar(x + (idx - 0.5) * w, vals, width=w, label=label, color=color, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ATTACKS)
    ax.set_ylabel("ECE-15 (x 10^-3, lower is better)")
    ax.set_title("Probability Calibration (ECE-15): PaperRF vs Ours Improved RF")
    ax.set_ylim(0, max(calib_df["ECE_15"].max() * 1000.0 * 1.32, 2.5))
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 2: F1 Point Estimate + Stratified Bootstrap 95% CI (F1-Optimal)
    ax2 = axes[1]
    f1_sub = boot_df[boot_df["Operating_Point"] == "F1-Optimal"]
    offsets = {"PaperRF": -0.14, "RF": 0.14}
    colors = {"PaperRF": "#8c96a5", "RF": "#2b5c8f"}
    markers = {"PaperRF": "s", "RF": "o"}

    for m_key in ("PaperRF", "RF"):
        pts, err_lo, err_hi = [], [], []
        for att in TARGET_ATTACKS:
            row = f1_sub[(f1_sub["Attack"] == att) & (f1_sub["Model_A"] == "RF") & (f1_sub["Model_B"] == "PaperRF")].iloc[0]
            prefix = "Model_A" if m_key == "RF" else "Model_B"
            f1 = float(row[f"{prefix}_F1"])
            lo = float(row[f"{prefix}_F1_CI_Low"])
            hi = float(row[f"{prefix}_F1_CI_High"])
            pts.append(f1)
            err_lo.append(max(0.0, f1 - lo))
            err_hi.append(max(0.0, hi - f1))

        ax2.errorbar(
            x + offsets[m_key],
            pts,
            yerr=[err_lo, err_hi],
            fmt=markers[m_key],
            color=colors[m_key],
            capsize=4,
            linewidth=1.5,
            markersize=6.5,
            label=f"{m_key} (95% CI)",
        )

    # Annotate RF vs PaperRF McNemar significance
    for idx, att in enumerate(TARGET_ATTACKS):
        r = f1_sub[
            (f1_sub["Attack"] == att)
            & (f1_sub["Model_A"] == "RF")
            & (f1_sub["Model_B"] == "PaperRF")
        ].iloc[0]
        p_val = float(r["McNemar_Exact_p"])
        sig_str = f"p={p_val:.1e}" if p_val < 0.001 else f"p={p_val:.3f}"
        y_top = max(float(r["Model_A_F1_CI_High"]), float(r["Model_B_F1_CI_High"]))
        ax2.text(
            idx,
            min(0.99, y_top + 0.015),
            f"RF vs PaperRF\n{sig_str}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#333333",
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(TARGET_ATTACKS)
    ax2.set_ylabel("F1-Score (95% Stratified Bootstrap CI, B=1000)")
    ax2.set_title("F1-Score Uncertainty & Exact McNemar Significance (F1-Optimal)")
    ax2.set_ylim(0.75, 1.03)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_png = FIGURES_DIR / "fig14_calibration_and_bootstrap.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def plot_fig15_mixed_flow_orchestration(
    eval_df: pd.DataFrame, summary_df: pd.DataFrame
) -> None:
    """Plot Fig 15: Full 363,035-flow Any-of-5 detection across Target-5, Unseen-4 & Benign."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8))

    # Panel 1: Per-class Any-of-5 flag rate (%) for Cost-Opt c=10 across PaperRF & RF
    ax = axes[0]
    classes = ["Benign"] + TARGET_ATTACKS + UNSEEN_ATTACKS
    x = np.arange(len(classes))
    w = 0.34
    c10_df = eval_df[eval_df["Operating_Point"] == "Cost-Opt c=10"]

    for idx, (m_key, color) in enumerate(
        [("PaperRF", "#8c96a5"), ("RF", "#2b5c8f")]
    ):
        rates = [
            float(
                c10_df[
                    (c10_df["Model_Family"] == m_key)
                    & (c10_df["Traffic_Class"] == cls)
                ]["Any5_Flag_Rate_Pct"].iloc[0]
            )
            for cls in classes
        ]
        ax.bar(
            x + (idx - 0.5) * w,
            rates,
            width=w,
            label=f"{m_key} (Cost-Opt c=10)",
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.axvline(0.5, color="#444444", linestyle=":", linewidth=1.1)
    ax.axvline(5.5, color="#c0392b", linestyle="--", linewidth=1.2)
    ax.text(3.0, 103, "5 Target Classes", ha="center", fontsize=9, fontweight="bold", color="#2b5c8f")
    ax.text(7.5, 103, "4 Unseen Zero-Day Classes", ha="center", fontsize=9, fontweight="bold", color="#c0392b")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=28, ha="right")
    ax.set_ylabel("Any-of-5 Specialist Flag Rate (%)")
    ax.set_title("Mixed-Flow Orchestration (All 363,035 Heldout Flows, c=10)")
    ax.set_ylim(0, 112)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 2: Benign FPR vs Target-5 Recall & All-Attack Recall across 6 regimes
    ax2 = axes[1]
    regimes = [
        f"{r['Model_Family']}\n({r['Operating_Point'].replace('Cost-Opt ', '')})"
        for _, r in summary_df.iterrows()
    ]
    xr = np.arange(len(regimes))
    w2 = 0.26
    t5_rec = summary_df["Target5_Recall_Pct"].to_numpy()
    all_rec = summary_df["AllAttacks_Recall_Pct"].to_numpy()
    fpr = summary_df["Benign_FPR_Pct"].to_numpy()

    b1 = ax2.bar(xr - w2, t5_rec, width=w2, label="Target-5 Recall (%)", color="#2b5c8f", edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(xr, all_rec, width=w2, label="All-9 Attacks Recall (%)", color="#16a085", edgecolor="black", linewidth=0.5)
    b3 = ax2.bar(xr + w2, fpr, width=w2, label="Benign FPR (%)", color="#e74c3c", edgecolor="black", linewidth=0.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90,
            )

    ax2.set_xticks(xr)
    ax2.set_xticklabels(regimes, fontsize=8.5)
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("System-Wide Tradeoff: Target-5 vs All-Attack Recall vs Benign FPR")
    ax2.set_ylim(0, 114)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y", linestyle="--", alpha=0.35)

    plt.tight_layout()
    out_png = FIGURES_DIR / "fig15_mixed_flow_orchestration.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_png}")


def generate_markdown_report(
    calib_df: pd.DataFrame,
    boot_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """Write scientific_diagnostics_findings.md purely from computed DataFrames."""
    out_md = IMPROVED_DIR / "scientific_diagnostics_findings.md"
    lines = [
        "# Scientific Diagnostics Report (Random Forest-Only Analytical & Full Held-Out Evaluation)",
        "",
        "All metrics below were computed strictly from existing `.npy`, `.joblib`, `.json`, `.csv` artifacts and `data/processed/heldout_test.parquet` without fitting or modifying any model.",
        "",
        "## 1. Probability Calibration Diagnostics (`PaperRF` vs `Ours Improved RF`)",
        "",
        "| Attack | Model | Brier | LogLoss | ECE-15 | MCE-15 | `t*_F1` | Test F1 | `t*_c10` | Cost `c=10` |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in calib_df.iterrows():
        lines.append(
            f"| {r['Attack']} | {r['Model']} | "
            f"{r['Brier_Score']:.6f} | {r['Log_Loss']:.6f} | {r['ECE_15']:.6f} | {r['MCE_15']:.6f} | "
            f"{r['Threshold_F1Opt']:.3f} | {r['Test_F1_at_F1Opt']:.4f} | "
            f"{r['Threshold_Cost_c10']:.3f} | {int(r['Test_Cost_c10']):,} |"
        )

    lines.extend(
        [
            "",
            "## 2. Stratified Bootstrap 95% CIs (B=1000) & Exact Two-Sided McNemar Tests",
            "",
            "| Attack | Operating Point | Pair | Model A F1 (95% CI) | Model B F1 (95% CI) | Disc (A+/B-, A-/B+) | Exact `p`-value |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for _, r in boot_df.iterrows():
        lines.append(
            f"| {r['Attack']} | {r['Operating_Point']} | {r['Model_A']} vs {r['Model_B']} | "
            f"{r['Model_A_F1']:.4f} [{r['Model_A_F1_CI_Low']:.4f}, {r['Model_A_F1_CI_High']:.4f}] | "
            f"{r['Model_B_F1']:.4f} [{r['Model_B_F1_CI_Low']:.4f}, {r['Model_B_F1_CI_High']:.4f}] | "
            f"({int(r['Discordant_A_Right_B_Wrong'])}, {int(r['Discordant_A_Wrong_B_Right'])}) | "
            f"{r['McNemar_Exact_p']:.3e} |"
        )

    lines.extend(
        [
            "",
            "## 3. Mixed-Flow 5-Specialist Orchestration (All 363,035 Held-Out Flows)",
            "",
            "| Model | Operating Point | Benign FPR (%) | Target-5 Recall (%) | Target-5 F1 | Unseen-4 Flagged (%) | All-9 Recall (%) | All-9 F1 | Inference (s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['Model_Family']} | {r['Operating_Point']} | {r['Benign_FPR_Pct']:.2f}% | "
            f"{r['Target5_Recall_Pct']:.2f}% | {r['Target5_F1']:.4f} | {r['Unseen4_Recall_Pct']:.2f}% | "
            f"{r['AllAttacks_Recall_Pct']:.2f}% | {r['AllAttacks_F1']:.4f} | {r['Inference_Time_Sec_363k']:.2f}s |"
        )

    lines.extend(["", "### 3.1 Per-Class Unseen-4 & Target-5 Detection (`Cost-Opt c=10`)", ""])
    c10 = eval_df[eval_df["Operating_Point"] == "Cost-Opt c=10"]
    pivot = c10.pivot(index="Traffic_Class", columns="Model_Family", values="Any5_Flag_Rate_Pct")
    tot_map = c10.drop_duplicates("Traffic_Class").set_index("Traffic_Class")["Total_Flows"].to_dict()
    lines.append("| Traffic Class | Flows | PaperRF (%) | Ours RF (%) |")
    lines.append("|---|---:|---:|---:|")
    order = ["Benign"] + TARGET_ATTACKS + UNSEEN_ATTACKS + ["TARGET_5_TOTAL", "UNSEEN_4_TOTAL", "ALL_ATTACKS_TOTAL"]
    for cls in order:
        lines.append(
            f"| {cls} | {int(tot_map[cls]):,} | {pivot.loc[cls, 'PaperRF']:.2f}% | "
            f"{pivot.loc[cls, 'RF']:.2f}% |"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Saved {out_md}")


def main() -> None:
    os.makedirs(IMPROVED_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    saved_thresholds = _load_thresholds()
    calib_df = run_calibration_diagnostics(saved_thresholds)
    boot_df = run_bootstrap_and_mcnemar(saved_thresholds, n_boot=1000)
    eval_df, summary_df = run_mixed_flow_orchestration(saved_thresholds)

    plot_fig14_calibration_and_bootstrap(calib_df, boot_df)
    plot_fig15_mixed_flow_orchestration(eval_df, summary_df)
    generate_markdown_report(calib_df, boot_df, eval_df, summary_df)
    print("\n  scientific_diagnostics.py completed cleanly (ZERO models trained/modified).")


if __name__ == "__main__":
    main()
