#!/usr/bin/env python3
"""
advanced_evaluation.py
Giai đoạn 2 - Phần 2:
1. Tối ưu ngưỡng theo Recall mục tiêu (Recall-Targeted Thresholding: R >= 0.95, 0.98, 0.99).
2. Tối ưu ngưỡng nhạy chi phí (Cost-Sensitive Threshold Optimization: c in {1, 5, 10, 20}).
3. Giải phẫu chuyên sâu mẫu bị bỏ sót (False Negative Forensic Analysis: Near-Miss vs Deep Stealth, phân tích phân phối đặc trưng NetFlow).
4. Xuất 3 biểu đồ học thuật chuẩn 300 DPI (fig11, fig12, fig13) và báo cáo Markdown.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Thiết lập đường dẫn
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
BASELINE_DIR = BASE_DIR / "results" / "baseline"
IMPROVED_DIR = BASE_DIR / "results" / "improved"
PAPER_DIR = BASE_DIR / "results" / "paper_original"
FIGURES_DIR = BASE_DIR / "figures"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessor import get_binary_data
from pipeline_utils import (
    TARGET_ATTACKS,
    evaluate_binary_predictions,
    trim_memory,
    vectorized_threshold_metrics,
)


def get_near_miss_bounds(thresh: float) -> tuple[float, float]:
    """
    Tính cặp cận vùng Near-Miss (near_low, t_active) đồng bộ với web_app/inference_engine.py:
    đảm bảo 3 vùng [0, near_low), [near_low, t_active), [t_active, 1] luôn tách rời nhau
    ngay cả khi ngưỡng t*_cost < 0.10.
    """
    t = float(thresh)
    near_low = round(min(0.10, 0.5 * t), 4) if t > 0.0 else 0.0
    return (near_low, t)

def setup_matplotlib_style():
    """Thiết lập cấu hình đồ họa chuẩn học thuật."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
    })

# ==============================================================================
# 1. TỐI ƯU NGƯỠNG THEO RECALL MỤC TIÊU (RECALL-TARGETED THRESHOLDING)
# ==============================================================================
def run_recall_targeted_evaluation():
    print("\n" + "=" * 75)
    print(" 1. KHỞI CHẠY TỐI ƯU NGƯỠNG THEO RECALL MỤC TIÊU (R >= 0.95, 0.98, 0.99)")
    print("=" * 75)

    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")

    target_recalls = [0.95, 0.98, 0.99]
    records = []
    thresholds_scan = np.round(np.linspace(0.99, 0.001, 990), 4)

    for att in TARGET_ATTACKS:
        for model_name, dir_path, sum_df in [("RF", BASELINE_DIR, rf_sum)]:
            prefix = model_name.lower()
            y_val = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_y_val.npy")
            p_val = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_best_proba_val.npy")
            y_test = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_y_test.npy")
            p_test = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_best_proba_test.npy")

            match_row = sum_df[(sum_df["Attack"] == att) & (sum_df["Condition"] == "Without SMOTE")]
            t_f1_opt = float(match_row["Threshold"].iloc[0])

            eval_f1 = evaluate_binary_predictions(y_test, p_test, t_f1_opt)
            val_rec_f1 = float(vectorized_threshold_metrics(y_val, p_val, [t_f1_opt])["recall"][0])
            near_low_f1, _ = get_near_miss_bounds(t_f1_opt)

            records.append({
                "Attack": att,
                "Model": model_name,
                "Mode": "F1-Optimal",
                "Target_Recall": np.nan,
                "Threshold": round(t_f1_opt, 4),
                "NearMiss_Lower_Bound": near_low_f1,
                "Val_Recall": round(val_rec_f1, 4),
                "Test_Precision": round(float(eval_f1["precision"]), 4),
                "Test_Recall": round(float(eval_f1["recall"]), 4),
                "Test_F1": round(float(eval_f1["f1"]), 4),
                "TP": int(eval_f1["TP"]),
                "FP": int(eval_f1["FP"]),
                "FN": int(eval_f1["FN"]),
                "TN": int(eval_f1["TN"]),
                "Delta_FN_vs_F1Opt": 0,
                "Delta_FP_vs_F1Opt": 0,
            })

            # Quét vector hóa O(N log N) qua 990 ngưỡng
            val_sweep = vectorized_threshold_metrics(y_val, p_val, thresholds_scan)
            val_recalls = val_sweep["recall"]

            for tr in target_recalls:
                hits = np.flatnonzero(val_recalls >= tr)
                t_found = float(thresholds_scan[int(hits[0])]) if len(hits) > 0 else 0.001
                t_found = min(t_f1_opt, t_found)

                eval_t = evaluate_binary_predictions(y_test, p_test, t_found)
                val_rec_act = float(vectorized_threshold_metrics(y_val, p_val, [t_found])["recall"][0])
                near_low_t, _ = get_near_miss_bounds(t_found)

                records.append({
                    "Attack": att,
                    "Model": model_name,
                    "Mode": f"Recall>={int(tr*100)}%",
                    "Target_Recall": float(tr),
                    "Threshold": round(t_found, 4),
                    "NearMiss_Lower_Bound": near_low_t,
                    "Val_Recall": round(val_rec_act, 4),
                    "Test_Precision": round(float(eval_t["precision"]), 4),
                    "Test_Recall": round(float(eval_t["recall"]), 4),
                    "Test_F1": round(float(eval_t["f1"]), 4),
                    "TP": int(eval_t["TP"]),
                    "FP": int(eval_t["FP"]),
                    "FN": int(eval_t["FN"]),
                    "TN": int(eval_t["TN"]),
                    "Delta_FN_vs_F1Opt": int(eval_t["FN"]) - int(eval_f1["FN"]),
                    "Delta_FP_vs_F1Opt": int(eval_t["FP"]) - int(eval_f1["FP"]),
                })

    df_recall = pd.DataFrame(records)
    out_file = IMPROVED_DIR / "recall_targeted_thresholds.csv"
    df_recall.to_csv(out_file, index=False)
    print(f"[+] Đã lưu bảng kết quả Recall-Targeted: {out_file}")
    return df_recall


# ==============================================================================
# 2. TỐI ƯU NGƯỠNG NHẠY CHI PHÍ (COST-SENSITIVE OPTIMIZATION)
# ==============================================================================
def run_cost_sensitive_evaluation():
    print("\n" + "=" * 75)
    print(" 2. KHỞI CHẠY TỐI ƯU NGƯỠNG NHẠY CHI PHÍ (c in {1, 5, 10, 20})")
    print("=" * 75)

    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")

    c_ratios = [1, 5, 10, 20]
    records = []
    thresholds_scan = np.round(np.linspace(0.005, 0.995, 199), 3)

    for att in TARGET_ATTACKS:
        for model_name, dir_path, sum_df in [("RF", BASELINE_DIR, rf_sum)]:
            prefix = model_name.lower()
            y_val = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_y_val.npy")
            p_val = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_best_proba_val.npy")
            y_test = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_y_test.npy")
            p_test = np.load(dir_path / f"{prefix}_{att.lower()}_nosmote_best_proba_test.npy")

            match_row = sum_df[(sum_df["Attack"] == att) & (sum_df["Condition"] == "Without SMOTE")]
            t_f1_opt = float(match_row["Threshold"].iloc[0])
            eval_f1 = evaluate_binary_predictions(y_test, p_test, t_f1_opt)
            fn_f1 = int(eval_f1["FN"])
            fp_f1 = int(eval_f1["FP"])

            # Quét vector hóa O(N log N) trên tập Validation
            val_sweep = vectorized_threshold_metrics(y_val, p_val, thresholds_scan)
            fn_val_arr = val_sweep["fn"]
            fp_val_arr = val_sweep["fp"]

            for c in c_ratios:
                costs_val = c * fn_val_arr + fp_val_arr
                t_best_cost = float(thresholds_scan[int(np.argmin(costs_val))])
                near_low, _ = get_near_miss_bounds(t_best_cost)

                eval_t = evaluate_binary_predictions(y_test, p_test, t_best_cost)
                tp_t = int(eval_t["TP"])
                fp_t = int(eval_t["FP"])
                fn_t = int(eval_t["FN"])
                tn_t = int(eval_t["TN"])

                cost_optimal = c * fn_t + fp_t
                cost_at_f1 = c * fn_f1 + fp_f1
                red_pct = ((cost_at_f1 - cost_optimal) / cost_at_f1 * 100) if cost_at_f1 > 0 else 0.0

                records.append({
                    "Attack": att,
                    "Model": model_name,
                    "Cost_Ratio_c": int(c),
                    "Optimal_Threshold": round(t_best_cost, 3),
                    "NearMiss_Lower_Bound": near_low,
                    "F1Opt_Threshold": round(t_f1_opt, 3),
                    "Test_Precision": round(float(eval_t["precision"]), 4),
                    "Test_Recall": round(float(eval_t["recall"]), 4),
                    "Test_F1": round(float(eval_t["f1"]), 4),
                    "TP": tp_t,
                    "FP": fp_t,
                    "FN": fn_t,
                    "TN": tn_t,
                    "Cost_Optimal": int(cost_optimal),
                    "Cost_at_F1Opt": int(cost_at_f1),
                    "Cost_Reduction_Pct": round(red_pct, 2),
                })

    df_cost = pd.DataFrame(records)
    out_file = IMPROVED_DIR / "cost_sensitive_evaluation.csv"
    df_cost.to_csv(out_file, index=False)
    print(f"[+] Đã lưu bảng kết quả Cost-Sensitive: {out_file}")
    return df_cost


# ==============================================================================
# 3. GIẢI PHẪU CHUYÊN SÂU MẪU BỊ BỎ SÓT (FALSE NEGATIVE FORENSIC ANALYSIS)
# ==============================================================================
def _collect_forensic_rows(
    att: str,
    model_name: str,
    mode_name: str,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    p_test: np.ndarray,
    t_opt: float,
) -> list[dict]:
    """Thu thập 4 nhóm (TP, FN, FP, TN) với ngưỡng Near-Miss đồng bộ Web UI."""
    near_low, t_active = get_near_miss_bounds(t_opt)
    pred_test = (p_test >= t_active).astype(int)

    groups = {
        "TP": (y_test == 1) & (pred_test == 1),
        "FN": (y_test == 1) & (pred_test == 0),
        "FP": (y_test == 0) & (pred_test == 1),
        "TN": (y_test == 0) & (pred_test == 0),
    }

    total_attack = int(np.sum(y_test == 1))
    fn_mask = groups["FN"]
    p_fn = p_test[fn_mask]
    n_near_miss = int(np.sum((p_fn >= near_low) & (p_fn < t_active)))
    n_deep_stealth = int(np.sum(p_fn < near_low))
    pct_near_miss = (n_near_miss / len(p_fn) * 100.0) if len(p_fn) > 0 else 0.0
    pct_deep_stealth = (n_deep_stealth / len(p_fn) * 100.0) if len(p_fn) > 0 else 0.0

    rows = []
    for g_name, mask in groups.items():
        sub_X = X_test[mask]
        count = len(sub_X)
        pct_base = (
            (count / total_attack * 100.0)
            if (g_name in ["TP", "FN"] and total_attack > 0)
            else (count / len(X_test) * 100.0 if len(X_test) > 0 else 0.0)
        )

        in_b_med = float(sub_X["IN_BYTES"].median()) if count > 0 else 0.0
        in_b_mean = float(sub_X["IN_BYTES"].mean()) if count > 0 else 0.0
        out_b_med = float(sub_X["OUT_BYTES"].median()) if count > 0 else 0.0
        out_b_mean = float(sub_X["OUT_BYTES"].mean()) if count > 0 else 0.0
        dur_med = float(sub_X["FLOW_DURATION_MILLISECONDS"].median()) if count > 0 else 0.0
        dur_mean = float(sub_X["FLOW_DURATION_MILLISECONDS"].mean()) if count > 0 else 0.0

        zero_dur_pct = float((sub_X["FLOW_DURATION_MILLISECONDS"] == 0).mean() * 100.0) if count > 0 else 0.0
        zero_out_b_pct = float((sub_X["OUT_BYTES"] == 0).mean() * 100.0) if count > 0 else 0.0

        top_ports = {int(k) if isinstance(k, (np.integer, int)) else float(k): int(v) for k, v in sub_X["L4_DST_PORT"].value_counts().head(3).items()} if count > 0 else {}
        top_proto = {int(k) if isinstance(k, (np.integer, int)) else float(k): int(v) for k, v in sub_X["PROTOCOL"].value_counts().head(3).items()} if count > 0 else {}
        top_l7 = {round(float(k), 2): int(v) for k, v in sub_X["L7_PROTO"].value_counts().head(3).items()} if count > 0 else {}

        rows.append({
            "Model": model_name,
            "Mode": mode_name,
            "Attack": att,
            "Threshold_Used": round(float(t_active), 4),
            "NearMiss_Lower_Bound": near_low,
            "Category": g_name,
            "Count": count,
            "Pct_Base": round(pct_base, 2),
            "Near_Miss_Count": n_near_miss if g_name == "FN" else np.nan,
            "Near_Miss_Pct": round(pct_near_miss, 2) if g_name == "FN" else np.nan,
            "Deep_Stealth_Count": n_deep_stealth if g_name == "FN" else np.nan,
            "Deep_Stealth_Pct": round(pct_deep_stealth, 2) if g_name == "FN" else np.nan,
            "IN_BYTES_median": round(in_b_med, 1),
            "IN_BYTES_mean": round(in_b_mean, 1),
            "OUT_BYTES_median": round(out_b_med, 1),
            "OUT_BYTES_mean": round(out_b_mean, 1),
            "DURATION_median": round(dur_med, 1),
            "DURATION_mean": round(dur_mean, 1),
            "Zero_Duration_Pct": round(zero_dur_pct, 2),
            "Zero_OutBytes_Pct": round(zero_out_b_pct, 2),
            "Top_Dst_Ports": str(top_ports),
            "Top_Protocols": str(top_proto),
            "Top_L7_Proto": str(top_l7),
        })
    return rows


def run_false_negative_forensics(df_cost: pd.DataFrame | None = None):
    print("\n" + "=" * 75)
    print(" 3. KHỞI CHẠY GIẢI PHẪU MẪU BỊ BỎ SÓT (OURS RF + PAPER-EXACT RF)")
    print("=" * 75)

    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")
    paper_sum_path = PAPER_DIR / "summary_paper_rf_comparison.csv"
    paper_sum = pd.read_csv(paper_sum_path) if paper_sum_path.exists() else None
    if df_cost is None:
        df_cost = pd.read_csv(IMPROVED_DIR / "cost_sensitive_evaluation.csv")

    all_records = []
    for att in TARGET_ATTACKS:
        X_test, y_test = get_binary_data(att, split="heldout_test", drop_ip=True)

        p_rf = np.load(BASELINE_DIR / f"rf_{att.lower()}_nosmote_best_proba_test.npy")

        t_rf_f1 = float(rf_sum[(rf_sum["Attack"] == att) & (rf_sum["Condition"] == "Without SMOTE")]["Threshold"].iloc[0])
        t_rf_c10 = float(df_cost[(df_cost["Attack"] == att) & (df_cost["Model"] == "RF") & (df_cost["Cost_Ratio_c"] == 10)]["Optimal_Threshold"].iloc[0])

        all_records.extend(_collect_forensic_rows(att, "RF", "F1-Optimal", X_test, y_test, p_rf, t_rf_f1))

        paper_p_path = PAPER_DIR / f"paper_rf_{att.lower()}_nosmote_best_proba_test.npy"
        if paper_sum is not None and paper_p_path.exists():
            p_pr = np.load(paper_p_path)
            t_pr_f1 = float(paper_sum[(paper_sum["Attack"] == att) & (paper_sum["Condition"] == "Without SMOTE")]["Threshold"].iloc[0])
            all_records.extend(_collect_forensic_rows(att, "PaperRF", "F1-Optimal", X_test, y_test, p_pr, t_pr_f1))

        all_records.extend(_collect_forensic_rows(att, "RF", "Cost-Opt c=10", X_test, y_test, p_rf, t_rf_c10))

        del X_test, y_test
        trim_memory()

    df_all = pd.DataFrame(all_records)
    all_out_file = IMPROVED_DIR / "fn_forensic_analysis_all_models.csv"
    df_all.to_csv(all_out_file, index=False)
    print(f"[+] Đã lưu bảng giải phẫu FN đa mô hình: {all_out_file}")

    # Giữ backward-compatible schema cho fn_forensic_analysis.csv (Ours RF F1-Optimal)
    compat_cols = [
        "Attack", "Category", "Count", "Pct_Base",
        "Near_Miss_Count", "Near_Miss_Pct", "Deep_Stealth_Count", "Deep_Stealth_Pct",
        "IN_BYTES_median", "IN_BYTES_mean", "OUT_BYTES_median", "OUT_BYTES_mean",
        "DURATION_median", "DURATION_mean", "Zero_Duration_Pct", "Zero_OutBytes_Pct",
        "Top_Dst_Ports", "Top_Protocols", "Top_L7_Proto",
    ]
    df_rf_compat = df_all[(df_all["Model"] == "RF") & (df_all["Mode"] == "F1-Optimal")][compat_cols].copy()
    compat_out_file = IMPROVED_DIR / "fn_forensic_analysis.csv"
    df_rf_compat.to_csv(compat_out_file, index=False)
    print(f"[+] Đã lưu bảng giải phẫu FN chuẩn tương thích ngược (RF F1-Optimal): {compat_out_file}")
    return df_rf_compat, df_all

# ==============================================================================
# 4. HÌNH 11: ĐÁNH ĐỔI NGƯỠNG QUYẾT ĐỊNH - RECALL - PRECISION & GIẢM THIỂU FN
# ==============================================================================
def plot_fig11_recall_threshold_tradeoff(df_recall: pd.DataFrame, out_path: Path):
    print("\n[*] Đang vẽ Hình 11: Recall Threshold Tradeoff & FN Reduction...")
    setup_matplotlib_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---------------- PANEL TRÁI: ĐƯỜNG CONG P-R-F1 VS THRESHOLD CHO DOS ----------------
    y_test_dos = np.load(BASELINE_DIR / "rf_dos_nosmote_y_test.npy")
    p_test_dos = np.load(BASELINE_DIR / "rf_dos_nosmote_best_proba_test.npy")

    t_scan = np.linspace(0.01, 0.99, 150)
    sweep_dos = vectorized_threshold_metrics(y_test_dos, p_test_dos, t_scan)
    precs = sweep_dos["precision"]
    recs = sweep_dos["recall"]
    f1s = sweep_dos["f1"]

    ax1.plot(t_scan, recs, label="Recall (Tỷ lệ Bắt trúng)", color="#2ca02c", linewidth=2.2, zorder=3)
    ax1.plot(t_scan, precs, label="Precision (Độ Chuẩn xác)", color="#1f77b4", linewidth=2.2, zorder=3)
    ax1.plot(t_scan, f1s, label="F1-Score (Hài hòa)", color="#ff7f0e", linewidth=2.0, linestyle="--", zorder=3)

    sub_rf_dos = df_recall[(df_recall["Attack"] == "DoS") & (df_recall["Model"] == "RF")]
    t_f1 = sub_rf_dos[sub_rf_dos["Mode"] == "F1-Optimal"]["Threshold"].iloc[0]
    t_r95 = sub_rf_dos[sub_rf_dos["Mode"] == "Recall>=95%"]["Threshold"].iloc[0]
    t_r98 = sub_rf_dos[sub_rf_dos["Mode"] == "Recall>=98%"]["Threshold"].iloc[0]

    ax1.axvline(t_f1, color="#e74c3c", linestyle=":", linewidth=1.8, label=f"F1-Opt Threshold (t={t_f1:.2f})")
    ax1.axvline(t_r95, color="#9b59b6", linestyle="-.", linewidth=1.8, label=f"Recall >= 95% (t={t_r95:.2f})")
    ax1.axvline(t_r98, color="#d35400", linestyle="-.", linewidth=1.8, label=f"Recall >= 98% (t={t_r98:.2f})")

    ax1.set_xlabel("Ngưỡng Quyết định (Decision Threshold t)", fontweight="bold")
    ax1.set_ylabel("Giá trị Chỉ số (Chùm Tấn công DoS)", fontweight="bold")
    ax1.set_title("Biến thiên Precision - Recall - F1 theo Ngưỡng (DoS - RF)", pad=12)
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0.45, 1.03)
    ax1.grid(True, zorder=0)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.92)

    # ---------------- PANEL PHẢI: BIỂU ĐỒ CỘT SO SÁNH SỐ CA FN BỊ BỎ SÓT ----------------
    attacks = TARGET_ATTACKS
    x = np.arange(len(attacks))
    width = 0.26

    rf_rec_df = df_recall[df_recall["Model"] == "RF"]

    fn_f1 = [rf_rec_df[(rf_rec_df["Attack"] == a) & (rf_rec_df["Mode"] == "F1-Optimal")]["FN"].iloc[0] for a in attacks]
    fn_r95 = [rf_rec_df[(rf_rec_df["Attack"] == a) & (rf_rec_df["Mode"] == "Recall>=95%")]["FN"].iloc[0] for a in attacks]
    fn_r98 = [rf_rec_df[(rf_rec_df["Attack"] == a) & (rf_rec_df["Mode"] == "Recall>=98%")]["FN"].iloc[0] for a in attacks]

    r1 = ax2.bar(x - width, fn_f1, width, label="F1-Optimal (Mặc định)", color="#e74c3c", edgecolor="#2c3e50", zorder=3)
    r2 = ax2.bar(x, fn_r95, width, label="Mục tiêu Recall >= 95%", color="#3498db", edgecolor="#2c3e50", zorder=3)
    r3 = ax2.bar(x + width, fn_r98, width, label="Mục tiêu Recall >= 98%", color="#2ecc71", edgecolor="#2c3e50", zorder=3)

    ax2.set_xlabel("Loại Tấn công", fontweight="bold")
    ax2.set_ylabel("Số Mẫu Tấn công Bị Bỏ sót (False Negatives - FN)", fontweight="bold")
    ax2.set_title("Cắt giảm Số Ca Bỏ sót (FN) khi Điều chỉnh Ngưỡng theo Recall", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks, fontweight="bold")
    ax2.grid(axis="y", zorder=0)
    ax2.set_ylim(0, max(fn_f1) * 1.25)
    ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    for rects, color in [(r1, "#c0392b"), (r2, "#2980b9"), (r3, "#27ae60")]:
        for rect in rects:
            h = rect.get_height()
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h + 3, f"{int(h)}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color=color)

    plt.suptitle("TỐI ƯU HÓA NGƯỠNG QUYẾT ĐỊNH THEO MỤC TIÊU RECALL VÀ TRIỆT TIÊU BỎ SÓT (FN)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Hình 11 thành công: {out_path}")


# ==============================================================================
# 5. HÌNH 12: TỐI ƯU CHI PHÍ VẬN HÀNH SOC (COST-SENSITIVE NIDS)
# ==============================================================================
def plot_fig12_cost_sensitive_analysis(df_cost: pd.DataFrame, out_path: Path):
    print("\n[*] Đang vẽ Hình 12: Cost-Sensitive Optimization & Cost Reduction...")
    setup_matplotlib_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ---------------- PANEL TRÁI: ĐƯỜNG CONG TỔNG CHI PHÍ VS THRESHOLD CHO DOS ----------------
    y_test_dos = np.load(BASELINE_DIR / "rf_dos_nosmote_y_test.npy")
    p_test_dos = np.load(BASELINE_DIR / "rf_dos_nosmote_best_proba_test.npy")

    t_scan = np.round(np.linspace(0.05, 0.95, 120), 3)
    sweep_dos = vectorized_threshold_metrics(y_test_dos, p_test_dos, t_scan)
    fn_arr = sweep_dos["fn"]
    fp_arr = sweep_dos["fp"]

    c_colors = {1: "#95a5a6", 5: "#3498db", 10: "#f39c12", 20: "#e74c3c"}
    c_sub = df_cost[(df_cost["Attack"] == "DoS") & (df_cost["Model"] == "RF")]

    for c in [1, 5, 10, 20]:
        costs = c * fn_arr + fp_arr
        t_star = c_sub[c_sub["Cost_Ratio_c"] == c]["Optimal_Threshold"].iloc[0]
        cost_star = c_sub[c_sub["Cost_Ratio_c"] == c]["Cost_Optimal"].iloc[0]

        ax1.plot(t_scan, costs, label=f"Cost (c={c}: 1 FN = {c} FP)", color=c_colors[c], linewidth=2.0, zorder=3)
        ax1.scatter([t_star], [cost_star], marker="*", s=140, color=c_colors[c], edgecolor="#2c3e50", zorder=5)
        ax1.annotate(f"t*={t_star:.2f}", (t_star, cost_star), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, fontweight="bold", color=c_colors[c])

    ax1.set_xlabel("Ngưỡng Quyết định (Decision Threshold t)", fontweight="bold")
    ax1.set_ylabel("Tổng Chi phí Vận hành SOC (Total Cost)", fontweight="bold")
    ax1.set_title("Hàm Chi phí Vận hành theo Ngưỡng Quyết định (DoS - RF)", pad=12)
    ax1.set_xlim(0.02, 0.98)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", zorder=0)
    ax1.legend(loc="upper center", fontsize=8.5, framealpha=0.92)

    # ---------------- PANEL PHẢI: TIẾT KIỆM CHI PHÍ & THAY ĐỔI TẠI C=10 ----------------
    attacks = TARGET_ATTACKS
    rf_c10 = df_cost[(df_cost["Model"] == "RF") & (df_cost["Cost_Ratio_c"] == 10)]

    red_pcts = [rf_c10[rf_c10["Attack"] == a]["Cost_Reduction_Pct"].iloc[0] for a in attacks]
    fn_opt = [rf_c10[rf_c10["Attack"] == a]["FN"].iloc[0] for a in attacks]
    fp_opt = [rf_c10[rf_c10["Attack"] == a]["FP"].iloc[0] for a in attacks]

    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")
    fn_f1 = [rf_sum[(rf_sum["Attack"] == a) & (rf_sum["Condition"] == "Without SMOTE")]["FN"].iloc[0] for a in attacks]
    fp_f1 = [rf_sum[(rf_sum["Attack"] == a) & (rf_sum["Condition"] == "Without SMOTE")]["FP"].iloc[0] for a in attacks]

    y_pos = np.arange(len(attacks))
    bars = ax2.barh(y_pos, red_pcts, color="#27ae60", edgecolor="#2c3e50", height=0.55, zorder=3)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(attacks, fontweight="bold")
    ax2.set_xlabel("Tỷ lệ Tiết kiệm Chi phí Vận hành SOC (%)", fontweight="bold")
    ax2.set_title("Hiệu quả Giảm Thiệt hại tại Tỷ lệ Phạt c=10 (1 FN = 10 FP)", pad=12)
    ax2.grid(axis="x", zorder=0)
    ax2.set_xlim(0, max(red_pcts) * 1.35 if max(red_pcts) > 0 else 50)

    for bar, pct, fn_o, fn_f, fp_o, fp_f in zip(bars, red_pcts, fn_opt, fn_f1, fp_opt, fp_f1):
        w = bar.get_width()
        txt = f"Tiết kiệm {pct:.1f}%\n(FN: {fn_f} -> {fn_o} | FP: {fp_f} -> {fp_o})"
        ax2.text(w + 1.2, bar.get_y() + bar.get_height() / 2.0, txt,
                 ha="left", va="center", fontsize=8, fontweight="bold", color="#1e8449")

    plt.suptitle("TỐI ƯU HÓA CHI PHÍ AN NINH MẠNG THEO MÔ HÌNH NHẠY CHI PHÍ (COST-SENSITIVE NIDS)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Hình 12 thành công: {out_path}")


# ==============================================================================
# 6. HÌNH 13: GIẢI PHẪU MẪU TẤN CÔNG LỌT LƯỚI (FALSE NEGATIVE FORENSICS)
# ==============================================================================
def plot_fig13_false_negative_forensics(
    df_forensics: pd.DataFrame,
    df_forensics_all: pd.DataFrame,
    df_recall: pd.DataFrame,
    df_cost: pd.DataFrame,
    out_path: Path,
):
    print("\n[*] Đang vẽ Hình 13: False Negative Forensics & Network Physical Stealth...")
    setup_matplotlib_style()

    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5))
    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")

    # ---------------- Ô (1,1): NEAR-MISS VS DEEP STEALTH (OURS RF + PAPER-EXACT RF) ----------------
    ax1 = axes[0, 0]
    attacks = TARGET_ATTACKS
    fn_f1_all = df_forensics_all[(df_forensics_all["Category"] == "FN") & (df_forensics_all["Mode"] == "F1-Optimal")]

    x = np.arange(len(attacks))
    models_cfg = [
        ("RF", -0.19, "#f39c12", "#c0392b", "Ours RF"),
        ("PaperRF", 0.19, "#eb984e", "#78281f", "Paper RF"),
    ]
    bar_w = 0.34
    max_tot = 0

    for idx_m, (m_name, offset, nm_col, ds_col, m_label) in enumerate(models_cfg):
        sub_m = fn_f1_all[fn_f1_all["Model"] == m_name]
        if sub_m.empty:
            continue
        nm_vals = [int(sub_m[sub_m["Attack"] == a]["Near_Miss_Count"].iloc[0]) for a in attacks]
        ds_vals = [int(sub_m[sub_m["Attack"] == a]["Deep_Stealth_Count"].iloc[0]) for a in attacks]
        tots = [nm + ds for nm, ds in zip(nm_vals, ds_vals)]
        max_tot = max(max_tot, max(tots) if tots else 0)

        lbl_nm = "Near-Miss (t_low <= p < t*)" if idx_m == 0 else None
        lbl_ds = "Deep Stealth (p < t_low)" if idx_m == 0 else None
        ax1.bar(x + offset, nm_vals, bar_w, label=lbl_nm, color=nm_col, edgecolor="#2c3e50", zorder=3)
        ax1.bar(x + offset, ds_vals, bar_w, bottom=nm_vals, label=lbl_ds, color=ds_col, edgecolor="#2c3e50", zorder=3)

        for i, (nm, ds, tot) in enumerate(zip(nm_vals, ds_vals, tots)):
            if tot > 0:
                nm_pct_str = f"{nm / tot:.0%}"
                txt = f"{m_label.split()[0]}:{tot}\n({nm_pct_str})"
            else:
                txt = f"{m_label.split()[0]}:0"
            ax1.text(x[i] + offset, tot + 4, txt, ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#2c3e50")

    ax1.set_xlabel("Loại Tấn công (Cột trái: Ours Improved RF | Cột phải: Paper-Exact RF)", fontweight="bold")
    ax1.set_ylabel("Số lượng Mẫu FN (F1-Optimal)", fontweight="bold")
    ax1.set_title("So sánh FN Hai hệ thống RF: Near-Miss vs Deep Stealth", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(attacks, fontweight="bold")
    ax1.set_ylim(0, max(max_tot, 10) * 1.28)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="upper right", fontsize=8.2, framealpha=0.92)

    # ---------------- Ô (1,2): GIẢI PHẪU ĐẶC TRƯNG ANALYSIS (TP VS FN VS TN) ----------------
    ax2 = axes[0, 1]
    X_test_an, y_test_an = get_binary_data("Analysis", split="heldout_test", drop_ip=True)
    p_test_an = np.load(BASELINE_DIR / "rf_analysis_nosmote_best_proba_test.npy")
    t_an = float(rf_sum[(rf_sum["Attack"] == "Analysis") & (rf_sum["Condition"] == "Without SMOTE")]["Threshold"].iloc[0])
    pred_an = (p_test_an >= t_an).astype(int)

    tp_mask_an = (y_test_an == 1) & (pred_an == 1)
    fn_mask_an = (y_test_an == 1) & (pred_an == 0)
    tn_mask_an = (y_test_an == 0) & (pred_an == 0)

    tp_an = X_test_an[tp_mask_an]["IN_BYTES"].values
    fn_an = X_test_an[fn_mask_an]["IN_BYTES"].values
    n_tn_an_total = int(np.sum(tn_mask_an))
    tn_an = X_test_an[tn_mask_an]["IN_BYTES"].sample(n=min(10000, n_tn_an_total), random_state=10).values

    fn_an_sub = X_test_an[fn_mask_an]
    tp_an_sub = X_test_an[tp_mask_an]
    fn_in_med = float(fn_an_sub["IN_BYTES"].median()) if len(fn_an_sub) > 0 else 0.0
    fn_out_med = float(fn_an_sub["OUT_BYTES"].median()) if len(fn_an_sub) > 0 else 0.0
    fn_port80_pct = float((fn_an_sub["L4_DST_PORT"] == 80).mean() * 100.0) if len(fn_an_sub) > 0 else 0.0
    tp_port0_pct = float((tp_an_sub["L4_DST_PORT"] == 0).mean() * 100.0) if len(tp_an_sub) > 0 else 0.0
    tp_out0_pct = float((tp_an_sub["OUT_BYTES"] == 0).mean() * 100.0) if len(tp_an_sub) > 0 else 0.0

    bplot1 = ax2.boxplot([tp_an, fn_an, tn_an], patch_artist=True,
                         tick_labels=[
                             f"TP (Phát hiện: N={len(tp_an):,})",
                             f"FN (Bỏ sót: N={len(fn_an):,})",
                             f"TN (Benign: N={n_tn_an_total/1000:.1f}k)",
                         ],
                         whis=[5, 95], showmeans=True,
                         meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"))

    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    for patch, c in zip(bplot1["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#2c3e50")

    ax2.set_yscale("log")
    ax2.set_ylim(bottom=30, top=max(np.max(tp_an), np.max(fn_an), np.max(tn_an)) * 12)
    ax2.set_ylabel("IN_BYTES (Thang Logarit)", fontweight="bold")
    ax2.set_title(f"Giải phẫu Phân phối IN_BYTES: Analysis (RF, t={t_an:.2f})", pad=10)
    ax2.grid(True, axis="y", zorder=0)
    ax2.text(
        0.5, 0.94,
        f"Phát hiện: FN có IN_BYTES (median={fn_in_med:.0f}B) & Port 80 ({fn_port80_pct:.1f}%)\n"
        f"giống hệt lưu lượng Benign, khác xa TP (Port 0: {tp_port0_pct:.0f}%, Out=0B: {tp_out0_pct:.0f}%)",
        transform=ax2.transAxes, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9db", edgecolor="#f59f00", lw=1),
        fontsize=8.3,
    )
    del X_test_an, y_test_an

    # ---------------- Ô (2,1): GIẢI PHẪU ĐẶC TRƯNG DOS (TP VS FN VS TN) ----------------
    ax3 = axes[1, 0]
    X_test_d, y_test_d = get_binary_data("DoS", split="heldout_test", drop_ip=True)
    p_test_dos = np.load(BASELINE_DIR / "rf_dos_nosmote_best_proba_test.npy")
    t_dos = float(rf_sum[(rf_sum["Attack"] == "DoS") & (rf_sum["Condition"] == "Without SMOTE")]["Threshold"].iloc[0])
    pred_dos = (p_test_dos >= t_dos).astype(int)

    tp_mask_d = (y_test_d == 1) & (pred_dos == 1)
    fn_mask_d = (y_test_d == 1) & (pred_dos == 0)
    tn_mask_d = (y_test_d == 0) & (pred_dos == 0)

    tp_dos = X_test_d[tp_mask_d]["OUT_BYTES"].values
    fn_dos = X_test_d[fn_mask_d]["OUT_BYTES"].values
    n_tn_dos_total = int(np.sum(tn_mask_d))
    tn_dos = X_test_d[tn_mask_d]["OUT_BYTES"].sample(n=min(10000, n_tn_dos_total), random_state=10).values

    dos_fn_row = df_forensics[(df_forensics["Attack"] == "DoS") & (df_forensics["Category"] == "FN")].iloc[0]
    dos_nm_pct = float(dos_fn_row["Near_Miss_Pct"])
    dos_near_low, _ = get_near_miss_bounds(t_dos)

    rf_dos_rec = df_recall[(df_recall["Attack"] == "DoS") & (df_recall["Model"] == "RF")]
    saved_r98 = -int(rf_dos_rec[rf_dos_rec["Mode"] == "Recall>=98%"]["Delta_FN_vs_F1Opt"].iloc[0])
    saved_r99 = -int(rf_dos_rec[rf_dos_rec["Mode"] == "Recall>=99%"]["Delta_FN_vs_F1Opt"].iloc[0])

    bplot2 = ax3.boxplot([tp_dos + 1, fn_dos + 1, tn_dos + 1], patch_artist=True,
                         tick_labels=[
                             f"TP (Phát hiện: N={len(tp_dos):,})",
                             f"FN (Bỏ sót: N={len(fn_dos):,})",
                             f"TN (Benign: N={n_tn_dos_total/1000:.1f}k)",
                         ],
                         whis=[5, 95], showmeans=True,
                         meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"))

    for patch, c in zip(bplot2["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#2c3e50")

    ax3.set_yscale("log")
    ax3.set_ylim(bottom=0.7, top=max(np.max(tp_dos), np.max(fn_dos), np.max(tn_dos)) * 15)
    ax3.set_ylabel("OUT_BYTES + 1 (Thang Logarit)", fontweight="bold")
    ax3.set_title(f"Giải phẫu Phân phối OUT_BYTES: DoS (RF, t={t_dos:.2f})", pad=10)
    ax3.grid(True, axis="y", zorder=0)
    ax3.text(
        0.5, 0.94,
        f"Phát hiện: {dos_nm_pct:.1f}% FN của DoS là Near-Miss ({dos_near_low:.2f} <= p < {t_dos:.2f})\n"
        f"Chỉ bị sót do ngưỡng F1-Opt cao; hạ ngưỡng (R>=98% / R>=99%) cứu được {saved_r98} - {saved_r99} ca!",
        transform=ax3.transAxes, ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f8f5", edgecolor="#1abc9c", lw=1),
        fontsize=8.3,
    )
    del X_test_d, y_test_d

    # ---------------- Ô (2,2): HỘP TỔNG KẾT & MA TRẬN RỦI RO AN NINH MẠNG ----------------
    ax4 = axes[1, 1]
    ax4.axis("off")

    rf_c10 = df_cost[(df_cost["Model"] == "RF") & (df_cost["Cost_Ratio_c"] == 10)]
    rf_benign_pass = rf_c10["TN"] / (rf_c10["TN"] + rf_c10["FP"]) * 100.0
    rf_pass_min = float(rf_benign_pass.min())
    rf_pass_max = float(rf_benign_pass.max())
    an_rf_c10_t = float(df_cost[(df_cost["Attack"] == "Analysis") & (df_cost["Model"] == "RF") & (df_cost["Cost_Ratio_c"] == 10)]["Optimal_Threshold"].iloc[0])
    an_rf_c10_low, _ = get_near_miss_bounds(an_rf_c10_t)

    summary_text = (
        "MA TRẬN RỦI RO & BẢN CHẤT VẬT LÝ MẠNG CỦA MẪU BỎ SÓT (FN)\n"
        "────────────────────────────────────────────────────────────\n\n"
        "1. PHÂN CẤP NGUY CƠ AN NINH MẠNG KHI LỌT LƯỚI:\n"
        " • Mức Thảm họa (Critical): Backdoor & Exploits -> Chiếm quyền điều khiển máy chủ.\n"
        "   Bắt buộc đặt ngưỡng nhạy chi phí c >= 10 hoặc Recall >= 98% để ép FN về tối thiểu.\n"
        " • Mức Cảnh báo (Moderate): Reconnaissance & Analysis -> Dò quét ban đầu.\n\n"
        "2. NGUYÊN NHÂN VẬT LÝ KHIẾN FN ẨN MÌNH Ở TẦNG NETFLOW:\n"
        f" • Ở Analysis (RF): {fn_port80_pct:.1f}% mẫu FN đánh vào Port 80 (HTTP) với kích thước\n"
        f"   gói tin bình thường (IN ~ {fn_in_med:.0f}B, OUT ~ {fn_out_med:.0f}B), giống hệt web Benign.\n"
        f" • Ở DoS (RF): {dos_nm_pct:.1f}% FN nằm sát ngưỡng ({dos_near_low:.2f} <= p < {t_dos:.2f}).\n"
        "   Hạ ngưỡng theo Recall/Chi phí cứu được phần lớn các mẫu này.\n\n"
        "3. KIẾN TRÚC TRIỂN KHAI THỰC TẾ ĐỀ XUẤT (2-TIER HYBRID NIDS):\n"
        f" • Tầng 1 (Inline NetFlow RF): Ours RF ở c=10 cho qua đúng {rf_pass_min:.2f}%-{rf_pass_max:.2f}% Benign\n"
        "   (giữ 99.7%+ lưu lượng sạch không bị cảnh báo nhầm với chi phí vận hành tối ưu).\n"
        " • Tầng 2 (Kiến trúc Chuyển tiếp DPI): Luồng vùng nghi vấn [t_low, t*) với\n"
        f"   t_low = min(0.10, 0.5*t*) (vd Analysis RF c=10: [{an_rf_c10_low:.4f}, {an_rf_c10_t:.3f}))\n"
        "   được định tuyến sang bộ phân tích heuristic NetFlow / DPI tầng 7."
    )

    ax4.text(0.03, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=8.3, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.7", facecolor="#f8f9fa", edgecolor="#34495e", lw=1.2))

    plt.suptitle("GIẢI PHẪU CHUYÊN SÂU MẪU TẤN CÔNG BỊ BỎ SÓT (FALSE NEGATIVE FORENSICS) - NF-UNSW-NB15",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Hình 13 thành công: {out_path}")


# ==============================================================================
# 7. XUẤT BÁO CÁO TỔNG HỢP MARKDOWN (FN_AND_COST_FINDINGS.MD)
# ==============================================================================
def generate_markdown_report(
    df_recall: pd.DataFrame,
    df_cost: pd.DataFrame,
    df_forensics_all: pd.DataFrame,
    out_path: Path,
):
    print("\n[*] Đang khởi tạo báo cáo tổng hợp: fn_and_cost_findings.md...")

    rf_rec = df_recall[df_recall["Model"] == "RF"]

    # Rút trích động các nhận định cốt lõi Recall-Targeted
    an_rf_f1 = rf_rec[(rf_rec["Attack"] == "Analysis") & (rf_rec["Mode"] == "F1-Optimal")].iloc[0]
    an_rf_98 = rf_rec[(rf_rec["Attack"] == "Analysis") & (rf_rec["Mode"] == "Recall>=98%")].iloc[0]
    an_tot = int(an_rf_f1["TP"] + an_rf_f1["FN"])
    an_miss_pct = int(an_rf_f1["FN"]) / max(an_tot, 1) * 100.0
    an_fn_drop_pct = (int(an_rf_f1["FN"]) - int(an_rf_98["FN"])) / max(int(an_rf_f1["FN"]), 1) * 100.0

    dos_rf_f1 = rf_rec[(rf_rec["Attack"] == "DoS") & (rf_rec["Mode"] == "F1-Optimal")].iloc[0]
    dos_rf_98 = rf_rec[(rf_rec["Attack"] == "DoS") & (rf_rec["Mode"] == "Recall>=98%")].iloc[0]
    dos_rf_99 = rf_rec[(rf_rec["Attack"] == "DoS") & (rf_rec["Mode"] == "Recall>=99%")].iloc[0]
    dos_tot = int(dos_rf_f1["TP"] + dos_rf_f1["FN"])
    dos_saved_98 = int(dos_rf_f1["FN"]) - int(dos_rf_98["FN"])
    dos_saved_99 = int(dos_rf_f1["FN"]) - int(dos_rf_99["FN"])
    dos_drop_99_pct = dos_saved_99 / max(int(dos_rf_f1["FN"]), 1) * 100.0

    md_content = f"""# Báo cáo Khoa học Chuyên sâu: Tối ưu Ngưỡng Recall, Mô hình Nhạy chi phí & Giải phẫu Mẫu Bỏ sót (FN)

**Dự án:** Phân loại Cảnh báo Xâm nhập Mạng (NIDS) trên dữ liệu NetFlow `NF-UNSW-NB15` (1.62M bản ghi)  
**Phạm vi:** Giai đoạn 2 - Tối ưu vận hành SOC & Giải phẫu FN trên Random Forest (`Ours Improved RF` & `Paper-Exact RF` trên Held-out Test)  
**Ngày thực hiện:** {time.strftime('%Y-%m-%d %H:%M:%S')}  

---

## 1. Tối ưu hóa Ngưỡng theo Recall Mục tiêu (Recall-Targeted Thresholding)

Trong các Trung tâm Điều hành An ninh Mạng (SOC), mục tiêu quan trọng nhất là **không được bỏ sót cuộc tấn công nào** (Zero-Trust / High Recall). Dưới đây là bảng đối sánh chuyển dịch ngưỡng từ điểm cân bằng `F1-Optimal` sang các chế độ vận hành ưu tiên độ phủ `Recall >= 95%`, `Recall >= 98%`, và `Recall >= 99%` trên tập kiểm thử độc lập Held-out Test.

### Bảng 1: Kết quả Điều chỉnh Ngưỡng theo Mục tiêu Recall trên Random Forest (Without SMOTE)

| Loại Tấn công | Chế độ | Ngưỡng $t^*$ | Near-Miss $t_{{low}}$ | Test Recall | Test Precision | Test F1 | FN Bị sót | Bắt thêm ($\\Delta$FN) | Đánh đổi Báo giả ($\\Delta$FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for att in TARGET_ATTACKS:
        sub = rf_rec[rf_rec["Attack"] == att]
        for _, r in sub.iterrows():
            if r["Mode"] == "F1-Optimal":
                delta_fn_str = "0 (Gốc)"
                delta_fp_str = "0 (Gốc)"
            else:
                delta_fn_str = f"**{int(r['Delta_FN_vs_F1Opt'])}** (Giảm)" if r["Delta_FN_vs_F1Opt"] < 0 else f"+{int(r['Delta_FN_vs_F1Opt'])}"
                delta_fp_str = f"+{int(r['Delta_FP_vs_F1Opt'])}" if r["Delta_FP_vs_F1Opt"] > 0 else f"{int(r['Delta_FP_vs_F1Opt'])}"
            md_content += (
                f"| **{att}** | {r['Mode']} | {r['Threshold']} | {r['NearMiss_Lower_Bound']} | "
                f"{r['Test_Recall']:.4f} | {r['Test_Precision']:.4f} | {r['Test_F1']:.4f} | "
                f"**{int(r['FN'])}** | {delta_fn_str} | {delta_fp_str} |\n"
            )

    md_content += f"""
> [!IMPORTANT]
> **Nhận định Cốt lõi (Từ dữ liệu đo kiểm RF):**
> - Với **Analysis**: Ở ngưỡng F1-Optimal mặc định ($t={an_rf_f1['Threshold']:.2f}$), mô hình bỏ sót **{int(an_rf_f1['FN'])}/{an_tot} cuộc tấn công ({an_miss_pct:.1f}%)**. Khi chuyển sang chế độ `Recall >= 98%` ($t={an_rf_98['Threshold']:.3f}$, vùng Near-Miss thích ứng $t_{{low}}={an_rf_98['NearMiss_Lower_Bound']:.4f}$), số mẫu bỏ sót giảm xuống còn **{int(an_rf_98['FN'])} ca (giảm {an_fn_drop_pct:.1f}% số ca lọt lưới)**.
> - Với **DoS**: Chuyển sang chế độ `Recall >= 98%` ($t={dos_rf_98['Threshold']:.3f}$) giúp số ca bỏ sót giảm từ **{int(dos_rf_f1['FN'])} xuống {int(dos_rf_98['FN'])} ca (cứu được {dos_saved_98} cuộc tấn công)**, và ở chế độ `Recall >= 99%` ($t={dos_rf_99['Threshold']:.3f}$) giảm xuống chỉ còn **{int(dos_rf_99['FN'])} ca (cứu được {dos_saved_99}/{int(dos_rf_f1['FN'])} ca bỏ sót, giảm {dos_drop_99_pct:.1f}% FN trên {dos_tot:,} mẫu DoS)**.

---

## 2. Tối ưu Hóa Chi phí Vận hành SOC (Cost-Sensitive Threshold Optimization)

Hàm tổn thất chi phí vận hành an ninh mạng:
$$\\text{{Total\\_Cost}}(t) = c \\times \\text{{FN}}(t) + 1 \\times \\text{{FP}}(t)$$
Trong thực tế tác chiến mạng, hậu quả của 1 cuộc tấn công lọt lưới (`FN`) nguy hiểm gấp nhiều lần một báo động giả (`FP`) do SOC phân tích nhầm. Bảng dưới đây khảo sát các hệ số thiệt hại $c \\in \\{{1, 5, 10, 20\\}}$ cho **Ours Improved Random Forest**, đồng thời áp dụng cận dưới Near-Miss thích ứng $t_{{low}} = \\min(0.10, 0.5 \\cdot t^*)$ để không làm sụp đổ vùng chuyển tiếp Tầng 2 khi $t^*_{{\\text{{cost}}}} < 0.10$.

### Bảng 2: Tối ưu Chi phí Vận hành trên Random Forest (Without SMOTE)

| Loại Tấn công | Hệ số Phạt $c$ | Ngưỡng Tối ưu $t^*_{{\\text{{cost}}}}$ | Near-Miss $t_{{low}}$ | FN tại $t^*$ | FP tại $t^*$ | Chi phí $t^*$ | Chi phí F1-Opt | **Tiết kiệm Chi phí (%)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    rf_cost = df_cost[df_cost["Model"] == "RF"]
    for att in TARGET_ATTACKS:
        sub = rf_cost[rf_cost["Attack"] == att]
        for _, r in sub.iterrows():
            md_content += (
                f"| **{att}** | $c={int(r['Cost_Ratio_c'])}$ | {r['Optimal_Threshold']:.3f} | {r['NearMiss_Lower_Bound']:.4f} | "
                f"{int(r['FN'])} | {int(r['FP'])} | {int(r['Cost_Optimal']):,} | {int(r['Cost_at_F1Opt']):,} | "
                f"**{r['Cost_Reduction_Pct']:.2f}%** |\n"
            )

    # Tạo chuỗi tóm tắt động c=10 cho RF
    rf_c10_df = rf_cost[rf_cost["Cost_Ratio_c"] == 10].set_index("Attack")
    rf_f1_fn_map = rf_rec[rf_rec["Mode"] == "F1-Optimal"].set_index("Attack")["FN"].to_dict()

    rf_c10_parts = [
        f"**{a} giảm {rf_c10_df.loc[a, 'Cost_Reduction_Pct']:.2f}%** (FN: {int(rf_f1_fn_map[a])} -> {int(rf_c10_df.loc[a, 'FN'])})"
        for a in TARGET_ATTACKS
    ]
    sub_010 = df_cost[df_cost["Optimal_Threshold"] < 0.10]
    sub_010_list = ", ".join(
        f"`{r['Attack']} {r['Model']} c={int(r['Cost_Ratio_c'])}` ($t^*={r['Optimal_Threshold']:.3f} \\rightarrow t_{{low}}={r['NearMiss_Lower_Bound']:.4f}$)"
        for _, r in sub_010.iterrows()
    )

    md_content += f"""
> [!TIP]
> **Quy luật Dịch chuyển Ngưỡng ($c=10$):**
> - **Ours Improved Random Forest ($c=10$):** {', '.join(rf_c10_parts)}.
> - **Xử lý trường hợp $t^*_{{\\text{{cost}}}} < 0.10$:** Khi ngưỡng chi phí giảm sâu dưới `0.10` ({sub_010_list}), quy tắc thích ứng $t_{{low}} = \\min(0.10, 0.5 \\cdot t^*)$ giữ vùng Near-Miss $[t_{{low}}, t^*)$ luôn hợp lệ và không bị triệt tiêu về rỗng.

---

## 3. Giải phẫu Chuyên sâu Mẫu Tấn công Bị Bỏ sót (False Negative Forensics)

### Bảng 3: Đối sánh FN Forensics Hai Hệ thống Random Forest tại F1-Optimal (`Ours RF` vs `PaperRF`)

| Loại Tấn công | Mô hình | Ngưỡng $t^*$ | Near-Miss $t_{{low}}$ | Tổng Mẫu FN | Near-Miss ($t_{{low}} \\le p < t^*$) | Deep Stealth ($p < t_{{low}}$) | `IN_BYTES` med (FN) | `OUT_BYTES` med (FN) | Zero-OutBytes (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    fn_f1_rows = df_forensics_all[(df_forensics_all["Category"] == "FN") & (df_forensics_all["Mode"] == "F1-Optimal")]
    for att in TARGET_ATTACKS:
        for m_name in ["RF", "PaperRF"]:
            sub = fn_f1_rows[(fn_f1_rows["Attack"] == att) & (fn_f1_rows["Model"] == m_name)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            md_content += (
                f"| **{att}** | {m_name} | {r['Threshold_Used']:.2f} | {r['NearMiss_Lower_Bound']:.4f} | "
                f"**{int(r['Count'])}** | {int(r['Near_Miss_Count'])} ({r['Near_Miss_Pct']:.1f}%) | "
                f"{int(r['Deep_Stealth_Count'])} ({r['Deep_Stealth_Pct']:.1f}%) | "
                f"{r['IN_BYTES_median']:.1f} | {r['OUT_BYTES_median']:.1f} | {r['Zero_OutBytes_Pct']:.2f}% |\n"
            )

    md_content += """
### Bảng 3b: Phân rã Near-Miss tại Chế độ Vận hành Cost-Opt $c=10$ (Đảm bảo Vùng Tầng 2 Không Bị Sập)

| Loại Tấn công | Mô hình | Ngưỡng $t^*_{c=10}$ | Near-Miss $t_{low}$ | Tổng Mẫu FN | Near-Miss ($t_{low} \\le p < t^*$) | Deep Stealth ($p < t_{low}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    fn_c10_rows = df_forensics_all[(df_forensics_all["Category"] == "FN") & (df_forensics_all["Mode"] == "Cost-Opt c=10")]
    for att in TARGET_ATTACKS:
        for m_name in ["RF"]:
            sub = fn_c10_rows[(fn_c10_rows["Attack"] == att) & (fn_c10_rows["Model"] == m_name)]
            if sub.empty:
                continue
            r = sub.iloc[0]
            md_content += (
                f"| **{att}** | {m_name} | {r['Threshold_Used']:.3f} | {r['NearMiss_Lower_Bound']:.4f} | "
                f"**{int(r['Count'])}** | {int(r['Near_Miss_Count'])} ({r['Near_Miss_Pct']:.1f}%) | "
                f"{int(r['Deep_Stealth_Count'])} ({r['Deep_Stealth_Pct']:.1f}%) |\n"
            )

    # Tính thông số Analysis & RF Benign pass-through động cho Root Cause
    an_fn_rf = fn_f1_rows[(fn_f1_rows["Attack"] == "Analysis") & (fn_f1_rows["Model"] == "RF")].iloc[0]
    dos_fn_rf = fn_f1_rows[(fn_f1_rows["Attack"] == "DoS") & (fn_f1_rows["Model"] == "RF")].iloc[0]
    rf_c10_pass = rf_c10_df["TN"] / (rf_c10_df["TN"] + rf_c10_df["FP"]) * 100.0
    rf_pass_min = float(rf_c10_pass.min())
    rf_pass_max = float(rf_c10_pass.max())

    md_content += f"""
### Bản chất Vật lý Mạng của Hiện tượng Lọt lưới (Root Cause Analysis):
1. **Thiếu thông tin Payload tầng 7 ở NetFlow:** Kẻ tấn công thực hiện hành vi dò quét/khai thác ứng dụng web (`Analysis`) trên cổng phổ thông (`Top_Dst_Ports` của FN RF: `{an_fn_rf['Top_Dst_Ports']}`). Ở tầng NetFlow, nhóm FN này có `IN_BYTES` trung vị **{an_fn_rf['IN_BYTES_median']:.1f}B** và `OUT_BYTES` trung vị **{an_fn_rf['OUT_BYTES_median']:.1f}B**, trùng khớp với lưu lượng web `Benign` thông thường.
2. **Tỷ lệ Near-Miss cao ở F1-Optimal:** Với `DoS`, **{dos_fn_rf['Near_Miss_Pct']:.2f}%** mẫu FN của RF ({int(dos_fn_rf['Near_Miss_Count'])}/{int(dos_fn_rf['Count'])}) nằm sát ngưỡng (`{dos_fn_rf['NearMiss_Lower_Bound']:.2f} <= p < {dos_fn_rf['Threshold_Used']:.2f}`); hạ ngưỡng theo chi phí ($c=10$) hoặc mục tiêu Recall ($R \\ge 98\\%, 99\\%$) giúp giải phóng và bắt giữ phần lớn các mẫu này.
3. **Kiến trúc 2-Tier Hybrid NIDS Đề xuất:**
   - **Tầng 1 (Suy luận NetFlow tốc độ cao):** `Ours Improved RF` với ngưỡng chi phí $c=10$ cho qua đúng **{rf_pass_min:.2f}%–{rf_pass_max:.2f}%** lưu lượng `Benign` (giảm thiểu cảnh báo giả trên lưu lượng sạch).
   - **Tầng 2 (Kiến trúc Chuyển tiếp DPI & Heuristic NetFlow):** Các luồng nằm trong vùng nghi vấn $[t_{{low}}, t^*)$ (với $t_{{low}} = \\min(0.10, 0.5 \\cdot t^*)$) được định tuyến sang bộ phân tích heuristic NetFlow tầng L4/L7/TCP-flags (và kiến trúc tích hợp Zeek/Suricata DPI giải mã payload tầng 7).

---
*Báo cáo được tự động khởi tạo bởi `src/advanced_evaluation.py`.*
"""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[+] Đã lưu báo cáo Markdown hoàn chỉnh: {out_path}")


# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================
def main():
    start_time = time.perf_counter()
    print("=" * 80)
    print(" BẮT ĐẦU QUY TRÌNH ĐÁNH GIÁ NÂNG CAO & GIẢI PHẪU FORENSICS (BƯỚC 5)")
    print("=" * 80)

    df_recall = run_recall_targeted_evaluation()
    df_cost = run_cost_sensitive_evaluation()
    df_forensics, df_forensics_all = run_false_negative_forensics(df_cost=df_cost)

    plot_fig11_recall_threshold_tradeoff(df_recall, FIGURES_DIR / "fig11_recall_threshold_tradeoff.png")
    plot_fig12_cost_sensitive_analysis(df_cost, FIGURES_DIR / "fig12_cost_sensitive_analysis.png")
    plot_fig13_false_negative_forensics(
        df_forensics,
        df_forensics_all,
        df_recall,
        df_cost,
        FIGURES_DIR / "fig13_false_negative_forensics.png",
    )

    report_file = IMPROVED_DIR / "fn_and_cost_findings.md"
    generate_markdown_report(df_recall, df_cost, df_forensics_all, report_file)

    print("\n" + "=" * 80)
    print(f" HOÀN TẤT TOÀN BỘ BƯỚC 5 TRONG {time.perf_counter() - start_time:.2f} GIÂY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
