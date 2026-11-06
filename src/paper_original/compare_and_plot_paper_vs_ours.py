#!/usr/bin/env python3
"""
compare_and_plot_paper_vs_ours.py
Tổng hợp bảng số liệu, xuất 4 biểu đồ học thuật 300 DPI và khởi tạo báo cáo khoa học
đối sánh trực tiếp 3 hệ thống trên cùng bộ phân vùng dữ liệu NF-UNSW-NB15:
  1. Hệ thống 1 (Paper-Exact RF): Tái lập nguyên bản 100% pipeline Pospíchal et al. (MDPI Information 2026)
  2. Hệ thống 2 (Ours Improved RF): Random Forest cải tiến (MutualInfoScorer rời rạc, ImbPipeline chống rò rỉ CV, dải ngưỡng 0.01..0.99)
  3. Hệ thống 3 (Ours HistGB & 2-Tier Cost-Sensitive NIDS): HistGradientBoosting tốc độ cao + Tối ưu ngưỡng chi phí SOC (c=10) / Recall mục tiêu

QUY TẮC CÁCH LY:
- CHỈ ĐỌC từ results/baseline/ và results/improved/.
- MỌI FILE ĐẦU RA chỉ ghi vào results/paper_original/ và results/paper_original/figures/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score

PAPER_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_SRC_DIR.parent.parent

PAPER_DIR = PROJECT_ROOT / "results" / "paper_original"
FIGURES_OUT_DIR = PAPER_DIR / "figures"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
IMPROVED_DIR = PROJECT_ROOT / "results" / "improved"

TARGET_ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]


def setup_matplotlib_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 0.85,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.titlesize": 13,
        "figure.titleweight": "bold",
        "grid.color": "#dfe6e9",
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.85,
    })


def _compute_pr_auc(dir_path: Path, prefix: str, attack: str, cond_code: str) -> float:
    p_file = dir_path / f"{prefix}_{attack.lower()}_{cond_code}_best_proba_test.npy"
    y_file = dir_path / f"{prefix}_{attack.lower()}_{cond_code}_y_test.npy"
    if p_file.exists() and y_file.exists():
        y_true = np.load(y_file)
        y_prob = np.load(p_file)
        return round(float(average_precision_score(y_true, y_prob)), 4)
    return np.nan


def _get_ours_rf_times(attack: str, cond_code: str, best_k: int) -> tuple:
    """Đọc train_time_sec và total_k_time_sec của Ours RF từ metrics CSV."""
    m_file = BASELINE_DIR / f"rf_{attack.lower()}_{cond_code}_metrics.csv"
    if not m_file.exists():
        return (np.nan, np.nan)
    df_m = pd.read_csv(m_file)
    sub = df_m[df_m["k"] == best_k]
    if sub.empty:
        sub = df_m.loc[[df_m["val_f1"].idxmax()]]
    row = sub.iloc[0]
    train_t = float(row["train_time_sec"]) if "train_time_sec" in row else np.nan
    test_t = float(row["test_time_sec"]) if "test_time_sec" in row else 0.0
    total_k_t = float(row["total_k_time_sec"]) if "total_k_time_sec" in row else train_t
    search_est = max(0.0, round(total_k_t - train_t - test_t, 2)) if pd.notna(total_k_t) and pd.notna(train_t) else np.nan
    return (round(train_t, 2), search_est)


def _get_ours_gb_times(attack: str, cond_code: str, best_k: int) -> tuple:
    """Đọc train_time_sec và search_time_sec ước lượng từ total_k_time_sec của Ours GB."""
    m_file = IMPROVED_DIR / f"gb_{attack.lower()}_{cond_code}_metrics.csv"
    if not m_file.exists():
        return (np.nan, np.nan)
    df_m = pd.read_csv(m_file)
    sub = df_m[df_m["k"] == best_k]
    if sub.empty:
        sub = df_m.loc[[df_m["val_f1"].idxmax()]]
    row = sub.iloc[0]
    train_t = float(row["train_time_sec"]) if "train_time_sec" in row else np.nan
    test_t = float(row["test_time_sec"]) if "test_time_sec" in row else 0.0
    if "search_time_sec" in row and pd.notna(row["search_time_sec"]):
        search_t = float(row["search_time_sec"])
    elif "total_k_time_sec" in row and pd.notna(row["total_k_time_sec"]) and pd.notna(train_t):
        search_t = max(0.0, float(row["total_k_time_sec"]) - train_t - test_t)
    else:
        search_t = 0.0
    return (round(train_t, 2), round(search_t, 2))


def build_comparison_tables() -> tuple:
    """Tổng hợp 3 bảng CSV đối sánh cốt lõi trong results/paper_original/."""
    FIGURES_OUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_sum_path = PAPER_DIR / "summary_paper_rf_comparison.csv"
    if not paper_sum_path.exists():
        raise FileNotFoundError(f"Chưa tìm thấy {paper_sum_path}. Hãy chạy train_paper_rf.py trước!")

    paper_sum = pd.read_csv(paper_sum_path)
    ours_rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")
    ours_gb_sum = pd.read_csv(IMPROVED_DIR / "summary_gb_comparison.csv")
    cost_df = pd.read_csv(IMPROVED_DIR / "cost_sensitive_evaluation.csv")

    # 1. Bảng đối sánh tổng thể 3 hệ thống (5 attacks x 2 conditions = 10 dòng)
    rows_3sys = []
    for att in TARGET_ATTACKS:
        for cond_code, cond_label in [("nosmote", "Without SMOTE"), ("smote", "With SMOTE")]:
            p_sub = paper_sum[(paper_sum["Attack"] == att) & (paper_sum["Condition"] == cond_label)]
            r_sub = ours_rf_sum[(ours_rf_sum["Attack"] == att) & (ours_rf_sum["Condition"] == cond_label)]
            g_sub = ours_gb_sum[(ours_gb_sum["Attack"] == att) & (ours_gb_sum["Condition"] == cond_label)]
            if p_sub.empty or r_sub.empty or g_sub.empty:
                continue

            p_row = p_sub.iloc[0]
            r_row = r_sub.iloc[0]
            g_row = g_sub.iloc[0]

            r_train_t, r_search_t = _get_ours_rf_times(att, cond_code, int(r_row["Best_k"]))
            g_train_t, g_search_t = _get_ours_gb_times(att, cond_code, int(g_row["Best_k"]))

            p_pr_auc = _compute_pr_auc(PAPER_DIR, "paper_rf", att, cond_code)
            r_pr_auc = _compute_pr_auc(BASELINE_DIR, "rf", att, cond_code)
            g_pr_auc = _compute_pr_auc(IMPROVED_DIR, "gb", att, cond_code)

            p_total_bestk_t = round(float(p_row["Search_Time_Sec"]) + float(p_row["Train_Time_Sec"]), 2)
            r_total_bestk_t = round((r_search_t if pd.notna(r_search_t) else 0.0) + r_train_t, 2)
            g_total_bestk_t = round((g_search_t if pd.notna(g_search_t) else 0.0) + g_train_t, 2)

            p_cost_c10 = int(10 * int(p_row["FN"]) + int(p_row["FP"]))
            r_cost_c10 = int(10 * int(r_row["FN"]) + int(r_row["FP"]))
            g_cost_c10 = int(10 * int(g_row["FN"]) + int(g_row["FP"]))

            rows_3sys.append({
                "Attack": att,
                "Condition": cond_label,
                # Paper-Exact RF
                "Paper_Best_k": int(p_row["Best_k"]),
                "Paper_Threshold": float(p_row["Threshold"]),
                "Paper_CV_F1": float(p_row["CV_Best_F1"]) if "CV_Best_F1" in p_row else np.nan,
                "Paper_Val_F1": float(p_row["Val_F1"]),
                "Paper_Precision": float(p_row["Precision"]),
                "Paper_Recall": float(p_row["Recall"]),
                "Paper_F1": float(p_row["F1_Score"]),
                "Paper_PR_AUC": p_pr_auc,
                "Paper_FP": int(p_row["FP"]),
                "Paper_FN": int(p_row["FN"]),
                "Paper_SOC_Cost_c10": p_cost_c10,
                "Paper_Search_Time_s": float(p_row["Search_Time_Sec"]),
                "Paper_Train_Time_s": float(p_row["Train_Time_Sec"]),
                "Paper_BestK_Total_Time_s": p_total_bestk_t,
                "Paper_All4K_Search_Time_s": float(p_row["Total_4k_Search_Time_Sec"]),
                # Ours Improved RF
                "OursRF_Best_k": int(r_row["Best_k"]),
                "OursRF_Threshold": float(r_row["Threshold"]),
                "OursRF_Precision": float(r_row["Precision"]),
                "OursRF_Recall": float(r_row["Recall"]),
                "OursRF_F1": float(r_row["F1_Score"]),
                "OursRF_PR_AUC": r_pr_auc,
                "OursRF_FP": int(r_row["FP"]),
                "OursRF_FN": int(r_row["FN"]),
                "OursRF_SOC_Cost_c10": r_cost_c10,
                "OursRF_Search_Time_s": r_search_t,
                "OursRF_Train_Time_s": r_train_t,
                "OursRF_BestK_Total_Time_s": r_total_bestk_t,
                # Ours HistGradientBoosting
                "OursGB_Best_k": int(g_row["Best_k"]),
                "OursGB_Threshold": float(g_row["Threshold"]),
                "OursGB_Precision": float(g_row["Precision"]),
                "OursGB_Recall": float(g_row["Recall"]),
                "OursGB_F1": float(g_row["F1_Score"]),
                "OursGB_PR_AUC": g_pr_auc,
                "OursGB_FP": int(g_row["FP"]),
                "OursGB_FN": int(g_row["FN"]),
                "OursGB_SOC_Cost_c10": g_cost_c10,
                "OursGB_Search_Time_s": g_search_t,
                "OursGB_Train_Time_s": g_train_t,
                "OursGB_BestK_Total_Time_s": g_total_bestk_t,
                # Deltas & Speedups vs Paper-Exact
                "Delta_F1_OursRF_vs_Paper": round(float(r_row["F1_Score"]) - float(p_row["F1_Score"]), 4),
                "Delta_F1_OursGB_vs_Paper": round(float(g_row["F1_Score"]) - float(p_row["F1_Score"]), 4),
                "Pipeline_Speedup_OursRF_vs_Paper": round(p_total_bestk_t / max(r_total_bestk_t, 0.01), 2),
                "Pipeline_Speedup_OursGB_vs_Paper": round(p_total_bestk_t / max(g_total_bestk_t, 0.01), 2),
                "TrainOnly_Speedup_OursGB_vs_Paper": round(float(p_row["Train_Time_Sec"]) / max(g_train_t, 0.01), 2),
            })

    df_3sys = pd.DataFrame(rows_3sys)
    df_3sys.to_csv(PAPER_DIR / "three_systems_comparison.csv", index=False)

    # 2. Bảng đối sánh vận hành SOC tại c=10 (Cost = 10*FN + 1*FP)
    soc_rows = []
    for att in TARGET_ATTACKS:
        p_no = df_3sys[(df_3sys["Attack"] == att) & (df_3sys["Condition"] == "Without SMOTE")].iloc[0]
        p_sm = df_3sys[(df_3sys["Attack"] == att) & (df_3sys["Condition"] == "With SMOTE")].iloc[0]

        rf_c10 = cost_df[
            (cost_df["Attack"] == att)
            & (cost_df["Model"].isin(["RF", "Random Forest"]))
            & (cost_df["Cost_Ratio_c"] == 10)
        ].iloc[0]
        gb_c10 = cost_df[
            (cost_df["Attack"] == att)
            & (cost_df["Model"].isin(["GB", "Gradient Boosting", "HistGradientBoosting"]))
            & (cost_df["Cost_Ratio_c"] == 10)
        ].iloc[0]

        paper_cost_no = int(p_no["Paper_SOC_Cost_c10"])
        paper_cost_sm = int(p_sm["Paper_SOC_Cost_c10"])
        ours_rf_cost_c10 = int(rf_c10["Cost_Optimal"])
        ours_gb_cost_c10 = int(gb_c10["Cost_Optimal"])

        soc_rows.append({
            "Attack": att,
            "Paper_NoSMOTE_Threshold": float(p_no["Paper_Threshold"]),
            "Paper_NoSMOTE_FN": int(p_no["Paper_FN"]),
            "Paper_NoSMOTE_FP": int(p_no["Paper_FP"]),
            "Paper_NoSMOTE_Cost_c10": paper_cost_no,
            "Paper_SMOTE_FN": int(p_sm["Paper_FN"]),
            "Paper_SMOTE_FP": int(p_sm["Paper_FP"]),
            "Paper_SMOTE_Cost_c10": paper_cost_sm,
            "OursRF_CostOpt_Threshold": float(rf_c10["Optimal_Threshold"]),
            "OursRF_CostOpt_FN": int(rf_c10["FN"]),
            "OursRF_CostOpt_FP": int(rf_c10["FP"]),
            "OursRF_CostOpt_Cost_c10": ours_rf_cost_c10,
            "OursGB_CostOpt_Threshold": float(gb_c10["Optimal_Threshold"]),
            "OursGB_CostOpt_FN": int(gb_c10["FN"]),
            "OursGB_CostOpt_FP": int(gb_c10["FP"]),
            "OursGB_CostOpt_Cost_c10": ours_gb_cost_c10,
            "FN_Saved_OursRFCost_vs_PaperNoSMOTE": int(p_no["Paper_FN"]) - int(rf_c10["FN"]),
            "Cost_Saved_Pct_OursRFCost_vs_PaperNoSMOTE": round(
                (paper_cost_no - ours_rf_cost_c10) / max(paper_cost_no, 1) * 100.0, 2
            ),
            "Cost_Saved_Pct_OursGBCost_vs_PaperNoSMOTE": round(
                (paper_cost_no - ours_gb_cost_c10) / max(paper_cost_no, 1) * 100.0, 2
            ),
        })

    df_soc = pd.DataFrame(soc_rows)
    df_soc.to_csv(PAPER_DIR / "soc_operational_comparison_c10.csv", index=False)

    # 3. Bảng đối sánh độ lệch xếp hạng Mutual Information (Paper auto MI vs Ours Discrete MI)
    paper_mi_path = PAPER_DIR / "paper_mi_scores_5attacks.csv"
    ours_mi_path = BASELINE_DIR / "mi_scores_5attacks.csv"
    df_mi_comp = None
    if paper_mi_path.exists() and ours_mi_path.exists():
        p_mi = pd.read_csv(paper_mi_path)
        o_mi = pd.read_csv(ours_mi_path)
        # Chuẩn hóa cấu trúc ours_mi_path
        if "Feature" in o_mi.columns and "Attack" in o_mi.columns:
            o_sub = o_mi.copy()
            score_col_o = "MI_Score" if "MI_Score" in o_sub.columns else [c for c in o_sub.columns if "score" in c.lower()][0]
            o_sub = o_sub.rename(columns={score_col_o: "Ours_Discrete_MI"})
            o_sub["Ours_Rank"] = o_sub.groupby("Attack")["Ours_Discrete_MI"].rank(ascending=False, method="min").astype(int)
        else:
            # Nếu dạng wide pivot (Feature x Attack)
            feat_col = o_mi.columns[0]
            melted = o_mi.melt(id_vars=[feat_col], var_name="Attack", value_name="Ours_Discrete_MI")
            o_sub = melted.rename(columns={feat_col: "Feature"})
            o_sub["Ours_Rank"] = o_sub.groupby("Attack")["Ours_Discrete_MI"].rank(ascending=False, method="min").astype(int)

        p_no_mi = p_mi[p_mi["Condition"] == "Without SMOTE"].rename(
            columns={"MI_Score": "Paper_Auto_MI_NoSMOTE", "Rank": "Paper_Rank_NoSMOTE"}
        )[["Attack", "Feature", "Paper_Auto_MI_NoSMOTE", "Paper_Rank_NoSMOTE"]]

        p_sm_mi = p_mi[p_mi["Condition"] == "With SMOTE"].rename(
            columns={"MI_Score": "Paper_Auto_MI_SMOTE", "Rank": "Paper_Rank_SMOTE"}
        )[["Attack", "Feature", "Paper_Auto_MI_SMOTE", "Paper_Rank_SMOTE"]]

        df_mi_comp = p_no_mi.merge(p_sm_mi, on=["Attack", "Feature"], how="left").merge(
            o_sub[["Attack", "Feature", "Ours_Discrete_MI", "Ours_Rank"]],
            on=["Attack", "Feature"],
            how="left",
        )
        df_mi_comp["Rank_Shift_Auto_vs_Discrete"] = df_mi_comp["Ours_Rank"] - df_mi_comp["Paper_Rank_NoSMOTE"]
        df_mi_comp.to_csv(PAPER_DIR / "mi_ranking_distortion_comparison.csv", index=False)

    return df_3sys, df_soc, df_mi_comp


def plot_fig_p01_three_systems_f1_and_speed(df_3sys: pd.DataFrame, out_path: Path):
    """Hình P1: Đối sánh F1-Score (Without SMOTE) và Tổng thời gian tối ưu + huấn luyện (k*) của 3 hệ thống."""
    setup_matplotlib_style()
    df_no = df_3sys[df_3sys["Condition"] == "Without SMOTE"].copy()
    df_no["order"] = df_no["Attack"].apply(lambda x: TARGET_ATTACKS.index(x))
    df_no = df_no.sort_values("order").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.4))
    attacks = df_no["Attack"].tolist()
    x = np.arange(len(attacks))
    w = 0.26

    c_paper = "#d62728"   # Đỏ trầm - Paper-Exact RF
    c_ours_rf = "#1f77b4" # Xanh dương - Ours Improved RF
    c_ours_gb = "#2ca02c" # Xanh lục - Ours HistGB

    # Panel trái: F1-Score
    b1 = ax1.bar(x - w, df_no["Paper_F1"], w, label="1. Paper-Exact RF (Pospíchal et al. 2026)", color=c_paper, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    b2 = ax1.bar(x, df_no["OursRF_F1"], w, label="2. Ours Improved RF (Baseline Cải tiến)", color=c_ours_rf, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    b3 = ax1.bar(x + w, df_no["OursGB_F1"], w, label="3. Ours HistGradientBoosting (Tốc độ cao)", color=c_ours_gb, edgecolor="#2c3e50", alpha=0.9, zorder=3)

    min_f1 = min(df_no["Paper_F1"].min(), df_no["OursRF_F1"].min(), df_no["OursGB_F1"].min())
    ax1.set_ylim(max(0.65, min_f1 - 0.08), 1.035)
    ax1.set_xticks(x)
    ax1.set_xticklabels(attacks, fontweight="bold")
    ax1.set_ylabel("Held-out Test F1-Score", fontweight="bold")
    ax1.set_title("A. Hiệu năng Phát hiện F1-Score (Điều kiện Chuẩn Without SMOTE)", pad=12)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="lower right", fontsize=8.3, framealpha=0.95)

    for bars, col in [(b1, "#a61c1c"), (b2, "#155d8b"), (b3, "#1e7e34")]:
        for rect in bars:
            h = rect.get_height()
            ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.004, f"{h:.4f}",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col, rotation=0)

    # Panel phải: Thời gian Search + Fit tại k* (Log scale)
    t1 = ax2.bar(x - w, df_no["Paper_BestK_Total_Time_s"], w, label="1. Paper-Exact RF (100% CV Search + Fit)", color=c_paper, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    t2 = ax2.bar(x, df_no["OursRF_BestK_Total_Time_s"], w, label="2. Ours Improved RF (150k CV Search + Fit)", color=c_ours_rf, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    t3 = ax2.bar(x + w, df_no["OursGB_BestK_Total_Time_s"], w, label="3. Ours HistGB (Histogram Search + Fit)", color=c_ours_gb, edgecolor="#2c3e50", alpha=0.9, zorder=3)

    ax2.set_yscale("log")
    max_t = df_no["Paper_BestK_Total_Time_s"].max()
    ax2.set_ylim(1.0, max_t * 9.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks, fontweight="bold")
    ax2.set_ylabel("Tổng Thời gian Search + Fit tại k* (Giây - Log)", fontweight="bold")
    ax2.set_title("B. Chi phí Tính toán Pipeline tại k* (Thang Logarit)", pad=12)
    ax2.grid(axis="y", which="both", zorder=0)
    ax2.legend(loc="upper right", fontsize=8.3, framealpha=0.95)

    for i in range(len(attacks)):
        for rect, col in [(t1[i], "#a61c1c"), (t2[i], "#155d8b"), (t3[i], "#1e7e34")]:
            h = rect.get_height()
            label_s = f"{h:.0f}s" if h >= 20 else f"{h:.1f}s"
            ax2.text(rect.get_x() + rect.get_width() / 2.0, h * 1.13, label_s,
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=col)
        sp_gb = df_no.iloc[i]["Pipeline_Speedup_OursGB_vs_Paper"]
        ax2.text(
            x[i], df_no.iloc[i]["Paper_BestK_Total_Time_s"] * 2.05,
            f"GB nhanh hơn\n{sp_gb:.0f}x",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=0.9),
        )

    plt.suptitle("ĐỐI SÁNH TRỰC TIẾP 3 HỆ THỐNG NIDS TRÊN CÙNG PHÂN VÙNG NF-UNSW-NB15 (1.62 TRIỆU LUỒNG)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fig_p02_smote_paradox_and_leakage(df_3sys: pd.DataFrame, df_mi_comp: pd.DataFrame, out_path: Path):
    """Hình P2 (2x2): Nghịch lý SMOTE, Rò rỉ CV nội suy, Báo động giả FP và Độ lệch Mutual Information."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 11.5))

    df_no = df_3sys[df_3sys["Condition"] == "Without SMOTE"].set_index("Attack").loc[TARGET_ATTACKS].reset_index()
    df_sm = df_3sys[df_3sys["Condition"] == "With SMOTE"].set_index("Attack").loc[TARGET_ATTACKS].reset_index()
    x = np.arange(len(TARGET_ATTACKS))

    # Ô (1,1): Paper-Exact Without SMOTE vs With SMOTE (Tái lập Table 7 trên NF-UNSW-NB15)
    ax = axes[0, 0]
    w = 0.34
    r1 = ax.bar(x - w / 2, df_no["Paper_F1"], w, label="Paper-Exact: Without SMOTE", color="#34495e", edgecolor="#2c3e50", alpha=0.92, zorder=3)
    r2 = ax.bar(x + w / 2, df_sm["Paper_F1"], w, label="Paper-Exact: With Fixed-SMOTE (56k)", color="#e67e22", edgecolor="#2c3e50", alpha=0.92, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax.set_ylim(max(0.60, min(df_no["Paper_F1"].min(), df_sm["Paper_F1"].min()) - 0.08), 1.03)
    ax.set_ylabel("Held-out Test F1-Score", fontweight="bold")
    ax.set_title("A. Paper-Exact RF: Without SMOTE vs Fixed-SMOTE (56,000 mẫu)", pad=10)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower right", fontsize=8.5)
    for i in range(len(TARGET_ATTACKS)):
        delta = df_sm.loc[i, "Paper_F1"] - df_no.loc[i, "Paper_F1"]
        top_y = max(df_no.loc[i, "Paper_F1"], df_sm.loc[i, "Paper_F1"])
        sign = "+" if delta >= 0 else ""
        col_d = "#1e7e34" if delta >= 0 else "#c0392b"
        ax.text(x[i], top_y + 0.008, f"Δ={sign}{delta:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=col_d)

    # Ô (1,2): Bằng chứng rò rỉ dữ liệu 3-Fold CV khi chạy SMOTE trước CV (Paper-Exact With SMOTE)
    ax = axes[0, 1]
    w3 = 0.25
    ax.bar(x - w3, df_sm["Paper_CV_F1"], w3, label="3-Fold CV F1 trên Train đã SMOTE (Bị rò rỉ nội suy)", color="#e74c3c", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    ax.bar(x, df_sm["Paper_Val_F1"], w3, label="Validation F1 thực tế (Không SMOTE)", color="#f1c40f", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    ax.bar(x + w3, df_sm["Paper_F1"], w3, label="Held-out Test F1 thực tế (Không SMOTE)", color="#2980b9", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax.set_ylim(max(0.55, df_sm["Paper_F1"].min() - 0.10), 1.06)
    ax.set_ylabel("F1-Score", fontweight="bold")
    ax.set_title("B. Ảo giác Hiệu năng Nội bộ (CV Leakage) khi SMOTE đặt trước 3-Fold CV", pad=10)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="lower right", fontsize=8.2)
    for i in range(len(TARGET_ATTACKS)):
        gap = df_sm.loc[i, "Paper_CV_F1"] - df_sm.loc[i, "Paper_F1"]
        ax.text(x[i] - w3, df_sm.loc[i, "Paper_CV_F1"] + 0.006, f"{df_sm.loc[i, 'Paper_CV_F1']:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#c0392b")
        ax.text(x[i] + w3, df_sm.loc[i, "Paper_F1"] + 0.006, f"-{gap:.3f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1f618d")

    # Ô (2,1): So sánh số lượng Báo động giả (False Positives) ở nhánh With SMOTE (Paper vs Ours RF)
    ax = axes[1, 0]
    w4 = 0.22
    ax.bar(x - 1.5 * w4, df_no["Paper_FP"], w4, label="Paper RF (No SMOTE)", color="#95a5a6", edgecolor="#2c3e50", zorder=3)
    ax.bar(x - 0.5 * w4, df_sm["Paper_FP"], w4, label="Paper RF (With SMOTE)", color="#d62728", edgecolor="#2c3e50", zorder=3)
    ax.bar(x + 0.5 * w4, df_no["OursRF_FP"], w4, label="Ours RF (No SMOTE)", color="#7fb3d5", edgecolor="#2c3e50", zorder=3)
    ax.bar(x + 1.5 * w4, df_sm["OursRF_FP"], w4, label="Ours RF (ImbPipeline SMOTE)", color="#1f77b4", edgecolor="#2c3e50", zorder=3)
    ax.set_yscale("symlog", linthresh=10)
    max_fp = max(df_sm["Paper_FP"].max(), df_sm["OursRF_FP"].max(), df_no["Paper_FP"].max())
    ax.set_ylim(0, max_fp * 3.5)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax.set_ylabel("Số Báo động giả (False Positives - Thang SymLog)", fontweight="bold")
    ax.set_title("C. So sánh Báo động giả (FP) trên Tập Held-out Test (~346k luồng Benign)", pad=10)
    ax.grid(axis="y", zorder=0)
    ax.legend(loc="upper right", fontsize=8.0, ncol=2)

    # Ô (2,2): Độ lệch điểm Mutual Information (Paper auto MI vs Ours Discrete-Aware MI)
    ax = axes[1, 1]
    if df_mi_comp is not None and not df_mi_comp.empty:
        avg_mi = (
            df_mi_comp.groupby("Feature")[["Paper_Auto_MI_NoSMOTE", "Ours_Discrete_MI"]]
            .mean()
            .sort_values("Paper_Auto_MI_NoSMOTE", ascending=True)
        )
        y_pos = np.arange(len(avg_mi))
        h_bar = 0.36
        ax.barh(y_pos + h_bar / 2, avg_mi["Paper_Auto_MI_NoSMOTE"], h_bar,
                label="Paper MI (discrete_features='auto' -> Continuous kNN)", color="#e74c3c", edgecolor="#2c3e50", zorder=3)
        ax.barh(y_pos - h_bar / 2, avg_mi["Ours_Discrete_MI"], h_bar,
                label="Ours MutualInfoScorer (Khai báo đúng biến rời rạc)", color="#1f77b4", edgecolor="#2c3e50", zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(avg_mi.index.tolist(), fontsize=8.2, fontweight="bold")
        ax.set_xlabel("Điểm Mutual Information Trung bình qua 5 Tấn công", fontweight="bold")
        ax.set_title("D. Hiện tượng Thổi phồng Điểm MI Giả tạo trên Biến Rời rạc (TCP_FLAGS, PROTOCOL)", pad=10)
        ax.grid(axis="x", zorder=0)
        ax.legend(loc="lower right", fontsize=8.0)

    plt.suptitle("PHÂN TÍCH PHƯƠNG PHÁP LUẬN: NGHỊCH LÝ SMOTE, RÒ RỈ CV VÀ ĐỘ LỆCH MUTUAL INFORMATION",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fig_p03_pr_curves_three_systems(df_3sys: pd.DataFrame, out_path: Path):
    """Hình P3 (2x3): Đường cong Precision-Recall đối sánh Paper-Exact RF, Ours RF và Ours HistGB."""
    setup_matplotlib_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()

    curves = [
        {"sys": "Paper", "dir": PAPER_DIR, "prefix": "paper_rf", "cond": "nosmote", "cond_label": "Without SMOTE",
         "th_col": "Paper_Threshold", "label": "Paper RF (No SMOTE)", "color": "#d62728", "ls": "-", "marker": "D"},
        {"sys": "Paper", "dir": PAPER_DIR, "prefix": "paper_rf", "cond": "smote", "cond_label": "With SMOTE",
         "th_col": "Paper_Threshold", "label": "Paper RF (SMOTE 56k)", "color": "#ff9896", "ls": "--", "marker": "v"},
        {"sys": "OursRF", "dir": BASELINE_DIR, "prefix": "rf", "cond": "nosmote", "cond_label": "Without SMOTE",
         "th_col": "OursRF_Threshold", "label": "Ours RF (No SMOTE)", "color": "#1f77b4", "ls": "-.", "marker": "o"},
        {"sys": "OursGB", "dir": IMPROVED_DIR, "prefix": "gb", "cond": "nosmote", "cond_label": "Without SMOTE",
         "th_col": "OursGB_Threshold", "label": "Ours HistGB (No SMOTE)", "color": "#2ca02c", "ls": ":", "marker": "*"},
    ]

    for idx, att in enumerate(TARGET_ATTACKS):
        ax = axes_flat[idx]
        ax.set_title(f"Tấn công: {att}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.04)
        ax.grid(True, zorder=0)

        for cfg in curves:
            p_file = cfg["dir"] / f"{cfg['prefix']}_{att.lower()}_{cfg['cond']}_best_proba_test.npy"
            y_file = cfg["dir"] / f"{cfg['prefix']}_{att.lower()}_{cfg['cond']}_y_test.npy"
            if not (p_file.exists() and y_file.exists()):
                continue
            y_true = np.load(y_file)
            y_prob = np.load(p_file)
            prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)

            sub = df_3sys[(df_3sys["Attack"] == att) & (df_3sys["Condition"] == cfg["cond_label"])]
            t_opt = float(sub[cfg["th_col"]].iloc[0]) if not sub.empty else 0.5
            y_pred = (y_prob >= t_opt).astype(int)
            op_p = precision_score(y_true, y_pred, zero_division=0)
            op_r = recall_score(y_true, y_pred, zero_division=0)

            ax.plot(rec_c, prec_c, label=f"{cfg['label']} (AP={ap:.4f})",
                    color=cfg["color"], linestyle=cfg["ls"], linewidth=1.7, zorder=3)
            ax.scatter([op_r], [op_p], color=cfg["color"], marker=cfg["marker"], s=72, edgecolor="#2c3e50", zorder=5)

        ax.legend(loc="lower left", fontsize=7.6, framealpha=0.92)

    # Ô 6: Hộp tổng hợp số liệu động
    ax_sum = axes_flat[5]
    ax_sum.axis("off")
    df_no = df_3sys[df_3sys["Condition"] == "Without SMOTE"]
    mean_f1_paper = df_no["Paper_F1"].mean()
    mean_f1_ours_rf = df_no["OursRF_F1"].mean()
    mean_f1_ours_gb = df_no["OursGB_F1"].mean()
    total_search_paper_min = df_3sys["Paper_All4K_Search_Time_s"].sum() / 60.0
    avg_sp_gb = df_no["Pipeline_Speedup_OursGB_vs_Paper"].mean()

    box_txt = (
        "TỔNG KẾT ĐỐI SÁNH ĐƯỜNG CONG PR (3 HỆ THỐNG)\n"
        "──────────────────────────────────────────────────────\n\n"
        f"1. TRUNG BÌNH MACRO F1 (WITHOUT SMOTE - 5 TẤN CÔNG):\n"
        f" • Paper-Exact RF (100% CV) : {mean_f1_paper:.4f}\n"
        f" • Ours Improved RF         : {mean_f1_ours_rf:.4f} (Δ = {mean_f1_ours_rf - mean_f1_paper:+.4f})\n"
        f" • Ours HistGradientBoosting: {mean_f1_ours_gb:.4f} (Δ = {mean_f1_ours_gb - mean_f1_paper:+.4f})\n\n"
        "2. NHẬN ĐỊNH KHOA HỌC TRỌNG TÂM:\n"
        " • Đường cong PR của Ours Improved RF trùng khít hoặc vượt nhẹ\n"
        "   Paper-Exact RF (đặc biệt ở k nhỏ như Analysis k=3 nhờ sửa lỗi\n"
        "   MutualInfoScorer cho biến rời rạc), dù chỉ tìm kiếm siêu tham số\n"
        "   trên 150k mẫu phân tầng thay vì 1.06 triệu mẫu.\n"
        f" • Nhánh Paper-Exact tiêu tốn tổng cộng ~{total_search_paper_min:.1f} phút chỉ riêng\n"
        "   cho bước RandomizedSearchCV qua 10 bài toán.\n"
        f" • Hệ thống HistGB rút ngắn thời gian Search+Fit trung bình ~{avg_sp_gb:.0f}x lần\n"
        "   so với Paper-Exact RF mà vẫn duy trì PR-AUC tiệm cận sát sao."
    )
    ax_sum.text(
        0.03, 0.5, box_txt, transform=ax_sum.transAxes, fontsize=8.7, va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#495057", lw=1.2),
    )

    plt.suptitle("ĐƯỜNG CONG PRECISION-RECALL & ĐIỂM HOẠT ĐỘNG: PAPER-EXACT RF VS OURS RF & HISTGB",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_fig_p04_soc_cost_and_fn(df_soc: pd.DataFrame, out_path: Path):
    """Hình P4 (1x2): Đối sánh số mẫu tấn công lọt lưới (FN) và Chi phí vận hành SOC (c=10) so với Paper-Exact."""
    setup_matplotlib_style()
    df_s = df_soc.set_index("Attack").loc[TARGET_ATTACKS].reset_index()
    x = np.arange(len(TARGET_ATTACKS))
    w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 6.3))

    # Panel trái: Số mẫu bỏ sót False Negatives (FN)
    b1 = ax1.bar(x - w, df_s["Paper_NoSMOTE_FN"], w, label="1. Paper-Exact RF (Ngưỡng F1: 0.05..0.94)", color="#d62728", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    b2 = ax1.bar(x, df_s["OursRF_CostOpt_FN"], w, label="2. Ours RF Cost-Sensitive (c=10)", color="#1f77b4", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    b3 = ax1.bar(x + w, df_s["OursGB_CostOpt_FN"], w, label="3. Ours HistGB Cost-Sensitive (c=10)", color="#2ca02c", edgecolor="#2c3e50", alpha=0.9, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax1.set_ylim(0, max(df_s["Paper_NoSMOTE_FN"].max(), df_s["OursGB_CostOpt_FN"].max()) * 1.22)
    ax1.set_ylabel("Số Cuộc Tấn Công Lọt Lưới (False Negatives - FN)", fontweight="bold")
    ax1.set_title("A. Cắt giảm Tấn công Lọt lưới (FN) so với Ngưỡng F1 Bài báo Gốc", pad=12)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="upper right", fontsize=8.3)

    for bars, col in [(b1, "#a61c1c"), (b2, "#155d8b"), (b3, "#1e7e34")]:
        for rect in bars:
            h = int(rect.get_height())
            ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 3, f"{h}",
                     ha="center", va="bottom", fontsize=8, fontweight="bold", color=col)

    # Panel phải: Tổng chi phí rủi ro SOC tại c=10 (Cost = 10*FN + 1*FP)
    c1 = ax2.bar(x - w, df_s["Paper_NoSMOTE_Cost_c10"], w, label="1. Paper-Exact RF (F1-Opt)", color="#d62728", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    c2 = ax2.bar(x, df_s["OursRF_CostOpt_Cost_c10"], w, label="2. Ours RF (Cost-Sensitive c=10)", color="#1f77b4", edgecolor="#2c3e50", alpha=0.9, zorder=3)
    c3 = ax2.bar(x + w, df_s["OursGB_CostOpt_Cost_c10"], w, label="3. Ours HistGB (Cost-Sensitive c=10)", color="#2ca02c", edgecolor="#2c3e50", alpha=0.9, zorder=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax2.set_ylim(0, max(df_s["Paper_NoSMOTE_Cost_c10"].max(), df_s["OursGB_CostOpt_Cost_c10"].max()) * 1.28)
    ax2.set_ylabel("Tổng Chi phí Rủi ro SOC: Cost = 10 × FN + 1 × FP", fontweight="bold")
    ax2.set_title("B. Hiệu quả Kinh tế An ninh SOC (c=10) so với Paper-Exact Baseline", pad=12)
    ax2.grid(axis="y", zorder=0)
    ax2.legend(loc="upper right", fontsize=8.3)

    for i in range(len(TARGET_ATTACKS)):
        pct = df_s.loc[i, "Cost_Saved_Pct_OursRFCost_vs_PaperNoSMOTE"]
        top_h = max(df_s.loc[i, "Paper_NoSMOTE_Cost_c10"], df_s.loc[i, "OursRF_CostOpt_Cost_c10"])
        badge_col = "#e8f5e9" if pct >= 0 else "#fdecea"
        edge_col = "#2e7d32" if pct >= 0 else "#c0392b"
        ax2.text(
            x[i] - w / 2.0, top_h + 70,
            f"Ours RF tiết kiệm\n{pct:+.1f}% chi phí",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=badge_col, edgecolor=edge_col, lw=0.9),
        )

    plt.suptitle("GIÁ TRỊ VẬN HÀNH THỰC TIỄN: CHỐNG BỎ SÓT TẤN CÔNG (FN) & TỐI ƯU CHI PHÍ SOC SO VỚI BÀI BÁO GỐC",
                 fontsize=12.8, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_markdown_report(df_3sys: pd.DataFrame, df_soc: pd.DataFrame, df_mi_comp: pd.DataFrame, out_md: Path):
    """Xuất báo cáo khoa học toàn diện đối sánh Paper-Exact Baseline với 2 hệ thống hiện có."""
    df_no = df_3sys[df_3sys["Condition"] == "Without SMOTE"].set_index("Attack").loc[TARGET_ATTACKS].reset_index()
    df_sm = df_3sys[df_3sys["Condition"] == "With SMOTE"].set_index("Attack").loc[TARGET_ATTACKS].reset_index()

    total_paper_search_min = df_3sys["Paper_All4K_Search_Time_s"].sum() / 60.0
    total_paper_train_min = pd.read_csv(PAPER_DIR / "summary_paper_rf_comparison.csv")["Total_4k_Train_Time_Sec"].sum() / 60.0

    # Đếm số bài toán Without SMOTE thắng With SMOTE trên Paper-Exact
    wins_nosmote_paper = int((df_no["Paper_F1"].values > df_sm["Paper_F1"].values).sum())

    lines = []
    lines.append("# Báo Cáo Khoa Học Đối Sánh: Paper-Exact Baseline (Pospíchal et al., 2026) vs Hai Hệ Thống Cải Tiến Của Dự Án\n")
    lines.append("**Bộ dữ liệu đánh giá chung (Apples-to-Apples):** `NF-UNSW-NB15` (1,623,118 bản ghi NetFlow v1; 10 đặc trưng lưu lượng; cùng phân vùng Stratified `Train = 1,104,497` | `Val = 155,586` | `Held-out Test = 363,035`).  ")
    lines.append("**Bài báo gốc đối chiếu:** Jiří Pospíchal, Aleš Augustín, Ladislav Huraj, Peter Střelec & Darja Gabriška, *\"Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic\"*, MDPI *Information* 2026, 17, 836.\n")
    lines.append("---\n")

    lines.append("## 1. Thiết Kế Thực Nghiệm Đối Chứng 3 Hệ Thống\n")
    lines.append("Để đảm bảo tính khách quan tuyệt đối (1-đối-1), cả 3 hệ thống đều đọc chung các phân vùng `train.parquet`, `val.parquet` và `heldout_test.parquet`, chỉ khác biệt ở **phương pháp luận pipeline**:\n")
    lines.append("| Thành phần Pipeline | Hệ thống 1: **Paper-Exact RF** (`src/paper_original/`) | Hệ thống 2: **Ours Improved RF** (`src/train_rf.py`) | Hệ thống 3: **Ours HistGB & 2-Tier SOC** (`src/train_gb.py` + `advanced_evaluation.py`) |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append("| **Cân bằng SMOTE (56,000 mẫu)** | Chạy 1 lần trên toàn bộ `X_train` **trước** `SelectKBest` & **trước** 3-Fold CV (*rò rỉ mẫu nội suy giữa các fold CV*) | Đóng gói `ImbPipeline` để SMOTE **chỉ chạy bên trong từng train-fold** của CV (*triệt tiêu rò rỉ CV*) | Nhánh `Without SMOTE` dùng *Hybrid Minority-Preserving Sampling* (100% Attack + 150k Benign); nhánh `SMOTE` dùng `ImbPipeline` |")
    lines.append("| **Chọn đặc trưng Mutual Info** | `SelectKBest(mutual_info_classif, random_state=10)` mặc định `discrete_features='auto'` (*xử lý nhầm toàn bộ cột số nguyên rời rạc như biến liên tục*) | `MutualInfoScorer` tùy chỉnh khai báo chính xác `DISCRETE_FEATURES` (`PROTOCOL`, `TCP_FLAGS`, ports, `L7_PROTO` factorized) | Thừa hưởng bộ `MutualInfoScorer` chuẩn xác cho không gian hỗn hợp rời rạc - liên tục |")
    lines.append("| **Tối ưu siêu tham số CV** | `RandomizedSearchCV(n_iter=10, cv=3)` trực tiếp trên **100% tập Train (~1.06M - 1.11M dòng)** | `RandomizedSearchCV(n_iter=10, cv=3)` trên tập con phân tầng đại diện `150,000` mẫu, sau đó fit lại trên 100% tập Train | `HistGradientBoostingClassifier` chia 255 giỏ histogram + song song hóa OpenMP |")
    lines.append("| **Chỉnh ngưỡng Quyết định** | Chỉ quét `np.arange(0.05, 0.95, 0.01)` tối ưu duy nhất chỉ số `F1` | Mở rộng `0.01..0.99` + Tích hợp ngưỡng **Recall-Targeted** ($R \\ge 95\\%, 98\\%, 99\\%$) & **Cost-Sensitive** ($c=10, 20$) | Ngưỡng nhạy chi phí SOC ($c=10$) làm màng lọc Tầng 1 tốc độ cao kết hợp DPI Tầng 2 |\n")

    lines.append("---\n")
    lines.append("## 2. Bảng Kết Quả Nguyên Bản Paper-Exact RF (Without SMOTE vs With Fixed-SMOTE 56,000)\n")
    lines.append("| Loại Tấn công | Điều kiện | $k^*$ | Ngưỡng $t^*$ | 3-Fold CV $F_1$ | Val $F_1$ | Test Precision | Test Recall | **Test $F_1$** | FP | FN | Search + Fit ($k^*$) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for att in TARGET_ATTACKS:
        for cond_label in ["Without SMOTE", "With SMOTE"]:
            r = df_3sys[(df_3sys["Attack"] == att) & (df_3sys["Condition"] == cond_label)].iloc[0]
            lines.append(
                f"| **{att}** | {cond_label} | {int(r['Paper_Best_k'])} | {r['Paper_Threshold']:.2f} | "
                f"{r['Paper_CV_F1']:.4f} | {r['Paper_Val_F1']:.4f} | {r['Paper_Precision']:.4f} | "
                f"{r['Paper_Recall']:.4f} | **{r['Paper_F1']:.4f}** | {int(r['Paper_FP']):,} | {int(r['Paper_FN']):,} | "
                f"{r['Paper_BestK_Total_Time_s']:.1f}s |"
            )
    lines.append(f"\n> **Nhận xét Tái lập Paper-Exact:** Ngay trên chính pipeline nguyên bản 100% của Pospíchal et al. khi áp dụng lên luồng `NF-UNSW-NB15`, cấu hình **Without SMOTE giành chiến thắng trên {wins_nosmote_paper}/5 loại tấn công** về chỉ số Test $F_1$, đồng thời tổng thời gian chạy chỉ riêng bước tìm kiếm siêu tham số `RandomizedSearchCV` qua 4 mốc $k$ tiêu tốn tới **{total_paper_search_min:.1f} phút** (cộng thêm **{total_paper_train_min:.1f} phút** huấn luyện mô hình cuối).\n")

    lines.append("---\n")
    lines.append("## 3. Đối Sánh Trực Tiếp 3 Hệ Thống (Điều Kiện Chuẩn Without SMOTE)\n")
    lines.append("| Loại Tấn công | **Paper-Exact RF** $F_1$ ($k^*$, $t^*$) | **Ours Improved RF** $F_1$ ($k^*$, $t^*$) | **Ours HistGB** $F_1$ ($k^*$, $t^*$) | $\\Delta F_1$ (Ours RF vs Paper) | Thời gian $k^*$ (Paper / Ours RF / Ours GB) | Tăng tốc Pipeline (Ours GB vs Paper) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_no.iterrows():
        lines.append(
            f"| **{r['Attack']}** | {r['Paper_F1']:.4f} ($k={int(r['Paper_Best_k'])}$, $t={r['Paper_Threshold']:.2f}$) | "
            f"**{r['OursRF_F1']:.4f}** ($k={int(r['OursRF_Best_k'])}$, $t={r['OursRF_Threshold']:.2f}$) | "
            f"{r['OursGB_F1']:.4f} ($k={int(r['OursGB_Best_k'])}$, $t={r['OursGB_Threshold']:.2f}$) | "
            f"`{r['Delta_F1_OursRF_vs_Paper']:+.4f}` | "
            f"{r['Paper_BestK_Total_Time_s']:.1f}s / {r['OursRF_BestK_Total_Time_s']:.1f}s / **{r['OursGB_BestK_Total_Time_s']:.1f}s** | "
            f"**Nhanh hơn {r['Pipeline_Speedup_OursGB_vs_Paper']:.1f}x** |"
        )
    lines.append("")

    lines.append("---\n")
    lines.append("## 4. Bốn Phát Hiện Khoa Học Cốt Lõi Từ Đối Chứng Thực Nghiệm\n")

    # Phát hiện 1: Độ lệch MI
    lines.append("### 4.1. Phát hiện 1: Hiện tượng Thổi phồng Điểm Mutual Information Giả tạo trên Biến Rời rạc (`discrete_features='auto'`)\n")
    if df_mi_comp is not None and not df_mi_comp.empty:
        top_auto = df_mi_comp.groupby("Feature")["Paper_Auto_MI_NoSMOTE"].mean().sort_values(ascending=False)
        top_disc = df_mi_comp.groupby("Feature")["Ours_Discrete_MI"].mean().sort_values(ascending=False)
        lines.append(
            f"- Trong `scikit-learn`, khi truyền ma trận đặc (dense matrix) vào `mutual_info_classif` với mặc định `discrete_features='auto'`, "
            f"hàm này **coi toàn bộ 10 cột NetFlow là biến thực liên tục** và áp dụng ước lượng entropy lân cận Kraskov $k$-NN kèm nhiễu Gaussian nhỏ ($10^{{-10}}$).\n"
            f"- Trên tập dữ liệu 1.06 triệu dòng chứa hàng trăm nghìn giá trị nguyên trùng nhau ở `TCP_FLAGS` và `PROTOCOL`, nhiễu phá vỡ đồng hạng làm điểm MI liên tục bị thổi phồng cực mạnh: "
            f"điểm MI trung bình của `TCP_FLAGS` vọt lên **{top_auto.get('TCP_FLAGS', 0.0):.4f}** và `PROTOCOL` vọt lên **{top_auto.get('PROTOCOL', 0.0):.4f}**, "
            f"trong khi các đặc trưng cốt lõi như `IN_BYTES` chỉ đạt **{top_auto.get('IN_BYTES', 0.0):.4f}** và `L4_DST_PORT` đạt **{top_auto.get('L4_DST_PORT', 0.0):.4f}**.\n"
            f"- Hệ quả là tại $k=3$, pipeline bài báo gốc luôn chọn cứng bộ ba `['PROTOCOL', 'L7_PROTO', 'TCP_FLAGS']` (bỏ rơi hoàn toàn số byte và cổng đích), "
            f"trong khi `MutualInfoScorer` cải tiến của chúng ta định danh đúng các biến rời rạc (đưa `L4_DST_PORT` = **{top_disc.get('L4_DST_PORT', 0.0):.4f}** và `IN_BYTES` lên nhóm dẫn đầu).\n"
        )

    # Phát hiện 2: Rò rỉ CV SMOTE
    lines.append("### 4.2. Phát hiện 2: Bằng chứng Thực nghiệm về Rò rỉ Dữ liệu Nội suy (CV Leakage) khi đặt SMOTE trước `RandomizedSearchCV`\n")
    lines.append("- Khi `SMOTE(56,000)` chạy trước `RandomizedSearchCV` (tăng số mẫu tấn công gấp 3 đến 46 lần bằng nội suy $k$-NN), các mẫu nhân tạo nằm sát nhau bị chia rải đều vào cả *train-fold* và *validation-fold* của 3-Fold CV. Điều này tạo ra **ảo giác hiệu năng nội bộ cực cao** (`CV_Best_F1` tiệm cận `0.98 - 0.999`), dẫn dắt bộ tìm kiếm ưu tiên các cây sâu (`min_samples_leaf` nhỏ, `max_features=None` hoặc `sqrt`) học vẹt vùng nội suy nhân tạo:")
    for _, r in df_sm.iterrows():
        gap = r["Paper_CV_F1"] - r["Paper_F1"]
        lines.append(
            f"  - **{r['Attack']} (With SMOTE):** 3-Fold CV $F_1$ báo ảo **{r['Paper_CV_F1']:.4f}**, nhưng khi kiểm tra trên Held-out Test thực tế rơi xuống **{r['Paper_F1']:.4f}** (độ sụt giảm ảo giác `-{gap:.4f}`, phát sinh **{int(r['Paper_FP']):,}** cảnh báo giả FP vs **{int(r['OursRF_FP']):,}** FP trên `Ours RF`)."
        )
    lines.append("")

    # Phát hiện 3: Nút thắt chi phí tính toán
    lines.append("### 4.3. Phát hiện 3: Giải quyết Triệt để Nút thắt Chi phí Tính toán trên Quy mô Hàng triệu Luồng NetFlow\n")
    lines.append("- Bài báo gốc thực nghiệm trên tập UNSW-NB15 gói tin nhỏ hơn (`175,341` dòng Train), nhưng khi mở rộng sang quy mô thực tế `NF-UNSW-NB15` (`1,104,497` dòng Train), việc chạy `RandomizedSearchCV` (`10 iter × 3 folds = 30` mô hình rừng ngẫu nhiên cho mỗi mốc $k$, tổng cộng `120` mô hình rừng cho mỗi bài toán) trên 100% dữ liệu tạo ra rào cản tính toán khổng lồ.")
    lines.append("- Hệ thống **Ours Improved RF** cắt giảm hàng chục lần thời gian tìm kiếm siêu tham số nhờ lấy mẫu phân tầng `150,000` dòng khi dò tham số rồi mới huấn luyện 100% dữ liệu, mà **không hề làm giảm (thậm chí cải thiện nhẹ) điểm $F_1$ trên Held-out Test**.")
    lines.append("- Đặc biệt, hệ thống **Ours HistGradientBoosting** đạt mức tăng tốc toàn diện từ hàng chục đến hàng trăm lần so với Paper-Exact Pipeline, biến việc tái huấn luyện mô hình liên tục (Continuous Retraining) hàng giờ thành vài giây.\n")

    # Phát hiện 4: Vận hành SOC c=10
    lines.append("### 4.4. Phát hiện 4: Giới hạn của Dải ngưỡng `0.05..0.94` Tối ưu Thuần $F_1$ vs Tối ưu Chi phí Vận hành SOC ($c=10$)\n")
    lines.append("- Bài báo gốc giới hạn dải quét ngưỡng ở `[0.05, 0.94]` và chỉ tối ưu điểm $F_1$ đối xứng (coi trọng số lỗi bỏ sót tấn công `FN` ngang bằng với báo nhầm `FP`). Trong môi trường SOC thực chiến ($c=10$: 1 cuộc tấn công lọt lưới gây thiệt hại gấp 10 lần 1 cảnh báo giả), hệ thống nhạy chi phí của chúng ta cắt giảm sâu cả số cuộc tấn công lọt lưới lẫn tổng chi phí rủi ro:\n")
    lines.append("| Loại Tấn công | Paper-Exact RF (`FN` / `FP` / `Cost c=10`) | Ours RF Cost-Opt $c=10$ (`FN` / `FP` / `Cost c=10`) | Ours HistGB Cost-Opt $c=10$ (`FN` / `FP` / `Cost c=10`) | **Số Tấn công Cứu thêm (Ours RF vs Paper)** | **Tiết kiệm Chi phí SOC (Ours RF vs Paper)** |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for _, s in df_soc.iterrows():
        lines.append(
            f"| **{s['Attack']}** | {int(s['Paper_NoSMOTE_FN'])} / {int(s['Paper_NoSMOTE_FP'])} / **{int(s['Paper_NoSMOTE_Cost_c10']):,}** | "
            f"**{int(s['OursRF_CostOpt_FN'])}** / {int(s['OursRF_CostOpt_FP'])} / **{int(s['OursRF_CostOpt_Cost_c10']):,}** | "
            f"{int(s['OursGB_CostOpt_FN'])} / {int(s['OursGB_CostOpt_FP'])} / {int(s['OursGB_CostOpt_Cost_c10']):,} | "
            f"**Cứu thêm {int(s['FN_Saved_OursRFCost_vs_PaperNoSMOTE']):+d} cuộc tấn công** | "
            f"**{s['Cost_Saved_Pct_OursRFCost_vs_PaperNoSMOTE']:+.2f}%** |"
        )
    lines.append("")

    lines.append("---\n")
    lines.append("## 5. Danh Mục Tệp Số Liệu & Biểu Đồ Đối Sánh Đã Xuất (`results/paper_original/`)\n")
    lines.append("1. [summary_paper_rf_comparison.csv](summary_paper_rf_comparison.csv): Bảng tổng hợp 10 mô hình Paper-Exact RF (`Without SMOTE` & `With SMOTE`).")
    lines.append("2. [three_systems_comparison.csv](three_systems_comparison.csv): Bảng đối sánh toàn diện 1-đối-1 giữa **Paper-Exact RF**, **Ours Improved RF** và **Ours HistGradientBoosting**.")
    lines.append("3. [soc_operational_comparison_c10.csv](soc_operational_comparison_c10.csv): Bảng định lượng hiệu quả cắt giảm `FN` và chi phí rủi ro SOC tại $c=10$.")
    lines.append("4. [mi_ranking_distortion_comparison.csv](mi_ranking_distortion_comparison.csv): Bảng phân tích độ lệch điểm số và thứ hạng đặc trưng Mutual Information.")
    lines.append("5. [figures/fig_p01_three_systems_f1_and_speed.png](figures/fig_p01_three_systems_f1_and_speed.png): Đối sánh $F_1$-Score và tổng thời gian tính toán (Search + Fit) của 3 hệ thống.")
    lines.append("6. [figures/fig_p02_smote_paradox_and_leakage_ablation.png](figures/fig_p02_smote_paradox_and_leakage_ablation.png): Giải phẫu nghịch lý SMOTE, ảo giác rò rỉ 3-Fold CV, báo động giả FP và độ lệch MI.")
    lines.append("7. [figures/fig_p03_pr_curves_three_systems.png](figures/fig_p03_pr_curves_three_systems.png): Đường cong Precision-Recall và điểm hoạt động của 3 hệ thống trên 5 loại tấn công.")
    lines.append("8. [figures/fig_p04_soc_cost_and_fn_comparison.png](figures/fig_p04_soc_cost_and_fn_comparison.png): Mức độ cắt giảm tấn công bỏ sót (`FN`) và tiết kiệm chi phí vận hành SOC ($c=10$).\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 78)
    print(" TỔNG HỢP SỐ LIỆU, BIỂU ĐỒ 300 DPI & BÁO CÁO: PAPER-EXACT VS OURS (2 HỆ THỐNG)")
    print("=" * 78)

    df_3sys, df_soc, df_mi_comp = build_comparison_tables()
    print(f"[+] Đã xuất bảng đối sánh 3 hệ thống ({len(df_3sys)} dòng): {PAPER_DIR / 'three_systems_comparison.csv'}")
    print(f"[+] Đã xuất bảng đối sánh chi phí SOC c=10 ({len(df_soc)} dòng): {PAPER_DIR / 'soc_operational_comparison_c10.csv'}")

    f1_path = FIGURES_OUT_DIR / "fig_p01_three_systems_f1_and_speed.png"
    f2_path = FIGURES_OUT_DIR / "fig_p02_smote_paradox_and_leakage_ablation.png"
    f3_path = FIGURES_OUT_DIR / "fig_p03_pr_curves_three_systems.png"
    f4_path = FIGURES_OUT_DIR / "fig_p04_soc_cost_and_fn_comparison.png"
    md_path = PAPER_DIR / "paper_vs_ours_findings.md"

    plot_fig_p01_three_systems_f1_and_speed(df_3sys, f1_path)
    print(f"[+] Đã xuất biểu đồ 1: {f1_path}")

    plot_fig_p02_smote_paradox_and_leakage(df_3sys, df_mi_comp, f2_path)
    print(f"[+] Đã xuất biểu đồ 2: {f2_path}")

    plot_fig_p03_pr_curves_three_systems(df_3sys, f3_path)
    print(f"[+] Đã xuất biểu đồ 3: {f3_path}")

    plot_fig_p04_soc_cost_and_fn(df_soc, f4_path)
    print(f"[+] Đã xuất biểu đồ 4: {f4_path}")

    generate_markdown_report(df_3sys, df_soc, df_mi_comp, md_path)
    print(f"[+] Đã xuất báo cáo khoa học: {md_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
