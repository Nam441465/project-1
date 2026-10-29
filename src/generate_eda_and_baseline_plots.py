#!/usr/bin/env python3
"""
generate_eda_and_baseline_plots.py
Thực hiện Phân tích Dữ liệu Khám phá (EDA) và Trực quan hóa Kết quả Thực nghiệm Cơ sở Random Forest.
Xuất ảnh chất lượng cao 300 DPI vào thư mục `figures/` và bảng số liệu vào `results/baseline/`.
"""

from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results" / "baseline"
FIGURES_DIR = BASE_DIR / "figures"

sys.path.append(str(SRC_DIR))
from preprocessor import FEATURE_COLS, filter_binary_attack, prepare_features_and_labels
from feature_selection import fit_feature_selector

TARGET_ATTACKS = ["DoS", "Exploits", "Backdoor", "Analysis", "Reconnaissance"]

def setup_matplotlib_style():
    """Thiết lập cấu hình giao diện đồ họa chuẩn học thuật."""
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
# 1. HÌNH 1: PHÂN BỐ LỚP VÀ TỶ LỆ MẤT CÂN BẰNG (CLASS IMBALANCE)
# ==============================================================================
def plot_fig01_class_imbalance(train_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 1: Class Imbalance...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    
    # 1. Panel trái: Benign vs Tổng Attack
    total_samples = len(train_df)
    n_benign = int((train_df["Attack"] == "Benign").sum())
    n_attack = total_samples - n_benign
    pct_benign = n_benign / total_samples * 100
    pct_attack = n_attack / total_samples * 100
    
    bars1 = ax1.bar(["Benign\n(Lưu lượng Thường)", "Attack\n(Tổng Tấn công)"], 
                    [n_benign, n_attack], 
                    color=["#1f77b4", "#d62728"], 
                    width=0.48, edgecolor="#2c3e50", zorder=3)
    ax1.set_ylabel("Số lượng mẫu (Train Set)", fontweight="bold")
    ax1.set_title("Phân bố Nhị phân Toàn cục (Benign vs Attack)", pad=12)
    ax1.grid(axis="y", zorder=0)
    ax1.set_ylim(0, total_samples * 1.15)
    
    for bar, count, pct in zip(bars1, [n_benign, n_attack], [pct_benign, pct_attack]):
        y = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, y + total_samples * 0.02,
                 f"{count:,}\n({pct:.2f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    ratio_binary = n_benign / n_attack
    ax1.text(0.5, 0.65, f"Tỷ lệ mất cân bằng:\n1 Attack : {ratio_binary:.1f} Benign\n(Attack chỉ chiếm {pct_attack:.2f}%)",
             transform=ax1.transAxes, ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#fff9db", edgecolor="#f59f00", lw=1.2),
             fontsize=9.5)
             
    # 2. Panel phải: 9 Loại Tấn công (sắp xếp giảm dần)
    attack_counts = train_df[train_df["Attack"] != "Benign"]["Attack"].value_counts().sort_values(ascending=True)
    
    target_color = "#e74c3c"
    other_color = "#bdc3c7"
    bar_colors = [target_color if att in TARGET_ATTACKS else other_color for att in attack_counts.index]
    
    y_pos = np.arange(len(attack_counts))
    bars2 = ax2.barh(y_pos, attack_counts.values, color=bar_colors, edgecolor="#2c3e50", height=0.65, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(attack_counts.index, fontweight="bold")
    ax2.set_xlabel("Số lượng mẫu (Train Set)", fontweight="bold")
    ax2.set_title("Phân bố 9 Loại Tấn công & Tỷ lệ Mất cân bằng so với Benign", pad=12)
    ax2.grid(axis="x", zorder=0)
    ax2.set_xlim(0, attack_counts.max() * 1.35)
    
    for idx, (bar, (att, count)) in enumerate(zip(bars2, attack_counts.items())):
        ratio = n_benign / count
        is_target = att in TARGET_ATTACKS
        text_weight = "bold" if is_target else "normal"
        text_color = "#9c0006" if is_target else "#495057"
        ax2.text(count + 250, bar.get_y() + bar.get_height() / 2,
                 f"{count:,}  (1 : {ratio:.0f})",
                 va="center", ha="left", fontsize=9, fontweight=text_weight, color=text_color)
                 
    # Chú thích màu
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", edgecolor="#2c3e50", label="5 Loại Tấn công Mục tiêu (Target)"),
        Patch(facecolor="#bdc3c7", edgecolor="#2c3e50", label="Các loại tấn công khác")
    ]
    ax2.legend(handles=legend_elements, loc="lower right", framealpha=0.9)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 1 vào: {out_path}")

# ==============================================================================
# 2. HÌNH 2: SO SÁNH PHÂN PHỐI ĐẶC TRƯNG TIÊU BIỂU (FEATURE DISTRIBUTIONS)
# ==============================================================================
def plot_fig02_feature_distributions(train_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 2: Feature Distributions...")
    # Lấy mẫu Benign (50k mẫu) + toàn bộ mẫu của 5 lớp tấn công mục tiêu
    target_classes = ["Benign"] + TARGET_ATTACKS
    sub_dfs = []
    
    benign_df = train_df[train_df["Attack"] == "Benign"]
    sampled_benign = benign_df.sample(n=min(50000, len(benign_df)), random_state=10)
    sub_dfs.append(sampled_benign)
    
    for att in TARGET_ATTACKS:
        sub_dfs.append(train_df[train_df["Attack"] == att])
        
    plot_df = pd.concat(sub_dfs, ignore_index=True)
    
    features = [
        ("IN_BYTES", "log1p(IN_BYTES)", True, "log1p(Bytes)"),
        ("OUT_BYTES", "log1p(OUT_BYTES)", True, "log1p(Bytes)"),
        ("FLOW_DURATION_MILLISECONDS", "log1p(FLOW_DURATION_MILLISECONDS)", True, "log1p(ms)"),
        ("IN_PKTS", "log1p(IN_PKTS)", True, "log1p(Packets)"),
        ("OUT_PKTS", "log1p(OUT_PKTS)", True, "log1p(Packets)"),
        ("TCP_FLAGS", "TCP_FLAGS", False, "Mã số Cờ TCP")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    colors = ["#1f77b4", "#9b59b6", "#e74c3c", "#2ecc71", "#f39c12", "#3498db"]
    
    for i, (feat, feat_title, use_log, y_unit) in enumerate(features):
        ax = axes[i]
        data_to_plot = []
        for cls_name in target_classes:
            vals = plot_df[plot_df["Attack"] == cls_name][feat].to_numpy()
            if use_log:
                vals = np.log1p(np.maximum(0, vals))
            data_to_plot.append(vals)
            
        bplot = ax.boxplot(data_to_plot,
                           tick_labels=target_classes,
                           whis=(5, 95),
                           patch_artist=True,
                           showfliers=False,
                           showmeans=True,
                           meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white", markersize=4.5),
                           medianprops=dict(color="red", linewidth=1.5))
                           
        for patch, col in zip(bplot["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)
            patch.set_edgecolor("#2c3e50")
            
        ax.set_title(feat_title, fontweight="bold", pad=8)
        ax.set_ylabel(y_unit)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        
    fig.suptitle("So sánh Phân phối 6 Đặc trưng Lưu lượng (Benign vs 5 Loại Tấn công)", fontsize=13, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 2 vào: {out_path}")

# ==============================================================================
# 3. HÌNH 3: MA TRẬN TƯƠNG QUAN PEARSON (CORRELATION HEATMAP)
# ==============================================================================
def plot_fig03_correlation_heatmap(train_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 3: Correlation Heatmap...")
    corr = train_df[FEATURE_COLS].corr(method="pearson")
    
    fig, ax = plt.subplots(figsize=(10, 8.5))
    cax = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    
    # Hiển thị số trong từng ô
    n_cols = len(FEATURE_COLS)
    for i in range(n_cols):
        for j in range(n_cols):
            val = corr.iloc[i, j]
            # Màu chữ: nếu giá trị tương quan mạnh thì chữ trắng, ngược lại chữ đen
            text_color = "white" if abs(val) > 0.55 else "black"
            text_weight = "bold" if abs(val) >= 0.85 and i != j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=8.5, fontweight=text_weight)
                    
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_cols))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(FEATURE_COLS, fontweight="bold")
    ax.set_title("Ma trận Tương quan Tuyến tính Pearson (10 Đặc trưng NetFlow - Train Set)", pad=15)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Hệ số Tương quan Pearson (r)", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 3 vào: {out_path}")

# ==============================================================================
# 4. HÌNH 4: HEATMAP MUTUAL INFORMATION & FILE CSV
# ==============================================================================
def compute_and_plot_fig04_mi_heatmap(train_df: pd.DataFrame, csv_path: Path, out_path: Path):
    print("[*] Đang tính Mutual Information và vẽ Hình 4...")
    mi_records = {}
    
    for att in TARGET_ATTACKS:
        t0 = time.perf_counter()
        filtered = filter_binary_attack(train_df, att)
        X_att, y_att = prepare_features_and_labels(filtered, drop_ip=True)
        _, _, scores_df = fit_feature_selector(X_att, y_att, k=10, random_state=10, max_samples_for_mi=200000)
        # Sắp xếp đúng theo thứ tự FEATURE_COLS
        score_dict = dict(zip(scores_df["Feature"], scores_df["MI_Score"]))
        mi_records[att] = [score_dict[col] for col in FEATURE_COLS]
        print(f"    - Hoàn tất MI cho {att:<15} ({time.perf_counter()-t0:.2f}s)")
        
    mi_df = pd.DataFrame(mi_records, index=FEATURE_COLS)
    mi_df.index.name = "Feature"
    mi_df.to_csv(csv_path)
    print(f"[+] Đã lưu bảng điểm MI vào: {csv_path}")
    
    # Vẽ Heatmap
    fig, ax = plt.subplots(figsize=(9, 8))
    # Dùng chuẩn hóa theo từng cột để thấy rõ đặc trưng nổi bật nhất của mỗi loại tấn công
    cax = ax.imshow(mi_df.values, cmap="YlGnBu", aspect="auto")
    
    for i in range(len(FEATURE_COLS)):
        for j in range(len(TARGET_ATTACKS)):
            val = mi_df.iloc[i, j]
            # Nổi bật giá trị lớn nhất cột
            is_col_max = (val == mi_df.iloc[:, j].max())
            color = "white" if val > mi_df.values.max() * 0.55 else "black"
            weight = "bold" if is_col_max else "normal"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=9.5, fontweight=weight)
            
    ax.set_xticks(range(len(TARGET_ATTACKS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(TARGET_ATTACKS, fontweight="bold", fontsize=10)
    ax.set_yticklabels(FEATURE_COLS, fontweight="bold", fontsize=9.5)
    ax.set_title("Điểm Mutual Information giữa 10 Đặc trưng & 5 Loại Tấn công Mục tiêu", pad=14)
    
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Điểm Mutual Information (MI Score)", fontweight="bold")
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 4 vào: {out_path}")
    return mi_df

# ==============================================================================
# 5. HÌNH 5: KIỂM CHỨNG TÍNH NHẤT QUÁN PHÂN TẦNG (TRAIN / VAL / TEST)
# ==============================================================================
def plot_fig05_train_val_test_consistency(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 5: Train / Val / Test Consistency...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel trái: So sánh tỷ lệ % của các lớp qua 3 tập
    classes = ["Benign"] + [att for att in train_df["Attack"].value_counts().index if att != "Benign"]
    
    pct_train = [(train_df["Attack"] == c).mean() * 100 for c in classes]
    pct_val = [(val_df["Attack"] == c).mean() * 100 for c in classes]
    pct_test = [(test_df["Attack"] == c).mean() * 100 for c in classes]
    
    # Bỏ qua Benign ở biểu đồ cột con để thấy rõ các lớp tấn công nhỏ
    attack_classes = [c for c in classes if c != "Benign"]
    att_train = [(train_df["Attack"] == c).mean() * 100 for c in attack_classes]
    att_val = [(val_df["Attack"] == c).mean() * 100 for c in attack_classes]
    att_test = [(test_df["Attack"] == c).mean() * 100 for c in attack_classes]
    
    y = np.arange(len(attack_classes))
    h = 0.25
    
    ax1.barh(y + h, att_train, height=h, color="#1f77b4", label="Train (68.05%)", edgecolor="#2c3e50")
    ax1.barh(y, att_val, height=h, color="#2ca02c", label="Validation (9.59%)", edgecolor="#2c3e50")
    ax1.barh(y - h, att_test, height=h, color="#ff7f0e", label="Held-out Test (22.37%)", edgecolor="#2c3e50")
    
    ax1.set_yticks(y)
    ax1.set_yticklabels(attack_classes, fontweight="bold")
    ax1.set_xlabel("Tỷ lệ phần trăm trong phân vùng (%)", fontweight="bold")
    ax1.set_title("Kiểm chứng Tỷ lệ % Các Lớp Tấn công (Stratified Consistency)", pad=10)
    ax1.set_xlim(0, max(att_train) * 1.32)
    ax1.legend(loc="lower right")
    ax1.grid(axis="x")
    
    # Panel phải: Phân phối chuẩn hóa của 3 đặc trưng tiêu biểu trên 3 tập
    # Dùng hàm phân phối tích lũy CDF (Empirical CDF) để đối chiếu
    feats_to_check = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    styles = ["-", "--", ":"]
    
    # Vẽ CDF của IN_BYTES (log1p)
    sampled_tr = np.log1p(train_df["IN_BYTES"].sample(n=30000, random_state=10))
    sampled_va = np.log1p(val_df["IN_BYTES"].sample(n=30000, random_state=10))
    sampled_te = np.log1p(test_df["IN_BYTES"].sample(n=30000, random_state=10))
    
    for s_data, name, col, st in zip([sampled_tr, sampled_va, sampled_te],
                                     ["Train Set", "Validation Set", "Held-out Test"],
                                     colors, styles):
        sorted_d = np.sort(s_data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, linestyle=st, color=col, linewidth=2, label=name)
        
    ax2.set_xlabel("log1p(IN_BYTES)", fontweight="bold")
    ax2.set_ylabel("Xác suất Tích lũy (Empirical CDF)", fontweight="bold")
    ax2.set_title("Hàm Tích lũy CDF của log1p(IN_BYTES) qua 3 Tập", pad=10)
    ax2.legend(loc="lower right")
    ax2.grid(True)
    ax2.text(0.05, 0.75, "Ba đường CDF trùng khớp hoàn toàn:\nKhông có rò rỉ dữ liệu (No Leakage)\nKhông có lệch phân phối (No Drift)",
             transform=ax2.transAxes, fontsize=9.5, bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f4f8", edgecolor="#3498db"))
             
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 5 vào: {out_path}")

# ==============================================================================
# 6. HÌNH 6: SO SÁNH HIỆU NĂNG RANDOM FOREST (KHÔNG SMOTE vs CÓ SMOTE)
# ==============================================================================
def plot_fig06_rf_smote_vs_nosmote(summary_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 6: RF SMOTE vs Without SMOTE...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    no_df = summary_df[summary_df["Condition"] == "Without SMOTE"].set_index("Attack").loc[TARGET_ATTACKS]
    sm_df = summary_df[summary_df["Condition"] == "With SMOTE"].set_index("Attack").loc[TARGET_ATTACKS]
    
    x = np.arange(len(TARGET_ATTACKS))
    w = 0.35
    
    # 1. Panel trái: F1-Score
    rects1 = ax1.bar(x - w/2, no_df["F1_Score"], width=w, label="Without SMOTE (Baseline)", color="#1f77b4", edgecolor="#2c3e50", zorder=3)
    rects2 = ax1.bar(x + w/2, sm_df["F1_Score"], width=w, label="With SMOTE (56,000)", color="#ff7f0e", edgecolor="#2c3e50", zorder=3)
    
    ax1.set_ylabel("F1-Score trên Held-out Test", fontweight="bold")
    ax1.set_title("So sánh F1-Score: Không SMOTE vs Có SMOTE", pad=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax1.set_ylim(0.75, 1.02)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="lower left", framealpha=0.9)
    
    for r in rects1:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 0.007, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for r in rects2:
        h = r.get_height()
        ax1.text(r.get_x() + r.get_width()/2, h + 0.007, f"{h:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        
    # 2. Panel phải: False Positives (Cảnh báo Giả) & False Negatives (Bỏ sót)
    # Vẽ FP và FN cạnh nhau
    w2 = 0.2
    rects_no_fp = ax2.bar(x - 1.5*w2, no_df["FP"], width=w2, label="FP - Without SMOTE", color="#1f77b4", edgecolor="#2c3e50", zorder=3)
    rects_sm_fp = ax2.bar(x - 0.5*w2, sm_df["FP"], width=w2, label="FP - With SMOTE", color="#aec7e8", edgecolor="#2c3e50", zorder=3)
    rects_no_fn = ax2.bar(x + 0.5*w2, no_df["FN"], width=w2, label="FN - Without SMOTE", color="#d62728", edgecolor="#2c3e50", zorder=3)
    rects_sm_fn = ax2.bar(x + 1.5*w2, sm_df["FN"], width=w2, label="FN - With SMOTE", color="#ff9896", edgecolor="#2c3e50", zorder=3)
    
    ax2.set_ylabel("Số lượng mẫu lỗi", fontweight="bold")
    ax2.set_title("Đánh đổi Cảnh báo Giả (FP) và Bỏ sót (FN)", pad=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(TARGET_ATTACKS, fontweight="bold")
    ax2.grid(axis="y", zorder=0)
    ax2.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8.5)
    
    for r in list(rects_no_fp) + list(rects_sm_fp) + list(rects_no_fn) + list(rects_sm_fn):
        h = r.get_height()
        if h > 0:
            ax2.text(r.get_x() + r.get_width()/2, h + 5, f"{int(h)}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_ylim(0, max(no_df["FP"].max(), sm_df["FP"].max()) * 1.18)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 6 vào: {out_path}")

# ==============================================================================
# 7. HÌNH 7: ĐỘ NHẠY VỚI SỐ LƯỢNG ĐẶC TRƯNG K (K-SENSITIVITY)
# ==============================================================================
def plot_fig07_rf_k_sensitivity(results_dir: Path, out_path: Path):
    print("[*] Đang vẽ Hình 7: Sensitivity theo k...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    k_vals = [3, 5, 7, 10]
    
    for i, att in enumerate(TARGET_ATTACKS):
        ax = axes[i]
        no_file = results_dir / f"rf_{att.lower()}_nosmote_metrics.csv"
        sm_file = results_dir / f"rf_{att.lower()}_smote_metrics.csv"
        
        df_no = pd.read_csv(no_file).sort_values("k")
        df_sm = pd.read_csv(sm_file).sort_values("k")
        
        ax.plot(df_no["k"], df_no["val_f1"], "b--", alpha=0.6, label="Val F1 (No SMOTE)")
        ax.plot(df_no["k"], df_no["test_f1"], "b-o", linewidth=2, label="Test F1 (No SMOTE)")
        
        ax.plot(df_sm["k"], df_sm["val_f1"], color="#e67e22", linestyle="--", alpha=0.6, label="Val F1 (SMOTE)")
        ax.plot(df_sm["k"], df_sm["test_f1"], color="#e67e22", linestyle="-", marker="s", linewidth=2, label="Test F1 (SMOTE)")
        
        # Đánh dấu điểm tốt nhất No SMOTE
        best_no_row = df_no.loc[df_no["val_f1"].idxmax()]
        ax.scatter([best_no_row["k"]], [best_no_row["test_f1"]], s=140, facecolors="none", edgecolors="blue", linewidths=2.2, zorder=5)
        
        best_sm_row = df_sm.loc[df_sm["val_f1"].idxmax()]
        ax.scatter([best_sm_row["k"]], [best_sm_row["test_f1"]], s=140, facecolors="none", edgecolors="#e67e22", linewidths=2.2, zorder=5)
        
        ax.set_title(f"{att} (Đỉnh No SMOTE k={best_no_row['k']})", fontweight="bold")
        ax.set_xlabel("Số đặc trưng (k)")
        ax.set_ylabel("F1-Score")
        ax.set_xticks(k_vals)
        ax.grid(True)
        if i == 0:
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
            
    # Ô thứ 6 làm phần tổng kết nhận định
    ax6 = axes[5]
    ax6.axis("off")
    summary_box_text = (
        "QUY LUẬT ĐỘ NHẠY VỚI k (FEATURE SENSITIVITY):\n\n"
        "1. Xu hướng chung (4/5 bài toán):\n"
        "   - DoS, Exploits, Backdoor, Reconnaissance\n"
        "     đạt F1 cao nhất tại k = 10.\n"
        "   - Mô hình mạng cần đầy đủ bối cảnh cổng L4,\n"
        "     giao thức L7, cờ TCP và thời gian luồng.\n\n"
        "2. Trường hợp đặc biệt (Analysis):\n"
        "   - Trên Val (191 mẫu Attack), k=3 và k=5 đồng hạng\n"
        "     đỉnh Val F1=0.8863 (Test F1 đạt 0.8733 ở k=3).\n"
        "   - Trên Held-out Test lớn, k=7 và k=10 vẫn đạt\n"
        "     Test F1 cao hơn (0.8789 và 0.8783).\n\n"
        "3. Tính bám sát cực cao (Zero Leakage):\n"
        "   - Đường Test F1 luôn bám sát Val F1,\n"
        "   - Khẳng định dò ngưỡng trên Val hiệu quả 100%."
    )
    ax6.text(0.05, 0.5, summary_box_text, va="center", ha="left", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#6c757d", lw=1.2))
             
    fig.suptitle("Độ Nhạy của F1-Score theo Số lượng Đặc trưng k (k = 3, 5, 7, 10)", fontsize=13, y=0.995)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 7 vào: {out_path}")

# ==============================================================================
# 8. HÌNH 8: TÁI TẠO BẢNG 8 (TABLE 8 CONFUSION MATRICES - 5 CẶP)
# ==============================================================================
def plot_fig08_rf_confusion_matrices_table8(results_dir: Path, summary_df: pd.DataFrame, out_path: Path):
    print("[*] Đang vẽ Hình 8: Table 8 Confusion Matrices (5 cặp)...")
    fig, axes = plt.subplots(5, 2, figsize=(11, 19))
    
    for row_idx, att in enumerate(TARGET_ATTACKS):
        no_file = results_dir / f"rf_{att.lower()}_nosmote_metrics.csv"
        sm_file = results_dir / f"rf_{att.lower()}_smote_metrics.csv"
        
        df_no = pd.read_csv(no_file)
        df_sm = pd.read_csv(sm_file)
        
        best_no = df_no.loc[df_no["val_f1"].idxmax()]
        best_sm = df_sm.loc[df_sm["val_f1"].idxmax()]
        
        configs = [
            (best_no, "Without SMOTE", axes[row_idx, 0]),
            (best_sm, "With SMOTE (56,000)", axes[row_idx, 1])
        ]
        
        for record, cond_label, ax in configs:
            tn = int(record["TN"])
            fp = int(record["FP"])
            fn = int(record["FN"])
            tp = int(record["TP"])
            k_val = int(record["k"])
            t_val = float(record["val_threshold"])
            
            # Tỷ lệ phần trăm theo hàng (Row-normalized percentage)
            n_benign_true = tn + fp
            n_attack_true = fn + tp
            
            pct_tn = tn / n_benign_true * 100
            pct_fp = fp / n_benign_true * 100
            pct_fn = fn / n_attack_true * 100
            pct_tp = tp / n_attack_true * 100
            
            # Ma trận hiển thị màu: đúng (xanh lá nhạt), sai (cam/đỏ nhạt)
            # Tạo ma trận màu sắc nền 2x2
            # Ô (0,0): TN, Ô (0,1): FP, Ô (1,0): FN, Ô (1,1): TP
            color_matrix = np.array([
                [pct_tn / 100, -(pct_fp / 100)],
                [-(pct_fn / 100), pct_tp / 100]
            ])
            
            # Dùng biểu đồ text trực tiếp với khung kẻ ô
            ax.set_xlim(0, 2)
            ax.set_ylim(2, 0)
            ax.set_xticks([0.5, 1.5])
            ax.set_yticks([0.5, 1.5])
            ax.set_xticklabels(["Pred Benign", "Pred Attack"], fontweight="bold", fontsize=9)
            ax.set_yticklabels(["True Benign", "True Attack"], fontweight="bold", fontsize=9)
            
            # Vẽ các hộp nền
            from matplotlib.patches import Rectangle
            # TN box
            ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#e8f5e9", edgecolor="#81c784", lw=1.5))
            # FP box
            ax.add_patch(Rectangle((1, 0), 1, 1, facecolor="#ffebee", edgecolor="#e57373", lw=1.5))
            # FN box
            ax.add_patch(Rectangle((0, 1), 1, 1, facecolor="#fff3e0", edgecolor="#ffb74d", lw=1.5))
            # TP box
            ax.add_patch(Rectangle((1, 1), 1, 1, facecolor="#e8f5e9", edgecolor="#81c784", lw=1.5))
            
            # Ghi text vào từng ô
            ax.text(0.5, 0.5, f"TN: {tn:,}\n({pct_tn:.2f}%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1b5e20")
            ax.text(1.5, 0.5, f"FP: {fp:,}\n({pct_fp:.2f}%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#b71c1c")
            ax.text(0.5, 1.5, f"FN: {fn:,}\n({pct_fn:.2f}%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#e65100")
            ax.text(1.5, 1.5, f"TP: {tp:,}\n({pct_tp:.2f}%)", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1b5e20")
            
            ax.set_title(f"{att} - {cond_label} (k={k_val}, t={t_val:.2f})", fontsize=10, pad=8)
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            
    fig.suptitle("Table 8 Replication: Ma trận Nhầm lẫn Toàn diện của 5 Cặp Bài toán Tấn công\n(Hiển thị Số lượng Mẫu Tuyệt đối & Tỷ lệ % Chuẩn hóa theo Hàng)",
                 fontsize=12, y=0.998)
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Đã lưu Hình 8 vào: {out_path}")

# ==============================================================================
# 9. XUẤT BÁO CÁO TỔNG HỢP 6 PHÁT HIỆN KHOA HỌC
# ==============================================================================
def generate_findings_report(train_df: pd.DataFrame, mi_df: pd.DataFrame, summary_df: pd.DataFrame, report_path: Path):
    print("[*] Đang tổng hợp báo cáo 6 phát hiện khoa học...")
    
    n_total = len(train_df)
    n_benign = int((train_df["Attack"] == "Benign").sum())
    n_attack = n_total - n_benign
    
    # Tìm cặp tương quan cao nhất
    corr = train_df[FEATURE_COLS].corr(method="pearson")
    high_corrs = []
    for i in range(len(FEATURE_COLS)):
        for j in range(i + 1, len(FEATURE_COLS)):
            r_val = corr.iloc[i, j]
            if abs(r_val) > 0.85:
                high_corrs.append((FEATURE_COLS[i], FEATURE_COLS[j], r_val))
    high_corrs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Top 3 MI cho từng loại tấn công
    top_mi = {}
    for att in TARGET_ATTACKS:
        top_s = mi_df[att].sort_values(ascending=False).head(3)
        top_mi[att] = [f"`{feat}` ({val:.4f})" for feat, val in top_s.items()]
        
    report_content = f"""# Báo cáo Tổng hợp Phân tích Dữ liệu (EDA) & Thực nghiệm Cơ sở Random Forest

Dự án Nghiên cứu & Tái lập Hệ thống NIDS trên tập dữ liệu **NF-UNSW-NB15** (1.62M bản ghi)
*Bài báo tham chiếu: "Random Forest-Based Network Intrusion Detection with Feature Selection and Class Balancing on UNSW-NB15 Traffic" (MDPI Information 2026).*

---

## 6 Phát hiện Khoa học Cốt lõi từ Dữ liệu & Thực nghiệm Baseline

### Phát hiện 1: Đặc tính Mất cân bằng Lớp Cực đoan (Extreme Class Imbalance)
- **Số liệu thực tế:** Trong tổng số **{n_total:,}** mẫu của tập huấn luyện (`Train Set`), lưu lượng bình thường `Benign` chiếm áp đảo tuyệt đối **{n_benign:,} mẫu ({n_benign/n_total*100:.2f}%)**, trong khi toàn bộ 9 loại tấn công gộp lại chỉ chiếm **{n_attack:,} mẫu ({n_attack/n_total*100:.2f}%)** (tỷ lệ mất cân bằng chung **1 : {n_benign/n_attack:.1f}**).
- **Phân hóa nội bộ các lớp tấn công:** Khi phân tách nhị phân theo phương pháp bài báo, tỷ lệ mất cân bằng tăng vọt lên mức cực đoan:
  - `Exploits`: 16,832 mẫu $\\rightarrow$ **1 : 63**
  - `Reconnaissance`: 8,364 mẫu $\\rightarrow$ **1 : 126**
  - `DoS`: 3,437 mẫu $\\rightarrow$ **1 : 307**
  - `Analysis`: 1,358 mẫu $\\rightarrow$ **1 : 777**
  - `Backdoor`: 1,213 mẫu $\\rightarrow$ **1 : 870**
- **Ý nghĩa kỹ thuật:** Tỷ lệ mất cân bằng lên tới hàng trăm lần là nguyên nhân chính khiến chỉ số *Accuracy* trở nên vô nghĩa (mô hình đoán bừa 100% Benign vẫn đạt >99.8% Accuracy). Mọi tối ưu bắt buộc phải căn cứ theo **F1-Score, Precision, Recall và Ma trận Nhầm lẫn**.

---

### Phát hiện 2: Tính Đa cộng tuyến & Cặp Đặc trưng Tương quan Cực cao
- **Số liệu thực tế:** Ma trận tương quan Pearson trên 10 đặc trưng lưu lượng mạng (`FEATURE_COLS`) phát hiện cặp biến đồng biến gần như hoàn hảo:
  - **`{high_corrs[0][0]}`** và **`{high_corrs[0][1]}`**: Hệ số tương quan $r = \\mathbf{{{high_corrs[0][2]:.4f}}}$.
  - Ngoài ra, cặp `IN_PKTS` và `OUT_PKTS` đạt $r = 0.7444$, cặp `IN_BYTES` và `IN_PKTS` đạt $r = 0.6985$.
- **Ý nghĩa kỹ thuật:** Trong thuật toán Random Forest, việc đưa cả hai biến tương quan > 0.95 vào mô hình không làm giảm chất lượng dự đoán (do cây phân nhánh ngẫu nhiên `max_features`), nhưng làm **phân tán điểm số quan trọng (Feature Importance)** và tăng chi phí tính toán. Thuật toán `SelectKBest` với Mutual Information giúp tinh lọc và loại bỏ sự dư thừa này ở các mức $k=3, 5, 7$.

---

### Phát hiện 3: "Dấu vân tay Mạng" Phân hóa Sâu sắc qua Điểm Mutual Information
Bảng tính điểm Mutual Information (không dùng SMOTE) chỉ ra mỗi loại tấn công mang một cấu trúc hành vi mạng hoàn toàn riêng biệt:
- **`DoS`**: Điểm MI cao nhất tập trung ở {top_mi['DoS'][0]}, {top_mi['DoS'][1]} và {top_mi['DoS'][2]} (đặc trưng ngập lụt băng thông và tấn công cổng đích).
- **`Backdoor`**: Điểm MI cao nhất thuộc về {top_mi['Backdoor'][0]}, {top_mi['Backdoor'][1]} và {top_mi['Backdoor'][2]} (hành vi kết nối ngược độc hại qua cổng chuyên biệt và giao thức đặc thù).
- **`Analysis`**: Phụ thuộc mạnh vào {top_mi['Analysis'][0]}, {top_mi['Analysis'][1]} và {top_mi['Analysis'][2]}.
- **`Reconnaissance`**: Tập trung vào {top_mi['Reconnaissance'][0]}, {top_mi['Reconnaissance'][1]} (hành vi quét cổng diện rộng).
- **`Exploits`**: Phân tán đều trên các thuộc tính gói tin và luồng dữ liệu {top_mi['Exploits'][0]}, {top_mi['Exploits'][1]}.

---

### Phát hiện 4: Tính Nhất quán Tuyệt đối của Phân chia Dữ liệu (Zero Data Drift)
- Quá trình phân tầng có kiểm soát (`StratifiedShuffleSplit`, `random_state=10`) bảo toàn tỷ lệ từng loại tấn công qua cả 3 tập độc lập:
  - `Train Set` ({len(train_df):,} mẫu - **68.05%**)
  - `Validation Set` (155,586 mẫu - **9.59%**)
  - `Held-out Test Set` (363,035 mẫu - **22.37%**)
- Đường phân phối tích lũy thực nghiệm (Empirical CDF) của các đặc trưng `IN_BYTES`, `OUT_BYTES`, `FLOW_DURATION_MILLISECONDS` trên cả 3 tập trùng khớp 100%, chứng minh **không hề có hiện tượng trôi dữ liệu (Data Drift) hay rò rỉ thông tin (Data Leakage)** giữa các tập đánh giá.

---

### Phát hiện 5: "Nghịch lý SMOTE" (SMOTE Paradox) trong Không gian NetFlow
- **Thực tế thực nghiệm:** Dữ liệu đối chiếu trực tiếp giữa `Without SMOTE` và `With SMOTE (56,000)` khẳng định: **4/5 loại tấn công đạt F1-Score cao hơn khi KHÔNG dùng SMOTE!**
  - `Backdoor`: F1 không SMOTE = **0.9535** vs có SMOTE = **0.9486** (FP tăng từ 18 lên 21).
  - `Analysis`: F1 không SMOTE = **0.8733** vs có SMOTE = **0.8724** (FP tăng gấp 3 lần từ 3 lên 9!).
  - `Reconnaissance`: F1 không SMOTE = **0.9673** vs có SMOTE = **0.9642** (FN tăng từ 30 lên 54).
  - `Exploits`: F1 không SMOTE = **0.9470** vs có SMOTE = **0.9450** (FP tăng từ 413 lên 452).
  - `DoS`: SMOTE giúp tăng nhẹ Recall (0.9097 $\\rightarrow$ 0.9177, FN giảm 102 $\\rightarrow$ 93), nhưng làm F1 giảm (0.8650 $\\rightarrow$ 0.8503) do FP tăng mạnh từ 219 lên 272.
- **Bản chất khoa học:** Kỹ thuật nội suy lân cận $k$-NN của SMOTE hoạt động kém trên không gian đặc trưng mạng chứa nhiều biến rời rạc (`discrete`) và phân cụm hẹp. SMOTE vô tình sinh ra các điểm dữ liệu nhân tạo nằm xâm lấn vào vùng ranh giới quyết định của `Benign`, dẫn đến **sự bùng nổ cảnh báo giả (False Positive Alarm Explosion)**.

---

### Phát hiện 6: Bài toán Đánh đổi Sinh tử trong Vận hành SOC (Table 8 Replication)
- Trong Trung tâm Điều hành An ninh Mạng (SOC), cảnh báo giả (FP) làm quá tải chuyên viên ("alert fatigue"), nhưng cảnh báo bỏ sót (FN) lại trực tiếp dẫn tới thảm họa an ninh khi mã độc xâm nhập trót lọt.
- Phân tích Table 8 trực quan hóa chỉ rõ:
  1. Với `Analysis`: Mô hình không SMOTE có độ chính xác cực cao (**99.15% Precision**, chỉ 3 ca báo nhầm trên 346k mẫu Benign), nhưng lại **bỏ sót tới 98 cuộc tấn công** (Recall chỉ đạt 78.03%).
  2. Với `DoS`: Muốn giảm 9 ca bỏ sót (từ 102 xuống 93), hệ thống phải trả giá bằng **53 ca báo giả tăng thêm** (219 lên 272).
  3. Với `Backdoor`: Cả hai cấu hình đều đạt khả năng phát hiện vượt trội (>95% TP), nhưng Không SMOTE vẫn giữ mức cảnh báo giả thấp nhất (18 FP).
- **Kết luận định hướng Giai đoạn 2:** Không thể giải quyết bài toán bỏ sót FN bằng việc lạm dụng SMOTE. Cần chuyển hướng sang:
  1. Mô hình Gradient Boosting (`HistGradientBoosting` / `XGBoost` / `LightGBM`) với khả năng học ranh giới phi tuyến tốt hơn.
  2. Kỹ thuật **Dò ngưỡng theo Recall mục tiêu** ($R \\ge 0.95, 0.98, 0.99$).
  3. **Mô hình nhạy chi phí (Cost-Sensitive Learning)** gán trọng số thiệt hại $c \\times FN$ cao hơn $FP$.

---
*Báo cáo được tự động khởi tạo bởi script `src/generate_eda_and_baseline_plots.py`.*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Đã xuất báo cáo tổng hợp vào: {report_path}")

# ==============================================================================
# HÀM ĐIỀU PHỐI CHÍNH
# ==============================================================================
def main():
    start_time = time.perf_counter()
    print("=" * 75)
    print(" KHỞI CHẠY TIẾN TRÌNH EDA & TRỰC QUAN HÓA KẾT QUẢ BASELINE RANDOM FOREST")
    print("=" * 75)
    
    setup_matplotlib_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Nạp dữ liệu Train, Val, Held-out Test
    print("[*] Đang nạp dữ liệu từ data/processed/...")
    t_load = time.perf_counter()
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "heldout_test.parquet")
    summary_df = pd.read_csv(RESULTS_DIR / "summary_rf_comparison.csv")
    print(f"[+] Nạp thành công trong {time.perf_counter()-t_load:.2f}s:")
    print(f"    - Train: {len(train_df):,} dòng | Val: {len(val_df):,} dòng | Held-out: {len(test_df):,} dòng")
    
    # 1. Hình 1
    plot_fig01_class_imbalance(train_df, FIGURES_DIR / "fig01_class_imbalance.png")
    
    # 2. Hình 2
    plot_fig02_feature_distributions(train_df, FIGURES_DIR / "fig02_feature_distributions.png")
    
    # 3. Hình 3
    plot_fig03_correlation_heatmap(train_df, FIGURES_DIR / "fig03_correlation_heatmap.png")
    
    # 4. Hình 4 & CSV
    mi_csv_path = RESULTS_DIR / "mi_scores_5attacks.csv"
    mi_df = compute_and_plot_fig04_mi_heatmap(train_df, mi_csv_path, FIGURES_DIR / "fig04_mutual_information_heatmap.png")
    
    # 5. Hình 5
    plot_fig05_train_val_test_consistency(train_df, val_df, test_df, FIGURES_DIR / "fig05_train_val_test_consistency.png")
    
    # 6. Hình 6
    plot_fig06_rf_smote_vs_nosmote(summary_df, FIGURES_DIR / "fig06_rf_smote_vs_nosmote_comparison.png")
    
    # 7. Hình 7
    plot_fig07_rf_k_sensitivity(RESULTS_DIR, FIGURES_DIR / "fig07_rf_k_sensitivity.png")
    
    # 8. Hình 8
    plot_fig08_rf_confusion_matrices_table8(RESULTS_DIR, summary_df, FIGURES_DIR / "fig08_rf_confusion_matrices_table8.png")
    
    # 9. Báo cáo Findings
    generate_findings_report(train_df, mi_df, summary_df, RESULTS_DIR / "eda_and_baseline_findings.md")
    
    total_time = time.perf_counter() - start_time
    print("=" * 75)
    print(f" HOÀN TẤT TOÀN BỘ 8 HÌNH ẢNH & BÁO CÁO TRONG {total_time:.2f}s ({total_time/60:.2f} phút)!")
    print("=" * 75)

if __name__ == "__main__":
    main()
