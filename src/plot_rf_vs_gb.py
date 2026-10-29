#!/usr/bin/env python3
"""
plot_rf_vs_gb.py
Trực quan hóa đối sánh toàn diện giữa Random Forest (Baseline) và Gradient Boosting (Improved - HistGradientBoosting).
Sinh Hình 9 (Đối sánh F1-Score & Thời gian Huấn luyện) và Hình 10 (Đường cong Precision-Recall Curves 5 loại tấn công).
Xuất ảnh 300 DPI vào thư mục `figures/`.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score

# Đường dẫn thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
BASELINE_DIR = BASE_DIR / "results" / "baseline"
IMPROVED_DIR = BASE_DIR / "results" / "improved"
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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
# HÌNH 9: ĐỐI SÁNH F1-SCORE VÀ THỜI GIAN HUẤN LUYỆN (RF vs GB)
# ==============================================================================
def plot_fig09_rf_vs_gb_comparison(comparison_df: pd.DataFrame, out_path: Path):
    """
    Vẽ Hình 9: 2 panel ngang
    - Panel trái: So sánh F1-Score (Without SMOTE) giữa RF và GB trên 5 loại tấn công.
    - Panel phải: So sánh Train Time (sec) giữa RF và GB và hiển thị hệ số tăng tốc.
    """
    print("[*] Đang vẽ Hình 9: Đối sánh F1-Score & Thời gian huấn luyện (RF vs GB)...")
    setup_matplotlib_style()
    
    # Lọc lấy điều kiện Without SMOTE chuẩn mực
    df_no = comparison_df[comparison_df["Condition"] == "Without SMOTE"].copy()
    
    # Đảm bảo thứ tự tấn công đúng chuẩn
    df_no["sort_idx"] = df_no["Attack"].apply(lambda x: TARGET_ATTACKS.index(x) if x in TARGET_ATTACKS else 99)
    df_no = df_no.sort_values("sort_idx").reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    attacks = df_no["Attack"].tolist()
    x = np.arange(len(attacks))
    width = 0.35
    
    color_rf = "#1f77b4"  # Xanh dương đậm chuẩn RF
    color_gb = "#2ca02c"  # Xanh lục bảo chuẩn GB
    
    # ---------------- PANEL TRÁI: F1-SCORE ----------------
    rects1_f1 = ax1.bar(x - width/2, df_no["RF_F1_Score"], width, label="Random Forest (Baseline)",
                        color=color_rf, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    rects2_f1 = ax1.bar(x + width/2, df_no["GB_F1_Score"], width, label="Gradient Boosting (HistGB)",
                        color=color_gb, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    
    ax1.set_ylabel("F1-Score (Held-out Test)", fontweight="bold")
    ax1.set_title("So sánh Hiệu năng Phát hiện (F1-Score - Without SMOTE)", pad=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(attacks, fontweight="bold")
    ax1.set_ylim(min(df_no["RF_F1_Score"].min(), df_no["GB_F1_Score"].min()) - 0.06, 1.02)
    ax1.grid(axis="y", zorder=0)
    ax1.legend(loc="lower right", framealpha=0.95, edgecolor="#bdc3c7")
    
    # Ghi giá trị 4 chữ số trên đầu mỗi cột
    for rect in rects1_f1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.005, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f77b4")
                 
    for rect in rects2_f1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.005, f"{h:.4f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1e7e34")
                 
    # ---------------- PANEL PHẢI: TRAINING TIME & SPEEDUP ----------------
    rects1_t = ax2.bar(x - width/2, df_no["RF_Train_Time_Sec"], width, label="Random Forest (Baseline)",
                       color=color_rf, edgecolor="#2c3e50", alpha=0.9, zorder=3)
    rects2_t = ax2.bar(x + width/2, df_no["GB_Train_Time_Sec"], width, label="Gradient Boosting (HistGB)",
                       color=color_gb, edgecolor="#2c3e50", alpha=0.9, zorder=3)
                       
    ax2.set_ylabel("Thời gian Huấn luyện (Giây) - Thang Logarit", fontweight="bold")
    ax2.set_title("So sánh Chi phí Tính toán & Tốc độ Hội tụ (Thang Logarit)", pad=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attacks, fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(axis="y", which="both", zorder=0)
    ax2.legend(loc="upper right", framealpha=0.95, edgecolor="#bdc3c7")
    
    # Đặt giới hạn y hợp lý cho thang logarit để legend không bị đè bởi badge Speedup
    max_rf_time = df_no["RF_Train_Time_Sec"].max()
    ax2.set_ylim(0.5, max_rf_time * 8.0)
    
    # Ghi số giây và hệ số tăng tốc
    for i, (rf_rect, gb_rect) in enumerate(zip(rects1_t, rects2_t)):
        rf_h = rf_rect.get_height()
        gb_h = gb_rect.get_height()
        speedup = df_no.iloc[i]["Speedup_Factor"]
        
        # Ghi số giây trên từng cột
        ax2.text(rf_rect.get_x() + rf_rect.get_width()/2., rf_h * 1.15, f"{rf_h:.1f}s",
                 ha="center", va="bottom", fontsize=8, color="#1f77b4", fontweight="bold")
        ax2.text(gb_rect.get_x() + gb_rect.get_width()/2., gb_h * 1.15, f"{gb_h:.1f}s",
                 ha="center", va="bottom", fontsize=8, color="#1e7e34", fontweight="bold")
                 
        # Badge hệ số tăng tốc phía trên cặp cột
        center_x = (rf_rect.get_x() + gb_rect.get_x() + gb_rect.get_width()) / 2.
        ax2.text(center_x, rf_h * 1.6, f"Nhanh hơn\n~{speedup:.1f}x",
                 ha="center", va="bottom", fontsize=8, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2e7d32", lw=1))

    plt.suptitle("ĐỐI SÁNH TOÀN DIỆN RANDOM FOREST VS GRADIENT BOOSTING TRÊN DỮ LIỆU NF-UNSW-NB15",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Hình 9 thành công: {out_path}")

# ==============================================================================
# HÌNH 10: ĐƯỜNG CONG PRECISION-RECALL (PR CURVES) 5 LOẠI TẤN CÔNG
# ==============================================================================
def plot_fig10_pr_curves_rf_vs_gb(out_path: Path):
    """
    Vẽ Hình 10: Lưới 2x3 gồm 5 đồ thị PR curves cho 5 loại tấn công và 1 ô tổng kết khoa học.
    Vẽ 4 đường cong PR cho mỗi loại tấn công:
      1. RF (Without SMOTE)
      2. RF (With SMOTE)
      3. GB (Without SMOTE)
      4. GB (With SMOTE)
    Kèm điểm hoạt động (Operating Point) và PR-AUC (Average Precision).
    """
    print("[*] Đang vẽ Hình 10: Precision-Recall Curves (RF vs GB) cho 5 loại tấn công...")
    setup_matplotlib_style()
    
    # Đọc ngưỡng tối ưu từ các bảng summary
    rf_sum = pd.read_csv(BASELINE_DIR / "summary_rf_comparison.csv")
    gb_sum = pd.read_csv(IMPROVED_DIR / "summary_gb_comparison.csv")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    
    # Định nghĩa cấu hình đường cong
    curve_configs = [
        {"model": "RF", "cond": "nosmote", "cond_name": "Without SMOTE", "label": "RF (No SMOTE)", 
         "color": "#1f77b4", "linestyle": "-", "marker": "o"},
        {"model": "RF", "cond": "smote", "cond_name": "With SMOTE", "label": "RF (SMOTE)", 
         "color": "#17becf", "linestyle": "--", "marker": "s"},
        {"model": "GB", "cond": "nosmote", "cond_name": "Without SMOTE", "label": "GB (No SMOTE)", 
         "color": "#2ca02c", "linestyle": "-", "marker": "*"},
        {"model": "GB", "cond": "smote", "cond_name": "With SMOTE", "label": "GB (SMOTE)", 
         "color": "#ff7f0e", "linestyle": "--", "marker": "^"}
    ]
    
    for idx, attack in enumerate(TARGET_ATTACKS):
        ax = axes_flat[idx]
        ax.set_title(f"Tấn công: {attack}", fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.grid(True, zorder=0)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        
        for cfg in curve_configs:
            m_code = cfg["model"].lower()
            cond_code = cfg["cond"]
            dir_path = BASELINE_DIR if cfg["model"] == "RF" else IMPROVED_DIR
            
            proba_file = dir_path / f"{m_code}_{attack.lower()}_{cond_code}_best_proba_test.npy"
            y_file = dir_path / f"{m_code}_{attack.lower()}_{cond_code}_y_test.npy"
            
            if not proba_file.exists() or not y_file.exists():
                continue
                
            y_true = np.load(y_file)
            y_proba = np.load(proba_file)
            
            # Tính PR Curve và Average Precision (PR-AUC)
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
            ap = average_precision_score(y_true, y_proba)
            
            # Lấy ngưỡng tối ưu đã chọn trên Validation
            sum_df = rf_sum if cfg["model"] == "RF" else gb_sum
            match_row = sum_df[(sum_df["Attack"] == attack) & (sum_df["Condition"] == cfg["cond_name"])]
            t_opt = float(match_row["Threshold"].iloc[0]) if not match_row.empty else 0.5
            
            # Tính điểm hoạt động tại ngưỡng t_opt
            y_pred = (y_proba >= t_opt).astype(int)
            oper_p = precision_score(y_true, y_pred, zero_division=0)
            oper_r = recall_score(y_true, y_pred, zero_division=0)
            
            # Vẽ đường cong
            ax.plot(recalls, precisions, label=f"{cfg['label']} (AP={ap:.4f})",
                    color=cfg["color"], linestyle=cfg["linestyle"], linewidth=1.6, zorder=3)
                    
            # Đánh dấu điểm hoạt động
            ax.scatter([oper_r], [oper_p], color=cfg["color"], marker=cfg["marker"],
                       s=70, edgecolor="#2c3e50", zorder=5)
                       
        ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9, edgecolor="#bdc3c7")
        
    # Ô thứ 6 (Hàng 2, Cột 3): Hộp Tổng kết Khoa học So sánh RF vs GB
    ax_summary = axes_flat[5]
    ax_summary.axis("off")
    
    summary_box_text = (
        "TỔNG KẾT KHOA HỌC: RANDOM FOREST VS GRADIENT BOOSTING\n"
        "────────────────────────────────────────────────────────\n\n"
        "1. HIỆU NĂNG NGUYÊN BẢN (WITHOUT SMOTE):\n"
        " • Random Forest đạt F1 nhỉnh hơn một chút ở trạng thái nguyên bản nhờ cơ chế\n"
        "   chia nhánh chính xác tuyệt đối (exact splits) trên các cổng dịch vụ rời rạc\n"
        "   và giao thức tầng 7 (L4_DST_PORT, L7_PROTO).\n"
        " • HistGradientBoosting đạt F1 tiệm cận rất sát (Reconnaissance 0.9611 vs 0.9673;\n"
        "   Analysis 0.8690 vs 0.8733; Backdoor 0.9386 vs 0.9535) và xây dựng ranh giới phi tuyến mượt mà.\n\n"
        "2. SỰ KHÁC BIỆT THÚ VỊ GIỮA 2 HỌ MÔ HÌNH VỚI SMOTE:\n"
        " • Random Forest bị suy giảm F1 ở 4/5 loại tấn công khi dùng SMOTE do nhiễu nội suy\n"
        "   trong không gian đặc trưng mạng rời rạc (Nghịch lý SMOTE).\n"
        " • Ngược lại, HistGradientBoosting (sử dụng 255 histogram bins) được SMOTE cải\n"
        "   thiện cả PR-AUC và F1 (Backdoor: 0.9386 -> 0.9440; Analysis: 0.8690 -> 0.8703;\n"
        "   Reconnaissance: 0.9611 -> 0.9630; và PR-AUC của DoS tăng từ 0.8899 -> 0.8983)\n"
        "   nhờ tăng mật độ mẫu tấn công trong quá trình chia bin lượng tử hóa.\n\n"
        "3. ĐỘT PHÁ TỐC ĐỘ & KHẢ NĂNG MỞ RỘNG (SCALABILITY):\n"
        " • HistGradientBoosting nhanh hơn Random Forest từ 10x đến 50x lần nhờ kỹ thuật\n"
        "   rời rạc hóa histogram và song song hóa OpenMP.\n"
        " • Phù hợp làm tuyến phòng thủ đầu tiên cho luồng NIDS lưu lượng cao thời gian thực."
    )
    
    ax_summary.text(0.04, 0.5, summary_box_text, transform=ax_summary.transAxes,
                    fontsize=8.5, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#495057", lw=1.2))
                    
    plt.suptitle("ĐƯỜNG CONG PRECISION-RECALL (PR CURVES) VÀ ĐIỂM HOẠT ĐỘNG: RF VS GB",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[+] Đã lưu Hình 10 thành công: {out_path}")

if __name__ == "__main__":
    rf_vs_gb_file = IMPROVED_DIR / "rf_vs_gb_comparison.csv"
    if rf_vs_gb_file.exists():
        comp_df = pd.read_csv(rf_vs_gb_file)
        plot_fig09_rf_vs_gb_comparison(comp_df, FIGURES_DIR / "fig09_rf_vs_gb_comparison.png")
        plot_fig10_pr_curves_rf_vs_gb(FIGURES_DIR / "fig10_precision_recall_curves_rf_vs_gb.png")
    else:
        print(f"[!] Chưa tìm thấy file đối sánh {rf_vs_gb_file}. Vui lòng chạy train_gb.py trước.")
