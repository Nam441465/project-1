#!/usr/bin/env python3
"""
tabs/registry_findings.py
Tab 4: Danh mục Mô hình (Model Registry) và Thư viện Biểu đồ / Báo cáo Nghiên cứu.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

import streamlit as st

from inference_engine import NIDSInferenceEngine

FIGURES_DICT: Dict[str, str] = {
    "fig01_class_imbalance.png": "Hình 1: Phân bố lớp & Mất cân bằng dữ liệu NF-UNSW-NB15",
    "fig02_feature_distributions.png": "Hình 2: Phân phối 6 đặc trưng lưu lượng tiêu biểu theo từng loại tấn công",
    "fig03_correlation_heatmap.png": "Hình 3: Ma trận tương quan Pearson giữa 10 đặc trưng",
    "fig04_mutual_information_heatmap.png": "Hình 4: Xếp hạng đặc trưng theo Mutual Information (MI)",
    "fig05_train_val_test_consistency.png": "Hình 5: Kiểm định độ nhất quán phân bố Train / Validation / Test",
    "fig06_rf_smote_vs_nosmote_comparison.png": "Hình 6: So sánh hiệu năng Random Forest Without SMOTE vs With SMOTE",
    "fig07_rf_k_sensitivity.png": "Hình 7: Phân tích độ nhạy của số đặc trưng k đối với F1-Score (RF)",
    "fig08_rf_confusion_matrices_table8.png": "Hình 8: Ma trận nhầm lẫn 5 bài toán RF Baseline (Table 8 MDPI)",
    "fig11_recall_threshold_tradeoff.png": "Hình 11: Tối ưu hóa ngưỡng theo Recall mục tiêu (95%, 98%, 99%)",
    "fig12_cost_sensitive_analysis.png": "Hình 12: Phân tích chi phí bất đối xứng Cost-Sensitive (c=1, 5, 10, 20)",
    "fig13_false_negative_forensics.png": "Hình 13: Phân tích đặc trưng các mẫu bị bỏ sót (False Negative Forensics)",
    "fig14_calibration_and_bootstrap.png": "Hình 14: Hiệu chuẩn xác suất (ECE-15) & Bootstrap 95% CI / McNemar (PaperRF vs Ours RF)",
    "fig15_mixed_flow_orchestration.png": "Hình 15: Đánh giá điều phối hỗn hợp Any-of-5 (Benign FPR, Target-5 & Unseen-4)",
    "fig_p01_two_systems_f1_and_speed.png": "Hình P01: Đối sánh F1 và tốc độ huấn luyện của 2 hệ thống RF (Paper-Exact vs Ours RF)",
    "fig_p02_smote_paradox_and_leakage_ablation.png": "Hình P02: Tác động của SMOTE và thực nghiệm chống rò rỉ kiểm định chéo (CV Leakage)",
    "fig_p03_pr_curves_two_systems.png": "Hình P03: Đường cong Precision-Recall của 2 hệ thống RF trên 5 lớp tấn công",
    "fig_p04_soc_cost_and_fn_comparison.png": "Hình P04: Đối sánh chi phí rủi ro SOC (c=10) và số luồng tấn công bị bỏ sót (FN)",
}


@st.cache_data(show_spinner=False)
def _read_static_markdown(report_path_str: str) -> str:
    """Đọc và cache nội dung báo cáo Markdown tĩnh."""
    with open(report_path_str, "r", encoding="utf-8") as rf:
        return rf.read()


def render_registry_findings_tab(
    engine: NIDSInferenceEngine,
    figures_dir: Path,
    results_dir: Path,
    on_reload_registry: Callable[[], None],
) -> None:
    """Render toàn bộ nội dung Tab 4."""
    st.markdown("#### Danh mục Mô hình & Kết quả Nghiên cứu")

    sub_t1, sub_t2 = st.tabs(["Danh mục Mô hình (Model Registry)", "Thư viện Biểu đồ (300 DPI) & Báo cáo"])

    with sub_t1:
        st.markdown("**Danh mục mô hình tự động phát hiện từ `results/*/models/`**")
        st.caption(
            "Khi cập nhật tệp mô hình hoặc metadata mới vào thư mục kết quả, "
            "nhấn nút bên dưới để làm mới danh mục và bộ nhớ đệm."
        )

        if st.button("Làm mới danh mục mô hình", key="btn_reload_registry"):
            on_reload_registry()

        if not engine.models_registry:
            st.warning("Không tìm thấy mô hình nào trong danh mục.")
        else:
            reg_summary = engine.get_registered_models_summary()
            st.dataframe(reg_summary, width="stretch", height=350)

            csv_export = reg_summary.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Tải bảng Model Registry (CSV)",
                data=csv_export,
                file_name="nids_models_registry_summary.csv",
                mime="text/csv",
            )

            st.markdown("---")
            st.markdown("**Thông số cấu hình và bảng ngưỡng của từng mô hình**")
            sorted_keys = sorted(engine.models_registry.keys())
            selected_model_inspect = st.selectbox(
                "Chọn mô hình để xem chi tiết:",
                options=sorted_keys,
                key="sb_registry_inspect",
            )

            if selected_model_inspect:
                info = engine.models_registry[selected_model_inspect]
                c_meta, c_thresh = st.columns([1, 1])

                with c_meta:
                    st.markdown(f"**Thông số cấu hình `{selected_model_inspect}`**")
                    st.json({
                        "Họ mô hình": info["model_family"],
                        "Loại tấn công mục tiêu": info["attack"],
                        "Điều kiện dữ liệu": info["condition_label"],
                        "Số đặc trưng (k)": info["k_features"],
                        "Danh sách đặc trưng": info["selected_features"],
                        "Chỉ số thực nghiệm": info["metrics"],
                    })

                with c_thresh:
                    st.markdown("**Bảng ngưỡng quyết định tối ưu (Threshold Map)**")
                    st.json(info["thresholds"])

    with sub_t2:
        st.markdown("**Thư viện 17 biểu đồ thực nghiệm (300 DPI)**")

        fig_choice = st.selectbox(
            "Chọn biểu đồ cần xem:",
            options=list(FIGURES_DICT.keys()),
            format_func=lambda x: f"{x} — {FIGURES_DICT[x]}",
            key="sb_fig_choice",
        )

        fig_path = figures_dir / fig_choice
        if not fig_path.exists():
            fig_path = results_dir / "paper_original" / "figures" / fig_choice
        if fig_path.exists():
            st.image(str(fig_path), caption=FIGURES_DICT[fig_choice], width="stretch")
        else:
            st.warning(f"Không tìm thấy tệp ảnh tại: {fig_path}")

        st.markdown("---")
        st.markdown("**Báo cáo phân tích kết quả thực nghiệm**")

        rep_choice = st.radio(
            "Chọn báo cáo:",
            [
                "Đối sánh 2 hệ thống RF: Paper-Exact RF vs Ours Improved RF (paper_vs_ours_findings.md)",
                "Phân tích khám phá dữ liệu & Baseline (eda_and_baseline_findings.md)",
                "Phân tích bỏ sót FN & Tối ưu chi phí SOC (fn_and_cost_findings.md)",
                "Chẩn đoán khoa học Package A: Calibration, Bootstrap & Any-of-5 (scientific_diagnostics_findings.md)",
            ],
            key="rad_report_choice",
        )

        if "paper_vs_ours_findings.md" in rep_choice:
            report_file = results_dir / "paper_original" / "paper_vs_ours_findings.md"
        elif "eda_and_baseline_findings.md" in rep_choice:
            report_file = results_dir / "baseline" / "eda_and_baseline_findings.md"
        elif "fn_and_cost_findings.md" in rep_choice:
            report_file = results_dir / "improved" / "fn_and_cost_findings.md"
        else:
            report_file = results_dir / "improved" / "scientific_diagnostics_findings.md"

        if report_file.exists():
            md_content = _read_static_markdown(str(report_file))
            with st.expander("Xem toàn bộ nội dung báo cáo", expanded=False):
                st.markdown(md_content)
        else:
            st.info(f"Chưa có tệp báo cáo tại {report_file}")
