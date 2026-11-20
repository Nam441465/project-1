#!/usr/bin/env python3
"""
app.py - Giao diện Thực nghiệm Hệ thống Phát hiện Xâm nhập Mạng (NIDS - NF-UNSW-NB15)
Slim Streamlit entrypoint điều phối 4 tab chức năng:
  - Tab 1: Kiểm tra đơn luồng (Single-Flow Inspection & Tier-2 Heuristic)
  - Tab 2: Đối sánh mô hình (Paper-Exact RF vs Hệ thống đề xuất)
  - Tab 3: Đánh giá theo lô dữ liệu (Batch Evaluation)
  - Tab 4: Danh mục mô hình & Kết quả nghiên cứu
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference_engine import NIDSInferenceEngine
from protocol_decoder import DEFAULT_BENIGN_FLOW
from styles import inject_custom_css, render_page_header
from tabs import (
    apply_sample_to_state,
    render_ab_showdown_tab,
    render_batch_scanner_tab,
    render_registry_findings_tab,
    render_single_flow_tab,
)

st.set_page_config(
    page_title="Hệ thống Phát hiện Xâm nhập Mạng (NIDS) | NF-UNSW-NB15",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


@st.cache_resource
def get_inference_engine() -> NIDSInferenceEngine:
    """Khởi tạo NIDSInferenceEngine 1 lần duy nhất và cache lại."""
    return NIDSInferenceEngine(lazy_load=True)


def _handle_reload_registry() -> None:
    """Làm mới Registry an toàn giữa các session bằng cách xóa resource cache."""
    index_file = CURRENT_DIR / "models_index.json"
    if index_file.exists():
        try:
            index_file.unlink()
        except OSError:
            pass
    get_inference_engine.clear()
    st.cache_data.clear()
    st.toast("Đã làm mới danh mục mô hình và bộ nhớ đệm.")
    st.rerun()


engine = get_inference_engine()

# Khởi tạo session_state với mẫu tĩnh DEFAULT_BENIGN_FLOW (tránh đọc parquet khi khởi động)
if "current_flow" not in st.session_state or "inp_L4_SRC_PORT" not in st.session_state:
    apply_sample_to_state(DEFAULT_BENIGN_FLOW)

if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None
if "batch_target_attack" not in st.session_state:
    st.session_state["batch_target_attack"] = "DoS"
if "ab_results" not in st.session_state:
    st.session_state["ab_results"] = None


# ==============================================================================
# SIDEBAR - CẤU HÌNH THỰC NGHIỆM
# ==============================================================================
with st.sidebar:
    st.markdown("### Cấu hình Thực nghiệm")
    st.caption("Tập dữ liệu NF-UNSW-NB15 • 10 đặc trưng NetFlow v1")

    num_models = len(engine.models_registry)
    family_counts: Dict[str, int] = {}
    for meta in engine.models_registry.values():
        fam = meta["model_family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1

    family_short = {
        "PAPER": "Paper-Exact RF",
        "RF": "Ours-RF",
        "GB": "Ours-HistGB",
    }
    ordered_fams = [f for f in ["PAPER", "RF", "GB"] if f in family_counts] + sorted(
        f for f in family_counts if f not in {"PAPER", "RF", "GB"}
    )
    if num_models > 0:
        breakdown = " • ".join(f"{family_counts[f]} {family_short.get(f, f)}" for f in ordered_fams)
        st.markdown(
            f'<div class="sidebar-status"><strong>Đã nạp {num_models} mô hình</strong><br>{breakdown}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("Chưa phát hiện mô hình nào trong thư mục kết quả.")

    st.markdown("---")
    st.markdown("#### 1. Ngưỡng quyết định")

    threshold_options = [
        "F1-Optimal (Cân bằng trên Validation)",
        "Cost-Sensitive (c=10: Giảm bỏ sót FN)",
        "Cost-Sensitive (c=20: Ưu tiên độ nhạy cao)",
        "Recall >= 95% (Độ phủ mục tiêu 95%)",
        "Recall >= 98% (Độ phủ mục tiêu 98%)",
        "Recall >= 99% (Độ phủ mục tiêu 99%)",
        "Standard (0.50: Ngưỡng mặc định)",
        "Tùy chỉnh (Slider)",
    ]

    selected_mode_label = st.selectbox(
        "Chế độ ngưỡng vận hành:",
        options=threshold_options,
        index=1,
        help="Điều chỉnh đánh đổi giữa số mẫu tấn công bị bỏ sót (FN) và cảnh báo nhầm (FP).",
        key="sb_global_threshold_mode",
    )

    custom_threshold_val: Optional[float] = None
    if selected_mode_label == "Tùy chỉnh (Slider)":
        custom_threshold_val = st.slider("Giá trị ngưỡng:", 0.01, 0.99, 0.50, 0.01, key="sl_global_custom_thresh")
        active_mode_key = "Custom"
    else:
        mode_mapping = {
            "F1-Optimal (Cân bằng trên Validation)": "F1-Optimal",
            "Cost-Sensitive (c=10: Giảm bỏ sót FN)": "Cost-Sensitive (c=10)",
            "Cost-Sensitive (c=20: Ưu tiên độ nhạy cao)": "Cost-Sensitive (c=20)",
            "Recall >= 95% (Độ phủ mục tiêu 95%)": "Recall >= 95%",
            "Recall >= 98% (Độ phủ mục tiêu 98%)": "Recall >= 98%",
            "Recall >= 99% (Độ phủ mục tiêu 99%)": "Recall >= 99%",
            "Standard (0.50: Ngưỡng mặc định)": "Standard (0.50)",
        }
        active_mode_key = mode_mapping[selected_mode_label]

    st.markdown("#### 2. Dữ liệu huấn luyện")
    cond_choice = st.radio(
        "Điều kiện SMOTE:",
        options=["Without SMOTE (Khuyến nghị)", "With SMOTE (56,000 mẫu)", "Tất cả điều kiện"],
        index=0,
        key="rad_global_condition",
    )
    if "Without SMOTE" in cond_choice:
        cond_filter = "nosmote"
    elif "With SMOTE" in cond_choice:
        cond_filter = "smote"
    else:
        cond_filter = "all"

    st.markdown("---")
    with st.expander("Phương pháp phân luồng 2 tầng", expanded=False):
        st.markdown(
            "- **Tầng 1 (Inline Classifier):** Phân loại lưu lượng NetFlow bằng HistGB hoặc Random Forest.\n"
            "- **Vùng cận ngưỡng (Near-Miss):** Khoảng `[min(0.10, 0.5 × t), t)` chuyển sang **Tầng 2** kiểm tra quy tắc đặc trưng sâu nhằm giảm bỏ sót mà không tăng mạnh cảnh báo nhầm."
        )


# ==============================================================================
# TIÊU ĐỀ TRANG CHÍNH & 4 TAB CHỨC NĂNG
# ==============================================================================
render_page_header()

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Kiểm tra đơn luồng",
    "2. Đối sánh mô hình",
    "3. Đánh giá theo lô dữ liệu",
    "4. Danh mục mô hình & Kết quả",
])

with tab1:
    render_single_flow_tab(engine, active_mode_key, cond_filter, custom_threshold_val)

with tab2:
    render_ab_showdown_tab(engine)

with tab3:
    render_batch_scanner_tab(engine, active_mode_key, custom_threshold_val)

with tab4:
    render_registry_findings_tab(engine, FIGURES_DIR, RESULTS_DIR, _handle_reload_registry)
