#!/usr/bin/env python3
"""
app.py - SOC NIDS Live Console (NF-UNSW-NB15)
Giao diện Web tương tác Demo Hệ thống Phát hiện Xâm nhập Mạng (NIDS) 2 Tầng.
Hỗ trợ kiểm thử trực tiếp 1 luồng mạng (Live Single-Flow), Quét lưu lượng hàng loạt (Batch Traffic Scanner)
và Trung tâm Quản lý Mô hình & Biểu đồ Khoa học (Model Registry & Findings).
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thiết lập đường dẫn
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference_engine import NIDSInferenceEngine, NETFLOW_FEATURES, CANONICAL_ATTACKS

# ==============================================================================
# CẤU HÌNH TRANG STREAMLIT
# ==============================================================================
st.set_page_config(
    page_title="SOC NIDS Live Console | NF-UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện SOC hiện đại
st.markdown("""
<style>
    .reportview-container {
        background: #0E1117;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9E9E9E;
        margin-bottom: 20px;
    }
    .verdict-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 8px solid;
    }
    .verdict-attack {
        background-color: rgba(229, 62, 62, 0.15);
        border-color: #E53E3E;
    }
    .verdict-suspicious {
        background-color: rgba(221, 107, 32, 0.15);
        border-color: #DD6B20;
    }
    .verdict-benign {
        background-color: rgba(56, 161, 105, 0.15);
        border-color: #38A169;
    }
    .metric-box {
        background-color: #1A202C;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border: 1px solid #2D3748;
    }
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# KHỞI TẠO VÀ CACHE LÕI SUY LUẬN
# ==============================================================================
@st.cache_resource
def get_inference_engine():
    """Khởi tạo NIDSInferenceEngine 1 lần duy nhất và cache lại."""
    return NIDSInferenceEngine(lazy_load=True)

engine = get_inference_engine()


def _apply_sample_to_state(sample: Dict[str, Any]):
    """Đồng bộ mẫu luồng mạng vào session_state và toàn bộ các ô nhập liệu của Interactive Editor."""
    st.session_state["current_flow"] = sample
    st.session_state["scenario_info"] = sample.get("Sample_Scenario", "Mẫu Tùy chỉnh")
    st.session_state["true_attack"] = sample.get("True_Attack", "Không rõ")
    st.session_state["inp_L4_SRC_PORT"] = max(0, min(65535, int(sample.get("L4_SRC_PORT", 0))))
    st.session_state["inp_L4_DST_PORT"] = max(0, min(65535, int(sample.get("L4_DST_PORT", 0))))
    st.session_state["inp_PROTOCOL"] = max(0, min(255, int(sample.get("PROTOCOL", 6))))
    st.session_state["inp_L7_PROTO"] = float(sample.get("L7_PROTO", 0.0))
    st.session_state["inp_IN_BYTES"] = max(0, int(sample.get("IN_BYTES", 0)))
    st.session_state["inp_OUT_BYTES"] = max(0, int(sample.get("OUT_BYTES", 0)))
    st.session_state["inp_IN_PKTS"] = max(0, int(sample.get("IN_PKTS", 0)))
    st.session_state["inp_OUT_PKTS"] = max(0, int(sample.get("OUT_PKTS", 0)))
    st.session_state["inp_TCP_FLAGS"] = max(0, int(sample.get("TCP_FLAGS", 0)))
    st.session_state["inp_FLOW_DURATION_MS"] = max(0, int(sample.get("FLOW_DURATION_MILLISECONDS", 0)))


# Khởi tạo session state lưu luồng mạng đang kiểm thử
if "current_flow" not in st.session_state or "inp_L4_SRC_PORT" not in st.session_state:
    try:
        sample_init = engine.sample_real_test_flow(sample_type="benign", random_seed=42)
    except Exception:
        sample_init = {
            "L4_SRC_PORT": 5353,
            "L4_DST_PORT": 80,
            "PROTOCOL": 6,
            "L7_PROTO": 7.0,
            "IN_BYTES": 1284,
            "OUT_BYTES": 2568,
            "IN_PKTS": 8,
            "OUT_PKTS": 10,
            "TCP_FLAGS": 27,
            "FLOW_DURATION_MILLISECONDS": 120,
            "Sample_Scenario": "Khởi tạo mặc định",
            "True_Attack": "Benign"
        }
    _apply_sample_to_state(sample_init)

# Khởi tạo session state cho Tab 2 (Batch Traffic)
if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None
if "batch_target_attack" not in st.session_state:
    st.session_state["batch_target_attack"] = "DoS"


# ==============================================================================
# SIDEBAR - BẢNG ĐIỀU KHIỂN SOC LIVE CONSOLE
# ==============================================================================
with st.sidebar:
    st.markdown("## 🛡️ **SOC NIDS CONSOLE**")
    st.markdown("**Kiến trúc Phòng thủ 2 Tầng (2-Tier Hybrid NIDS)**")
    st.caption("Mô hình NetFlow v1 • NF-UNSW-NB15 Dataset")

    # 1. Trạng thái Registry
    num_models = len(engine.models_registry)
    rf_count = sum(1 for m in engine.models_registry.values() if m["model_family"] == "RF")
    gb_count = sum(1 for m in engine.models_registry.values() if m["model_family"] == "GB")
    st.success(f"🟢 **{num_models} Mô hình Trực tuyến** ({rf_count} RF & {gb_count} HistGB)")

    st.markdown("---")
    st.markdown("### ⚙️ **Cấu hình Ngưỡng SOC**")

    # 2. Bộ chọn Chế độ Ngưỡng
    threshold_options = [
        "F1-Optimal (Chuẩn cân bằng Validation)",
        "Cost-Sensitive (c=10: Ưu tiên chống bỏ sót SOC)",
        "Cost-Sensitive (c=20: Phòng thủ nghiêm ngặt)",
        "Recall >= 95% (Độ phủ mục tiêu 95%)",
        "Recall >= 98% (Độ phủ mục tiêu 98%)",
        "Recall >= 99% (Zero-Trust: Bắt tối đa)",
        "Standard (0.50: Ngưỡng mặc định lý thuyết)",
        "Tùy chỉnh (Custom Slider)"
    ]

    selected_mode_label = st.selectbox(
        "Chế độ Ngưỡng Quyết định:",
        options=threshold_options,
        index=0,
        help="Chọn chính sách ngưỡng để điều khiển sự đánh đổi giữa False Negatives (lọt lưới) và False Positives (báo động giả)."
    )

    custom_threshold_val = None
    if selected_mode_label == "Tùy chỉnh (Custom Slider)":
        custom_threshold_val = st.slider("Kéo Ngưỡng Quyết định:", 0.01, 0.99, 0.50, 0.01)
        active_mode_key = "Custom"
    else:
        # Ánh xạ nhãn hiển thị sang key của engine
        mode_mapping = {
            "F1-Optimal (Chuẩn cân bằng Validation)": "F1-Optimal",
            "Cost-Sensitive (c=10: Ưu tiên chống bỏ sót SOC)": "Cost-Sensitive (c=10)",
            "Cost-Sensitive (c=20: Phòng thủ nghiêm ngặt)": "Cost-Sensitive (c=20)",
            "Recall >= 95% (Độ phủ mục tiêu 95%)": "Recall >= 95%",
            "Recall >= 98% (Độ phủ mục tiêu 98%)": "Recall >= 98%",
            "Recall >= 99% (Zero-Trust: Bắt tối đa)": "Recall >= 99%",
            "Standard (0.50: Ngưỡng mặc định lý thuyết)": "Standard (0.50)"
        }
        active_mode_key = mode_mapping[selected_mode_label]

    # 3. Bộ lọc điều kiện huấn luyện
    st.markdown("### 🎯 **Bộ Lọc Huấn Luyện**")
    cond_choice = st.radio(
        "Điều kiện Dữ liệu Huấn luyện:",
        options=["Without SMOTE (Chuẩn khuyến nghị)", "With SMOTE (56,000 mẫu)", "Hiển thị Tất cả (So sánh No-SMOTE vs SMOTE)"],
        index=0
    )
    if "Without SMOTE" in cond_choice:
        cond_filter = "nosmote"
    elif "With SMOTE" in cond_choice:
        cond_filter = "smote"
    else:
        cond_filter = "all"

    st.markdown("---")
    st.info(
        "💡 **Kiến trúc 2 Tầng:**\n"
        "- **Tầng 1 (Inline ML):** Phát hiện tức thì luồng vượt ngưỡng chặn.\n"
        "- **Tầng 2 (DPI Inspection):** Kiểm tra gói tin sâu cho vùng Near-Miss (10% - Ngưỡng)."
    )


# ==============================================================================
# TIÊU ĐỀ TRANG CHÍNH
# ==============================================================================
st.markdown('<p class="main-header">🛡️ SOC NIDS LIVE CONSOLE</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hệ thống Giám sát & Phản ứng An ninh Mạng Đa Mô hình Thông minh (NF-UNSW-NB15)</p>', unsafe_allow_html=True)


# ==============================================================================
# KHU VỰC 3 TAB CHÍNH
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "🔍 TAB 1: KIỂM THỬ TRỰC TIẾP 1 LUỒNG (LIVE SINGLE-FLOW)",
    "📊 TAB 2: QUÉT LƯU LƯỢNG HÀNG LOẠT (BATCH TRAFFIC SCANNER)",
    "📚 TAB 3: TRUNG TÂM MÔ HÌNH & KHOA HỌC (REGISTRY & FIGURES)"
])


# ==============================================================================
# TAB 1: KIỂM THỬ TRỰC TIẾP 1 LUỒNG MẠNG (LIVE SINGLE-FLOW INSPECTOR)
# ==============================================================================
with tab1:
    st.markdown("### ⚡ **1. Nạp Mẫu Thử Nghiệm Nhanh (1-Click Real Presets)**")
    
    col_target_att, col_p1, col_p2, col_p3, col_p4 = st.columns([1.5, 1.2, 1.3, 1.8, 1.2])

    with col_target_att:
        preset_attack = st.selectbox("Mục tiêu thử nghiệm:", CANONICAL_ATTACKS, index=0)

    with col_p1:
        if st.button("🟢 Lưu Lượng Sạch", width="stretch", help="Lấy 1 mẫu lưu lượng Benign thực tế từ tập Test."):
            sample = engine.sample_real_test_flow(
                sample_type="benign",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p2:
        if st.button("🔴 Tấn Công Điển Hình", width="stretch", help="Lấy 1 mẫu tấn công có xác suất cao rõ ràng (TP)."):
            sample = engine.sample_real_test_flow(
                sample_type="attack_typical",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p3:
        if st.button("🟡 Tấn Công Ẩn Mình (Near-Miss FN)", width="stretch", help="Mẫu tấn công bị bỏ sót ở F1-Optimal nhưng bắt dính ở Cost-Sensitive hoặc Recall >= 98%!"):
            sample = engine.sample_real_test_flow(
                sample_type="attack_stealth_nearmiss",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p4:
        if st.button("🎲 Mẫu Ngẫu Nhiên", width="stretch", help="Rút ngẫu nhiên 1 kịch bản (Benign / Typical / Near-Miss) từ tập Test."):
            sample = engine.sample_real_test_flow(
                sample_type="random",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    # Hiển thị thông tin ngữ cảnh mẫu đang nạp
    st.caption(
        f"🏷️ **Ngữ cảnh Mẫu Hiện Tại:** `{st.session_state.get('scenario_info', 'Mẫu Tùy chỉnh')}` | "
        f"Nhãn Thực Tế: **{st.session_state.get('true_attack', 'Không rõ')}**"
    )

    # Bộ chọn Phạm vi Mô hình Suy luận
    col_scope, _ = st.columns([2, 1])
    with col_scope:
        eval_scope = st.radio(
            "Phạm vi Mô hình Phân tích:",
            options=[
                f"Chỉ đánh giá Mục tiêu thử nghiệm đang chọn ({preset_attack})",
                "Quét toàn diện 5 loại tấn công (All 5 Attack Detectors)"
            ],
            index=0,
            horizontal=True,
            help="Chọn 'Chỉ đánh giá Mục tiêu' để tập trung kiểm thử hành vi của mô hình đối với đúng loại tấn công đang thử."
        )
    att_filter_val = preset_attack if "Chỉ đánh giá Mục tiêu" in eval_scope else "all"

    st.markdown("---")
    st.markdown("### 🎛️ **2. Bảng Nhập & Điều Chỉnh 10 Thông Số NetFlow (Interactive Flow Editor)**")

    # Chia 5 cột nhập liệu cho 10 đặc trưng NetFlow (gắn key cố định đồng bộ session_state)
    row1_c1, row1_c2, row1_c3, row1_c4, row1_c5 = st.columns(5)
    row2_c1, row2_c2, row2_c3, row2_c4, row2_c5 = st.columns(5)

    with row1_c1:
        f_src_port = st.number_input("L4_SRC_PORT", min_value=0, max_value=65535, key="inp_L4_SRC_PORT")
    with row1_c2:
        f_dst_port = st.number_input("L4_DST_PORT", min_value=0, max_value=65535, key="inp_L4_DST_PORT")
    with row1_c3:
        f_proto = st.number_input("PROTOCOL", min_value=0, max_value=255, key="inp_PROTOCOL")
    with row1_c4:
        f_l7_proto = st.number_input("L7_PROTO", format="%.1f", key="inp_L7_PROTO")
    with row1_c5:
        f_in_bytes = st.number_input("IN_BYTES", min_value=0, key="inp_IN_BYTES")

    with row2_c1:
        f_out_bytes = st.number_input("OUT_BYTES", min_value=0, key="inp_OUT_BYTES")
    with row2_c2:
        f_in_pkts = st.number_input("IN_PKTS", min_value=0, key="inp_IN_PKTS")
    with row2_c3:
        f_out_pkts = st.number_input("OUT_PKTS", min_value=0, key="inp_OUT_PKTS")
    with row2_c4:
        f_tcp_flags = st.number_input("TCP_FLAGS", min_value=0, key="inp_TCP_FLAGS")
    with row2_c5:
        f_flow_dur = st.number_input("FLOW_DURATION_MS", min_value=0, key="inp_FLOW_DURATION_MS")

    # Đóng gói dữ liệu đầu vào hiện tại
    input_flow_data = {
        "L4_SRC_PORT": f_src_port,
        "L4_DST_PORT": f_dst_port,
        "PROTOCOL": f_proto,
        "L7_PROTO": f_l7_proto,
        "IN_BYTES": f_in_bytes,
        "OUT_BYTES": f_out_bytes,
        "IN_PKTS": f_in_pkts,
        "OUT_PKTS": f_out_pkts,
        "TCP_FLAGS": f_tcp_flags,
        "FLOW_DURATION_MILLISECONDS": f_flow_dur
    }

    # ==========================================================================
    # CHẠY SUY LUẬN TRÊN ENGINE
    # ==========================================================================
    try:
        prediction_output = engine.predict_single_flow(
            flow_data=input_flow_data,
            threshold_mode=active_mode_key,
            condition_filter=cond_filter,
            attack_filter=att_filter_val,
            custom_threshold=custom_threshold_val
        )
        verdict = prediction_output["verdict"]
        details_df = prediction_output["details"]
    except Exception as err:
        st.error(f"Lỗi khi thực hiện suy luận: {err}")
        verdict = None
        details_df = None

    if verdict is not None:
        st.markdown("---")
        st.markdown("### 🚨 **3. Phán Quyết Của Kiến Trúc Phòng Thủ 2 Tầng (2-Tier Hybrid NIDS Verdict)**")

        # Hiển thị Banner phán quyết màu sắc động
        status = verdict["status"]
        if status == "ATTACK":
            card_class = "verdict-card verdict-attack"
            icon = "🚨"
            badge = "🔴 TẦNG 1: CHẶN CỨNG (INLINE PREVENTION)"
        elif status == "SUSPICIOUS":
            card_class = "verdict-card verdict-suspicious"
            icon = "⚠️"
            badge = "🟡 TẦNG 2: KIỂM TRA SÂU DPI (NEAR-MISS INSPECTION)"
        else:
            card_class = "verdict-card verdict-benign"
            icon = "✅"
            badge = "🟢 FAST PATH: CHO PHÉP THÔNG QUA (BENIGN FORWARDING)"

        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0;">{icon} <b>{verdict['title']}</b> &nbsp; <span style="font-size:0.9rem; padding:4px 8px; border-radius:4px; background:#1A202C;">{badge}</span></h3>
            <p style="font-size: 1.1rem; margin-bottom: 5px;">{verdict['message']}</p>
            <p style="font-size: 0.9rem; color:#A0AEC0; margin-bottom: 0;"><b>Kích hoạt bởi:</b> {verdict['triggered_by']} | <b>Xác suất rủi ro tối đa:</b> {verdict['max_probability']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ======================================================================
        # BIỂU ĐỒ SO SÁNH ĐA MÔ HÌNH VÀ BẢNG CHI TIẾT
        # ======================================================================
        st.markdown("### ⚔️ **4. Đấu Trường Đa Mô Hình (Multi-Model Comparison & Risk Gauges)**")
        col_chart, col_table = st.columns([1.1, 1.4])

        with col_chart:
            st.markdown("**Biểu Đồ Mức Độ Rủi Ro & Ngưỡng Quyết Định:**")
            
            # Sắp xếp để hiển thị trực quan
            chart_data = details_df.sort_values(by="Probability", ascending=True).copy()
            
            fig, ax = plt.subplots(figsize=(6, max(3.5, len(chart_data) * 0.45)))
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1A202C')

            y_pos = np.arange(len(chart_data))
            probs = chart_data["Probability"].values * 100
            thresholds = chart_data["Threshold_Used"].values * 100

            # Màu sắc theo vùng rủi ro
            bar_colors = []
            for p, th in zip(probs, thresholds):
                if p >= th:
                    bar_colors.append('#E53E3E') # Đỏ
                elif p >= 10.0:
                    bar_colors.append('#DD6B20') # Cam
                else:
                    bar_colors.append('#38A169') # Xanh

            bars = ax.barh(y_pos, probs, color=bar_colors, height=0.6, edgecolor='none')
            
            # Vẽ ranh giới ngưỡng cho từng mô hình
            for i, th in enumerate(thresholds):
                ax.plot([th, th], [i - 0.35, i + 0.35], color='#FC8181', linestyle='--', linewidth=2.0)

            # Vạch chuẩn 10% Near-Miss
            ax.axvline(x=10.0, color='#F6AD55', linestyle=':', linewidth=1.2, label='Vùng Nghi vấn (10%)')

            # Gắn nhãn mô hình
            labels = [f"{r['Model_Family']} - {r['Attack']} ({r['Condition']})" for _, r in chart_data.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color='#E2E8F0', fontsize=9)
            ax.set_xlabel("Xác suất Tấn công (%)", color='#E2E8F0', fontsize=10)
            ax.set_xlim(0, 105)
            ax.tick_params(colors='#CBD5E0', labelsize=9)
            
            # Ẩn viền
            for spine in ax.spines.values():
                spine.set_color('#4A5568')

            # Ghi số % bên cạnh bar
            for bar, p in zip(bars, probs):
                ax.text(p + 1.5, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", 
                        va='center', ha='left', color='#FFFFFF', fontsize=8, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            st.markdown(f"**Bảng Chi Tiết Dự Đoán ({len(details_df)} Mô hình):**")
            
            # Format hiển thị cho DataFrame
            display_df = details_df[[
                "Model_Family", "Attack", "Condition", "Probability", 
                "Threshold_Used", "Risk_Zone", "Selected_Features_Count", "Latency_ms"
            ]].copy()
            
            display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Threshold_Used"] = display_df["Threshold_Used"].apply(lambda x: f"{x:.3f}")
            display_df["Latency_ms"] = display_df["Latency_ms"].apply(lambda x: f"{x:.2f} ms")
            
            st.dataframe(
                display_df,
                width="stretch",
                height=320,
                column_config={
                    "Model_Family": "Họ Model",
                    "Attack": "Mục Tiêu",
                    "Condition": "Điều Kiện",
                    "Probability": "Xác Suất",
                    "Threshold_Used": "Ngưỡng Áp Dụng",
                    "Risk_Zone": "Vùng Rủi Ro",
                    "Selected_Features_Count": "Số k Feats",
                    "Latency_ms": "Độ Trễ"
                }
            )


# ==============================================================================
# TAB 2: QUÉT LƯU LƯỢNG HÀNG LOẠT (BATCH TRAFFIC SCANNER)
# ==============================================================================
with tab2:
    st.markdown("### 📊 **Quét Lưu Lượng Mạng Hàng Loạt & Đánh Giá Hiệu Năng Thực Nghiệm**")
    st.caption("Cho phép kiểm thử đồng thời hàng nghìn luồng mạng từ Tập Test Độc Lập (Held-out Test) hoặc tải lên file CSV bất kỳ.")

    col_b_source, col_b_model = st.columns([1.2, 1.0])

    with col_b_source:
        st.markdown("**1. Nguồn Dữ Liệu Đầu Vào:**")
        data_source_mode = st.radio(
            "Chọn Nguồn Dữ Liệu:",
            options=["Rút mẫu từ Held-out Test Parquet (1-Click Preset)", "Tải lên file CSV từ máy tính"],
            index=0
        )

    with col_b_model:
        st.markdown("**2. Mô Hình Phân Tích & Quét Lưu Lượng:**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            batch_model_family = st.selectbox("Thuật toán:", ["RF", "GB"], index=0)
        with col_m2:
            batch_condition = st.selectbox("Điều kiện Train:", ["Without SMOTE", "With SMOTE"], index=0)
            cond_key = "nosmote" if batch_condition == "Without SMOTE" else "smote"

    st.markdown("---")

    target_attack_for_eval = "DoS"

    if data_source_mode == "Rút mẫu từ Held-out Test Parquet (1-Click Preset)":
        b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with b_c1:
            target_attack_for_eval = st.selectbox("Bài toán Tấn công:", CANONICAL_ATTACKS, index=0)
        with b_c2:
            num_samples = st.slider("Số lượng luồng quét:", min_value=100, max_value=5000, value=500, step=100)
        with b_c3:
            att_ratio = st.slider("Tỷ lệ tấn công pha trộn:", min_value=0.05, max_value=0.50, value=0.10, step=0.05)
        with b_c4:
            inc_other = st.checkbox("Pha trộn cả các tấn công khác", value=False, help="Thử thách khả năng chịu lỗi cross-attack.")

        col_run_b1, col_run_b2 = st.columns([3, 1])
        with col_run_b1:
            if st.button("🚀 BẮT ĐẦU QUÉT LƯU LƯỢNG HÀNG LOẠT", type="primary", width="stretch"):
                with st.spinner("Đang nạp dữ liệu phân tầng từ heldout_test.parquet và suy luận hàng loạt..."):
                    sampled_batch = engine.sample_batch_test_flows(
                        attack_name=target_attack_for_eval,
                        total_samples=num_samples,
                        attack_ratio=att_ratio,
                        include_other_attacks=inc_other,
                        random_seed=int(time.time() % 10000)
                    )
                    st.session_state["batch_df"] = sampled_batch
                    st.session_state["batch_target_attack"] = target_attack_for_eval

        with col_run_b2:
            if st.session_state.get("batch_df") is not None:
                if st.button("🗑️ Xóa Dữ Liệu Batch", width="stretch"):
                    st.session_state["batch_df"] = None
                    st.session_state["batch_target_attack"] = "DoS"
                    st.rerun()

    else:
        uploaded_file = st.file_uploader("Chọn file CSV chứa các đặc trưng NetFlow:", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.session_state["batch_df"] = uploaded_df
                st.success(f"Đã đọc file CSV thành công: {len(uploaded_df):,} dòng.")
                if "Attack" in uploaded_df.columns:
                    avail_attacks = [
                        str(a) for a in uploaded_df["Attack"].dropna().unique()
                        if str(a).strip().lower() != "benign"
                    ]
                    if avail_attacks:
                        target_attack_for_eval = st.selectbox("Chọn loại tấn công mục tiêu trong file CSV:", avail_attacks)
                    else:
                        target_attack_for_eval = st.selectbox(
                            "File CSV chỉ chứa lưu lượng Benign — Chọn bộ dò tấn công cần kiểm thử:",
                            CANONICAL_ATTACKS,
                            index=0
                        )
                    st.session_state["batch_target_attack"] = target_attack_for_eval
                else:
                    target_attack_for_eval = st.selectbox(
                        "Chọn bài toán Tấn công cần quét trên file CSV:",
                        CANONICAL_ATTACKS,
                        index=0
                    )
                    st.session_state["batch_target_attack"] = target_attack_for_eval
            except Exception as e:
                st.error(f"Lỗi đọc file CSV: {e}")

    # ==========================================================================
    # XỬ LÝ QUÉT VÀ HIỂN THỊ KẾT QUẢ TỪ SESSION_STATE
    # ==========================================================================
    active_batch_df = st.session_state.get("batch_df")
    active_target_attack = st.session_state.get("batch_target_attack") or target_attack_for_eval or "DoS"

    if active_batch_df is not None:
        with st.spinner("Đang chạy suy luận ma trận trên lõi NIDS..."):
            res_df, summary = engine.predict_batch_flows(
                df=active_batch_df,
                attack_name=active_target_attack,
                model_family=batch_model_family,
                condition=cond_key,
                threshold_mode=active_mode_key,
                custom_threshold=custom_threshold_val
            )

        if summary.get("missing_model_features"):
            st.warning(
                f"⚠️ File dữ liệu thiếu {len(summary['missing_model_features'])} cột đặc trưng cần thiết cho mô hình "
                f"(`{', '.join(summary['missing_model_features'])}`). Hệ thống đã tạm điền giá trị `0` để không gián đoạn quét, "
                f"nhưng kết quả dự đoán có thể bị sai lệch!"
            )
        elif summary.get("missing_features"):
            st.info(
                f"ℹ️ File dữ liệu thiếu cột NetFlow phụ (`{', '.join(summary['missing_features'])}`), "
                f"nhưng mô hình hiện tại không sử dụng các cột này nên kết quả vẫn chính xác 100%."
            )

        st.markdown("### 📈 **Kết Quả Phân Tích & Thống Kê Tổng Hợp**")

        # 4 Thẻ KPI vận hành
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Tổng Số Luồng Đã Quét", f"{summary['total_flows']:,} luồng")
        kpi2.metric("Số Luồng Chặn Ngay (Tầng 1)", f"{summary['attack_count']:,} luồng", delta="Chặn Cứng", delta_color="inverse")
        kpi3.metric("Số Luồng Nghi Vấn (Tầng 2 DPI)", f"{summary['suspicious_count']:,} luồng", delta="Near-Miss")
        kpi4.metric("Tốc Độ Suy Luận", f"{summary['throughput_flows_per_sec']:,.0f} luồng/s", delta=f"{summary['batch_inference_time_sec']:.3f}s")

        # Nếu có nhãn thực nghiệm thì hiển thị tiếp Thẻ Đánh giá & Ma trận
        if "metrics" in summary:
            m = summary["metrics"]
            st.markdown("#### 🎯 **Chỉ Số Thực Nghiệm (Ground-Truth Performance):**")
            
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Accuracy (Độ chính xác)", f"{m['Accuracy']*100:.2f}%")
            m_c2.metric("Precision (Độ tin cậy)", f"{m['Precision']*100:.2f}%")
            m_c3.metric("Recall (Độ nhạy bắt trúng)", f"{m['Recall']*100:.2f}%")
            m_c4.metric("F1-Score (Cân bằng)", f"{m['F1']*100:.2f}%")

            # Ma trận nhầm lẫn
            st.markdown(
                f"""
                <div style="background:#1A202C; padding:15px; border-radius:8px; border:1px solid #2D3748; margin-bottom:15px;">
                    <b>Bảng Phân Bố Quyết Định (Confusion Matrix):</b> &nbsp;
                    <span style="color:#48BB78;"><b>True Positives (Bắt đúng TP):</b> {m['TP']}</span> | 
                    <span style="color:#F56565;"><b>False Negatives (Bỏ sót FN):</b> {m['FN']}</span> | 
                    <span style="color:#ED8936;"><b>False Positives (Báo sai FP):</b> {m['FP']}</span> | 
                    <span style="color:#4299E1;"><b>True Negatives (Sạch đúng TN):</b> {m['TN']}</span>
                </div>
                """, unsafe_allow_html=True
            )

            if "cross_attack_total" in summary and summary["cross_attack_total"] > 0:
                st.info(f"🌐 **Khả Năng Phòng Thủ Liên Tấn Công (Cross-Attack Defense):** Mô hình đã phát hiện **{summary['cross_attack_detected']} / {summary['cross_attack_total']}** luồng thuộc các loại tấn công khác!")

        # Bảng chi tiết luồng
        st.markdown("#### 📋 **Danh Sách Dữ Liệu Luồng Đã Phân Tích:**")
        
        filter_col, download_col = st.columns([2, 1])
        with filter_col:
            view_filter = st.selectbox(
                "Lọc dữ liệu hiển thị:",
                options=["Tất cả luồng", "Chỉ xem ATTACK (Tầng 1)", "Chỉ xem SUSPICIOUS (Tầng 2 DPI)", "Chỉ xem FALSE NEGATIVES (Bị bỏ sót)", "Chỉ xem FALSE POSITIVES (Báo động giả)"],
                index=0
            )

        # Lọc dữ liệu theo lựa chọn (hỗ trợ cả cột Attack dạng chuỗi lẫn cột Label nhị phân 0/1)
        filtered_res = res_df.copy()
        target_att_eval = str(summary.get("target_attack", active_target_attack))
        if view_filter == "Chỉ xem ATTACK (Tầng 1)":
            filtered_res = filtered_res[filtered_res["Predicted_Label"] == 1]
        elif view_filter == "Chỉ xem SUSPICIOUS (Tầng 2 DPI)":
            filtered_res = filtered_res[filtered_res["Risk_Zone"].str.contains("SUSPICIOUS")]
        elif view_filter == "Chỉ xem FALSE NEGATIVES (Bị bỏ sót)":
            if "Attack" in filtered_res.columns:
                att_col = filtered_res["Attack"].fillna("Benign").astype(str)
                filtered_res = filtered_res[(att_col.str.lower() == target_att_eval.lower()) & (filtered_res["Predicted_Label"] == 0)]
            elif "Label" in filtered_res.columns:
                filtered_res = filtered_res[(filtered_res["Label"].astype(int) == 1) & (filtered_res["Predicted_Label"] == 0)]
        elif view_filter == "Chỉ xem FALSE POSITIVES (Báo động giả)":
            if "Attack" in filtered_res.columns:
                att_col = filtered_res["Attack"].fillna("Benign").astype(str)
                filtered_res = filtered_res[(att_col.str.lower() == "benign") & (filtered_res["Predicted_Label"] == 1)]
            elif "Label" in filtered_res.columns:
                filtered_res = filtered_res[(filtered_res["Label"].astype(int) == 0) & (filtered_res["Predicted_Label"] == 1)]

        st.dataframe(filtered_res.head(500), width="stretch", height=280)

        with download_col:
            csv_data = filtered_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Tải Kết Quả (CSV)",
                data=csv_data,
                file_name=f"nids_batch_scan_{active_target_attack}_{batch_model_family}.csv",
                mime="text/csv",
                width="stretch"
            )


# ==============================================================================
# TAB 3: TRUNG TÂM QUẢN LÝ MÔ HÌNH & KHOA HỌC (REGISTRY & FIGURES)
# ==============================================================================
with tab3:
    st.markdown("### 📚 **Trung Tâm Quản Lý Mô Hình (Model Registry) & Báo Cáo Nghiên Cứu**")

    sub_t1, sub_t2 = st.tabs(["🏛️ Auto-Discovered Model Registry", "🖼️ Thư Viện 13 Biểu Đồ 300 DPI & Báo Cáo"])

    # --------------------------------------------------------------------------
    # SUB-TAB 1: MODEL REGISTRY
    # --------------------------------------------------------------------------
    with sub_t1:
        st.markdown("#### **Bộ Đăng Ký Mô Hình Tự Động (Plug-and-Play Model Registry)**")
        st.caption("Kiến trúc linh hoạt: Khi huấn luyện mô hình mới, chỉ cần thả file .joblib và metadata.json vào results/*/models/ là hệ thống tự nạp ngay!")

        if st.button("🔄 Quét lại Thư mục Models (Reload Registry)"):
            with st.spinner("Đang xóa cache RAM và quét lại đĩa cứng..."):
                engine.reload_registry()
                st.toast(f"Đã làm mới Registry & Cache! Tìm thấy {len(engine.models_registry)} mô hình.")

        reg_summary = engine.get_registered_models_summary()
        st.dataframe(reg_summary, width="stretch", height=350)

        st.markdown("---")
        st.markdown("#### **Chi Tiết Siêu Tham Số & Bảng Ngưỡng Từng Mô Hình:**")
        selected_model_inspect = st.selectbox("Chọn mô hình để xem chi tiết:", list(engine.models_registry.keys()))
        
        if selected_model_inspect:
            info = engine.models_registry[selected_model_inspect]
            c_meta, c_thresh = st.columns([1, 1])

            with c_meta:
                st.markdown(f"**Thông tin Cơ bản `{selected_model_inspect}`:**")
                st.json({
                    "Họ Model": info["model_family"],
                    "Tấn Công Mục Tiêu": info["attack"],
                    "Điều Kiện": info["condition_label"],
                    "Số Đặc Trưng k": info["k_features"],
                    "Danh Sách Đặc Trưng (selected_features)": info["selected_features"],
                    "Chỉ Số Thực Nghiệm": info["metrics"]
                })

            with c_thresh:
                st.markdown("**Bảng Ngưỡng Quyết Định Tối Ưu Hóa (Threshold Map):**")
                st.json(info["thresholds"])

    # --------------------------------------------------------------------------
    # SUB-TAB 2: THƯ VIỆN BIỂU ĐỒ 300 DPI
    # --------------------------------------------------------------------------
    with sub_t2:
        st.markdown("#### **Thư Viện 13 Biểu Đồ Khoa Học Chuẩn Xuất Bản (300 DPI)**")

        figures_dict = {
            "fig01_class_imbalance.png": "Hình 1: Phân bố lớp & Mất cân bằng dữ liệu NF-UNSW-NB15",
            "fig02_feature_distributions.png": "Hình 2: Phân phối 6 đặc trưng lưu lượng tiêu biểu theo từng loại tấn công",
            "fig03_correlation_heatmap.png": "Hình 3: Ma trận tương quan Pearson giữa 10 đặc trưng",
            "fig04_mutual_information_heatmap.png": "Hình 4: Xếp hạng đặc trưng theo Mutual Information (MI)",
            "fig05_train_val_test_consistency.png": "Hình 5: Kiểm định độ nhất quán phân bố Train / Validation / Test",
            "fig06_rf_smote_vs_nosmote_comparison.png": "Hình 6: So sánh hiệu năng Random Forest Without SMOTE vs With SMOTE",
            "fig07_rf_k_sensitivity.png": "Hình 7: Phân tích độ nhạy của số đặc trưng k đối với F1-Score (RF)",
            "fig08_rf_confusion_matrices_table8.png": "Hình 8: Ma trận nhầm lẫn 5 bài toán RF Baseline (Table 8 MDPI)",
            "fig09_rf_vs_gb_comparison.png": "Hình 9: So sánh đối đầu Random Forest vs HistGradientBoosting",
            "fig10_precision_recall_curves_rf_vs_gb.png": "Hình 10: Đường cong Precision-Recall (RF vs GB trên 5 bài toán)",
            "fig11_recall_threshold_tradeoff.png": "Hình 11: Tối ưu hóa ngưỡng theo Recall mục tiêu (95%, 98%, 99%)",
            "fig12_cost_sensitive_analysis.png": "Hình 12: Phân tích Chi phí Bất đối xứng Cost-Sensitive (c=1, 5, 10, 20)",
            "fig13_false_negative_forensics.png": "Hình 13: Giải phẫu đặc trưng các mẫu bị bỏ sót (False Negative Forensics)"
        }

        fig_choice = st.selectbox(
            "Chọn Biểu Đồ Khoa Học Cần Xem:",
            options=list(figures_dict.keys()),
            format_func=lambda x: f"{x} - {figures_dict[x]}"
        )

        fig_path = FIGURES_DIR / fig_choice
        if fig_path.exists():
            st.image(str(fig_path), caption=figures_dict[fig_choice], width="stretch")
        else:
            st.warning(f"Không tìm thấy file ảnh tại: {fig_path}")

        st.markdown("---")
        st.markdown("#### 📑 **Đọc Báo Cáo Nghiên Cứu & Phát Hiện Khoa Học:**")
        
        rep_choice = st.radio(
            "Chọn Báo cáo:",
            ["Báo cáo Phân tích Khám phá Dữ liệu & Baseline (EDA & Baseline Findings)", "Báo cáo Giải phẫu Bỏ sót FN & Tối ưu Chi phí SOC (FN & Cost Findings)"]
        )
        
        if "EDA & Baseline" in rep_choice:
            report_file = RESULTS_DIR / "baseline" / "eda_and_baseline_findings.md"
        else:
            report_file = RESULTS_DIR / "improved" / "fn_and_cost_findings.md"

        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as rf:
                md_content = rf.read()
            with st.expander("📖 Nhấn vào đây để xem toàn bộ nội dung báo cáo nghiên cứu", expanded=False):
                st.markdown(md_content)
        else:
            st.info(f"Chưa có file báo cáo tại {report_file}")
