#!/usr/bin/env python3
"""
app.py - Enterprise SOC NIDS Live Console (NF-UNSW-NB15)
Giao diện Giám sát & Phản ứng An ninh Mạng Đa Mô hình Thông minh.
Hỗ trợ Kiến trúc Phòng thủ 2 Tầng (2-Tier Hybrid NIDS), Giải mã Giao thức Mạng Thời gian thực,
Đấu trường Đối đầu Trực tiếp (A/B Showdown: Paper vs Ours) và Quét Lưu lượng Hàng loạt.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Thiết lập đường dẫn dự án
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
# BẢNG ÁNH XẠ GIẢI MÃ GIAO THỨC MẠNG & THÔNG SỐ NETFLOW
# ==============================================================================
L4_PROTO_MAP = {
    1: {"name": "ICMP", "desc": "Internet Control Message (Ping/Traceroute)", "color": "#EC4899"},
    6: {"name": "TCP", "desc": "Transmission Control Protocol (Kết nối tin cậy L4)", "color": "#3B82F6"},
    17: {"name": "UDP", "desc": "User Datagram Protocol (Truyền tải phi kết nối nhanh)", "color": "#8B5CF6"},
    47: {"name": "GRE", "desc": "Generic Routing Encapsulation (VPN Tunnel)", "color": "#10B981"},
    50: {"name": "ESP", "desc": "IPsec Encapsulating Security Payload", "color": "#F59E0B"}
}

L7_PROTO_MAP = {
    0.0: {"name": "Unknown / L4 Raw", "desc": "Lưu lượng L4 thuần túy (Chưa định danh L7)", "category": "Transport"},
    3.0: {"name": "SMTP", "desc": "Simple Mail Transfer Protocol (Thư điện tử)", "category": "Email"},
    5.0: {"name": "DNS", "desc": "Domain Name System (Phân giải tên miền)", "category": "Network Core"},
    7.0: {"name": "HTTP", "desc": "HyperText Transfer Protocol (Lưu lượng Web)", "category": "Web Service"},
    11.0: {"name": "FTP", "desc": "File Transfer Protocol (Truyền tập tin)", "category": "File Transfer"},
    12.0: {"name": "SSH", "desc": "Secure Shell Remote Access (Quản trị từ xa)", "category": "Remote"},
    19.0: {"name": "SSL / TLS", "desc": "Transport Layer Security (Web mã hóa HTTPS)", "category": "Secure Web"},
    20.0: {"name": "MySQL", "desc": "MySQL Database Connection (Truy vấn CSDL)", "category": "Database"}
}

WELL_KNOWN_PORTS = {
    0: "ICMP / System Raw",
    20: "FTP-Data (File Data)",
    21: "FTP (Command Port)",
    22: "SSH (Secure Admin Shell)",
    23: "Telnet (Unencrypted Remote)",
    25: "SMTP (Mail Server)",
    53: "DNS (Name Server)",
    67: "DHCP Server",
    68: "DHCP Client",
    80: "HTTP (Standard Web Server)",
    110: "POP3 (Mail Client)",
    143: "IMAP (Mail Client)",
    443: "HTTPS (Encrypted Web Server)",
    3306: "MySQL Database Port",
    5353: "mDNS Multicast Discovery",
    8080: "HTTP-Proxy / Alt Web Server",
    8443: "HTTPS-Alt Web Service"
}

TCP_FLAG_BITS = [
    (0x01, "FIN", "Đóng kết nối bình thường"),
    (0x02, "SYN", "Bắt tay khởi tạo kết nối (Three-Way Handshake)"),
    (0x04, "RST", "Hủy bỏ/Ngắt kết nối khẩn cấp"),
    (0x08, "PSH", "Đẩy dữ liệu lập tức cho ứng dụng"),
    (0x10, "ACK", "Xác nhận đã nhận gói tin"),
    (0x20, "URG", "Dữ liệu khẩn cấp (Urgent Pointer)"),
    (0x40, "ECE", "Thông báo tắc nghẽn ECN"),
    (0x80, "CWR", "Giảm cửa sổ tắc nghẽn (Congestion Reduced)")
]


def decode_l4_protocol(proto: int) -> Dict[str, str]:
    """Giải mã giao thức tầng truyền vận L4."""
    return L4_PROTO_MAP.get(int(proto), {
        "name": f"IP-Proto {proto}",
        "desc": f"Giao thức mạng IP số hiệu {proto}",
        "color": "#64748B"
    })


def decode_l7_protocol(l7: float) -> Dict[str, str]:
    """Giải mã giao thức tầng ứng dụng L7."""
    return L7_PROTO_MAP.get(float(l7), {
        "name": f"L7 Proto {l7:.1f}",
        "desc": "Giao thức ứng dụng mở rộng / Tùy biến",
        "category": "Custom"
    })


def decode_port(port: int) -> str:
    """Giải mã số hiệu cổng sang tên dịch vụ chuẩn hóa."""
    p = int(port)
    if p in WELL_KNOWN_PORTS:
        return f"{p} ({WELL_KNOWN_PORTS[p]})"
    elif 1 <= p <= 1023:
        return f"{p} (Privileged System Port)"
    elif 1024 <= p <= 49151:
        return f"{p} (Registered Service Port)"
    else:
        return f"{p} (Dynamic / Ephemeral Client Port)"


def decode_tcp_flags(flags: int) -> List[str]:
    """Giải mã bitmask cờ TCP thành danh sách cờ tương ứng."""
    fl = int(flags)
    if fl <= 0:
        return ["NONE (0x00)"]
    active_flags = []
    for bit, name, _ in TCP_FLAG_BITS:
        if fl & bit:
            active_flags.append(name)
    return active_flags if active_flags else [f"Raw (0x{fl:02X})"]


def explain_scenario_context(flow: Dict[str, Any], true_attack: str) -> Dict[str, str]:
    """Tạo lời giải thích ngữ cảnh nghiệp vụ mạng phục vụ thuyết trình chuyên sâu."""
    dst_port = int(flow.get("L4_DST_PORT", 0))
    proto = int(flow.get("PROTOCOL", 6))
    l7 = float(flow.get("L7_PROTO", 0.0))
    flags = int(flow.get("TCP_FLAGS", 0))
    in_bytes = int(flow.get("IN_BYTES", 0))
    out_bytes = int(flow.get("OUT_BYTES", 0))
    duration = int(flow.get("FLOW_DURATION_MILLISECONDS", 0))
    att_name = str(true_attack).strip().lower()

    if att_name == "dos":
        if duration > 1000 and (in_bytes < 1000 or out_bytes < 1000):
            scenario_title = "Kịch bản: Tấn công Slow-Rate DoS (Slowloris / HTTP Keep-Alive Hang)"
            scenario_desc = f"Luồng kết nối kéo dài ({duration:,} ms) đến cổng đích {dst_port} nhưng dung lượng truyền rất nhỏ ({in_bytes:,}B / {out_bytes:,}B). Kẻ tấn công cố tình giữ kết nối mở lâu nhằm cạn kiệt Connection Pool của máy chủ Web."
        else:
            scenario_title = "Kịch bản: Tấn công Từ chối Dịch vụ Phân tán (High-Rate DoS Flood)"
            scenario_desc = f"Lưu lượng DoS dồn dập nhắm vào dịch vụ tại cổng {dst_port}. Mô hình NetFlow theo dõi sự mất cân bằng về tốc độ gửi gói tin và thời gian phiên."
    elif att_name == "exploits":
        scenario_title = "Kịch bản: Thăm dò & Khai thác Lỗ hổng Ứng dụng (Web/App Exploitation)"
        scenario_desc = f"Luồng dữ liệu mang đặc tính thăm dò dịch vụ tại cổng {dst_port} với chuỗi payload nhằm chiếm quyền kiểm soát hoặc thực thi mã từ xa (RCE / Buffer Overflow)."
    elif att_name == "backdoor":
        scenario_title = "Kịch bản: Kênh Kết nối Bí mật Command & Control (C2 Beaconing)"
        scenario_desc = f"Luồng thông tin bất thường kết nối từ cổng nguồn {flow.get('L4_SRC_PORT', 0)} ra bên ngoài, duy trì chu kỳ định kỳ để nhận chỉ thị từ máy chủ điều khiển ngầm."
    elif att_name == "reconnaissance":
        scenario_title = "Kịch bản: Quét Thám mã Dò cổng Mạng (Network & Port Sweep Reconnaissance)"
        scenario_desc = f"Luồng gửi các gói tin thăm dò (SYN Sweep / ACK Probe) tới cổng {dst_port} nhằm lập bản đồ các cổng đang mở và dịch vụ đang lắng nghe trên máy chủ đích."
    elif att_name == "analysis":
        scenario_title = "Kịch bản: Phân tích Cấu trúc & Quét Lỗ hổng Web (Fuzzing / Directory Traversal)"
        scenario_desc = f"Kẻ tấn công gửi các biến thể tham số truy vấn web hoặc thử nghiệm các endpoint lạ để tìm kiếm sơ hở cấu hình và rò rỉ dữ liệu nhạy cảm."
    else:
        scenario_title = "Kịch bản: Lưu lượng Người dùng Hợp lệ (Benign Production Traffic)"
        scenario_desc = f"Phiên truyền thông tiêu chuẩn giữa máy trạm và máy chủ ({decode_port(dst_port)}) với tỷ lệ trao đổi dữ liệu ({in_bytes:,}B In / {out_bytes:,}B Out) phù hợp với hành vi thông thường."

    return {
        "title": scenario_title,
        "description": scenario_desc
    }


# ==============================================================================
# CẤU HÌNH GIAO DIỆN & CUSTOM CSS HIỆN ĐẠI (ENTERPRISE SOC THEME)
# ==============================================================================
st.set_page_config(
    page_title="SOC NIDS Live Console | NF-UNSW-NB15",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme Global */
    .stApp {
        background: #0B0F19;
        color: #E2E8F0;
    }
    
    /* Typography Header */
    .soc-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .soc-sub {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 22px;
    }

    /* Cards & Containers */
    .soc-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    
    /* Phán Quyết Vận Hành (Verdicts) */
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 8px solid;
    }
    .verdict-attack {
        background: rgba(239, 68, 68, 0.12);
        border-color: #EF4444;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.15);
    }
    .verdict-suspicious {
        background: rgba(245, 158, 11, 0.12);
        border-color: #F59E0B;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.15);
    }
    .verdict-benign {
        background: rgba(16, 185, 129, 0.12);
        border-color: #10B981;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.15);
    }

    /* Protocol & Flag Badges */
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .badge-proto {
        background: #1E293B;
        color: #38BDF8;
        border: 1px solid #334155;
    }
    .badge-port {
        background: #1E293B;
        color: #FBBF24;
        border: 1px solid #334155;
    }
    .badge-flag {
        background: #312E81;
        color: #A5B4FC;
        border: 1px solid #4338CA;
    }
    
    /* 2-Tier Pipeline Flow Stepper */
    .stepper-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
        gap: 8px;
    }
    .stepper-node {
        flex: 1;
        min-width: 180px;
        background: #1F2937;
        border-radius: 8px;
        padding: 12px 14px;
        border: 1px solid #374151;
        text-align: center;
    }
    .stepper-active-attack {
        border-color: #EF4444;
        background: rgba(239, 68, 68, 0.2);
    }
    .stepper-active-suspicious {
        border-color: #F59E0B;
        background: rgba(245, 158, 11, 0.2);
    }
    .stepper-active-benign {
        border-color: #10B981;
        background: rgba(16, 185, 129, 0.2);
    }
    .stepper-arrow {
        font-size: 1.3rem;
        color: #64748B;
        font-weight: bold;
        padding: 0 4px;
    }

    /* A/B Showdown Battle Cards */
    .battle-card-paper {
        background: #18181B;
        border: 2px solid #52525B;
        border-radius: 10px;
        padding: 18px;
    }
    .battle-card-ours {
        background: #0F172A;
        border: 2px solid #0EA5E9;
        border-radius: 10px;
        padding: 18px;
        box-shadow: 0 0 16px rgba(14, 165, 233, 0.15);
    }
    .delta-win {
        color: #10B981;
        font-weight: 700;
    }
    .delta-loss {
        color: #EF4444;
        font-weight: 700;
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
    """Đồng bộ mẫu luồng mạng vào session_state và các ô nhập liệu."""
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

if "batch_df" not in st.session_state:
    st.session_state["batch_df"] = None
if "batch_target_attack" not in st.session_state:
    st.session_state["batch_target_attack"] = "DoS"

if "ab_results" not in st.session_state:
    st.session_state["ab_results"] = None


# ==============================================================================
# SIDEBAR - BẢNG ĐIỀU KHIỂN CHÍNH SÁCH SOC LIVE CONSOLE
# ==============================================================================
with st.sidebar:
    st.markdown("## 🛡️ **SOC NIDS CONSOLE**")
    st.markdown("**Kiến trúc Phòng thủ 2 Tầng (2-Tier Hybrid NIDS)**")
    st.caption("NetFlow v1 • NF-UNSW-NB15 Dataset • Pospíchal et al. (2026)")

    # 1. Trạng thái Registry
    num_models = len(engine.models_registry)
    paper_count = sum(1 for m in engine.models_registry.values() if m["model_family"] == "PAPER")
    rf_count = sum(1 for m in engine.models_registry.values() if m["model_family"] == "RF")
    gb_count = sum(1 for m in engine.models_registry.values() if m["model_family"] == "GB")
    
    st.success(
        f"🟢 **{num_models} Mô hình Trực tuyến**\n\n"
        f"• **{paper_count}** Bài báo gốc (Paper-Exact RF)\n\n"
        f"• **{rf_count}** Cải tiến ngẫu nhiên (Ours-RF)\n\n"
        f"• **{gb_count}** Tăng tốc phân tầng (Ours-HistGB)"
    )

    st.markdown("---")
    st.markdown("### ⚙️ **Chính Sách Ngưỡng Quyết Định**")

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
        "Chế độ Ngưỡng Vận Hành:",
        options=threshold_options,
        index=1,  # Mặc định Cost-Sensitive c=10
        help="Điều khiển sự đánh đổi giữa False Negatives (bỏ lọt tấn công) và False Positives (báo động giả)."
    )

    custom_threshold_val = None
    if selected_mode_label == "Tùy chỉnh (Custom Slider)":
        custom_threshold_val = st.slider("Ngưỡng Quyết định:", 0.01, 0.99, 0.50, 0.01)
        active_mode_key = "Custom"
    else:
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

    # Bộ lọc điều kiện huấn luyện
    st.markdown("### 🎯 **Điều Kiện Huấn Luyện**")
    cond_choice = st.radio(
        "Tập Huấn Luyện:",
        options=["Without SMOTE (Chuẩn khuyến nghị)", "With SMOTE (56,000 mẫu)", "Tất cả điều kiện"],
        index=0
    )
    if "Without SMOTE" in cond_choice:
        cond_filter = "nosmote"
    elif "With SMOTE" in cond_choice:
        cond_filter = "smote"
    else:
        cond_filter = "all"

    st.markdown("---")
    st.markdown("""
    <div style="background:#1E293B; padding:12px; border-radius:8px; border:1px solid #334155; font-size:0.85rem;">
        <b>💡 Nguyên lý Kiến trúc 2 Tầng:</b><br>
        • <b>Tầng 1 (Inline Classifier):</b> Mô hình HistGB/RF phân loại siêu tốc (&lt; 1ms).<br>
        • <b>Vùng Nghi Vấn (Near-Miss):</b> Khoảng cách từ 10% đến Ngưỡng Chặn được chuyển giao sang <b>Tầng 2 (Deep Packet Inspection)</b> để soi sâu tải trọng gói tin!
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TIÊU ĐỀ TRANG CHÍNH & BRANDING
# ==============================================================================
st.markdown('<p class="soc-header">🛡️ ENTERPRISE SOC NIDS LIVE CONSOLE</p>', unsafe_allow_html=True)
st.markdown('<p class="soc-sub">Hệ thống Giám sát & Phản ứng An ninh Mạng 2 Tầng Đa Mô hình Thông minh (NF-UNSW-NB15)</p>', unsafe_allow_html=True)


# ==============================================================================
# KHU VỰC 4 TAB CHUYÊN BIỆT
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 TAB 1: KIỂM THỬ TRỰC TIẾP 1 LUỒNG (LIVE SINGLE-FLOW)",
    "⚔️ TAB 2: ĐỐI ĐẦU TRỰC TIẾP (A/B SHOWDOWN: PAPER VS OURS)",
    "📊 TAB 3: QUÉT LƯU LƯỢNG HÀNG LOẠT (BATCH TRAFFIC SCANNER)",
    "📚 TAB 4: TRUNG TÂM MÔ HÌNH & KHOA HỌC (REGISTRY & FINDINGS)"
])


# ==============================================================================
# TAB 1: KIỂM THỬ TRỰC TIẾP 1 LUỒNG MẠNG (LIVE SINGLE-FLOW & PROTOCOL DECODER)
# ==============================================================================
with tab1:
    st.markdown("### ⚡ **1. Nạp Nhanh Mẫu Thực Tế Từ Tập Kiểm Thử Độc Lập (1-Click Real Presets)**")
    
    col_target_att, col_p1, col_p2, col_p3, col_p4 = st.columns([1.5, 1.2, 1.3, 1.8, 1.2])

    with col_target_att:
        preset_attack = st.selectbox("Bài toán Tấn công:", CANONICAL_ATTACKS, index=0, key="sb_preset_attack")

    with col_p1:
        if st.button("🟢 Lưu Lượng Sạch", width="stretch", help="Nạp 1 mẫu lưu lượng Benign thực tế từ tập Test."):
            sample = engine.sample_real_test_flow(
                sample_type="benign",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p2:
        if st.button("🔴 Tấn Công Rõ Ràng (TP)", width="stretch", help="Mẫu tấn công có xác suất cao rõ ràng."):
            sample = engine.sample_real_test_flow(
                sample_type="attack_typical",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p3:
        if st.button("🟡 Tấn Công Ẩn Mình (Near-Miss FN)", width="stretch", help="Mẫu tấn công bị bỏ sót ở F1-Optimal nhưng bắt dính ở Cost-Sensitive hoặc Tầng 2 DPI!"):
            sample = engine.sample_real_test_flow(
                sample_type="attack_stealth_nearmiss",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    with col_p4:
        if st.button("🎲 Mẫu Ngẫu Nhiên", width="stretch", help="Rút ngẫu nhiên 1 kịch bản từ tập Test."):
            sample = engine.sample_real_test_flow(
                sample_type="random",
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1))
            )
            _apply_sample_to_state(sample)
            st.rerun()

    # ==========================================================================
    # GIẢI MÃ GIAO THỨC MẠNG & THÔNG TIN NGỮ CẢNH THỜI GIAN THỰC
    # ==========================================================================
    cur_sample = st.session_state.get("current_flow", {})
    proto_info = decode_l4_protocol(cur_sample.get("PROTOCOL", 6))
    l7_info = decode_l7_protocol(cur_sample.get("L7_PROTO", 7.0))
    flags_list = decode_tcp_flags(cur_sample.get("TCP_FLAGS", 0))
    src_port_desc = decode_port(cur_sample.get("L4_SRC_PORT", 0))
    dst_port_desc = decode_port(cur_sample.get("L4_DST_PORT", 0))
    scenario_story = explain_scenario_context(cur_sample, st.session_state.get("true_attack", "Benign"))

    flags_badges_html = " ".join([f'<span class="badge-pill badge-flag">{f}</span>' for f in flags_list])

    st.markdown(f"""
    <div class="soc-card" style="border-left: 6px solid #38BDF8;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <span style="font-size: 1.05rem; font-weight: 700; color: #38BDF8;">🌐 THÔNG SỐ GIAO THỨC ĐÃ GIẢI MÃ (REAL-TIME PROTOCOL DECODER)</span>
                <div style="margin-top: 8px;">
                    <span class="badge-pill badge-proto">Giao Thức L4: {proto_info['name']} ({cur_sample.get('PROTOCOL', 6)})</span>
                    <span class="badge-pill badge-proto">Ứng Dụng L7: {l7_info['name']} ({l7_info['category']})</span>
                    <span class="badge-pill badge-port">Nguồn: {src_port_desc}</span>
                    <span class="badge-pill badge-port">Đích: {dst_port_desc}</span>
                </div>
                <div style="margin-top: 6px;">
                    <span style="font-size:0.85rem; color:#94A3B8; margin-right:8px;"><b>Cờ TCP:</b></span>
                    {flags_badges_html}
                </div>
            </div>
            <div style="text-align: right; min-width: 200px; margin-top: 6px;">
                <span style="font-size: 0.85rem; color: #94A3B8;">Nhãn Thực Tế Ground-Truth:</span><br>
                <span style="font-size: 1.1rem; font-weight: 700; color: {'#EF4444' if st.session_state.get('true_attack') != 'Benign' else '#10B981'};">
                    {st.session_state.get('true_attack', 'Không rõ')}
                </span>
                <div style="font-size:0.8rem; color:#64748B;">{st.session_state.get('scenario_info', '')}</div>
            </div>
        </div>
        <hr style="border-color: #1F2937; margin: 12px 0 8px 0;">
        <div style="font-size: 0.92rem; color: #CBD5E1;">
            <b>{scenario_story['title']}</b><br>
            <span style="color: #94A3B8;">{scenario_story['description']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bộ chọn Phạm vi Mô hình Suy luận
    col_scope, _ = st.columns([2.5, 1])
    with col_scope:
        eval_scope = st.radio(
            "Phạm vi Mô hình Phân tích:",
            options=[
                f"Chỉ đánh giá Mục tiêu tấn công đang chọn ({preset_attack})",
                "Quét toàn diện cả 5 loại tấn công (All 5 Attack Detectors)"
            ],
            index=0,
            horizontal=True,
            help="Tập trung phân tích luồng đối với đúng loại tấn công mục tiêu hoặc quét toàn bộ."
        )
    att_filter_val = preset_attack if "Chỉ đánh giá Mục tiêu" in eval_scope else "all"

    st.markdown("---")
    st.markdown("### 🎛️ **2. Bảng Tinh Chỉnh 10 Thông Số NetFlow (Interactive Flow Editor)**")

    # Bảng nhập 10 thông số NetFlow chuẩn
    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)

    with r1_c1:
        f_src_port = st.number_input("L4_SRC_PORT (Cổng Nguồn)", min_value=0, max_value=65535, key="inp_L4_SRC_PORT")
    with r1_c2:
        f_dst_port = st.number_input("L4_DST_PORT (Cổng Đích)", min_value=0, max_value=65535, key="inp_L4_DST_PORT")
    with r1_c3:
        f_proto = st.number_input("PROTOCOL (Giao Thức L4)", min_value=0, max_value=255, key="inp_PROTOCOL")
    with r1_c4:
        f_l7_proto = st.number_input("L7_PROTO (Giao Thức L7)", format="%.1f", key="inp_L7_PROTO")
    with r1_c5:
        f_in_bytes = st.number_input("IN_BYTES (Bytes Chiều Đến)", min_value=0, key="inp_IN_BYTES")

    with r2_c1:
        f_out_bytes = st.number_input("OUT_BYTES (Bytes Chiều Đi)", min_value=0, key="inp_OUT_BYTES")
    with r2_c2:
        f_in_pkts = st.number_input("IN_PKTS (Gói Tin Chiều Đến)", min_value=0, key="inp_IN_PKTS")
    with r2_c3:
        f_out_pkts = st.number_input("OUT_PKTS (Gói Tin Chiều Đi)", min_value=0, key="inp_OUT_PKTS")
    with r2_c4:
        f_tcp_flags = st.number_input("TCP_FLAGS (Bitmask Cờ TCP)", min_value=0, key="inp_TCP_FLAGS")
    with r2_c5:
        f_flow_dur = st.number_input("FLOW_DURATION_MS (Thời Lượng ms)", min_value=0, key="inp_FLOW_DURATION_MS")

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
        st.markdown("### 🚨 **3. Phán Quyết Vận Hành Của Kiến Trúc Phòng Thủ 2 Tầng**")

        status = verdict["status"]
        max_p = verdict["max_probability"]
        
        # Thiết lập màu sắc và trạng thái pipeline
        if status == "ATTACK":
            card_class = "verdict-box verdict-attack"
            icon = "🚨"
            badge = "🔴 TẦNG 1: CHẶN CỨNG (INLINE PREVENTION / DROP)"
            step1_cls = "stepper-node"
            step2_cls = "stepper-node stepper-active-attack"
            step3_cls = "stepper-node stepper-active-attack"
            step4_cls = "stepper-node"
            action_text = "Thực hiện DROP gói tin ngay lập tức tại Tầng 1 (Line-rate) để bảo vệ tài nguyên mạng. Không cần tốn chi phí Tầng 2."
        elif status == "SUSPICIOUS":
            card_class = "verdict-box verdict-suspicious"
            icon = "⚠️"
            badge = "🟡 CHUYỂN GIAO TẦNG 2: KIỂM TRA SÂU DPI (NEAR-MISS ESCALATION)"
            step1_cls = "stepper-node"
            step2_cls = "stepper-node stepper-active-suspicious"
            step3_cls = "stepper-node stepper-active-suspicious"
            step4_cls = "stepper-node stepper-active-suspicious"
            action_text = "Luồng rơi vào Vùng Cận Ngưỡng (Near-Miss 10% - Ngưỡng). Tầng 1 chuyển giao sang Tầng 2 để soi tải trọng (Deep Packet Inspection), ngăn chặn triệt để tấn công ẩn mình!"
        else:
            card_class = "verdict-box verdict-benign"
            icon = "✅"
            badge = "🟢 FAST PATH: CHO PHÉP THÔNG QUA (BENIGN FORWARDING)"
            step1_cls = "stepper-node"
            step2_cls = "stepper-node stepper-active-benign"
            step3_cls = "stepper-node stepper-active-benign"
            step4_cls = "stepper-node"
            action_text = "Xác suất rủi ro < 10%. Cho phép luồng dữ liệu chuyển tiếp siêu tốc qua Fast-Path với độ trễ tối thiểu (< 1ms)."

        # Sơ đồ tương tác Pipeline Stepper
        st.markdown(f"""
        <div class="stepper-container">
            <div class="{step1_cls}">
                <div style="font-size:0.8rem; color:#94A3B8;">BƯỚC 1: TIẾP NHẬN</div>
                <b>Thu Thập NetFlow</b><br>
                <span style="font-size:0.8rem; color:#38BDF8;">10 Đặc Trưng L4/L7</span>
            </div>
            <div class="stepper-arrow">➔</div>
            <div class="{step2_cls}">
                <div style="font-size:0.8rem; color:#94A3B8;">BƯỚC 2: TẦNG 1 INLINE</div>
                <b>Bộ Phân Loại Siêu Tốc</b><br>
                <span style="font-size:0.8rem; color:#CBD5E1;">HistGB / RF (&lt; 1ms)</span>
            </div>
            <div class="stepper-arrow">➔</div>
            <div class="{step3_cls}">
                <div style="font-size:0.8rem; color:#94A3B8;">BƯỚC 3: PHÂN VÙNG</div>
                <b>Rủi Ro: {max_p*100:.1f}%</b><br>
                <span style="font-size:0.8rem; color:#CBD5E1;">{status}</span>
            </div>
            <div class="stepper-arrow">➔</div>
            <div class="{step4_cls}">
                <div style="font-size:0.8rem; color:#94A3B8;">BƯỚC 4: HÀNH ĐỘNG SOC</div>
                <b>{badge.split(':')[1].split('(')[0].strip()}</b><br>
                <span style="font-size:0.8rem; color:#CBD5E1;">{badge.split('(')[-1].replace(')', '') if '(' in badge else ''}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Banner Phán Quyết
        st.markdown(f"""
        <div class="{card_class}">
            <h3 style="margin-top:0;">{icon} <b>{verdict['title']}</b> &nbsp; <span style="font-size:0.85rem; padding:4px 10px; border-radius:6px; background:#1A202C;">{badge}</span></h3>
            <p style="font-size: 1.05rem; margin-bottom: 6px;">{verdict['message']}</p>
            <p style="font-size: 0.9rem; color:#A0AEC0; margin-bottom: 6px;"><b>Kích hoạt bởi:</b> <code>{verdict['triggered_by']}</code> | <b>Xác suất rủi ro tối đa:</b> <b>{max_p*100:.1f}%</b></p>
            <div style="background:#111827; padding:8px 12px; border-radius:6px; font-size:0.88rem; border:1px solid #1F2937;">
                <b>Khuyến nghị Vận hành SOC:</b> {action_text}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ======================================================================
        # BIỂU ĐỒ SO SÁNH ĐA MÔ HÌNH VÀ BẢNG CHI TIẾT
        # ======================================================================
        st.markdown("### ⚔️ **4. Đấu Trường Đa Mô Hình (Multi-Model Risk Arena & Threshold Gauges)**")
        col_chart, col_table = st.columns([1.1, 1.4])

        with col_chart:
            st.markdown("**Biểu Đồ Mức Rủi Ro & Ngưỡng Từng Mô Hình:**")
            chart_data = details_df.sort_values(by="Probability", ascending=True).copy()
            
            fig, ax = plt.subplots(figsize=(6, max(3.5, len(chart_data) * 0.45)))
            fig.patch.set_facecolor('#0B0F19')
            ax.set_facecolor('#111827')

            y_pos = np.arange(len(chart_data))
            probs = chart_data["Probability"].values * 100
            thresholds = chart_data["Threshold_Used"].values * 100

            bar_colors = []
            for p, th in zip(probs, thresholds):
                if p >= th:
                    bar_colors.append('#EF4444')
                elif p >= 10.0:
                    bar_colors.append('#F59E0B')
                else:
                    bar_colors.append('#10B981')

            bars = ax.barh(y_pos, probs, color=bar_colors, height=0.6, edgecolor='none')
            
            for i, th in enumerate(thresholds):
                ax.plot([th, th], [i - 0.35, i + 0.35], color='#F87171', linestyle='--', linewidth=2.0)

            ax.axvline(x=10.0, color='#FBBF24', linestyle=':', linewidth=1.2, label='Vùng Nghi vấn (10%)')

            labels = [f"{r['Model_Family']} - {r['Attack']} ({r['Condition']})" for _, r in chart_data.iterrows()]
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, color='#E2E8F0', fontsize=8.5)
            ax.set_xlabel("Xác suất Tấn công P(Attack) (%)", color='#E2E8F0', fontsize=9.5)
            ax.set_xlim(0, 105)
            ax.tick_params(colors='#CBD5E0', labelsize=8.5)
            
            for spine in ax.spines.values():
                spine.set_color('#1F2937')

            for bar, p in zip(bars, probs):
                ax.text(p + 1.5, bar.get_y() + bar.get_height()/2, f"{p:.1f}%", 
                        va='center', ha='left', color='#FFFFFF', fontsize=8, fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_table:
            st.markdown(f"**Bảng Ma Trận Dự Đoán ({len(details_df)} Mô hình Đang Chạy):**")
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
# TAB 2: ĐỐI ĐẦU TRỰC TIẾP (A/B SHOWDOWN: PAPER-EXACT VS OURS 2-TIER NIDS)
# ==============================================================================
with tab2:
    st.markdown("### ⚔️ **Đấu Trường Đối Đầu: Mô Hình Bài Báo Gốc vs Hệ Thống Cải Tiến 2 Tầng**")
    st.caption(
        "Thực nghiệm đối sánh trực tiếp trên cùng một tập dữ liệu kiểm thử độc lập (Held-out Test). "
        "Chứng minh sự vượt trội của Hệ thống Cải tiến (Cost-Sensitive $c=10$ & Near-Miss Escalation) so với Mô hình Bài báo gốc (Pospíchal et al., 2026)."
    )

    ab_c1, ab_c2, ab_c3, ab_c4 = st.columns([1.5, 1.2, 1.3, 1.5])
    with ab_c1:
        ab_attack = st.selectbox("Bài toán Tấn công Thử thách:", CANONICAL_ATTACKS, index=0, key="sb_ab_attack")
    with ab_c2:
        ab_samples = st.slider("Số lượng luồng quét:", min_value=100, max_value=3000, value=500, step=100, key="sl_ab_samples")
    with ab_c3:
        ab_ratio = st.slider("Tỷ lệ tấn công pha trộn:", min_value=0.05, max_value=0.50, value=0.10, step=0.05, key="sl_ab_ratio")
    with ab_c4:
        ab_ours_family = st.selectbox("Thuật toán Hệ thống Ta:", ["GB (HistGradientBoosting - Khuyến nghị)", "RF (Random Forest Cải tiến)"], index=0)
        ours_family_key = "GB" if "HistGradientBoosting" in ab_ours_family else "RF"

    col_btn_ab, col_clear_ab = st.columns([3, 1])
    with col_btn_ab:
        if st.button("⚔️ BẮT ĐẦU ĐỐI ĐẦU THỰC NGHIỆM TRỰC TIẾP", type="primary", width="stretch"):
            with st.spinner("Đang nạp dữ liệu kiểm thử độc lập và thực thi song song 2 hệ thống..."):
                sampled_ab_df = engine.sample_batch_test_flows(
                    attack_name=ab_attack,
                    total_samples=ab_samples,
                    attack_ratio=ab_ratio,
                    include_other_attacks=False,
                    random_seed=int(time.time() % 10000)
                )

                # Chạy mô hình Bài báo gốc: Paper-Exact RF, F1-Optimal
                res_paper, sum_paper = engine.predict_batch_flows(
                    df=sampled_ab_df,
                    attack_name=ab_attack,
                    model_family="PAPER",
                    condition="nosmote",
                    threshold_mode="F1-Optimal"
                )

                # Chạy Hệ thống Cải tiến của chúng ta: HistGB / RF, Cost-Sensitive c=10
                res_ours, sum_ours = engine.predict_batch_flows(
                    df=sampled_ab_df,
                    attack_name=ab_attack,
                    model_family=ours_family_key,
                    condition="nosmote",
                    threshold_mode="Cost-Sensitive (c=10)"
                )

                st.session_state["ab_results"] = {
                    "batch_df": sampled_ab_df,
                    "target_attack": ab_attack,
                    "paper": {"res": res_paper, "sum": sum_paper},
                    "ours": {"res": res_ours, "sum": sum_ours, "family": ours_family_key}
                }

    with col_clear_ab:
        if st.session_state.get("ab_results") is not None:
            if st.button("🗑️ Xóa Kết Quả A/B", width="stretch"):
                st.session_state["ab_results"] = None
                st.rerun()

    # ==========================================================================
    # HIỂN THỊ KẾT QUẢ ĐỐI ĐẦU A/B SIDE-BY-SIDE
    # ==========================================================================
    ab_data = st.session_state.get("ab_results")
    if ab_data is not None:
        p_sum = ab_data["paper"]["sum"]
        o_sum = ab_data["ours"]["sum"]
        p_m = p_sum.get("metrics", {})
        o_m = o_sum.get("metrics", {})

        # Tính toán chi phí rủi ro SOC (c = 10)
        c_val = 10
        p_cost = c_val * p_m.get("FN", 0) + p_m.get("FP", 0)
        o_cost = c_val * o_m.get("FN", 0) + o_m.get("FP", 0)
        cost_diff = p_cost - o_cost
        cost_savings_pct = (cost_diff / p_cost * 100) if p_cost > 0 else 0.0

        fn_paper = p_m.get("FN", 0)
        fn_ours = o_m.get("FN", 0)
        fn_diff = fn_paper - fn_ours
        fn_red_pct = (fn_diff / fn_paper * 100) if fn_paper > 0 else 0.0

        speedup = (o_sum["throughput_flows_per_sec"] / p_sum["throughput_flows_per_sec"]) if p_sum["throughput_flows_per_sec"] > 0 else 1.0

        st.markdown("---")
        st.markdown("### 🏆 **Bảng Tổng Kết Hiệu Năng Đối Đầu Thực Nghiệm**")

        # 3 Thẻ Đột Phá Lớn
        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Số Cuộc Tấn Công Bỏ Sót (FN)",
            f"{fn_ours} vụ lọt lưới",
            delta=f"Giảm {fn_diff} vụ (-{fn_red_pct:.1f}%)" if fn_diff >= 0 else f"Tăng {abs(fn_diff)} vụ",
            delta_color="normal" if fn_diff >= 0 else "inverse"
        )
        b2.metric(
            "Chi Phí Rủi Ro SOC (c = 10)",
            f"{o_cost:,.0f} điểm phạt",
            delta=f"Tiết kiệm {cost_savings_pct:.1f}% chi phí rủi ro" if cost_savings_pct >= 0 else f"+{abs(cost_savings_pct):.1f}%",
            delta_color="normal" if cost_savings_pct >= 0 else "inverse"
        )
        b3.metric(
            "Tốc Độ Xử Lý Lưu Lượng",
            f"{o_sum['throughput_flows_per_sec']:,.0f} luồng/s",
            delta=f"Nhanh gấp {speedup:.1f}x lần" if speedup >= 1.0 else f"{speedup:.1f}x",
            delta_color="normal"
        )

        st.markdown("---")
        col_ab_left, col_ab_right = st.columns(2)

        with col_ab_left:
            st.markdown(f"""
            <div class="battle-card-paper">
                <h4 style="color:#A1A1AA; margin-top:0;">📜 <b>MÔ HÌNH BÀI BÁO GỐC (POSPÍCHAL ET AL.)</b></h4>
                <div style="font-size:0.88rem; color:#71717A; margin-bottom:12px;">
                    Thuật toán: <b>Random Forest Baseline</b> | Ngưỡng: <b>F1-Optimal ({p_sum.get('threshold_used', 0.5):.3f})</b>
                </div>
                <table style="width:100%; font-size:0.92rem; border-collapse:collapse;">
                    <tr><td style="padding:4px 0;">Bỏ sót tấn công (False Negatives - FN):</td><td style="text-align:right; color:#EF4444; font-weight:700;">{fn_paper} luồng</td></tr>
                    <tr><td style="padding:4px 0;">Báo động giả (False Positives - FP):</td><td style="text-align:right; color:#F59E0B; font-weight:700;">{p_m.get('FP', 0)} luồng</td></tr>
                    <tr><td style="padding:4px 0;">Tỷ lệ bắt trúng (Recall):</td><td style="text-align:right; font-weight:700;">{p_m.get('Recall', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:4px 0;">Độ tin cậy (Precision):</td><td style="text-align:right; font-weight:700;">{p_m.get('Precision', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:4px 0;">F1-Score:</td><td style="text-align:right; font-weight:700;">{p_m.get('F1', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:4px 0;">Tốc độ thực thi:</td><td style="text-align:right; font-weight:700;">{p_sum.get('throughput_flows_per_sec', 0):,.0f} luồng/s</td></tr>
                    <tr style="border-top:1px solid #3F3F46;"><td style="padding:8px 0; font-weight:bold;">Tổng Chi Phí Rủi Ro SOC:</td><td style="text-align:right; font-weight:bold; color:#EF4444;">{p_cost:,.0f}</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_ab_right:
            st.markdown(f"""
            <div class="battle-card-ours">
                <h4 style="color:#38BDF8; margin-top:0;">🛡️ <b>HỆ THỐNG CẢI TIẾN CỦA CHÚNG TA (OURS 2-TIER NIDS)</b></h4>
                <div style="font-size:0.88rem; color:#94A3B8; margin-bottom:12px;">
                    Thuật toán: <b>{ab_data['ours']['family']} + 2-Tier Pipeline</b> | Ngưỡng: <b>Cost-Sensitive c=10 ({o_sum.get('threshold_used', 0.5):.3f})</b>
                </div>
                <table style="width:100%; font-size:0.92rem; border-collapse:collapse;">
                    <tr><td style="padding:4px 0;">Bỏ sót tấn công (False Negatives - FN):</td><td style="text-align:right; color:#10B981; font-weight:700;">{fn_ours} luồng ({'-' if fn_diff >= 0 else '+'}{abs(fn_diff)})</td></tr>
                    <tr><td style="padding:4px 0;">Báo động giả (False Positives - FP):</td><td style="text-align:right; color:#F59E0B; font-weight:700;">{o_m.get('FP', 0)} luồng</td></tr>
                    <tr><td style="padding:4px 0;">Tỷ lệ bắt trúng (Recall):</td><td style="text-align:right; color:#10B981; font-weight:700;">{o_m.get('Recall', 0)*100:.2f}% ({'+' if (o_m.get('Recall', 0)-p_m.get('Recall', 0))>=0 else ''}{(o_m.get('Recall', 0)-p_m.get('Recall', 0))*100:.2f}%)</td></tr>
                    <tr><td style="padding:4px 0;">Độ tin cậy (Precision):</td><td style="text-align:right; font-weight:700;">{o_m.get('Precision', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:4px 0;">F1-Score:</td><td style="text-align:right; font-weight:700;">{o_m.get('F1', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:4px 0;">Tốc độ thực thi:</td><td style="text-align:right; color:#38BDF8; font-weight:700;">{o_sum.get('throughput_flows_per_sec', 0):,.0f} luồng/s ({speedup:.1f}x)</td></tr>
                    <tr style="border-top:1px solid #1E3A8A;"><td style="padding:8px 0; font-weight:bold;">Tổng Chi Phí Rủi Ro SOC:</td><td style="text-align:right; font-weight:bold; color:#10B981;">{o_cost:,.0f} (Tiết kiệm {cost_savings_pct:.1f}%)</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        # ======================================================================
        # DANH SÁCH CÁC MẪU BÀI BÁO GỐC BỎ SÓT NHƯNG HỆ THỐNG TA BẮT DÍNH
        # ======================================================================
        st.markdown("---")
        st.markdown("#### 🎯 **Giải Phẫu Mẫu Tấn Công Lọt Lưới (Stealth Attacks Missed by Paper but Caught by Ours)**")
        st.caption("Danh sách các luồng tấn công thực tế bị mô hình bài báo gốc gán nhãn Benign (FN), nhưng Hệ thống 2 Tầng của ta đã phát hiện hoặc chuyển giao sang Tầng 2 DPI thành công:")

        batch_raw = ab_data["batch_df"]
        res_p = ab_data["paper"]["res"]
        res_o = ab_data["ours"]["res"]

        # Điều kiện: Nhãn thực tế là tấn công, Paper dự đoán 0 (bỏ sót), nhưng Ours dự đoán 1 hoặc đưa vào Vùng Nghi Vấn (DPI)
        is_attack_groundtruth = (batch_raw["Label"] == 1) if "Label" in batch_raw.columns else (batch_raw["Attack"].astype(str).str.lower() == ab_attack.lower())
        mask_caught_by_ours = is_attack_groundtruth & (res_p["Predicted_Label"] == 0) & ((res_o["Predicted_Label"] == 1) | (res_o["Risk_Zone"].str.contains("SUSPICIOUS")))

        stealth_caught_df = batch_raw[mask_caught_by_ours].copy()
        
        if len(stealth_caught_df) > 0:
            stealth_caught_df["Paper_Probability"] = res_p.loc[mask_caught_by_ours, "Probability"].apply(lambda x: f"{x*100:.1f}%")
            stealth_caught_df["Ours_Probability"] = res_o.loc[mask_caught_by_ours, "Probability"].apply(lambda x: f"{x*100:.1f}%")
            stealth_caught_df["Ours_Action"] = res_o.loc[mask_caught_by_ours, "Risk_Zone"]

            cols_show = ["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "IN_BYTES", "OUT_BYTES", "TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "Paper_Probability", "Ours_Probability", "Ours_Action"]
            existing_cols_show = [c for c in cols_show if c in stealth_caught_df.columns]
            
            st.dataframe(stealth_caught_df[existing_cols_show].head(100), width="stretch", height=240)
            st.success(f"🎉 **Minh chứng Thực nghiệm:** Có **{len(stealth_caught_df)}** luồng tấn công ẩn mình mà Bài báo gốc bỏ lọt 100%, nhưng Hệ thống của ta đã ngăn chặn hoặc soi DPI thành công!")
        else:
            st.info("Trong đợt quét ngẫu nhiên này, mô hình bài báo gốc không bỏ sót mẫu nào đặc biệt.")


# ==============================================================================
# TAB 3: QUÉT LƯU LƯỢNG HÀNG LOẠT (BATCH TRAFFIC SCANNER & SOC CONSOLE)
# ==============================================================================
with tab3:
    st.markdown("### 📊 **Quét Lưu Lượng Mạng Hàng Loạt & Bảng Chỉ Huy Sự Cố SOC**")
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
            avail_families = ["GB", "RF", "PAPER"]
            batch_model_family = st.selectbox("Thuật toán:", avail_families, index=0)
        with col_m2:
            batch_condition = st.selectbox("Điều kiện Huấn luyện:", ["Without SMOTE", "With SMOTE"], index=0)
            cond_key = "nosmote" if batch_condition == "Without SMOTE" else "smote"

    st.markdown("---")

    target_attack_for_eval = "DoS"

    if data_source_mode == "Rút mẫu từ Held-out Test Parquet (1-Click Preset)":
        b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with b_c1:
            target_attack_for_eval = st.selectbox("Bài toán Tấn công:", CANONICAL_ATTACKS, index=0, key="sb_batch_eval_att")
        with b_c2:
            num_samples = st.slider("Số lượng luồng quét:", min_value=100, max_value=5000, value=500, step=100, key="sl_batch_samples")
        with b_c3:
            att_ratio = st.slider("Tỷ lệ tấn công pha trộn:", min_value=0.05, max_value=0.50, value=0.10, step=0.05, key="sl_batch_ratio")
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

        st.markdown("### 📈 **Kết Quả Phân Tích & Thống Kê Tổng Hợp**")

        # 4 Thẻ KPI vận hành
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Tổng Số Luồng Đã Quét", f"{summary['total_flows']:,} luồng")
        kpi2.metric("Số Luồng Chặn Ngay (Tầng 1)", f"{summary['attack_count']:,} luồng", delta="Chặn Cứng", delta_color="inverse")
        kpi3.metric("Số Luồng Nghi Vấn (Tầng 2 DPI)", f"{summary['suspicious_count']:,} luồng", delta="Near-Miss")
        kpi4.metric("Tốc Độ Suy Luận", f"{summary['throughput_flows_per_sec']:,.0f} luồng/s", delta=f"{summary['batch_inference_time_sec']:.3f}s")

        # Phân bố Mức độ Nghiêm trọng Sự cố (SOC Incident Severity Breakdown)
        crit_count = summary["attack_count"]
        med_count = summary["suspicious_count"]
        low_count = summary["benign_count"]
        
        st.markdown("#### 🚨 **Phân Bố Cảnh Báo An Ninh Theo Mức Độ Nghiêm Trọng (Incident Severity):**")
        s_c1, s_c2, s_c3 = st.columns(3)
        s_c1.markdown(f"""
        <div style="background:rgba(239,68,68,0.15); border:1px solid #EF4444; border-radius:8px; padding:12px; text-align:center;">
            <span style="color:#EF4444; font-weight:700; font-size:1.1rem;">🔴 CRITICAL (NGHIÊM TRỌNG)</span><br>
            <b style="font-size:1.4rem;">{crit_count:,}</b> luồng<br>
            <span style="font-size:0.8rem; color:#94A3B8;">Tầng 1 Inline Drop lập tức</span>
        </div>
        """, unsafe_allow_html=True)
        s_c2.markdown(f"""
        <div style="background:rgba(245,158,11,0.15); border:1px solid #F59E0B; border-radius:8px; padding:12px; text-align:center;">
            <span style="color:#F59E0B; font-weight:700; font-size:1.1rem;">🟡 MEDIUM (NGHI VẤN / NEAR-MISS)</span><br>
            <b style="font-size:1.4rem;">{med_count:,}</b> luồng<br>
            <span style="font-size:0.8rem; color:#94A3B8;">Chuyển giao Tầng 2 Soi DPI Payload</span>
        </div>
        """, unsafe_allow_html=True)
        s_c3.markdown(f"""
        <div style="background:rgba(16,185,129,0.15); border:1px solid #10B981; border-radius:8px; padding:12px; text-align:center;">
            <span style="color:#10B981; font-weight:700; font-size:1.1rem;">🟢 LOW (LƯU LƯỢNG SẠCH)</span><br>
            <b style="font-size:1.4rem;">{low_count:,}</b> luồng<br>
            <span style="font-size:0.8rem; color:#94A3B8;">Cho phép thông qua Fast-Path</span>
        </div>
        """, unsafe_allow_html=True)

        # Phân tích Cổng Dịch vụ Bị Tấn công Nhiều nhất (Top Targeted Ports)
        if "L4_DST_PORT" in res_df.columns:
            attack_flows_df = res_df[res_df["Predicted_Label"] == 1]
            if len(attack_flows_df) > 0:
                top_ports = attack_flows_df["L4_DST_PORT"].value_counts().head(5)
                top_ports_str = " | ".join([f"<b>{decode_port(p)}:</b> {cnt:,} đợt" for p, cnt in top_ports.items()])
                st.markdown(f"""
                <div style="background:#111827; padding:10px 14px; border-radius:8px; border:1px solid #1F2937; margin: 12px 0; font-size:0.9rem;">
                    🎯 <b>Top Cổng Dịch Vụ Bị Tấn Công Nhắm Đến Nhiều Nhất:</b> &nbsp; {top_ports_str}
                </div>
                """, unsafe_allow_html=True)

        # Khuyến nghị Vận hành SOC Tự động (Actionable Recommendations)
        rec_actions = []
        if crit_count > 0:
            rec_actions.append(f"Kích hoạt quy tắc Firewall Chặn Cứng (Drop Rule) cho {crit_count} luồng có nguy cơ cao.")
        if med_count > 0:
            rec_actions.append(f"Chuyển {med_count} luồng nghi vấn Near-Miss sang hệ thống Zeek/Snort DPI để phân tích gói tin sâu.")
        if not rec_actions:
            rec_actions.append("Lưu lượng mạng sạch, không phát hiện dấu hiệu xâm nhập nguy hiểm.")

        st.markdown(f"""
        <div style="background:#0F172A; border-left:6px solid #38BDF8; padding:12px 16px; border-radius:6px; margin-bottom:16px;">
            <b>🛡️ Kiến Nghị Vận Hành Phản Ứng Nhanh Của SOC:</b><br>
            {"<br>".join([f"• {act}" for act in rec_actions])}
        </div>
        """, unsafe_allow_html=True)

        # Nếu có nhãn thực nghiệm thì hiển thị Thẻ Đánh giá & Ma trận
        if "metrics" in summary:
            m = summary["metrics"]
            st.markdown("#### 🎯 **Chỉ Số Thực Nghiệm (Ground-Truth Performance):**")
            
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("Accuracy (Độ chính xác)", f"{m['Accuracy']*100:.2f}%")
            m_c2.metric("Precision (Độ tin cậy)", f"{m['Precision']*100:.2f}%")
            m_c3.metric("Recall (Độ nhạy bắt trúng)", f"{m['Recall']*100:.2f}%")
            m_c4.metric("F1-Score (Cân bằng)", f"{m['F1']*100:.2f}%")

            st.markdown(
                f"""
                <div style="background:#111827; padding:14px; border-radius:8px; border:1px solid #1F2937; margin-bottom:15px; font-size:0.92rem;">
                    <b>Bảng Phân Bố Quyết Định (Confusion Matrix):</b> &nbsp;
                    <span style="color:#10B981;"><b>True Positives (Bắt đúng TP):</b> {m['TP']}</span> | 
                    <span style="color:#EF4444;"><b>False Negatives (Bỏ sót FN):</b> {m['FN']}</span> | 
                    <span style="color:#F59E0B;"><b>False Positives (Báo sai FP):</b> {m['FP']}</span> | 
                    <span style="color:#38BDF8;"><b>True Negatives (Sạch đúng TN):</b> {m['TN']}</span>
                </div>
                """, unsafe_allow_html=True
            )

        # Bảng chi tiết luồng
        st.markdown("#### 📋 **Danh Sách Dữ Liệu Luồng Đã Phân Tích:**")
        filter_col, download_col = st.columns([2, 1])
        with filter_col:
            view_filter = st.selectbox(
                "Lọc dữ liệu hiển thị:",
                options=["Tất cả luồng", "Chỉ xem ATTACK (Tầng 1)", "Chỉ xem SUSPICIOUS (Tầng 2 DPI)", "Chỉ xem FALSE NEGATIVES (Bị bỏ sót)", "Chỉ xem FALSE POSITIVES (Báo động giả)"],
                index=0
            )

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
# TAB 4: TRUNG TÂM QUẢN LÝ MÔ HÌNH & KHOA HỌC (REGISTRY & FINDINGS)
# ==============================================================================
with tab4:
    st.markdown("### 📚 **Trung Tâm Quản Lý Mô Hình (Model Registry) & Báo Cáo Nghiên Cứu**")

    sub_t1, sub_t2 = st.tabs(["🏛️ Auto-Discovered Model Registry", "🖼️ Thư Viện 17 Biểu Đồ 300 DPI & Báo Cáo"])

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
        st.markdown("#### **Thư Viện 17 Biểu Đồ Khoa Học Chuẩn Xuất Bản (300 DPI)**")

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
            "fig13_false_negative_forensics.png": "Hình 13: Giải phẫu đặc trưng các mẫu bị bỏ sót (False Negative Forensics)",
            "fig_p01_three_systems_f1_and_speed.png": "Hình P01: Đối sánh F1 và tốc độ huấn luyện của 3 hệ thống (Paper-Exact vs Ours RF vs Ours GB)",
            "fig_p02_smote_paradox_and_leakage_ablation.png": "Hình P02: Tác động của SMOTE và thực nghiệm chống rò rỉ kiểm định chéo (CV Leakage)",
            "fig_p03_pr_curves_three_systems.png": "Hình P03: Đường cong Precision-Recall của 3 hệ thống trên 5 lớp tấn công",
            "fig_p04_soc_cost_and_fn_comparison.png": "Hình P04: Đối sánh chi phí rủi ro SOC (c=10) và số luồng tấn công bị bỏ sót (FN)"
        }

        fig_choice = st.selectbox(
            "Chọn Biểu Đồ Khoa Học Cần Xem:",
            options=list(figures_dict.keys()),
            format_func=lambda x: f"{x} - {figures_dict[x]}"
        )

        fig_path = FIGURES_DIR / fig_choice
        if not fig_path.exists():
            fig_path = RESULTS_DIR / "paper_original" / "figures" / fig_choice
        if fig_path.exists():
            st.image(str(fig_path), caption=figures_dict[fig_choice], width="stretch")
        else:
            st.warning(f"Không tìm thấy file ảnh tại: {fig_path}")

        st.markdown("---")
        st.markdown("#### 📑 **Đọc Báo Cáo Nghiên Cứu & Phát Hiện Khoa Học:**")
        
        rep_choice = st.radio(
            "Chọn Báo cáo Nghiên cứu:",
            [
                "Báo cáo Đối sánh 3 Hệ thống: Paper-Exact vs Ours RF vs Ours GB (Paper vs Ours Findings)",
                "Báo cáo Phân tích Khám phá Dữ liệu & Baseline (EDA & Baseline Findings)",
                "Báo cáo Giải phẫu Bỏ sót FN & Tối ưu Chi phí SOC (FN & Cost Findings)"
            ]
        )
        
        if "Paper vs Ours" in rep_choice:
            report_file = RESULTS_DIR / "paper_original" / "paper_vs_ours_findings.md"
        elif "EDA & Baseline" in rep_choice:
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
