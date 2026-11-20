#!/usr/bin/env python3
"""
styles.py
Cấu hình giao diện tối giản đơn sắc (Monochrome Academic UI) và các hàm dựng thẻ HTML
tái sử dụng cho ứng dụng phân tích NIDS.
"""

from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st

MONOCHROME_CSS = """
<style>
    /* Monochrome Academic Theme */
    .stApp {
        background-color: #FFFFFF;
        color: #111827;
    }

    section[data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 1px solid #E5E7EB;
    }

    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Typography Header */
    .nids-header {
        font-size: 1.75rem;
        font-weight: 700;
        color: #111827;
        letter-spacing: -0.015em;
        margin: 0 0 4px 0;
        line-height: 1.25;
    }
    .nids-sub {
        font-size: 0.95rem;
        color: #4B5563;
        margin: 0 0 18px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid #E5E7EB;
    }

    /* Cards & Containers */
    .mono-card {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
        color: #111827;
    }

    /* Khung Kết quả Phân loại */
    .verdict-box {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 16px;
        color: #111827;
    }
    .verdict-attack {
        border-left: 5px solid #111827;
        background-color: #F3F4F6;
    }
    .verdict-suspicious {
        border-left: 5px solid #6B7280;
        background-color: #F9FAFB;
    }
    .verdict-benign {
        border-left: 5px solid #D1D5DB;
        background-color: #FFFFFF;
    }

    .verdict-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 10px;
        margin: 12px 0;
        padding: 10px 12px;
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 4px;
    }
    .verdict-kv-label {
        font-size: 0.78rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .verdict-kv-value {
        font-size: 0.96rem;
        font-weight: 600;
        color: #111827;
        margin-top: 2px;
    }

    /* Monochrome Badges */
    .mono-badge {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 500;
        background-color: #FFFFFF;
        color: #111827;
        border: 1px solid #D1D5DB;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    .mono-badge-solid {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #111827;
        color: #FFFFFF;
    }
    .mono-badge-muted {
        display: inline-block;
        padding: 3px 9px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #E5E7EB;
        color: #111827;
    }

    /* Comparison Cards */
    .compare-card {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 16px 18px;
        color: #111827;
    }
    .compare-card-primary {
        background-color: #FFFFFF;
        border: 1.5px solid #111827;
        border-radius: 6px;
        padding: 16px 18px;
        color: #111827;
    }

    /* Tier-2 DPI Panels */
    .dpi-card {
        background-color: #F9FAFB;
        border: 1px solid #D1D5DB;
        border-left: 4px solid #374151;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 12px;
        color: #111827;
    }
    .dpi-finding {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 3px solid #6B7280;
        border-radius: 4px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }

    /* Sidebar status card */
    .sidebar-status {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 0.86rem;
        color: #111827;
        line-height: 1.5;
    }
</style>
"""


def inject_custom_css() -> None:
    """Tiêm CSS nền sáng đơn sắc tối giản vào ứng dụng Streamlit."""
    st.markdown(MONOCHROME_CSS, unsafe_allow_html=True)


def render_page_header() -> None:
    """Hiển thị tiêu đề chính chuẩn học thuật của ứng dụng."""
    st.markdown(
        '<h1 class="nids-header">Hệ Thống Phát Hiện Xâm Nhập Mạng (NIDS) — NF-UNSW-NB15</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="nids-sub">Thực nghiệm phân loại lưu lượng NetFlow và kiến trúc phân luồng hai tầng</p>',
        unsafe_allow_html=True,
    )


def render_protocol_decoder_card(
    proto_val: Any,
    proto_info: Dict[str, str],
    l7_info: Dict[str, str],
    src_port_desc: str,
    dst_port_desc: str,
    flags_list: List[str],
    true_attack: str,
    scenario_info: str,
    scenario_story: Dict[str, str],
) -> None:
    """Render thẻ tóm tắt thông số giao thức giải mã theo dạng thẻ đơn sắc."""
    flags_text = ", ".join(str(f) for f in flags_list)
    st.markdown(
        f"""
        <div class="mono-card">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
                <div>
                    <div style="font-size:0.92rem; font-weight:600; color:#111827; margin-bottom:8px;">
                        Thông số giao thức đã giải mã
                    </div>
                    <div>
                        <span class="mono-badge">Giao thức L4: {html.escape(proto_info['name'])} ({html.escape(str(proto_val))})</span>
                        <span class="mono-badge">Ứng dụng L7: {html.escape(l7_info['name'])} ({html.escape(l7_info['category'])})</span>
                        <span class="mono-badge">Cổng nguồn: {html.escape(src_port_desc)}</span>
                        <span class="mono-badge">Cổng đích: {html.escape(dst_port_desc)}</span>
                        <span class="mono-badge">Cờ TCP: {html.escape(flags_text)}</span>
                    </div>
                </div>
                <div style="text-align:right; min-width:200px;">
                    <div style="font-size:0.78rem; color:#6B7280; text-transform:uppercase;">Nhãn thực tế (Ground-Truth)</div>
                    <div style="font-size:1.05rem; font-weight:700; color:#111827; margin-top:2px;">
                        {html.escape(str(true_attack))}
                    </div>
                    <div style="font-size:0.8rem; color:#4B5563;">{html.escape(str(scenario_info))}</div>
                </div>
            </div>
            <hr style="border:none; border-top:1px solid #E5E7EB; margin:10px 0 8px 0;">
            <div style="font-size:0.88rem; color:#374151;">
                <strong>{html.escape(scenario_story['title'])}</strong> —
                <span style="color:#4B5563;">{html.escape(scenario_story['description'])}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_stepper_and_verdict(
    verdict: Dict[str, Any],
    card_class: str,
    icon: str,
    badge: str,
    step_classes: List[str],
    action_text: str,
) -> None:
    """Render khung Kết quả Phân loại & Quyết định Phân luồng theo phong cách học thuật đơn sắc."""
    del icon, step_classes  # Giữ tham số tương thích chữ ký hàm nhưng không render sơ đồ màu mè
    max_p = float(verdict["max_probability"])
    thresh = float(verdict.get("threshold_used", 0.50))
    near_low = float(verdict.get("near_miss_low", 0.10))
    status = str(verdict.get("status", "BENIGN"))
    primary_attack = str(verdict.get("primary_attack", "Benign"))
    triggered_by = str(verdict.get("triggered_by", "None"))

    if status == "ATTACK":
        conclusion_label = f"Tấn công ({primary_attack})"
        badge_cls = "mono-badge-solid"
    elif status == "SUSPICIOUS":
        conclusion_label = f"Nghi vấn cận ngưỡng ({primary_attack})"
        badge_cls = "mono-badge-muted"
    else:
        conclusion_label = "Lưu lượng bình thường (Benign)"
        badge_cls = "mono-badge"

    st.markdown(
        f"""
        <div class="{html.escape(card_class)}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div style="font-size:1.05rem; font-weight:700; color:#111827;">
                    Kết luận: {html.escape(conclusion_label)}
                </div>
                <span class="{badge_cls}">{html.escape(badge)}</span>
            </div>
            <div class="verdict-grid">
                <div>
                    <div class="verdict-kv-label">Xác suất dự đoán tối đa P(Attack)</div>
                    <div class="verdict-kv-value">{max_p*100:.2f}%</div>
                </div>
                <div>
                    <div class="verdict-kv-label">Ngưỡng áp dụng (t)</div>
                    <div class="verdict-kv-value">{thresh*100:.1f}% ({thresh:.3f})</div>
                </div>
                <div>
                    <div class="verdict-kv-label">Vùng cận ngưỡng (Near-Miss)</div>
                    <div class="verdict-kv-value">[{near_low*100:.1f}% – {thresh*100:.1f}%)</div>
                </div>
                <div>
                    <div class="verdict-kv-label">Mô hình ghi nhận</div>
                    <div class="verdict-kv-value">{html.escape(triggered_by)}</div>
                </div>
            </div>
            <div style="font-size:0.9rem; color:#374151; line-height:1.45;">
                <strong>Cơ chế xử lý:</strong> {html.escape(action_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dpi_inspection_panel(dpi_report: Dict[str, Any]) -> None:
    """Render nội dung bảng kiểm tra sâu Heuristic Tầng 2 theo chuẩn đơn sắc."""
    metrics = dpi_report["metrics"]
    border_map = {"danger": "#111827", "warning": "#4B5563", "success": "#D1D5DB"}
    accent = border_map.get(dpi_report.get("risk_badge", "warning"), "#4B5563")

    st.markdown(
        f"""
        <div class="dpi-card" style="border-left-color: {accent};">
            <div style="font-weight:700; font-size:0.96rem; color:#111827; margin-bottom:4px;">
                {html.escape(str(dpi_report['recommendation_label']))}
            </div>
            <div style="font-size:0.88rem; color:#374151;">
                {html.escape(str(dpi_report['summary']))}
            </div>
            <div style="font-size:0.82rem; color:#4B5563; margin-top:6px;">
                <strong>Tính nhất quán L4/L7-Port:</strong> {html.escape(str(metrics['l4_l7_consistency']))} |
                <strong>Cờ TCP:</strong> {html.escape(', '.join(metrics['tcp_flags_list']))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Bytes / Gói tin (TB)", f"{metrics['avg_pkt_size']:.1f} B")
    m2.metric("Tỷ lệ IN / OUT", f"{metrics['in_out_byte_ratio']:.2f}x")
    m3.metric("Tốc độ gói tin", f"{metrics['pkts_per_sec']:,.1f} pkts/s")
    m4.metric("Thời lượng phiên", f"{metrics['duration_ms']:,} ms")

    sev_border = {"HIGH": "#111827", "MEDIUM": "#6B7280", "LOW": "#9CA3AF", "INFO": "#D1D5DB"}
    for finding in dpi_report["findings"]:
        sc = sev_border.get(finding["severity"], "#9CA3AF")
        st.markdown(
            f"""
            <div class="dpi-finding" style="border-left: 3px solid {sc};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; color:#111827;">[{html.escape(finding['rule_id'])}] {html.escape(finding['title'])}</span>
                    <span style="font-size:0.78rem; font-weight:600; color:#4B5563;">{html.escape(finding['severity'])}</span>
                </div>
                <div style="font-size:0.86rem; color:#4B5563; margin-top:4px;">{html.escape(finding['detail'])}</div>
                <div style="font-size:0.8rem; color:#374151; margin-top:6px;"><code>{html.escape(finding['snort_hint'])}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_severity_breakdown(
    crit_count: int,
    med_count: int,
    low_count: int,
    near_low: float,
    thresh: float,
) -> None:
    """Render 3 thẻ phân bố mức độ rủi ro theo tông đơn sắc."""
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"""
        <div class="mono-card" style="border-left: 4px solid #111827; text-align:center;">
            <div style="font-size:0.85rem; font-weight:600; color:#111827;">Tấn công — Chặn Tầng 1</div>
            <div style="font-size:1.35rem; font-weight:700; color:#111827; margin:4px 0;">{crit_count:,} luồng</div>
            <div style="font-size:0.8rem; color:#4B5563;">Xác suất &ge; {thresh*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"""
        <div class="mono-card" style="border-left: 4px solid #6B7280; text-align:center;">
            <div style="font-size:0.85rem; font-weight:600; color:#111827;">Nghi vấn — Chuyển Tầng 2</div>
            <div style="font-size:1.35rem; font-weight:700; color:#111827; margin:4px 0;">{med_count:,} luồng</div>
            <div style="font-size:0.8rem; color:#4B5563;">Khoảng [{near_low*100:.1f}% – {thresh*100:.1f}%)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"""
        <div class="mono-card" style="border-left: 4px solid #D1D5DB; text-align:center;">
            <div style="font-size:0.85rem; font-weight:600; color:#111827;">Lưu lượng bình thường</div>
            <div style="font-size:1.35rem; font-weight:700; color:#111827; margin:4px 0;">{low_count:,} luồng</div>
            <div style="font-size:0.8rem; color:#4B5563;">Xác suất &lt; {near_low*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
