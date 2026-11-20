#!/usr/bin/env python3
"""
tabs/single_flow.py
Tab 1: Kiểm tra đơn luồng (Single-Flow Inspection), Giải mã Giao thức,
Kiểm tra Heuristic Tầng 2 và Ma trận Đồng thuận 5×3.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import streamlit as st

from inference_engine import (
    CANONICAL_ATTACKS,
    NIDSInferenceEngine,
    build_consensus_matrix,
    get_near_miss_bounds,
)
from protocol_decoder import (
    analyze_tier2_dpi_heuristics,
    decode_l4_protocol,
    decode_l7_protocol,
    decode_port,
    decode_tcp_flags,
    explain_scenario_context,
)
from styles import (
    render_dpi_inspection_panel,
    render_pipeline_stepper_and_verdict,
    render_protocol_decoder_card,
)


def _normalize_flow_features(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Chuẩn hóa 10 đặc trưng NetFlow về đúng kiểu và giới hạn widget."""
    return {
        "L4_SRC_PORT": max(0, min(65535, int(float(sample.get("L4_SRC_PORT", 0))))),
        "L4_DST_PORT": max(0, min(65535, int(float(sample.get("L4_DST_PORT", 0))))),
        "PROTOCOL": max(0, min(255, int(float(sample.get("PROTOCOL", 6))))),
        "L7_PROTO": float(sample.get("L7_PROTO", 0.0)),
        "IN_BYTES": max(0, int(float(sample.get("IN_BYTES", 0)))),
        "OUT_BYTES": max(0, int(float(sample.get("OUT_BYTES", 0)))),
        "IN_PKTS": max(0, int(float(sample.get("IN_PKTS", 0)))),
        "OUT_PKTS": max(0, int(float(sample.get("OUT_PKTS", 0)))),
        "TCP_FLAGS": max(0, int(float(sample.get("TCP_FLAGS", 0)))),
        "FLOW_DURATION_MILLISECONDS": max(0, int(float(sample.get("FLOW_DURATION_MILLISECONDS", 0)))),
    }


def apply_sample_to_state(sample: Dict[str, Any]) -> None:
    """Đồng bộ mẫu luồng mạng vào session_state, baseline và các ô nhập liệu."""
    normalized = _normalize_flow_features(sample)
    scenario = str(sample.get("Sample_Scenario", "Mẫu tùy chỉnh"))
    true_att = str(sample.get("True_Attack", "Không rõ"))

    st.session_state["current_flow"] = dict(normalized)
    st.session_state["preset_baseline"] = dict(normalized)
    st.session_state["preset_scenario_info"] = scenario
    st.session_state["preset_true_attack"] = true_att
    st.session_state["scenario_info"] = scenario
    st.session_state["true_attack"] = true_att

    st.session_state["inp_L4_SRC_PORT"] = normalized["L4_SRC_PORT"]
    st.session_state["inp_L4_DST_PORT"] = normalized["L4_DST_PORT"]
    st.session_state["inp_PROTOCOL"] = normalized["PROTOCOL"]
    st.session_state["inp_L7_PROTO"] = normalized["L7_PROTO"]
    st.session_state["inp_IN_BYTES"] = normalized["IN_BYTES"]
    st.session_state["inp_OUT_BYTES"] = normalized["OUT_BYTES"]
    st.session_state["inp_IN_PKTS"] = normalized["IN_PKTS"]
    st.session_state["inp_OUT_PKTS"] = normalized["OUT_PKTS"]
    st.session_state["inp_TCP_FLAGS"] = normalized["TCP_FLAGS"]
    st.session_state["inp_FLOW_DURATION_MS"] = normalized["FLOW_DURATION_MILLISECONDS"]


def _get_live_flow_from_widgets() -> Dict[str, Any]:
    """Đọc giá trị NetFlow trực tiếp từ live widget keys `st.session_state['inp_*']`."""
    return {
        "L4_SRC_PORT": int(st.session_state.get("inp_L4_SRC_PORT", 0)),
        "L4_DST_PORT": int(st.session_state.get("inp_L4_DST_PORT", 0)),
        "PROTOCOL": int(st.session_state.get("inp_PROTOCOL", 6)),
        "L7_PROTO": float(st.session_state.get("inp_L7_PROTO", 0.0)),
        "IN_BYTES": int(st.session_state.get("inp_IN_BYTES", 0)),
        "OUT_BYTES": int(st.session_state.get("inp_OUT_BYTES", 0)),
        "IN_PKTS": int(st.session_state.get("inp_IN_PKTS", 0)),
        "OUT_PKTS": int(st.session_state.get("inp_OUT_PKTS", 0)),
        "TCP_FLAGS": int(st.session_state.get("inp_TCP_FLAGS", 0)),
        "FLOW_DURATION_MILLISECONDS": int(st.session_state.get("inp_FLOW_DURATION_MS", 0)),
    }


def _is_manually_edited(live_flow: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    """Kiểm tra người dùng có chỉnh sửa thông số NetFlow khác với mẫu preset vừa nạp hay không."""
    if not baseline:
        return False
    for key, val in live_flow.items():
        base_val = baseline.get(key)
        if base_val is None:
            continue
        if key == "L7_PROTO":
            if abs(float(val) - float(base_val)) > 1e-6:
                return True
        elif int(val) != int(base_val):
            return True
    return False


def render_single_flow_tab(
    engine: NIDSInferenceEngine,
    active_mode_key: str,
    cond_filter: str,
    custom_threshold_val: Optional[float],
) -> None:
    """Render toàn bộ nội dung Tab 1."""
    st.markdown("#### 1. Chọn mẫu dữ liệu từ tập kiểm thử độc lập (Held-out Test)")

    col_target_att, col_p1, col_p2, col_p3, col_p4 = st.columns([1.4, 1.2, 1.4, 1.8, 1.2])

    with col_target_att:
        preset_attack = st.selectbox("Loại tấn công mục tiêu:", CANONICAL_ATTACKS, index=0, key="sb_preset_attack")

    def _load_preset(sample_type: str) -> None:
        try:
            sample = engine.sample_real_test_flow(
                sample_type=sample_type,
                attack_name=preset_attack,
                random_seed=int(time.time_ns() % (2**31 - 1)),
            )
            apply_sample_to_state(sample)
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể nạp mẫu '{sample_type}': {exc}")

    with col_p1:
        if st.button("Mẫu Lưu Lượng Sạch", width="stretch", help="Nạp 1 mẫu Benign thực tế từ tập kiểm thử."):
            _load_preset("benign")

    with col_p2:
        if st.button("Mẫu Tấn Công Rõ Ràng (TP)", width="stretch", help="Nạp 1 mẫu tấn công có xác suất nhận diện cao."):
            _load_preset("attack_typical")

    with col_p3:
        if st.button(
            "Mẫu Tấn Công Cận Ngưỡng (Near-Miss)",
            width="stretch",
            help="Nạp 1 mẫu tấn công khó nhận diện ở ngưỡng F1 nhưng được phát hiện ở ngưỡng nhạy chi phí hoặc Tầng 2.",
        ):
            _load_preset("attack_stealth_nearmiss")

    with col_p4:
        if st.button("Mẫu Ngẫu Nhiên", width="stretch", help="Rút ngẫu nhiên 1 mẫu từ tập kiểm thử."):
            _load_preset("random")

    # Đồng bộ trạng thái luồng hiện tại từ các widget inp_*
    live_flow = _get_live_flow_from_widgets()
    baseline = st.session_state.get("preset_baseline", {})
    manually_edited = _is_manually_edited(live_flow, baseline)

    if manually_edited:
        display_true_attack = "Tùy chỉnh (Manual Edit)"
        orig_sc = st.session_state.get("preset_scenario_info", "Mẫu gốc")
        display_scenario = f"Chỉnh sửa thủ công (Gốc: {orig_sc})"
    else:
        display_true_attack = str(
            st.session_state.get("preset_true_attack", st.session_state.get("true_attack", "Benign"))
        )
        display_scenario = str(
            st.session_state.get("preset_scenario_info", st.session_state.get("scenario_info", ""))
        )

    st.session_state["current_flow"] = live_flow
    st.session_state["true_attack"] = display_true_attack
    st.session_state["scenario_info"] = display_scenario

    col_scope, _ = st.columns([2.5, 1])
    with col_scope:
        eval_scope = st.radio(
            "Phạm vi Mô hình Phân tích:",
            options=[
                f"Chỉ đánh giá Mục tiêu tấn công đang chọn ({preset_attack})",
                "Quét toàn diện cả 5 loại tấn công (All 5 Attack Detectors)",
            ],
            index=0,
            horizontal=True,
            help="Phân tích luồng đối với riêng loại tấn công mục tiêu hoặc toàn bộ 5 nhóm tấn công.",
        )
    att_filter_val = preset_attack if "Chỉ đánh giá Mục tiêu" in eval_scope else "all"

    # Thực hiện suy luận ngay trên live_flow để hiển thị kết luận ngay dưới phần chọn mẫu
    try:
        prediction_output = engine.predict_single_flow(
            flow_data=live_flow,
            threshold_mode=active_mode_key,
            condition_filter=cond_filter,
            attack_filter=att_filter_val,
            custom_threshold=custom_threshold_val,
        )
        verdict = prediction_output["verdict"]
        details_df = prediction_output["details"]
    except Exception as err:
        st.error(f"Lỗi khi thực hiện suy luận: {err}")
        return

    st.markdown("#### 2. Kết quả phân loại & quyết định phân luồng")

    status = str(verdict["status"])
    near_low_pct = float(verdict.get("near_miss_low", 0.10)) * 100.0
    thresh_pct = float(verdict.get("threshold_used", 0.50)) * 100.0

    if status == "ATTACK":
        card_class = "verdict-box verdict-attack"
        badge = "Tầng 1: Chặn luồng"
        action_text = (
            f"Xác suất vượt ngưỡng chặn ({thresh_pct:.1f}%). "
            f"Thực hiện chặn trực tiếp tại Tầng 1 nhằm ngăn chặn xâm nhập ngay lập tức."
        )
    elif status == "SUSPICIOUS":
        card_class = "verdict-box verdict-suspicious"
        badge = "Tầng 2: Kiểm tra sâu"
        action_text = (
            f"Xác suất rơi vào vùng cận ngưỡng [{near_low_pct:.1f}% - {thresh_pct:.1f}%). "
            f"Chuyển tiếp sang Tầng 2 để kiểm tra bổ sung quy tắc hình thái và tải trọng gói tin."
        )
    else:
        card_class = "verdict-box verdict-benign"
        badge = "Cho phép thông qua"
        action_text = (
            f"Xác suất rủi ro dưới cận dưới vùng nghi vấn ({near_low_pct:.1f}%). "
            f"Cho phép luồng dữ liệu đi qua bình thường qua nhánh xử lý nhanh (Fast-Path)."
        )

    render_pipeline_stepper_and_verdict(
        verdict=dict(verdict),
        card_class=card_class,
        icon="",
        badge=badge,
        step_classes=[],
        action_text=action_text,
    )

    proto_info = decode_l4_protocol(live_flow["PROTOCOL"])
    l7_info = decode_l7_protocol(live_flow["L7_PROTO"])
    flags_list = decode_tcp_flags(live_flow["TCP_FLAGS"])
    src_port_desc = decode_port(live_flow["L4_SRC_PORT"])
    dst_port_desc = decode_port(live_flow["L4_DST_PORT"])
    scenario_story = explain_scenario_context(live_flow, display_true_attack)

    render_protocol_decoder_card(
        proto_val=live_flow["PROTOCOL"],
        proto_info=proto_info,
        l7_info=l7_info,
        src_port_desc=src_port_desc,
        dst_port_desc=dst_port_desc,
        flags_list=flags_list,
        true_attack=display_true_attack,
        scenario_info=display_scenario,
        scenario_story=scenario_story,
    )

    dpi_report = analyze_tier2_dpi_heuristics(live_flow, dict(verdict))
    dpi_title = (
        "Kiểm tra quy tắc Tầng 2 (Kích hoạt do xác suất thuộc vùng cận ngưỡng)"
        if status == "SUSPICIOUS"
        else "Kiểm tra quy tắc Tầng 2 (Chẩn đoán tất định trên 10 đặc trưng NetFlow)"
    )
    with st.expander(dpi_title, expanded=(status == "SUSPICIOUS")):
        render_dpi_inspection_panel(dict(dpi_report))

    st.markdown("---")
    st.markdown("#### 3. Thông số đặc trưng NetFlow (10 đặc trưng đầu vào)")

    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5 = st.columns(5)
    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)

    with r1_c1:
        st.number_input("L4_SRC_PORT (Cổng nguồn)", min_value=0, max_value=65535, key="inp_L4_SRC_PORT")
    with r1_c2:
        st.number_input("L4_DST_PORT (Cổng đích)", min_value=0, max_value=65535, key="inp_L4_DST_PORT")
    with r1_c3:
        st.number_input("PROTOCOL (Giao thức L4)", min_value=0, max_value=255, key="inp_PROTOCOL")
    with r1_c4:
        st.number_input("L7_PROTO (Giao thức L7)", format="%.1f", key="inp_L7_PROTO")
    with r1_c5:
        st.number_input("IN_BYTES (Bytes đến)", min_value=0, key="inp_IN_BYTES")

    with r2_c1:
        st.number_input("OUT_BYTES (Bytes đi)", min_value=0, key="inp_OUT_BYTES")
    with r2_c2:
        st.number_input("IN_PKTS (Gói đến)", min_value=0, key="inp_IN_PKTS")
    with r2_c3:
        st.number_input("OUT_PKTS (Gói đi)", min_value=0, key="inp_OUT_PKTS")
    with r2_c4:
        st.number_input("TCP_FLAGS (Cờ TCP)", min_value=0, key="inp_TCP_FLAGS")
    with r2_c5:
        st.number_input("FLOW_DURATION_MS (Thời lượng ms)", min_value=0, key="inp_FLOW_DURATION_MS")

    if att_filter_val == "all":
        st.markdown("---")
        st.markdown("#### Ma Trận Đồng Thuận Đa Mô Hình (5 loại tấn công × 3 họ mô hình)")
        st.caption(
            "Tổng hợp xác suất, biên độ lệch ngưỡng (`prob - thresh`), tỷ lệ đồng thuận (`k/3`) "
            "và độ trễ trên 5 loại tấn công và 3 họ mô hình (`PAPER`, `Ours-RF`, `Ours-HistGB`)."
        )
        matrices = build_consensus_matrix(details_df)
        for cond_label, matrix_df in matrices.items():
            if len(matrices) > 1:
                st.markdown(f"**Điều kiện huấn luyện: `{cond_label}`**")
            show_cols = [
                c
                for c in ["Attack", "PAPER", "Ours-RF", "Ours-HistGB", "Consensus", "Total_Latency_ms"]
                if c in matrix_df.columns
            ]
            st.dataframe(
                matrix_df[show_cols],
                width="stretch",
                hide_index=True,
                column_config={
                    "Attack": "Loại tấn công",
                    "PAPER": "PAPER (Xác suất / Biên ngưỡng)",
                    "Ours-RF": "Ours-RF (Xác suất / Biên ngưỡng)",
                    "Ours-HistGB": "Ours-HistGB (Xác suất / Biên ngưỡng)",
                    "Consensus": "Đồng thuận (k/3)",
                    "Total_Latency_ms": st.column_config.NumberColumn("Tổng độ trễ (ms)", format="%.2f ms"),
                },
            )

    st.markdown("---")
    st.markdown("#### 4. Đối sánh xác suất dự đoán và ngưỡng quyết định giữa các mô hình")
    col_chart, col_table = st.columns([1.1, 1.4])

    with col_chart:
        st.markdown("**Biểu đồ xác suất rủi ro theo từng mô hình**")
        chart_data = details_df.sort_values(by="Probability", ascending=True).copy()

        fig, ax = plt.subplots(figsize=(6, max(3.5, len(chart_data) * 0.45)))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        y_pos = np.arange(len(chart_data))
        probs = chart_data["Probability"].to_numpy() * 100.0
        thresholds = chart_data["Threshold_Used"].to_numpy() * 100.0
        near_lows = (
            chart_data["Near_Miss_Low"].to_numpy() * 100.0
            if "Near_Miss_Low" in chart_data.columns
            else np.array([get_near_miss_bounds(t / 100.0)[0] * 100.0 for t in thresholds])
        )

        # Thang xám đơn sắc theo mức độ phân vùng
        bar_colors: List[str] = []
        for p, nl, th in zip(probs, near_lows, thresholds):
            if p >= th:
                bar_colors.append("#111827")
            elif p >= nl:
                bar_colors.append("#6B7280")
            else:
                bar_colors.append("#D1D5DB")

        bars = ax.barh(y_pos, probs, color=bar_colors, height=0.58, edgecolor="#374151", linewidth=0.5)

        for i, (nl, th) in enumerate(zip(near_lows, thresholds)):
            ax.plot([th, th], [i - 0.35, i + 0.35], color="#111827", linestyle="--", linewidth=1.8)
            ax.plot([nl, nl], [i - 0.30, i + 0.30], color="#6B7280", linestyle=":", linewidth=1.5)

        unique_nl = np.unique(np.round(near_lows, 2))
        if len(unique_nl) == 1:
            ax.axvline(x=float(unique_nl[0]), color="#6B7280", linestyle=":", linewidth=1.1, alpha=0.8)

        legend_handles = [
            Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8, label="Ngưỡng chặn (t)"),
            Line2D([0], [0], color="#6B7280", linestyle=":", linewidth=1.5, label="Cận dưới Near-Miss"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="lower right",
            facecolor="#FFFFFF",
            edgecolor="#D1D5DB",
            labelcolor="#111827",
            fontsize=7.8,
        )

        labels = [f"{r['Model_Family']} - {r['Attack']} ({r['Condition']})" for _, r in chart_data.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, color="#111827", fontsize=8.5)
        ax.set_xlabel("Xác suất dự đoán tấn công P(Attack) (%)", color="#111827", fontsize=9.2)
        ax.set_xlim(0, 108)
        ax.tick_params(colors="#374151", labelsize=8.5)
        ax.grid(axis="x", color="#E5E7EB", linestyle="-", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#D1D5DB")
        ax.spines["bottom"].set_color("#D1D5DB")

        for bar, p in zip(bars, probs):
            ax.text(
                min(p + 1.5, 101.0),
                bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%",
                va="center",
                ha="left",
                color="#111827",
                fontsize=8,
                fontweight="bold",
            )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_table:
        st.markdown(f"**Bảng kết quả chi tiết ({len(details_df)} mô hình)**")
        cols_keep = [
            "Model_Family",
            "Attack",
            "Condition",
            "Probability",
            "Near_Miss_Low",
            "Threshold_Used",
            "Risk_Zone",
            "Selected_Features_Count",
            "Latency_ms",
        ]
        display_df = details_df[[c for c in cols_keep if c in details_df.columns]].copy()

        display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
        if "Near_Miss_Low" in display_df.columns:
            display_df["Near_Miss_Low"] = display_df["Near_Miss_Low"].apply(lambda x: f"{x*100:.1f}%")
        display_df["Threshold_Used"] = display_df["Threshold_Used"].apply(lambda x: f"{x:.3f}")
        display_df["Latency_ms"] = display_df["Latency_ms"].apply(lambda x: f"{x:.2f} ms")

        st.dataframe(
            display_df,
            width="stretch",
            height=320,
            column_config={
                "Model_Family": "Họ mô hình",
                "Attack": "Mục tiêu",
                "Condition": "Điều kiện",
                "Probability": "Xác suất",
                "Near_Miss_Low": "Cận Near-Miss",
                "Threshold_Used": "Ngưỡng áp dụng",
                "Risk_Zone": "Phân vùng",
                "Selected_Features_Count": "Số đặc trưng (k)",
                "Latency_ms": "Độ trễ",
            },
        )
