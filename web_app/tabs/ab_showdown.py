#!/usr/bin/env python3
"""
tabs/ab_showdown.py
Tab 2: Đối sánh thực nghiệm giữa Mô hình bài báo gốc (Paper-Exact RF)
và Hệ thống đề xuất (Ours 2-Tier NIDS).
"""

from __future__ import annotations

import html
import time
from typing import Dict

import streamlit as st

from inference_engine import CANONICAL_ATTACKS, NIDSInferenceEngine


_FAMILY_LABELS: Dict[str, str] = {
    "RF": "Random Forest cải tiến (Ours-RF)",
    "PAPER": "Random Forest bài báo gốc (Paper-Exact RF)",
}


def render_ab_showdown_tab(engine: NIDSInferenceEngine) -> None:
    """Render toàn bộ nội dung Tab 2: Đối sánh thực nghiệm hai hệ thống."""
    st.markdown("#### Đối sánh thực nghiệm: Mô hình bài báo gốc (Pospíchal et al., 2026) và Hệ thống đề xuất")
    st.caption(
        "Đánh giá song song trên cùng tập mẫu rút từ Held-out Test: "
        "Paper-Exact RF (ngưỡng F1-Optimal) so với Hệ thống đề xuất (ngưỡng nhạy chi phí c=10 và phân luồng Tầng 2)."
    )

    avail_families = engine.get_available_model_families()
    ours_candidates = [f for f in avail_families if f != "PAPER"] or avail_families
    if not ours_candidates:
        st.warning("Chưa phát hiện mô hình nào trong danh mục để thực hiện đối sánh.")
        return

    ab_c1, ab_c2, ab_c3, ab_c4 = st.columns([1.5, 1.2, 1.3, 1.5])
    with ab_c1:
        ab_attack = st.selectbox("Loại tấn công đánh giá:", CANONICAL_ATTACKS, index=0, key="sb_ab_attack")
    with ab_c2:
        ab_samples = st.slider(
            "Số lượng luồng kiểm thử:", min_value=100, max_value=3000, value=500, step=100, key="sl_ab_samples"
        )
    with ab_c3:
        ab_ratio = st.slider(
            "Tỷ lệ mẫu tấn công:", min_value=0.05, max_value=0.50, value=0.10, step=0.05, key="sl_ab_ratio"
        )
    with ab_c4:
        ab_ours_family = st.selectbox(
            "Mô hình hệ thống đề xuất:",
            options=ours_candidates,
            index=0,
            format_func=lambda f: _FAMILY_LABELS.get(f, f),
            key="sb_ab_ours_family",
        )
        ours_family_key = ab_ours_family

    col_btn_ab, col_clear_ab = st.columns([3, 1])
    with col_btn_ab:
        if st.button("Chạy đối sánh thực nghiệm", type="primary", width="stretch", key="btn_run_ab"):
            with st.spinner("Đang rút mẫu từ tập kiểm thử và thực thi suy luận song song..."):
                try:
                    sampled_ab_df = engine.sample_batch_test_flows(
                        attack_name=ab_attack,
                        total_samples=ab_samples,
                        attack_ratio=ab_ratio,
                        include_other_attacks=False,
                        random_seed=int(time.time() % 10000),
                    )

                    baseline_family = "PAPER" if "PAPER" in avail_families else ours_candidates[0]
                    res_paper, sum_paper = engine.predict_batch_flows(
                        df=sampled_ab_df,
                        attack_name=ab_attack,
                        model_family=baseline_family,
                        condition="nosmote",
                        threshold_mode="F1-Optimal",
                    )

                    res_ours, sum_ours = engine.predict_batch_flows(
                        df=sampled_ab_df,
                        attack_name=ab_attack,
                        model_family=ours_family_key,
                        condition="nosmote",
                        threshold_mode="Cost-Sensitive (c=10)",
                    )

                    st.session_state["ab_results"] = {
                        "batch_df": sampled_ab_df,
                        "target_attack": ab_attack,
                        "paper": {"res": res_paper, "sum": sum_paper},
                        "ours": {"res": res_ours, "sum": sum_ours, "family": ours_family_key},
                    }
                except Exception as exc:
                    st.error(f"Lỗi khi thực thi đối sánh: {exc}")

    with col_clear_ab:
        if st.session_state.get("ab_results") is not None:
            if st.button("Xóa kết quả", width="stretch", key="btn_clear_ab"):
                st.session_state["ab_results"] = None
                st.rerun()

    ab_data = st.session_state.get("ab_results")
    if ab_data is None:
        return

    p_sum = ab_data["paper"]["sum"]
    o_sum = ab_data["ours"]["sum"]
    p_m = p_sum.get("metrics", {})
    o_m = o_sum.get("metrics", {})

    c_val = 10
    p_cost = c_val * p_m.get("FN", 0) + p_m.get("FP", 0)
    o_cost = c_val * o_m.get("FN", 0) + o_m.get("FP", 0)
    cost_diff = p_cost - o_cost
    cost_savings_pct = (cost_diff / p_cost * 100.0) if p_cost > 0 else 0.0

    fn_paper = int(p_m.get("FN", 0))
    fn_ours = int(o_m.get("FN", 0))
    fn_diff = fn_paper - fn_ours
    fn_red_pct = (fn_diff / fn_paper * 100.0) if fn_paper > 0 else 0.0

    p_tput = float(p_sum.get("throughput_flows_per_sec", 0.0))
    o_tput = float(o_sum.get("throughput_flows_per_sec", 0.0))
    speedup = (o_tput / p_tput) if p_tput > 0 else 1.0

    st.markdown("---")
    st.markdown("#### Tổng hợp chỉ số đối sánh")

    b1, b2, b3 = st.columns(3)
    if fn_diff > 0:
        fn_delta = f"Giảm {fn_diff} mẫu (-{fn_red_pct:.1f}%)"
    elif fn_diff < 0:
        fn_delta = f"Tăng {abs(fn_diff)} mẫu"
    else:
        fn_delta = "Không đổi (0 mẫu)"

    b1.metric(
        "Số mẫu tấn công bị bỏ sót (FN)",
        f"{fn_ours} mẫu",
        delta=fn_delta,
        delta_color="off",
    )

    if cost_diff > 0:
        cost_delta = f"Giảm {cost_savings_pct:.1f}% (-{cost_diff:,.0f})"
    elif cost_diff < 0:
        cost_delta = f"Tăng {abs(cost_savings_pct):.1f}% (+{abs(cost_diff):,.0f})"
    else:
        cost_delta = "Không đổi"

    b2.metric(
        "Tổng chi phí rủi ro SOC (c = 10)",
        f"{o_cost:,.0f}",
        delta=cost_delta,
        delta_color="off",
    )

    if speedup >= 1.0:
        speed_delta = f"Nhanh hơn {speedup:.1f}x"
    else:
        speed_delta = f"Tỷ lệ {speedup:.2f}x"

    b3.metric(
        "Thông lượng suy luận",
        f"{o_tput:,.0f} luồng/s",
        delta=speed_delta,
        delta_color="off",
    )

    st.markdown("---")
    col_ab_left, col_ab_right = st.columns(2)

    with col_ab_left:
        st.markdown(
            f"""
            <div class="compare-card">
                <div style="font-size:0.98rem; font-weight:700; color:#111827; margin-bottom:4px;">
                    Mô hình bài báo gốc (Pospíchal et al., 2026)
                </div>
                <div style="font-size:0.84rem; color:#4B5563; margin-bottom:12px;">
                    Thuật toán: <strong>Random Forest Baseline</strong> | Ngưỡng: <strong>F1-Optimal ({p_sum.get('threshold_used', 0.5):.3f})</strong>
                </div>
                <table style="width:100%; font-size:0.9rem; border-collapse:collapse; color:#111827;">
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Bỏ sót tấn công (False Negatives - FN):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{fn_paper} luồng</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Báo động nhầm (False Positives - FP):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{p_m.get('FP', 0)} luồng</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Độ nhạy (Recall):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{p_m.get('Recall', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Độ chính xác dương (Precision):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{p_m.get('Precision', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Điểm F1-Score:</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{p_m.get('F1', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Thông lượng thực thi:</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{p_tput:,.0f} luồng/s</td></tr>
                    <tr><td style="padding:8px 0 2px 0; font-weight:700;">Tổng chi phí rủi ro SOC:</td><td style="text-align:right; font-weight:700;">{p_cost:,.0f}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ab_right:
        fn_sign = "-" if fn_diff >= 0 else "+"
        rec_delta = (o_m.get("Recall", 0) - p_m.get("Recall", 0)) * 100.0
        fam_display = html.escape(str(ab_data["ours"]["family"]))
        st.markdown(
            f"""
            <div class="compare-card-primary">
                <div style="font-size:0.98rem; font-weight:700; color:#111827; margin-bottom:4px;">
                    Hệ thống đề xuất (Ours 2-Tier NIDS)
                </div>
                <div style="font-size:0.84rem; color:#4B5563; margin-bottom:12px;">
                    Thuật toán: <strong>{fam_display} + 2-Tier</strong> | Ngưỡng: <strong>Cost-Sensitive c=10 ({o_sum.get('threshold_used', 0.5):.3f})</strong>
                </div>
                <table style="width:100%; font-size:0.9rem; border-collapse:collapse; color:#111827;">
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Bỏ sót tấn công (False Negatives - FN):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{fn_ours} luồng ({fn_sign}{abs(fn_diff)})</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Báo động nhầm (False Positives - FP):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{o_m.get('FP', 0)} luồng</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Độ nhạy (Recall):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{o_m.get('Recall', 0)*100:.2f}% ({'+' if rec_delta >= 0 else ''}{rec_delta:.2f}%)</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Độ chính xác dương (Precision):</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{o_m.get('Precision', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Điểm F1-Score:</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{o_m.get('F1', 0)*100:.2f}%</td></tr>
                    <tr><td style="padding:5px 0; border-bottom:1px solid #E5E7EB;">Thông lượng thực thi:</td><td style="text-align:right; font-weight:600; border-bottom:1px solid #E5E7EB;">{o_tput:,.0f} luồng/s ({speedup:.1f}x)</td></tr>
                    <tr><td style="padding:8px 0 2px 0; font-weight:700;">Tổng chi phí rủi ro SOC:</td><td style="text-align:right; font-weight:700;">{o_cost:,.0f} (Giảm {cost_savings_pct:.1f}%)</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Danh sách các mẫu tấn công bị mô hình bài báo gốc bỏ sót (FN) nhưng được hệ thống đề xuất phát hiện")
    st.caption(
        "Các luồng tấn công thực tế bị mô hình cơ sở phân loại nhầm là Benign, "
        "nhưng được hệ thống đề xuất chặn tại Tầng 1 hoặc chuyển kiểm tra Tầng 2."
    )

    batch_raw = ab_data["batch_df"]
    res_p = ab_data["paper"]["res"]
    res_o = ab_data["ours"]["res"]
    target_att = str(ab_data.get("target_attack", ab_attack))

    is_attack_groundtruth = (
        (batch_raw["Label"] == 1)
        if "Label" in batch_raw.columns
        else (batch_raw["Attack"].astype(str).str.lower() == target_att.lower())
    )
    mask_caught_by_ours = (
        is_attack_groundtruth
        & (res_p["Predicted_Label"] == 0)
        & ((res_o["Predicted_Label"] == 1) | (res_o["Risk_Zone"].str.contains("SUSPICIOUS")))
    )

    stealth_caught_df = batch_raw[mask_caught_by_ours].copy()

    if len(stealth_caught_df) > 0:
        stealth_caught_df["Paper_Risk_Score"] = res_p.loc[mask_caught_by_ours, "Risk_Score"].apply(
            lambda x: f"{x*100:.1f}%"
        )
        stealth_caught_df["Ours_Risk_Score"] = res_o.loc[mask_caught_by_ours, "Risk_Score"].apply(
            lambda x: f"{x*100:.1f}%"
        )
        stealth_caught_df["Ours_Action"] = res_o.loc[mask_caught_by_ours, "Risk_Zone"]

        cols_show = [
            "L4_SRC_PORT",
            "L4_DST_PORT",
            "PROTOCOL",
            "IN_BYTES",
            "OUT_BYTES",
            "TCP_FLAGS",
            "FLOW_DURATION_MILLISECONDS",
            "Paper_Risk_Score",
            "Ours_Risk_Score",
            "Ours_Action",
        ]
        existing_cols_show = [c for c in cols_show if c in stealth_caught_df.columns]

        st.dataframe(stealth_caught_df[existing_cols_show].head(100), width="stretch", height=240)
        st.info(
            f"Ghi nhận {len(stealth_caught_df)} luồng tấn công bị mô hình bài báo gốc bỏ sót "
            f"nhưng đã được hệ thống đề xuất phân loại hoặc chuyển kiểm tra Tầng 2."
        )
    else:
        st.info("Trong tập mẫu rút ngẫu nhiên lần này, mô hình bài báo gốc không phát sinh mẫu bỏ sót nào được phân loại lại.")
