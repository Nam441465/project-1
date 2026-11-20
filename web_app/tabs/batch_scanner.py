#!/usr/bin/env python3
"""
tabs/batch_scanner.py
Tab 3: Đánh giá theo lô dữ liệu (Batch Evaluation).
Hỗ trợ rút mẫu Parquet hoặc tải lên CSV, memoization suy luận theo khóa dữ liệu+tham số,
và phân tách state theo nguồn dữ liệu để tránh dữ liệu cũ bị kẹt.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from inference_engine import CANONICAL_ATTACKS, NIDSInferenceEngine
from protocol_decoder import analyze_tier2_dpi_heuristics, decode_port
from styles import render_dpi_inspection_panel


def _compute_df_hash(df: pd.DataFrame) -> str:
    """Tính mã băm ổn định cho DataFrame để memoize kết quả batch inference."""
    hasher = hashlib.sha256()
    hasher.update(f"{df.shape}|{list(df.columns)}".encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="uint64")
    hasher.update(row_hashes.tobytes())
    return hasher.hexdigest()


def _reset_batch_state() -> None:
    """Xóa sạch state của Tab 3 một cách nhất quán."""
    st.session_state["batch_df"] = None
    st.session_state["batch_data_hash"] = None
    st.session_state["batch_inference_cache"] = None
    st.session_state["batch_uploaded_sig"] = None
    st.session_state["batch_target_attack"] = "DoS"


def render_batch_scanner_tab(
    engine: NIDSInferenceEngine,
    active_mode_key: str,
    custom_threshold_val: Optional[float],
) -> None:
    """Render toàn bộ nội dung Tab 3."""
    st.markdown("#### Đánh giá lưu lượng theo lô dữ liệu (Batch Evaluation)")
    st.caption(
        "Kiểm thử đồng thời nhiều luồng mạng từ tập kiểm thử độc lập (Held-out Test) "
        "hoặc từ tệp CSV tải lên."
    )

    avail_families = engine.get_available_model_families()
    if not avail_families:
        st.warning("Chưa có mô hình nào trong danh mục để thực hiện đánh giá.")
        return

    avail_conditions = engine.get_available_conditions()
    if not avail_conditions:
        avail_conditions = [("nosmote", "Without SMOTE"), ("smote", "With SMOTE")]
    cond_label_to_key: Dict[str, str] = {label: key for key, label in avail_conditions}
    cond_labels = list(cond_label_to_key.keys())

    col_b_source, col_b_model = st.columns([1.2, 1.0])

    with col_b_source:
        st.markdown("**1. Nguồn dữ liệu đầu vào**")
        data_source_mode = st.radio(
            "Chọn nguồn dữ liệu:",
            options=[
                "Rút mẫu từ Held-out Test Parquet (1-Click Preset)",
                "Tải lên file CSV từ máy tính",
            ],
            index=0,
            key="rad_batch_source_mode",
        )

    # Xóa state cũ ngay khi chuyển đổi giữa Parquet và CSV Upload
    prev_source_mode = st.session_state.get("batch_source_mode_prev")
    if prev_source_mode is not None and prev_source_mode != data_source_mode:
        _reset_batch_state()
    st.session_state["batch_source_mode_prev"] = data_source_mode

    with col_b_model:
        st.markdown("**2. Mô hình phân tích**")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            batch_model_family = st.selectbox("Thuật toán:", avail_families, index=0, key="sb_batch_family")
        with col_m2:
            batch_condition = st.selectbox("Điều kiện huấn luyện:", cond_labels, index=0, key="sb_batch_condition")
            cond_key = cond_label_to_key[batch_condition]

    st.markdown("---")

    target_attack_for_eval = str(st.session_state.get("batch_target_attack") or "DoS")

    if data_source_mode == "Rút mẫu từ Held-out Test Parquet (1-Click Preset)":
        b_c1, b_c2, b_c3, b_c4 = st.columns([1.5, 1.5, 1.5, 1.5])
        with b_c1:
            target_attack_for_eval = st.selectbox(
                "Loại tấn công:", CANONICAL_ATTACKS, index=0, key="sb_batch_eval_att"
            )
        with b_c2:
            num_samples = st.slider(
                "Số lượng luồng:", min_value=100, max_value=5000, value=500, step=100, key="sl_batch_samples"
            )
        with b_c3:
            att_ratio = st.slider(
                "Tỷ lệ tấn công pha trộn:", min_value=0.05, max_value=0.50, value=0.10, step=0.05, key="sl_batch_ratio"
            )
        with b_c4:
            inc_other = st.checkbox(
                "Pha trộn các lớp tấn công khác",
                value=False,
                help="Đánh giá phản ứng của bộ phân loại khi có thêm các lớp tấn công ngoài mục tiêu.",
                key="chk_batch_inc_other",
            )

        col_run_b1, col_run_b2 = st.columns([3, 1])
        with col_run_b1:
            if st.button("Bắt đầu quét lưu lượng hàng loạt", type="primary", width="stretch", key="btn_run_batch"):
                with st.spinner("Đang nạp dữ liệu phân tầng từ heldout_test.parquet..."):
                    try:
                        sampled_batch = engine.sample_batch_test_flows(
                            attack_name=target_attack_for_eval,
                            total_samples=num_samples,
                            attack_ratio=att_ratio,
                            include_other_attacks=inc_other,
                            random_seed=int(time.time() % 10000),
                        )
                        st.session_state["batch_df"] = sampled_batch
                        st.session_state["batch_data_hash"] = _compute_df_hash(sampled_batch)
                        st.session_state["batch_target_attack"] = target_attack_for_eval
                        st.session_state["batch_inference_cache"] = None
                    except Exception as exc:
                        st.error(f"Lỗi khi rút mẫu dữ liệu: {exc}")

        with col_run_b2:
            if st.session_state.get("batch_df") is not None:
                if st.button("Xóa dữ liệu", width="stretch", key="btn_clear_batch_parquet"):
                    _reset_batch_state()
                    st.rerun()

    else:
        uploaded_file = st.file_uploader(
            "Chọn tệp CSV chứa các đặc trưng NetFlow:", type=["csv"], key="fu_batch_csv"
        )
        if uploaded_file is None:
            if st.session_state.get("batch_df") is not None or st.session_state.get("batch_uploaded_sig") is not None:
                _reset_batch_state()
        else:
            file_sig = (uploaded_file.name, getattr(uploaded_file, "size", 0))
            if file_sig != st.session_state.get("batch_uploaded_sig"):
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    if uploaded_df.empty:
                        _reset_batch_state()
                        st.error("Tệp CSV rỗng (0 dòng dữ liệu). Vui lòng chọn tệp hợp lệ.")
                    else:
                        st.session_state["batch_df"] = uploaded_df
                        st.session_state["batch_data_hash"] = _compute_df_hash(uploaded_df)
                        st.session_state["batch_uploaded_sig"] = file_sig
                        st.session_state["batch_inference_cache"] = None
                except Exception as exc:
                    _reset_batch_state()
                    st.error(f"Lỗi đọc tệp CSV: {exc}")

            active_csv_df = st.session_state.get("batch_df")
            if active_csv_df is not None:
                st.info(f"Đã nạp tệp CSV `{uploaded_file.name}`: {len(active_csv_df):,} dòng.")
                if "Attack" in active_csv_df.columns:
                    avail_attacks = [
                        str(a)
                        for a in active_csv_df["Attack"].dropna().unique()
                        if str(a).strip().lower() != "benign"
                    ]
                    if avail_attacks:
                        target_attack_for_eval = st.selectbox(
                            "Chọn loại tấn công mục tiêu trong tệp CSV:",
                            avail_attacks,
                            key="sb_csv_attack_present",
                        )
                    else:
                        target_attack_for_eval = st.selectbox(
                            "Tệp CSV chỉ chứa lưu lượng Benign — Chọn bộ phân loại cần kiểm thử:",
                            CANONICAL_ATTACKS,
                            index=0,
                            key="sb_csv_attack_benign",
                        )
                else:
                    target_attack_for_eval = st.selectbox(
                        "Chọn loại tấn công cần quét trên tệp CSV:",
                        CANONICAL_ATTACKS,
                        index=0,
                        key="sb_csv_attack_nolabel",
                    )
                st.session_state["batch_target_attack"] = target_attack_for_eval

                if st.button("Xóa dữ liệu", width="content", key="btn_clear_batch_csv"):
                    _reset_batch_state()
                    st.rerun()

    # ==========================================================================
    # XỬ LÝ QUÉT (MEMOIZED) VÀ HIỂN THỊ KẾT QUẢ TỪ SESSION_STATE
    # ==========================================================================
    active_batch_df = st.session_state.get("batch_df")
    if active_batch_df is None:
        return

    active_target_attack = str(st.session_state.get("batch_target_attack") or target_attack_for_eval or "DoS")
    data_hash = st.session_state.get("batch_data_hash") or _compute_df_hash(active_batch_df)
    st.session_state["batch_data_hash"] = data_hash

    cache_key: Tuple[Any, ...] = (
        data_hash,
        batch_model_family,
        cond_key,
        active_mode_key,
        custom_threshold_val,
        active_target_attack,
    )
    cached_bundle = st.session_state.get("batch_inference_cache")

    if isinstance(cached_bundle, dict) and cached_bundle.get("key") == cache_key:
        res_df = cached_bundle["res_df"]
        summary = cached_bundle["summary"]
    else:
        with st.spinner("Đang thực hiện suy luận theo lô..."):
            try:
                res_df, summary = engine.predict_batch_flows(
                    df=active_batch_df,
                    attack_name=active_target_attack,
                    model_family=batch_model_family,
                    condition=cond_key,
                    threshold_mode=active_mode_key,
                    custom_threshold=custom_threshold_val,
                    strict=True,
                )
                st.session_state["batch_inference_cache"] = {
                    "key": cache_key,
                    "res_df": res_df,
                    "summary": summary,
                }
            except Exception as exc:
                st.error(f"Không thể thực hiện suy luận hàng loạt: {exc}")
                return

    if summary.get("missing_model_features"):
        st.warning(
            f"Tệp dữ liệu thiếu {len(summary['missing_model_features'])} cột đặc trưng cần thiết cho mô hình "
            f"(`{', '.join(summary['missing_model_features'])}`). Hệ thống đã điền giá trị `0` tạm thời; "
            f"kết quả dự đoán có thể bị ảnh hưởng."
        )
    if summary.get("coerced_columns"):
        st.info(
            f"Đã chuẩn hóa các giá trị không phải số/NaN về `0` ở cột: `{', '.join(summary['coerced_columns'])}`."
        )

    crit_count = int(summary["attack_count"])
    med_count = int(summary["suspicious_count"])
    low_count = int(summary["benign_count"])
    near_low = float(summary.get("near_miss_low", 0.10))
    thresh_used = float(summary.get("threshold_used", 0.50))

    st.markdown("#### Tổng hợp kết quả phân loại")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Tổng số luồng", f"{summary['total_flows']:,}")
    kpi2.metric(
        "Cảnh báo Tầng 1",
        f"{crit_count:,}",
        delta=f">= {thresh_used*100:.1f}%",
        delta_color="off",
    )
    kpi3.metric(
        "Nghi vấn Tầng 2",
        f"{med_count:,}",
        delta=f"[{near_low*100:.1f}% - {thresh_used*100:.1f}%)",
        delta_color="off",
    )
    kpi4.metric(
        "Lưu lượng bình thường",
        f"{low_count:,}",
        delta=f"< {near_low*100:.1f}%",
        delta_color="off",
    )
    kpi5.metric(
        "Thông lượng suy luận",
        f"{summary['throughput_flows_per_sec']:,.0f}/s",
        delta=f"{summary['batch_inference_time_sec']:.3f}s",
        delta_color="off",
    )

    if "L4_DST_PORT" in res_df.columns:
        attack_flows_df = res_df[res_df["Predicted_Label"] == 1]
        if len(attack_flows_df) > 0:
            top_ports = attack_flows_df["L4_DST_PORT"].dropna().value_counts().head(5)
            if len(top_ports) > 0:
                top_ports_str = " | ".join(
                    f"<strong>{decode_port(p)}:</strong> {cnt:,} luồng" for p, cnt in top_ports.items()
                )
                st.markdown(
                    f"""
                    <div class="mono-card" style="font-size:0.88rem;">
                        <strong>Phân bố cổng đích bị cảnh báo nhiều nhất:</strong> &nbsp; {top_ports_str}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    rec_actions = []
    if crit_count > 0:
        rec_actions.append(f"Chặn tại Tầng 1 đối với {crit_count:,} luồng vượt ngưỡng ({thresh_used*100:.1f}%).")
    if med_count > 0:
        rec_actions.append(
            f"Chuyển {med_count:,} luồng thuộc vùng cận ngưỡng [{near_low*100:.1f}% - {thresh_used*100:.1f}%) sang kiểm tra sâu tại Tầng 2."
        )
    if not rec_actions:
        rec_actions.append("Không ghi nhận luồng vượt ngưỡng hoặc nằm trong vùng nghi vấn.")

    st.markdown(
        f"""
        <div class="mono-card" style="border-left: 4px solid #111827; font-size:0.88rem;">
            <strong>Khuyến nghị xử lý theo quy trình 2 tầng:</strong><br>
            {"<br>".join(f"• {act}" for act in rec_actions)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "metrics" in summary:
        m = summary["metrics"]
        st.markdown("#### Chỉ số đánh giá trên nhãn thực tế (Ground-Truth)")

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Accuracy", f"{m['Accuracy']*100:.2f}%")
        m_c2.metric("Precision", f"{m['Precision']*100:.2f}%")
        m_c3.metric("Recall", f"{m['Recall']*100:.2f}%")
        m_c4.metric("F1-Score", f"{m['F1']*100:.2f}%")

        st.markdown(
            f"""
            <div class="mono-card" style="font-size:0.9rem;">
                <strong>Ma trận nhầm lẫn (Confusion Matrix):</strong> &nbsp;
                <span><strong>TP:</strong> {m['TP']}</span> &nbsp;|&nbsp;
                <span><strong>FN:</strong> {m['FN']}</span> &nbsp;|&nbsp;
                <span><strong>FP:</strong> {m['FP']}</span> &nbsp;|&nbsp;
                <span><strong>TN:</strong> {m['TN']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    suspicious_subset = res_df[res_df["Risk_Zone"].str.contains("SUSPICIOUS", na=False)]
    if not suspicious_subset.empty:
        with st.expander(f"Kiểm tra quy tắc Tầng 2 cho các luồng nghi vấn trong lô ({len(suspicious_subset)} luồng)", expanded=False):
            sample_idx = st.selectbox(
                "Chọn chỉ số dòng SUSPICIOUS để kiểm tra chi tiết:",
                options=list(suspicious_subset.index[:50]),
                key="sb_batch_dpi_idx",
            )
            chosen_row = suspicious_subset.loc[sample_idx].to_dict()
            dpi_report = analyze_tier2_dpi_heuristics(chosen_row, {"status": "SUSPICIOUS"})
            render_dpi_inspection_panel(dict(dpi_report))

    st.markdown("#### Danh sách luồng dữ liệu đã phân tích")
    filter_col, download_col = st.columns([2, 1])
    with filter_col:
        view_filter = st.selectbox(
            "Lọc dữ liệu hiển thị:",
            options=[
                "Tất cả luồng",
                "Chỉ xem ATTACK (Tầng 1)",
                "Chỉ xem SUSPICIOUS (Tầng 2 DPI)",
                "Chỉ xem FALSE NEGATIVES (Bị bỏ sót)",
                "Chỉ xem FALSE POSITIVES (Báo động giả)",
            ],
            index=0,
            key="sb_batch_view_filter",
        )

    filtered_res = res_df.copy()
    target_att_eval = str(summary.get("target_attack", active_target_attack))
    if view_filter == "Chỉ xem ATTACK (Tầng 1)":
        filtered_res = filtered_res[filtered_res["Predicted_Label"] == 1]
    elif view_filter == "Chỉ xem SUSPICIOUS (Tầng 2 DPI)":
        filtered_res = filtered_res[filtered_res["Risk_Zone"].str.contains("SUSPICIOUS", na=False)]
    elif view_filter == "Chỉ xem FALSE NEGATIVES (Bị bỏ sót)":
        if "Attack" in filtered_res.columns:
            att_col = filtered_res["Attack"].fillna("Benign").astype(str)
            filtered_res = filtered_res[
                (att_col.str.lower() == target_att_eval.lower()) & (filtered_res["Predicted_Label"] == 0)
            ]
        elif "Label" in filtered_res.columns:
            lbl_num = pd.to_numeric(filtered_res["Label"], errors="coerce").fillna(0).astype(int)
            filtered_res = filtered_res[(lbl_num == 1) & (filtered_res["Predicted_Label"] == 0)]
    elif view_filter == "Chỉ xem FALSE POSITIVES (Báo động giả)":
        if "Attack" in filtered_res.columns:
            att_col = filtered_res["Attack"].fillna("Benign").astype(str)
            filtered_res = filtered_res[(att_col.str.lower() == "benign") & (filtered_res["Predicted_Label"] == 1)]
        elif "Label" in filtered_res.columns:
            lbl_num = pd.to_numeric(filtered_res["Label"], errors="coerce").fillna(0).astype(int)
            filtered_res = filtered_res[(lbl_num == 0) & (filtered_res["Predicted_Label"] == 1)]

    st.dataframe(filtered_res.head(500), width="stretch", height=280)

    with download_col:
        csv_data = filtered_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Tải kết quả (CSV)",
            data=csv_data,
            file_name=f"nids_batch_scan_{active_target_attack}_{batch_model_family}.csv",
            mime="text/csv",
            width="stretch",
        )
