from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "web_app" / "app.py")


def _new_app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=60)


@pytest.mark.slow
def test_app_cold_boot_and_tab1_protocol_decoder_live_sync():
    at = _new_app()
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6  # 4 main tabs + 2 sub-tabs in Tab 4
    assert at.session_state["true_attack"] == "Benign"

    # Mutating a live input widget updates current_flow and flips true_attack to Manual Edit
    at.number_input(key="inp_L4_DST_PORT").set_value(22).run()
    assert not at.exception
    assert at.session_state["current_flow"]["L4_DST_PORT"] == 22
    assert at.session_state["true_attack"] == "Tùy chỉnh (Manual Edit)"


@pytest.mark.slow
def test_app_tab1_preset_and_ux2_five_attack_consensus_matrix():
    at = _new_app()
    at.run()

    # Load a typical DoS preset
    btn = next(b for b in at.button if "Tấn Công Rõ Ràng" in b.label)
    btn.click().run()
    assert not at.exception
    assert at.session_state["true_attack"] == "DoS"

    # Switch scope to all 5 attack detectors (UX-2 Consensus Matrix)
    scope_radio = next(r for r in at.radio if r.label == "Phạm vi Mô hình Phân tích:")
    scope_radio.set_value("Quét toàn diện cả 5 loại tấn công (All 5 Attack Detectors)").run()
    assert not at.exception
    assert any("Ma Trận Đồng Thuận" in md.value for md in at.markdown)


@pytest.mark.slow
def test_app_tab2_ab_showdown_flow():
    at = _new_app()
    at.run()

    ab_btn = next(b for b in at.button if "Chạy đối sánh thực nghiệm" in b.label)
    ab_btn.click().run()
    assert not at.exception
    ab_res = at.session_state["ab_results"]
    assert ab_res is not None
    assert ab_res["target_attack"] == "DoS"
    assert len(ab_res["batch_df"]) == 500


@pytest.mark.slow
def test_app_tab3_batch_scanner_and_source_switch_clears_state():
    at = _new_app()
    at.run()

    scan_btn = next(b for b in at.button if "Bắt đầu quét lưu lượng hàng loạt" in b.label)
    scan_btn.click().run()
    assert not at.exception
    assert at.session_state["batch_df"] is not None
    assert len(at.session_state["batch_df"]) == 500

    # Switching data source to CSV Upload must reset stale batch_df
    src_radio = at.radio(key="rad_batch_source_mode")
    src_radio.set_value("Tải lên file CSV từ máy tính").run()
    assert not at.exception
    assert at.session_state["batch_df"] is None


@pytest.mark.slow
def test_app_tab4_registry_sorted_and_reload():
    at = _new_app()
    at.run()

    inspect_sb = at.selectbox(key="sb_registry_inspect")
    assert list(inspect_sb.options) == sorted(inspect_sb.options)
    assert len(inspect_sb.options) == 30

    reload_btn = at.button(key="btn_reload_registry")
    reload_btn.click().run()
    assert not at.exception
