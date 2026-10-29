#!/usr/bin/env bash
# ==============================================================================
# run_web.sh - Khởi động Web Demo SOC NIDS Live Console (Streamlit)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tự động dò tìm Python interpreter theo thứ tự ưu tiên môi trường ảo (venv)
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif [ -x "$HOME/venv/bin/python" ]; then
    PYTHON_BIN="$HOME/venv/bin/python"
else
    PYTHON_BIN="python3"
fi

echo "================================================================================"
echo " 🛡️ KHỞI ĐỘNG HỆ THỐNG SOC NIDS LIVE CONSOLE (STREAMLIT)"
echo "================================================================================"
echo "[*] Thư mục dự án: $SCRIPT_DIR"
echo "[*] Môi trường Python: $PYTHON_BIN"
echo "[*] Khởi chạy Streamlit trên cổng 8501 (Tất cả giao diện mạng 0.0.0.0)..."
echo "[*] Truy cập từ trình duyệt trên máy Host (Windows): http://localhost:8501"
echo "================================================================================"

"$PYTHON_BIN" -m streamlit run web_app/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
