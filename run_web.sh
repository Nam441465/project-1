#!/usr/bin/env bash
# ==============================================================================
# run_web.sh - Khởi động Web Demo SOC NIDS Live Console (Streamlit)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT="${PORT:-8501}"
HOST="${HOST:-0.0.0.0}"

# 1. Tự động dò tìm Python interpreter theo thứ tự ưu tiên môi trường ảo (venv)
PYTHON_BIN=""
if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/venv/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/venv/bin/python"
elif [ -x "$HOME/.venv/bin/python" ]; then
    PYTHON_BIN="$HOME/.venv/bin/python"
elif [ -x "$HOME/venv/bin/python" ]; then
    PYTHON_BIN="$HOME/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[LỖI] Không tìm thấy Python interpreter hợp lệ (.venv, venv, \$HOME/.venv, \$HOME/venv hoặc python3)." >&2
    exit 1
fi

# 2. Kiểm tra thư viện bắt buộc
if ! "$PYTHON_BIN" -c "import streamlit, pandas, sklearn, joblib" >/dev/null 2>&1; then
    echo "[LỖI] Python ($PYTHON_BIN) thiếu một hoặc nhiều gói: streamlit, pandas, scikit-learn, joblib." >&2
    echo "      Hãy cài đặt bằng: $PYTHON_BIN -m pip install -r requirements.txt" >&2
    exit 1
fi

# 3. Kiểm tra dữ liệu và mô hình tiền huấn luyện (pre-flight check)
HELDOUT_PATH="$SCRIPT_DIR/data/processed/heldout_test.parquet"
if [ ! -f "$HELDOUT_PATH" ]; then
    echo "[LỖI] Không tìm thấy tập kiểm thử: $HELDOUT_PATH" >&2
    exit 1
fi

if ! compgen -G "$SCRIPT_DIR/results/*/models/*_metadata.json" >/dev/null 2>&1 && \
   ! compgen -G "$SCRIPT_DIR/results/models/*_metadata.json" >/dev/null 2>&1; then
    echo "[LỖI] Không tìm thấy file *_metadata.json nào trong $SCRIPT_DIR/results/*/models/." >&2
    exit 1
fi

# 4. Kiểm tra cổng mạng đang khả dụng
if ! "$PYTHON_BIN" - "$HOST" "$PORT" <<'PYEOF'
import socket, sys
host = sys.argv[1]
port = int(sys.argv[2])
bind_host = "127.0.0.1" if host == "0.0.0.0" else host
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    if s.connect_ex((bind_host, port)) == 0:
        sys.exit(1)
PYEOF
then
    echo "[LỖI] Cổng $PORT trên $HOST đang được tiến trình khác sử dụng. Hãy đặt PORT=<port_khác> ./run_web.sh" >&2
    exit 1
fi

echo "================================================================================"
echo " 🛡️ KHỞI ĐỘNG HỆ THỐNG SOC NIDS LIVE CONSOLE (STREAMLIT)"
echo "================================================================================"
echo "[*] Thư mục dự án: $SCRIPT_DIR"
echo "[*] Môi trường Python: $PYTHON_BIN"
echo "[*] Khởi chạy Streamlit tại $HOST:$PORT (fileWatcherType=none)..."
echo "[*] Truy cập từ trình duyệt: http://localhost:$PORT"
echo "================================================================================"

exec "$PYTHON_BIN" -m streamlit run web_app/app.py \
    --server.port "$PORT" \
    --server.address "$HOST" \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false
