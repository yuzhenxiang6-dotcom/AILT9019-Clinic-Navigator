#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh - launch the Streamlit app (macOS / Linux)
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")" || exit 1

if [ ! -d "venv" ]; then
    echo ""
    echo "[ERROR] No virtual environment found."
    echo "        Run  bash setup.sh  first."
    echo ""
    read -r -p "Press Enter to close..."
    exit 1
fi

# shellcheck disable=SC1091
source venv/bin/activate

if [ ! -f ".env" ]; then
    echo ""
    echo "[NOTE] No .env file found."
    echo "       The app will start in OFFLINE MODE and show retrieved text."
    echo "       To enable online mode, copy .env.example to .env and add your key."
    echo ""
fi

echo "Starting UHS Retrieval QA at http://localhost:8501"
echo "Press Ctrl+C in this terminal to stop the server."
echo ""

exec streamlit run app.py