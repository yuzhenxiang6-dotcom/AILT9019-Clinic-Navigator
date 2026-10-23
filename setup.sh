#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh - one-click environment setup for macOS / Linux
#
# Creates a virtual environment and installs everything from requirements.txt.
# Safe to run more than once.
# ---------------------------------------------------------------------------
set -u

cd "$(dirname "$0")" || exit 1

echo ""
echo "==========================================================="
echo "  AILT9019 - UHS Retrieval QA | Environment Setup"
echo "==========================================================="
echo ""

PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[ERROR] Python 3.10 or newer is required but was not found."
    echo "        Install Python from https://www.python.org/downloads/"
    echo "        then run this script again."
    echo ""
    read -r -p "Press Enter to close..."
    exit 1
fi

echo "[1/3] Using $PY ($("$PY" --version 2>&1))"

if [ ! -d "venv" ]; then
    echo "[2/3] Creating virtual environment (venv)..."
    "$PY" -m venv venv || {
        echo ""
        echo "[ERROR] Could not create the virtual environment."
        echo "        On Ubuntu you may need: sudo apt install python3-venv"
        echo ""
        read -r -p "Press Enter to close..."
        exit 1
    }
else
    echo "[2/3] Virtual environment already exists, reusing it."
fi

# shellcheck disable=SC1091
source venv/bin/activate || {
    echo "[ERROR] Could not activate the virtual environment."
    read -r -p "Press Enter to close..."
    exit 1
}

echo "[3/3] Installing dependencies (this can take several minutes)..."
echo ""

if pip install --upgrade pip >/dev/null 2>&1; then
    echo "      pip updated."
fi

if pip install -r requirements.txt; then
    echo ""
    echo "==========================================================="
    echo "  Setup complete."
    echo ""
    echo "  Next:  bash start.sh          (to run the app)"
    echo "         python run_tests.py    (to run the 12 test cases)"
    echo ""
    echo "  NOTE:  .env is required for online mode."
    echo "         Copy .env.example to .env and paste your API key."
    echo "         Without it the app still runs, in offline mode."
    echo "==========================================================="
    echo ""
    read -r -p "Press Enter to close..."
    exit 0
else
    echo ""
    echo "[ERROR] Installation failed. Common causes:"
    echo "        - No internet connection"
    echo "        - Corporate proxy or firewall blocking PyPI"
    echo "        Try:  pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple"
    echo ""
    read -r -p "Press Enter to close..."
    exit 1
fi