#!/bin/bash
# =============================================================
# setup.sh — Automated Environment Setup for Two-Stage RAG
# =============================================================
# Run this script ONCE to set up the environment:
#   chmod +x setup.sh && ./setup.sh
# =============================================================

set -e  # Exit immediately on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo ""
echo "============================================================"
echo " TWO-STAGE RAG PIPELINE — Environment Setup"
echo "============================================================"

# ----------------------------------------------------------
# Step 1: Create Python Virtual Environment
# ----------------------------------------------------------
echo ""
echo "[Step 1] Creating Python virtual environment at: $VENV_DIR"

if [ -d "$VENV_DIR" ]; then
    echo "  ⚠ Virtual environment already exists. Skipping creation."
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created"
fi

# ----------------------------------------------------------
# Step 2: Activate Virtual Environment
# ----------------------------------------------------------
echo ""
echo "[Step 2] Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "  ✓ Activated: $VIRTUAL_ENV"

# ----------------------------------------------------------
# Step 3: Upgrade pip
# ----------------------------------------------------------
echo ""
echo "[Step 3] Upgrading pip..."
pip install --upgrade pip --quiet
echo "  ✓ pip upgraded"

# ----------------------------------------------------------
# Step 4: Install all dependencies
# ----------------------------------------------------------
echo ""
echo "[Step 4] Installing dependencies from requirements.txt..."
echo "  This may take a few minutes on first run (downloading models)..."
pip install -r "$SCRIPT_DIR/requirements.txt"
echo ""
echo "  ✓ All dependencies installed"

# ----------------------------------------------------------
# Step 5: Check .env file
# ----------------------------------------------------------
echo ""
echo "[Step 5] Checking .env configuration..."
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    if grep -q "your_google_api_key_here" "$ENV_FILE"; then
        echo "  ⚠ WARNING: .env file contains placeholder API key!"
        echo "  Please edit .env and replace 'your_google_api_key_here'"
        echo "  with your actual Google API key."
        echo "  Get your key at: https://aistudio.google.com/app/apikey"
    else
        echo "  ✓ .env file found with API key configured"
    fi
else
    echo "  ⚠ .env file not found. Please create it:"
    echo "    echo 'GOOGLE_API_KEY=your_key_here' > .env"
fi

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo " SETUP COMPLETE!"
echo "============================================================"
echo ""
echo " Next steps:"
echo ""
echo "  1. Set your API key in .env:"
echo "     GOOGLE_API_KEY=your_actual_key_here"
echo ""
echo "  2. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  3. Ingest your documents:"
echo "     python main.py --ingest sample_docs/sample.txt"
echo ""
echo "  4. Ask questions:"
echo "     python main.py"
echo "     > Enter your query: What is a transformer model?"
echo ""
echo "============================================================"
