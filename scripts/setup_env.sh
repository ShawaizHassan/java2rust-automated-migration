#!/usr/bin/env bash
set -e

echo "[*] Creating Python environment (conda if available, else venv)..."

if command -v conda >/dev/null 2>&1; then
  conda create -y -n java2rust_env python=3.11
  echo "Run: conda activate java2rust_env"
else
  python3 -m venv .venv
  echo "Run: source .venv/bin/activate"
fi

echo "[*] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt
[ -f requirements-dev.txt ] && pip install -r requirements-dev.txt || true

echo "[*] CUDA note: If you plan GPU inference, ensure your CUDA drivers/toolkit match your framework (vLLM, etc.)."
echo "[*] Setup complete."
echo "[*] Remember to activate the environment before use."
echo "    If using conda: conda activate java2rust_env"
echo "    If using venv: source .venv/bin/activate" 