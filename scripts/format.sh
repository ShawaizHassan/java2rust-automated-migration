#!/usr/bin/env bash
set -e

echo "[*] Python formatters..."
black .
isort .

echo "[*] Python linter (ruff) fix..."
ruff check . --fix || true

echo "[*] Rust formatters..."
if command -v cargo >/dev/null 2>&1; then
  cargo fmt || true
fi
echo "[*] Format complete."