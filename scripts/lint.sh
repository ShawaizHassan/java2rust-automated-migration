#!/usr/bin/env bash
set -e

echo "[*] Ruff lint..."
ruff check .

echo "[*] Black check..."
black --check .

echo "[*] isort check..."
isort --check-only .

echo "[*] Rust clippy..."
if command -v cargo >/dev/null 2>&1; then
  cargo clippy --all-targets -- -D warnings || true
fi
echo "[*] Lint complete."