#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_compile_logs.sh /path/to/compile_logs.py
#
# This script runs your compile-error logger on all listed JSONL files and writes:
#   /data/hassan/java2rust-automated-migration/experiments/lustre/error-logs/no-pass/<method>/<model-name>.jsonl
#
# where <method> is the parent folder name (ragcat-hgl-gptsummary or ragcat-references-gptsummary)
# and <model-name> is the input filename.

LOGGER_PY="${1:-compile_logs.py}"

OUT_ROOT="/data/hassan/java2rust-automated-migration/experiments/javarusttrans/error-logs/fourth-pass"

WORKERS="${WORKERS:-8}"
TIMEOUT="${TIMEOUT:-8}"
MAX_ERR_CHARS="${MAX_ERR_CHARS:-10000}"

inputs=(
## JRT 4x refine
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fourth-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# ## JRT 3x refine
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/third-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# ## JRT 2x refine
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/second-pass/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# ## JRT 1x refine
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/ragcat-signals-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

#base rag
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/deepseek-coder-1.3b-instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/RAG/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-hgl-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl

# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/CodeLlama-7b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/CodeLlama-34b-Instruct-hf.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/Deepseek-Coder-6.7b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/Deepseek-Coder-33b-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/Qwen2.5-Coder-7B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/Qwen2.5-Coder-14B-Instruct.jsonl
# /data/hassan/java2rust-automated-migration/experiments/lustre/no-feedback/ragcat-references-gptsummary/Qwen2.5-Coder-32B-Instruct.jsonl
)

echo "Logger: ${LOGGER_PY}"
echo "Workers=${WORKERS} Timeout=${TIMEOUT} MaxErrChars=${MAX_ERR_CHARS}"
echo

for in_file in "${inputs[@]}"; do
  if [[ ! -f "$in_file" ]]; then
    echo "[SKIP] Missing: $in_file"
    continue
  fi

  method="$(basename "$(dirname "$in_file")")"       # ragcat-hgl-gptsummary / ragcat-references-gptsummary
  model_file="$(basename "$in_file")"                # e.g., Qwen2.5-Coder-32B-Instruct.jsonl
  out_dir="${OUT_ROOT}/${method}"
  out_file="${out_dir}/${model_file}"

  mkdir -p "$out_dir"

  echo "[RUN] ${method} / ${model_file}"
  python3 "$LOGGER_PY" \
    --input "$in_file" \
    --out "$out_file" \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --max_err_chars "$MAX_ERR_CHARS"
  echo "[OK ] -> ${out_file}"
  echo
done

echo "All done."
