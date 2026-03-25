#!/usr/bin/env bash
set -euo pipefail

# Batch runner for refinement.py (compile refinement by default).
# Runs sequentially, binding to GPUs "0,1" (TP=2).
#
# Usage:
#   bash run_baseline_refinement_rag.sh
#
# Optional:
#   UNITTEST=1 bash run_baseline_refinement_rag.sh   # adds -u (runtime/unit-test refinement)

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT="${SCRIPT:-/data/hassan/java2rust-automated-migration/my-algo/refinement.py}"

# Use 2 GPUs (0,1)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# If you want unittest refinement, set UNITTEST=1 in env.
UNITTEST_FLAG=""
if [[ "${UNITTEST:-0}" == "1" ]]; then
  UNITTEST_FLAG="-u"
fi

# ---------------- Paths ----------------
IN_BASE="/data/hassan/java2rust-automated-migration/experiments/baseline/error-logs/no-pass/RAG"
OUT_BASE="/data/hassan/java2rust-automated-migration/experiments/baseline/refinement/compile-repair/RAG"

# ---------------- Model paths ----------------
# MODEL_CODELLAMA_7B="/data/hassan/models/CodeLlama-7b-Instruct-hf"
# MODEL_CODELLAMA_34B="/data/hassan/models/CodeLlama-34b-Instruct-hf"

MODEL_DEEPSEEK_1B3="/data/hassan/models/deepseek-coder-1.3b-instruct"
MODEL_DEEPSEEK_6B7="/data/jfeng/models/Deepseek-Coder-6.7b-Instruct"
MODEL_DEEPSEEK_33B="/data/jfeng/models/Deepseek-Coder-33b-Instruct"

MODEL_QWEN_7B="/data/jfeng/models/Qwen2.5-Coder-7B-Instruct"
MODEL_QWEN_14B="/data/jfeng/models/Qwen2.5-Coder-14B-Instruct"
MODEL_QWEN_32B="/data/jfeng/models/Qwen2.5-Coder-32B-Instruct"

# ---------------- Jobs ----------------
# Each job: method | model_name | model_path | input_jsonl | output_jsonl
JOBS=(
  # "RAG|CodeLlama-7b-Instruct-hf|${MODEL_CODELLAMA_7B}|${IN_BASE}/CodeLlama-7b-Instruct-hf.jsonl|${OUT_BASE}/CodeLlama-7b-Instruct-hf.jsonl"
  # "RAG|CodeLlama-34b-Instruct-hf|${MODEL_CODELLAMA_34B}|${IN_BASE}/CodeLlama-34b-Instruct-hf.jsonl|${OUT_BASE}/CodeLlama-34b-Instruct-hf.jsonl"

  # NOTE: your input filename is lowercase here; keep it exactly as your file name
  "RAG|deepseek-coder-1.3b-instruct|${MODEL_DEEPSEEK_1B3}|${IN_BASE}/deepseek-coder-1.3b-instruct.jsonl|${OUT_BASE}/deepseek-coder-1.3b-instruct.jsonl"
  "RAG|Deepseek-Coder-6.7b-Instruct|${MODEL_DEEPSEEK_6B7}|${IN_BASE}/Deepseek-Coder-6.7b-Instruct.jsonl|${OUT_BASE}/Deepseek-Coder-6.7b-Instruct.jsonl"
  "RAG|Deepseek-Coder-33b-Instruct|${MODEL_DEEPSEEK_33B}|${IN_BASE}/Deepseek-Coder-33b-Instruct.jsonl|${OUT_BASE}/Deepseek-Coder-33b-Instruct.jsonl"

  "RAG|Qwen2.5-Coder-7B-Instruct|${MODEL_QWEN_7B}|${IN_BASE}/Qwen2.5-Coder-7B-Instruct.jsonl|${OUT_BASE}/Qwen2.5-Coder-7B-Instruct.jsonl"
  "RAG|Qwen2.5-Coder-14B-Instruct|${MODEL_QWEN_14B}|${IN_BASE}/Qwen2.5-Coder-14B-Instruct.jsonl|${OUT_BASE}/Qwen2.5-Coder-14B-Instruct.jsonl"
  "RAG|Qwen2.5-Coder-32B-Instruct|${MODEL_QWEN_32B}|${IN_BASE}/Qwen2.5-Coder-32B-Instruct.jsonl|${OUT_BASE}/Qwen2.5-Coder-32B-Instruct.jsonl"
)

# ---------------- Run ----------------
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "SCRIPT=${SCRIPT}"
echo "UNITTEST_FLAG='${UNITTEST_FLAG}'"
echo "IN_BASE=${IN_BASE}"
echo "OUT_BASE=${OUT_BASE}"
echo "========================================================"

mkdir -p "${OUT_BASE}"

for job in "${JOBS[@]}"; do
  IFS="|" read -r method model_name model_path in_jsonl out_jsonl <<< "${job}"

  echo
  echo ">>> METHOD: ${method}"
  echo ">>> MODEL : ${model_name}"
  echo ">>> IN    : ${in_jsonl}"
  echo ">>> OUT   : ${out_jsonl}"
  echo ">>> PATH  : ${model_path}"
  echo "--------------------------------------------------------"

  # Input must exist
  if [[ ! -f "${in_jsonl}" ]]; then
    echo "ERROR: missing input file: ${in_jsonl}"
    exit 1
  fi

  # Model path must exist
  if [[ ! -d "${model_path}" ]]; then
    echo "ERROR: missing model dir: ${model_path}"
    exit 1
  fi

  mkdir -p "$(dirname "${out_jsonl}")"

  # Safety: ensure input/output basenames match (prevents accidental misalignment)
  if [[ "$(basename "${in_jsonl}")" != "$(basename "${out_jsonl}")" ]]; then
    echo "ERROR: basename mismatch!"
    echo "  IN : $(basename "${in_jsonl}")"
    echo "  OUT: $(basename "${out_jsonl}")"
    exit 1
  fi

  ${PYTHON_BIN} "${SCRIPT}" \
    --input "${in_jsonl}" \
    --out "${out_jsonl}" \
    ${UNITTEST_FLAG} \
    --model_path "${model_path}" \
    --tensor_parallel_size 2

  echo "✓ Done: ${method}/${model_name}"
done

echo
echo "ALL DONE."
