#!/usr/bin/env bash
set -euo pipefail

# Batch runner for refinement.py (compile refinement by default).
# Runs sequentially, binding to GPUs "0,1" (TP=2).
#
# Usage:
#   bash run_refinement_batch_fourthpass.sh
#
# Optional:
#   UNITTEST=1 bash run_refinement_batch_fourthpass.sh   # add -u (runtime/unit-test refinement)

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

# ---------------- Paths (FOURTH-PASS error logs) ----------------
IN_BASE="/data/hassan/java2rust-automated-migration/experiments/javarusttrans/error-logs/fourth-pass"
METHOD="ragcat-signals-gptsummary"
IN_DIR="${IN_BASE}/${METHOD}"

# Output: keep aligned under refinement/fifth-pass/<method>/<model>.jsonl
OUT_BASE="/data/hassan/java2rust-automated-migration/experiments/javarusttrans/refinement/fifth-pass"
OUT_DIR="${OUT_BASE}/${METHOD}"

# ---------------- Model paths ----------------
MODEL_CODELLAMA_7B="/data/hassan/models/CodeLlama-7b-Instruct-hf"
MODEL_CODELLAMA_34B="/data/hassan/models/CodeLlama-34b-Instruct-hf"

MODEL_QWEN_7B="/data/jfeng/models/Qwen2.5-Coder-7B-Instruct"
MODEL_QWEN_14B="/data/jfeng/models/Qwen2.5-Coder-14B-Instruct"
MODEL_QWEN_32B="/data/jfeng/models/Qwen2.5-Coder-32B-Instruct"

MODEL_DEEPSEEK_6B7="/data/jfeng/models/Deepseek-Coder-6.7b-Instruct"
MODEL_DEEPSEEK_33B="/data/jfeng/models/Deepseek-Coder-33b-Instruct"

# ---------------- Jobs (ORDER matters) ----------------
# Each job: method | model_name | model_path | input_jsonl | output_jsonl
JOBS=(
  "${METHOD}|CodeLlama-7b-Instruct-hf|${MODEL_CODELLAMA_7B}|${IN_DIR}/CodeLlama-7b-Instruct-hf.jsonl|${OUT_DIR}/CodeLlama-7b-Instruct-hf.jsonl"
  "${METHOD}|CodeLlama-34b-Instruct-hf|${MODEL_CODELLAMA_34B}|${IN_DIR}/CodeLlama-34b-Instruct-hf.jsonl|${OUT_DIR}/CodeLlama-34b-Instruct-hf.jsonl"

  "${METHOD}|Deepseek-Coder-6.7b-Instruct|${MODEL_DEEPSEEK_6B7}|${IN_DIR}/Deepseek-Coder-6.7b-Instruct.jsonl|${OUT_DIR}/Deepseek-Coder-6.7b-Instruct.jsonl"
  "${METHOD}|Deepseek-Coder-33b-Instruct|${MODEL_DEEPSEEK_33B}|${IN_DIR}/Deepseek-Coder-33b-Instruct.jsonl|${OUT_DIR}/Deepseek-Coder-33b-Instruct.jsonl"

  "${METHOD}|Qwen2.5-Coder-7B-Instruct|${MODEL_QWEN_7B}|${IN_DIR}/Qwen2.5-Coder-7B-Instruct.jsonl|${OUT_DIR}/Qwen2.5-Coder-7B-Instruct.jsonl"
  "${METHOD}|Qwen2.5-Coder-14B-Instruct|${MODEL_QWEN_14B}|${IN_DIR}/Qwen2.5-Coder-14B-Instruct.jsonl|${OUT_DIR}/Qwen2.5-Coder-14B-Instruct.jsonl"
  "${METHOD}|Qwen2.5-Coder-32B-Instruct|${MODEL_QWEN_32B}|${IN_DIR}/Qwen2.5-Coder-32B-Instruct.jsonl|${OUT_DIR}/Qwen2.5-Coder-32B-Instruct.jsonl"
)

# ---------------- Run ----------------
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "SCRIPT=${SCRIPT}"
echo "UNITTEST_FLAG='${UNITTEST_FLAG}'"
echo "IN_DIR=${IN_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "========================================================"

mkdir -p "${OUT_DIR}"

for job in "${JOBS[@]}"; do
  IFS="|" read -r method model_name model_path in_jsonl out_jsonl <<< "${job}"

  echo
  echo ">>> METHOD: ${method}"
  echo ">>> MODEL : ${model_name}"
  echo ">>> IN    : ${in_jsonl}"
  echo ">>> OUT   : ${out_jsonl}"
  echo ">>> PATH  : ${model_path}"
  echo "--------------------------------------------------------"

  # Safety checks to catch mis-alignment early
  if [[ ! -f "${in_jsonl}" ]]; then
    echo "ERROR: input file not found: ${in_jsonl}" >&2
    exit 1
  fi
  if [[ ! -d "${model_path}" ]]; then
    echo "ERROR: model_path not found (not a dir): ${model_path}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${out_jsonl}")"

  # Run sequentially on GPUs 0,1 (TP=2).
  ${PYTHON_BIN} "${SCRIPT}" \
    --input "${in_jsonl}" \
    --out "${out_jsonl}" \
    ${UNITTEST_FLAG} \
    --model_path "${model_path}" \
    --tensor_parallel_size 2 \
    --batch_size 8

  echo "✓ Done: ${method}/${model_name}"
done

echo
echo "ALL DONE."
