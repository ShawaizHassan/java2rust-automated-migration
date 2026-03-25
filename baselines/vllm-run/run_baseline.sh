#!/usr/bin/env bash
set -euo pipefail

# =========================
# Sequential batch runner
# - Runs methods in order: COT -> Vert -> Instruction
# - For each method, runs all models (one-by-one)
# - Uses GPUs 0,1 (tensor_parallel_size=2 inside your python script)
# - Saves:
#   /data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement/<method>/<model-name>.jsonl
# =========================

export CUDA_VISIBLE_DEVICES=0,1

DATASET="/data/hassan/java2rust-automated-migration/data/processed/final_test.jsonl"
SCRIPT="baseline.py"

OUT_BASE="/data/hassan/java2rust-automated-migration/experiments/baseline/no-refinement"

BATCH_SIZE=64
MAX_TOKENS=2048
RAG_K=2
ICL_K=2

MODELS=(
  "/data/hassan/models/CodeLlama-34b-Instruct-hf"
  "/data/jfeng/models/Qwen2.5-Coder-32B-Instruct"
  "/data/jfeng/models/Deepseek-Coder-33b-Instruct"
  "/data/jfeng/models/Deepseek-Coder-6.7b-Instruct"
  "/data/hassan/models/deepseek-coder-1.3b-instruct"
  "/data/hassan/models/CodeLlama-7b-Instruct-hf"
  "/data/jfeng/models/Qwen2.5-Coder-7B-Instruct"
  "/data/jfeng/models/Qwen2.5-Coder-14B-Instruct"
)

METHODS=(
  "Instruction"
)

for METHOD in "${METHODS[@]}"; do
  mkdir -p "${OUT_BASE}/${METHOD}"

  echo "============================================================"
  echo "[START METHOD] ${METHOD}"
  echo "============================================================"

  for HF_PATH in "${MODELS[@]}"; do
    MODEL_NAME="$(basename "${HF_PATH}")"
    OUT_FILE="${OUT_BASE}/${METHOD}/${MODEL_NAME}.jsonl"

    echo "------------------------------------------------------------"
    echo "[RUN] method=${METHOD} model=${MODEL_NAME}"
    echo "[OUT] ${OUT_FILE}"
    echo "------------------------------------------------------------"

    python3 "${SCRIPT}" \
      --dataset "${DATASET}" \
      --output "${OUT_FILE}" \
      --model_name "${MODEL_NAME}" \
      --method "${METHOD}" \
      --hf_model_path "${HF_PATH}" \
      --batch_size "${BATCH_SIZE}" \
      --max_tokens "${MAX_TOKENS}" \
      --rag_k "${RAG_K}" \
      --icl_k "${ICL_K}"

    echo "[DONE] ${METHOD} :: ${MODEL_NAME}"
  done

  echo "============================================================"
  echo "[DONE METHOD] ${METHOD}"
  echo "============================================================"
done

echo "All runs finished."
