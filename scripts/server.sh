#!/usr/bin/env bash
set -euo pipefail

# GPU & port
if [[ $# -ge 1 ]]; then
  GPU_ID="$1"
else
  GPU_ID="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
            | awk '{print NR-1, $1}' | sort -k2 -nr | head -n1 | awk '{print $1}')"
fi
PORT="${2:-8000}"

MODEL_DIR="/data/hassan/models/Qwen2.5-Coder-7B-Instruct"
NAME="Qwen2.5-Coder-7B-Instruct"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Disable CUDA graphs + torch compile (avoid the crashy path)
export TORCHINDUCTOR_CUDAGRAPHS=0
export TORCH_CUDAGRAPHS=0
export VLLM_TORCH_COMPILE=0

# Choose ONE attention backend (enum names!)
# If you DON'T have flash-attn, use TRITON_ATTN (works out-of-the-box):
export VLLM_ATTENTION_BACKEND=TRITON_ATTN

# If you DO have flash-attn installed, you can use this instead:
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# (Optional) ensure FlexAttention is not chosen implicitly:
export VLLM_USE_FLEX_ATTENTION=0

echo "[INFO] GPU ${GPU_ID}, port ${PORT}"
echo "[INFO] TORCHINDUCTOR_CUDAGRAPHS=$TORCHINDUCTOR_CUDAGRAPHS TORCH_CUDAGRAPHS=$TORCH_CUDAGRAPHS VLLM_TORCH_COMPILE=$VLLM_TORCH_COMPILE"
echo "[INFO] VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"

vllm serve "${MODEL_DIR}" \
  --tokenizer "${MODEL_DIR}" \
  --dtype float16 \
  --host 127.0.0.1 --port "${PORT}" \
  --served-model-name "${NAME}" \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --swap-space 16 \
  --enforce-eager \
  --api-key hassan2024