export CUDA_VISIBLE_DEVICES=3
vllm serve /data/hassan/java2rust-automated-migration/models/Qwen2.5-Coder-7B-Instruct\
    --dtype float16 \
    --port 8000 \
    --served-model-name Qwen2.5-Coder-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --api-key hassan2024