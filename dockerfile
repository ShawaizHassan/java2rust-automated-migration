FROM vllm/vllm-openai:latest

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# System utilities that are commonly useful for research repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for better layer caching
COPY requirements.txt /workspace/requirements.txt

# Install your repo's Python dependencies.
# Keep vllm OUT of requirements.txt if you use this base image.
RUN if [ -f /workspace/requirements.txt ]; then \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt; \
    fi

# Copy the rest of the project
COPY . /workspace

# Optional: make shell scripts executable if you use them
RUN find /workspace -type f -name "*.sh" -exec chmod +x {} \; || true

# Default to an interactive shell for experiments/debugging
CMD ["/bin/bash"]