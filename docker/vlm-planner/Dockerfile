# VLM Planner Server (CogVLM2)
# High-level task planner for robot control
#
# Build:
#   docker build -t vlm-planner -f docker/vlm-planner/Dockerfile .
#
# Run:
#   docker run --gpus '"device=1"' -p 5600:5600 vlm-planner

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Transformers and VLM dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=0.25.0 \
    bitsandbytes>=0.42.0 \
    huggingface_hub>=0.20.0 \
    sentencepiece \
    pillow \
    numpy

# CogVLM2 specific dependencies
RUN pip install --no-cache-dir \
    xformers \
    einops \
    timm

WORKDIR /app

# Copy simulator code for imports
COPY simulator/ /app/simulator/
COPY examples/vlm_planner_server.py /app/server.py

EXPOSE 5600

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5600/health || exit 1

CMD ["python", "/app/server.py", "--host", "0.0.0.0", "--port", "5600"]
