# CogACT Inference Server
# Keeps model loaded in GPU memory, serves actions over HTTP
#
# Build:
#   docker build -t cogact-server -f docker/cogact-server/Dockerfile .
#
# Run:
#   docker run --gpus '"device=0"' -p 5500:5500 --name cogact cogact-server

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
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# TensorFlow (required by CogACT data pipeline)
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    tensorflow_datasets==4.9.3 \
    tensorflow_graphics==2021.12.3

# CogACT core dependencies
RUN pip install --no-cache-dir \
    accelerate>=0.25.0 \
    draccus>=0.8.0 \
    einops \
    huggingface_hub \
    json-numpy \
    jsonlines \
    matplotlib \
    peft==0.11.1 \
    protobuf \
    rich \
    sentencepiece==0.1.99 \
    timm==0.9.10 \
    tokenizers==0.19.1 \
    transformers==4.40.1

# Additional deps for server
RUN pip install --no-cache-dir \
    Pillow \
    numpy

# Install dlimp and openvla from git
RUN pip install --no-cache-dir \
    "dlimp @ git+https://github.com/moojink/dlimp_openvla" \
    "openvla @ git+https://github.com/arnoldland/openvla"

# Copy CogACT source
WORKDIR /app
COPY extern/CogACT /app/cogact
RUN pip install --no-cache-dir -e /app/cogact

# Copy server script
COPY examples/cogact_server.py /app/server.py

# Pre-download model weights (optional - comment out to download at runtime)
# RUN python -c "from vla import load_vla; load_vla('CogACT/CogACT-Small', load_for_training=False)"

EXPOSE 5500

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

CMD ["python", "/app/server.py", "--host", "0.0.0.0", "--port", "5500"]
