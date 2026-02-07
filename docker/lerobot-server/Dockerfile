# LeRobot VLA Server (SmolVLA, Pi0, GR00T)
# End-to-end VLA models for direct robot control
#
# Build:
#   docker build -t lerobot-server -f docker/lerobot-server/Dockerfile .
#
# Run:
#   LEROBOT_MODEL=smolvla docker run --gpus '"device=2"' -p 5700:5700 lerobot-server
#   LEROBOT_MODEL=pi0 docker run --gpus '"device=2"' -p 5700:5700 lerobot-server

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch with CUDA 12.1
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --index-url https://download.pytorch.org/whl/cu121

# LeRobot with SmolVLA backend
# Note: Install from git for latest model support
RUN pip install --no-cache-dir \
    "lerobot[smolvla] @ git+https://github.com/huggingface/lerobot.git" \
    transformers>=4.40.0 \
    accelerate>=0.25.0 \
    huggingface_hub>=0.20.0 \
    pillow \
    numpy

RUN pip install --no-cache-dir \
    sentencepiece \
    protobuf

WORKDIR /app

COPY simulator/ /app/simulator/
COPY examples/lerobot_server.py /app/server.py

EXPOSE 5700

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:5700/health || exit 1

CMD ["python", "/app/server.py", "--host", "0.0.0.0", "--port", "5700"]
