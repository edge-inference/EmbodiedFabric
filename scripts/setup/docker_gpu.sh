#!/bin/bash
set -e

echo "=== PhysicAI Docker GPU Setup ==="

# Test GPU access
echo "Checking Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi -L 2>/dev/null; then
    echo "      GPU access ready"
else
    echo "      GPU access failed. Ensure nvidia-container-toolkit is installed:"
    echo "        apt-get install -y nvidia-container-toolkit"
    echo "        systemctl restart docker"
fi

# Verify submodules
echo "Checking submodules..."
git submodule update --init --recursive

echo "=== Setup Complete ==="
