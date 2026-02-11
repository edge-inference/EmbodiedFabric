#!/bin/bash
set -e

echo "=== PhysicAI Setup ==="

# 1. Add user to docker group
# USER_NAME=${SUDO_USER:-$USER}
# if ! groups $USER_NAME | grep -q docker; then
#     echo "[1/4] Adding $USER_NAME to docker group..."
#     usermod -aG docker $USER_NAME
# else
#     echo "already in group"
# fi

# # 2. Install docker-compose plugin
# if ! docker compose version &>/dev/null; then
#     echo "find and install docker compose plugin"

# 3. Test GPU access
echo "Checking Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi -L 2>/dev/null; then
    echo "      GPU access ready"
else
    echo "        GPU access failed. See ifnvidia-container-toolkit is installed:"
    echo "        apt-get install -y nvidia-container-toolkit"
    echo "        systemctl restart docker"
fi

# 4. Verify submodule, could be anything needing to run in the container.
echo "[4/4] Checking CogACT submodule..."
if [ -f "extern/CogACT/pyproject.toml" ]; then
    echo "      CogACT submodule present"
else
    echo "      Initializing submodule..."
    git submodule update --init --recursive
fi

echo "=== Setup Complete ==="
