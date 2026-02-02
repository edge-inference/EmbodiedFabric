#!/bin/bash
# PhysicAI Environment Setup Script
# Run with: sudo bash setup.sh
set -e

echo "=== PhysicAI Setup ==="

# 1. Add user to docker group
# USER_NAME=${SUDO_USER:-$USER}
# if ! groups $USER_NAME | grep -q docker; then
#     echo "[1/4] Adding $USER_NAME to docker group..."
#     usermod -aG docker $USER_NAME
#     echo "      NOTE: Log out and back in for group changes to take effect"
# else
#     echo "[1/4] User already in docker group"
# fi

# # 2. Install docker-compose plugin (v2)
# if ! docker compose version &>/dev/null; then
#     echo "[2/4] Installing Docker Compose plugin..."
#     DOCKER_CONFIG=${DOCKER_CONFIG:-/usr/local/lib/docker}
#     mkdir -p $DOCKER_CONFIG/cli-plugins
#     curl -SL https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-linux-x86_64 \
#         -o $DOCKER_CONFIG/cli-plugins/docker-compose
#     chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose
#     echo "      Installed: $(docker compose version)"
# else
#     echo "[2/4] Docker Compose already installed: $(docker compose version)"
# fi

# 3. Test GPU access
echo "[3/4] Testing Docker GPU access..."
if docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi -L 2>/dev/null; then
    echo "      GPU access confirmed"
else
    echo "      WARNING: GPU access failed. Ensure nvidia-container-toolkit is installed:"
    echo "        apt-get install -y nvidia-container-toolkit"
    echo "        systemctl restart docker"
fi

# 4. Verify submodule
echo "[4/4] Checking CogACT submodule..."
if [ -f "extern/CogACT/pyproject.toml" ]; then
    echo "      CogACT submodule present"
else
    echo "      Initializing submodule..."
    git submodule update --init --recursive
fi

echo ""
echo "=== Setup Complete ==="
