#!/usr/bin/env bash
# Setup conda "behavior" env: OmniGibson + Isaac Sim + bddl + GR00T.
# Requires: conda, GPU with RT cores, ~80GB free disk for BEHAVIOR-1K dataset.
# Set OMNIGIBSON_DATA_PATH to use a different disk for dataset extraction.
set -euo pipefail

GROOT_DIR="/home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6"
BEHAVIOR_DIR="${GROOT_DIR}/BEHAVIOR-1K"
[ -d "${GROOT_DIR}/external_dependencies/BEHAVIOR-1K" ] && BEHAVIOR_DIR="${GROOT_DIR}/external_dependencies/BEHAVIOR-1K"

if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found."
  exit 1
fi

if [ ! -d "$BEHAVIOR_DIR" ]; then
  mkdir -p "$(dirname "$BEHAVIOR_DIR")"
  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$BEHAVIOR_DIR"
fi
cd "$BEHAVIOR_DIR"
git checkout feat/task-progress 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -q "^behavior "; then
  conda activate behavior
  if ! python -c "import omnigibson" 2>/dev/null; then
    echo "WARNING: env 'behavior' exists but omnigibson missing. Re-run: cd $BEHAVIOR_DIR && ./setup.sh --new-env --omnigibson --bddl --dataset --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos"
    exit 1
  fi
else
  echo "Installing OmniGibson + Isaac Sim via BEHAVIOR-1K setup.sh..."
  ./setup.sh --new-env --omnigibson --bddl --dataset \
    --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
  conda activate behavior
fi

# GR00T: torch first, then prebuilt flash-attn wheel, then editable install
cd "$GROOT_DIR"
pip install "torch==2.7.1" "torchvision==0.22.1" -q
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl" -q
pip install -e . -q

python gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py

echo "Done. Usage: conda activate behavior && cd $GROOT_DIR && python gr00t/eval/rollout_policy.py ..."
