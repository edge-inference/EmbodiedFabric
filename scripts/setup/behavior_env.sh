#!/usr/bin/env bash
# setup BEHAVIOR-1K sim environment (OmniGibson / Isaac Sim).
# Use an GPU with RT cores.
set -euxo pipefail

GROOT_DIR="/home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6"
BEHAVIOR_DIR="$GROOT_DIR/external_dependencies/BEHAVIOR-1K"

cd "$GROOT_DIR"

# uv env setup 
uv sync --python 3.10
uv pip install -e .

# Clone BEHAVIOR-1K
if [ ! -d "$BEHAVIOR_DIR" ]; then
  git clone https://github.com/StanfordVL/BEHAVIOR-1K.git "$BEHAVIOR_DIR"
fi

cd "$BEHAVIOR_DIR"
git checkout feat/task-progress

source "$GROOT_DIR/.venv/bin/activate"
bash ./setup_uv.sh

cd "$GROOT_DIR"
python gr00t/eval/sim/BEHAVIOR/prepare_test_instances.py

echo "BEHAVIOR env setup complete."
