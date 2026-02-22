#!/usr/bin/env bash
# Profiled G1 loco-manipulation run with optional torch.profiler tracing.
# Usage:
#   ./g1_groot_profiled.sh              # wall-clock timing only
#   ./g1_groot_profiled.sh --torch      # + torch.profiler on first 5 VLA inference calls

TORCH_FLAG=""
if [[ "$1" == "--torch" ]]; then
    TORCH_FLAG="--torch_profile --profile_steps 5"
    echo "[INFO] torch.profiler enabled for 5 inference calls"
fi

cd /home/modfi/models/vla_simu/extern/IsaacLab-Arena

python /home/modfi/models/vla_simu/scripts/profiling/vla_pipeline.py \
  --policy_type gr00t_closedloop \
  --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
  --num_steps 200 \
  --device cuda:0 \
  --policy_device cuda:1 \
  --headless \
  --enable_cameras \
  $TORCH_FLAG \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_joint 2>&1 | tee /home/modfi/models/vla_simu/logs/profiling/output.log
