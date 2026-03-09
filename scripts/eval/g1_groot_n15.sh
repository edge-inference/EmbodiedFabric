#!/usr/bin/env bash
# G1 loco-manipulation with GR00T N1.5 (3B) -- walk, pick box, place in bin.
# Run from the IsaacLab-Arena directory.

set -euo pipefail

cd /home/modfi/models/vla_simu/extern/IsaacLab-Arena

LOG_ROOT="/home/modfi/models/vla_simu/logs"
VIDEO_FOLDER="${VIDEO_FOLDER:-${LOG_ROOT}/videos/g1_groot_n15}"
RECORD_VIDEO="${RECORD_VIDEO:-1}"
NUM_STEPS="${NUM_STEPS:-1000}"
VIDEO_LENGTH="${VIDEO_LENGTH:-${NUM_STEPS}}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-0}"

VIDEO_ARGS=()
if [[ "${RECORD_VIDEO}" == "1" ]]; then
  mkdir -p "${VIDEO_FOLDER}"
  VIDEO_ARGS=(
    --video
    --video_length "${VIDEO_LENGTH}"
    --video_interval "${VIDEO_INTERVAL}"
    --video_folder "${VIDEO_FOLDER}"
  )
fi

python isaaclab_arena/examples/policy_runner.py \
  --policy_type gr00t_closedloop \
  --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
  --num_steps "${NUM_STEPS}" \
  --device cuda:0 \
  --policy_device cuda:1 \
  --enable_cameras \
  "${VIDEO_ARGS[@]}" \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_joint 2>&1 | tee /home/modfi/models/vla_simu/logs/g1_groot_eval.log
