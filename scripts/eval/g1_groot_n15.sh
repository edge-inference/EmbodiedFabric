#!/usr/bin/env bash
# G1 loco-manipulation with GR00T N1.5 (3B) -- walk, pick box, place in bin.
# Run from the IsaacLab-Arena directory.

cd /home/modfi/models/vla_simu/extern/IsaacLab-Arena

python isaaclab_arena/examples/policy_runner.py \
  --policy_type gr00t_closedloop \
  --policy_config_yaml_path isaaclab_arena_gr00t/g1_locomanip_gr00t_closedloop_config.yaml \
  --num_steps 3600 \
  --device cuda:0 \
  --policy_device cuda:1 \
  --enable_cameras \
  galileo_g1_locomanip_pick_and_place \
  --object brown_box \
  --embodiment g1_wbc_joint 2>&1 | tee /home/modfi/models/vla_simu/logs/g1_groot_eval.log
