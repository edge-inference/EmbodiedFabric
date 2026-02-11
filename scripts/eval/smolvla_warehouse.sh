#!/usr/bin/env bash
# Warehouse pick-and-place: GR1 + SmolVLA in a warehouse with shelves. (still WIP)

lerobot-eval \
  --policy.path=nvidia/smolvla-arena-gr1-microwave \
  --env.type=isaaclab_arena \
  --env.hub_path=nvidia/isaaclab-arena-envs \
  --rename_map='{"observation.images.robot_pov_cam_rgb": "observation.images.robot_pov_cam"}' \
  --policy.device=cuda \
  --env.environment=warehouse_pnp \
  --env.embodiment=gr1_pink \
  --env.object=cracker_box \
  --env.task="Pick up the box from the shelf and place it in the sorting bin." \
  --env.headless=false \
  --env.enable_cameras=true \
  --env.video=true \
  --env.video_length=10 \
  --env.video_interval=15 \
  --env.state_keys=robot_joint_pos \
  --env.camera_keys=robot_pov_cam_rgb \
  --trust_remote_code=True \
  --eval.batch_size=1 | tee logs/warehouse_eval.log
