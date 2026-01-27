#!/usr/bin/env python3
"""
OpenVLA Batch Inference Test

This script tests whether OpenVLA can accept batched inputs.
Run with:
  CUDA_VISIBLE_DEVICES=0,1 python examples/openvla_batch_test.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.vla.openvla import OpenVLAModel
from simulator.vla.interface import VLAObservation


def main():
    print("Loading OpenVLA...")
    model = OpenVLAModel(quantization="none")
    if not model.load():
        print("Failed to load OpenVLA.")
        return 1
    
    # Create fake observations (random images)
    obs_list = []
    for i in range(3):
        rgb = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        obs = VLAObservation(
            rgb_image=rgb,
            depth_image=np.zeros((256, 256), dtype=np.float32),
            instruction=f"Move forward {i}",
            proprioception=np.zeros(8, dtype=np.float32)
        )
        obs_list.append(obs)
    
    print("Running batch inference...")
    actions = model.predict_batch(obs_list)
    
    for i, action in enumerate(actions):
        print(f"Action {i}: velocity={action.base_velocity}, gripper={action.gripper_action}, conf={action.confidence}")
    
    print("Batch test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
