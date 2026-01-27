#!/usr/bin/env python3
"""
Test TensorRT-OpenVLA Server

This sends a single image + instruction to the TRT-OpenVLA server.

Usage:
  TRT_OPENVLA_URL=http://127.0.0.1:8000/act python examples/trt_openvla_test.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.vla.trt_openvla import TRTOpenVLAClient
from simulator.vla.interface import VLAObservation


def main():
    url = os.getenv("TRT_OPENVLA_URL", "http://127.0.0.1:8000/act")
    print(f"Testing TRT-OpenVLA server at: {url}")

    # Create a dummy observation
    rgb = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    obs = VLAObservation(
        rgb_image=rgb,
        depth_image=np.zeros((256, 256), dtype=np.float32),
        instruction="Move forward",
        proprioception=np.zeros(8, dtype=np.float32)
    )

    client = TRTOpenVLAClient(url=url)
    action = client.predict(obs)

    print(f"Action: base_velocity={action.base_velocity}, gripper={action.gripper_action}")
    if action.reasoning:
        print(f"Reasoning: {action.reasoning}")


if __name__ == "__main__":
    main()
