#!/usr/bin/env python3
"""
VLA Batch Inference Test

Generic batch test for the selected VLA backend.

Usage:
  VLA_MODEL=cogact CUDA_VISIBLE_DEVICES=0,1 python examples/vla_batch_test.py
  VLA_MODEL=openvla CUDA_VISIBLE_DEVICES=0,1 python examples/vla_batch_test.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.vla.interface import create_vla, VLAObservation


def main():
    vla_model = os.getenv("VLA_MODEL", "cogact")
    print(f"Loading VLA model: {vla_model}")

    kwargs = {}
    if vla_model == "cogact":
        kwargs["model_id"] = os.getenv("COGACT_MODEL_ID", "CogACT/CogACT-Small")
        kwargs["action_model_type"] = os.getenv("COGACT_ACTION_MODEL_TYPE", "DiT-S")
        kwargs["device"] = os.getenv("COGACT_DEVICE", "cuda:0")
    elif vla_model == "openvla":
        kwargs["quantization"] = os.getenv("OPENVLA_QUANTIZATION", "none")

    model = create_vla(vla_model, **kwargs)

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

    # Force load TO report device placement
    if hasattr(model, "load"):
        model.load()

    try:
        import torch
        if hasattr(model, "_model") and model._model is not None:
            print("Model device:", next(model._model.parameters()).device)
        elif hasattr(model, "parameters"):
            print("Model device:", next(model.parameters()).device)
        else:
            print("Model device: unknown")
    except Exception as e:
        print("Model device: unknown (", e, ")")

    print("Running batch inference...")
    if hasattr(model, "predict_batch"):
        actions = model.predict_batch(obs_list)
    else:
        actions = [model.predict(obs) for obs in obs_list]

    for i, action in enumerate(actions):
        print(f"Action {i}: velocity={action.base_velocity}, gripper={action.gripper_action}, conf={action.confidence}")

    print("Batch test complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
