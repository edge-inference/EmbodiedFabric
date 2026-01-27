"""
TensorRT-OpenVLA Client

REST client for TensorRT-OpenVLA inference server.
See: https://github.com/rail-berkeley/tensorrt-openvla
"""

from typing import Optional, Dict, Any
import os
import time
import numpy as np
import requests
import json_numpy

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class TRTOpenVLAClient(VLAInterface):
    """
    Client for TensorRT-OpenVLA server.
    Expects server endpoint that accepts JSON with:
      - image: np.ndarray (H,W,3) uint8
      - instruction: str
    Returns:
      - action: np.ndarray (7,)
      - reasoning: str (optional)
    """

    def __init__(self,
                 url: Optional[str] = None,
                 latency_budget_ms: float = 100.0):
        self._url = url or os.getenv("TRT_OPENVLA_URL", "http://127.0.0.1:8000/act")
        self._latency_budget_ms = latency_budget_ms
        self._last_metrics = VLAMetrics()

    def predict(self, observation: VLAObservation) -> VLAAction:
        start = time.perf_counter()

        # Ensure uint8 image
        rgb = observation.rgb_image
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        payload = {
            "image": rgb,
            "instruction": observation.instruction,
            "return_ids": False
        }

        try:
            # json_numpy handles numpy arrays
            resp = requests.post(self._url, data=json_numpy.dumps(payload), headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = json_numpy.loads(resp.text)

            action = data.get("action", None)
            reasoning = data.get("reasoning", None)

            if action is None:
                raise ValueError("No action returned from TRT-OpenVLA server")

            # Convert action array to RobotCommand-style action
            action = np.array(action).astype(np.float32).flatten()
            vla_action = VLAAction(
                base_velocity=(float(action[0]), float(action[1])),
                gripper_action=float(action[6]) if action.shape[0] >= 7 else 0.5,
                confidence=1.0,
                reasoning=reasoning
            )

        except Exception as e:
            vla_action = VLAAction(done=True, confidence=0.0, reasoning=str(e))

        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = VLAMetrics(
            latency_ms=latency_ms,
            flops=0.0,
            memory_bytes=0,
            tokens_processed=0
        )

        return vla_action

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "TensorRT-OpenVLA"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
