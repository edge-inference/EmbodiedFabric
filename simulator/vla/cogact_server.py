"""
CogACT Server Client

Lightweight REST client for a local CogACT server.
"""

from typing import List, Optional
import os
import time
import numpy as np
import requests
import json_numpy

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class CogACTServerClient(VLAInterface):
    """Client for a CogACT HTTP server."""

    def __init__(self,
                 url: Optional[str] = None,
                 latency_budget_ms: float = 100.0):
        self._url = url or os.getenv("COGACT_SERVER_URL", "http://127.0.0.1:5500/act_batch")
        self._latency_budget_ms = latency_budget_ms
        self._last_metrics = VLAMetrics()

    def _pack_image(self, rgb: np.ndarray) -> np.ndarray:
        if rgb.dtype != np.uint8:
            return (rgb * 255).astype(np.uint8)
        return rgb

    def predict(self, observation: VLAObservation) -> VLAAction:
        return self.predict_batch([observation])[0]

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        if not observations:
            return []

        start = time.perf_counter()
        images = [self._pack_image(obs.rgb_image) for obs in observations]
        instructions = [obs.instruction for obs in observations]

        payload = {
            "images": images,
            "instructions": instructions,
        }

        try:
            resp = requests.post(
                self._url,
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = json_numpy.loads(resp.text)
            actions = data.get("actions", [])
        except Exception:
            actions = []

        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = VLAMetrics(
            latency_ms=latency_ms,
            flops=0.0,
            memory_bytes=0,
            tokens_processed=0
        )

        out: List[VLAAction] = []
        for action in actions:
            action = np.array(action).astype(np.float32).flatten()
            out.append(VLAAction(
                base_velocity=(float(action[0]), float(action[1])),
                gripper_action=float(action[6]) if action.shape[0] >= 7 else 0.5,
                confidence=1.0
            ))
        if not out:
            out = [VLAAction(done=True, confidence=0.0) for _ in observations]
        return out

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return "CogACT-Server"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
