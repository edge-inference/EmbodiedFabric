"""HTTP client for LeRobot VLA servers."""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from ..interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics
from .base import VLAControlMode

logger = logging.getLogger(__name__)


class LeRobotServerClient(VLAInterface):
    def __init__(
        self,
        server_url: Optional[str] = None,
        model_type: str = "smolvla",
        control_mode: str = VLAControlMode.LOW_LEVEL,
        latency_budget_ms: float = 100.0,
        **_: object,
    ):
        self._server_url = server_url or os.getenv("LEROBOT_SERVER_URL", "http://localhost:5700/predict")
        self._batch_url = self._server_url.replace("/predict", "/predict_batch")
        self._model_type = model_type
        self._control_mode = control_mode
        self._latency_budget_ms = latency_budget_ms
        self._last_metrics = VLAMetrics()

    def predict(self, observation: VLAObservation) -> VLAAction:
        import requests
        import base64
        from io import BytesIO

        start = time.perf_counter()

        image = observation.rgb_image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        response = requests.post(
            self._server_url,
            json={
                "image": img_b64,
                "instruction": observation.instruction,
                "proprioception": observation.proprioception.tolist(),
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)

        return VLAAction(
            base_velocity=tuple(data.get("base_velocity", (0.0, 0.0))),
            arm_action=np.array(data["arm_action"]) if data.get("arm_action") else None,
            gripper_action=data.get("gripper_action", 0.5),
            done=data.get("done", False),
            confidence=data.get("confidence", 1.0),
            control_mode=data.get("control_mode", self._control_mode),
            joint_velocities=np.array(data["joint_velocities"]) if data.get("joint_velocities") else None,
            joint_positions=np.array(data["joint_positions"]) if data.get("joint_positions") else None,
            reasoning=f"[{self._model_type.upper()}_SERVER]",
        )

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        import requests
        import base64
        from io import BytesIO

        if not observations:
            return []

        start = time.perf_counter()

        batch = []
        for obs in observations:
            image = obs.rgb_image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)

            buffer = BytesIO()
            Image.fromarray(image).save(buffer, format="JPEG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            batch.append(
                {
                    "image": img_b64,
                    "instruction": obs.instruction,
                    "proprioception": obs.proprioception.tolist(),
                }
            )

        response = requests.post(self._batch_url, json={"batch": batch}, timeout=60)
        response.raise_for_status()
        results_data = response.json().get("results", [])

        self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)

        actions: List[VLAAction] = []
        for data in results_data:
            actions.append(
                VLAAction(
                    base_velocity=tuple(data.get("base_velocity", (0.0, 0.0))),
                    arm_action=np.array(data["arm_action"]) if data.get("arm_action") else None,
                    gripper_action=data.get("gripper_action", 0.5),
                    done=data.get("done", False),
                    confidence=data.get("confidence", 1.0),
                    control_mode=data.get("control_mode", self._control_mode),
                    joint_velocities=np.array(data["joint_velocities"]) if data.get("joint_velocities") else None,
                    joint_positions=np.array(data["joint_positions"]) if data.get("joint_positions") else None,
                    reasoning=f"[{self._model_type.upper()}_SERVER]",
                )
            )
        return actions

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        return None

    @property
    def model_name(self) -> str:
        return f"LeRobotServer({self._model_type})"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms

