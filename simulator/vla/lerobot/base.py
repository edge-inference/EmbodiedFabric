"""Common utilities for LeRobot-backed VLA models."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class VLAControlMode:
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"


@dataclass
class LeRobotConfig:
    model_type: str
    model_path: str
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    compile_model: bool = False
    action_chunk_size: int = 10
    latency_budget_ms: float = 100.0
    control_mode: str = VLAControlMode.LOW_LEVEL


class BaseLeRobotVLA(VLAInterface, ABC):
    def __init__(self, config: LeRobotConfig):
        self._config = config
        self._policy = None
        self._loaded = False
        self._last_metrics = VLAMetrics()
        self._action_buffer: List[np.ndarray] = []
        self._buffer_idx = 0

    def _load_model(self) -> None:
        raise NotImplementedError

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def _preprocess_observation(self, obs: VLAObservation) -> Dict[str, Any]:
        image = obs.rgb_image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_img = Image.fromarray(image)

        return {
            "observation.images.camera1": pil_img,
            "observation.images.camera2": pil_img,
            "observation.images.camera3": pil_img,
            "observation.state": obs.proprioception,
            "task": obs.instruction,
        }

    def _postprocess_action(self, raw_action: np.ndarray) -> VLAAction:
        control_mode = self._config.control_mode

        if control_mode == VLAControlMode.LOW_LEVEL:
            if len(raw_action) >= 2:
                forward = abs(float(raw_action[0])) * 0.3
                turn = float(raw_action[1]) * 0.5
                base_vel = (forward, turn)
            else:
                base_vel = (0.0, 0.0)

            return VLAAction(
                base_velocity=base_vel,
                arm_action=raw_action,
                gripper_action=float(raw_action[-1]) if len(raw_action) > 0 else 0.5,
                confidence=1.0,
                control_mode=control_mode,
                joint_velocities=raw_action,
                reasoning=f"[{self._config.model_type.upper()}] low-level",
            )

        if len(raw_action) >= 7:
            base_vel = (float(raw_action[0]), float(raw_action[1]))
            arm_action = raw_action[2:6]
            gripper = float(raw_action[6])
        elif len(raw_action) >= 2:
            base_vel = (float(raw_action[0]), float(raw_action[1]))
            arm_action = None
            gripper = 0.5
        else:
            base_vel = (0.0, 0.0)
            arm_action = raw_action if len(raw_action) > 0 else None
            gripper = 0.5

        return VLAAction(
            base_velocity=base_vel,
            arm_action=arm_action,
            gripper_action=gripper,
            confidence=1.0,
            control_mode=control_mode,
            reasoning=f"[{self._config.model_type.upper()}] high-level",
        )

    @property
    def control_mode(self) -> str:
        return self._config.control_mode

    def set_control_mode(self, mode: str) -> None:
        if mode in (VLAControlMode.HIGH_LEVEL, VLAControlMode.LOW_LEVEL):
            self._config.control_mode = mode

    def predict(self, observation: VLAObservation) -> VLAAction:
        if not self._loaded:
            self._load_model()

        start = time.perf_counter()

        if self._action_buffer and self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
        else:
            inputs = self._preprocess_observation(observation)
            raw_actions = self._run_inference(inputs)

            if raw_actions.ndim == 2:
                self._action_buffer = [raw_actions[i] for i in range(raw_actions.shape[0])]
            else:
                self._action_buffer = [raw_actions]

            self._buffer_idx = 1
            action = self._action_buffer[0]

        result = self._postprocess_action(action)
        self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)
        return result

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        if not observations:
            return []
        return [self.predict(obs) for obs in observations]

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        self._action_buffer.clear()
        self._buffer_idx = 0

    @property
    def latency_budget_ms(self) -> float:
        return self._config.latency_budget_ms

