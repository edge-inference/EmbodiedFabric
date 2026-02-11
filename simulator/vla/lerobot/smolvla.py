"""SmolVLA wrapper."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseLeRobotVLA, LeRobotConfig, VLAControlMode

logger = logging.getLogger(__name__)


class SmolVLAModel(BaseLeRobotVLA):
    DEFAULT_MODEL = "lerobot/smolvla_base"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        latency_budget_ms: float = 50.0,
        control_mode: str = VLAControlMode.LOW_LEVEL,
        **_: Any,
    ):
        super().__init__(
            LeRobotConfig(
                model_type="smolvla",
                model_path=model_path,
                device=device,
                latency_budget_ms=latency_budget_ms,
                control_mode=control_mode,
            )
        )
        self._preprocess = None
        self._postprocess = None

    def _load_model(self) -> None:
        import torch

        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError:
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.common.policies.factory import make_pre_post_processors

        logger.info("Loading SmolVLA: %s", self._config.model_path)

        self._policy = SmolVLAPolicy.from_pretrained(self._config.model_path).to(self._config.device).eval()

        self._preprocess, self._postprocess = make_pre_post_processors(
            self._policy.config,
            self._config.model_path,
            preprocessor_overrides={"device_processor": {"device": self._config.device}},
        )

        if self._config.compile_model:
            self._policy = torch.compile(self._policy)

        self._loaded = True

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch

        if self._preprocess is None or self._postprocess is None:
            raise RuntimeError("SmolVLA preprocess/postprocess not initialized")

        with torch.no_grad():
            frame = self._build_frame(inputs)
            batch = self._preprocess(frame)
            action = self._policy.select_action(batch)
            action = self._postprocess(action)
            return action.cpu().numpy()

    def _build_frame(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )

        frame: Dict[str, Any] = {}
        frame["observation.state"] = torch.tensor(inputs["observation.state"], dtype=torch.float32)
        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            if key in inputs:
                frame[key] = transform(inputs[key])
        frame["task"] = inputs.get("task", "")
        return frame

    @property
    def model_name(self) -> str:
        return f"SmolVLA({self._config.model_path})"

