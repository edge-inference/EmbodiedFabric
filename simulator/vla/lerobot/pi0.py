"""Pi0 wrapper."""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from .base import BaseLeRobotVLA, LeRobotConfig, VLAControlMode

logger = logging.getLogger(__name__)


class Pi0Model(BaseLeRobotVLA):
    DEFAULT_MODEL = "lerobot/pi0_base"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        latency_budget_ms: float = 100.0,
        control_mode: str = VLAControlMode.LOW_LEVEL,
        **_: Any,
    ):
        super().__init__(
            LeRobotConfig(
                model_type="pi0",
                model_path=model_path,
                device=device,
                latency_budget_ms=latency_budget_ms,
                control_mode=control_mode,
            )
        )

    def _load_model(self) -> None:
        import torch

        try:
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        except ImportError:
            from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

        logger.info("Loading Pi0: %s", self._config.model_path)

        self._policy = PI0Policy.from_pretrained(self._config.model_path).to(self._config.device).eval()
        if self._config.compile_model:
            self._policy = torch.compile(self._policy)
        self._loaded = True

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch

        with torch.no_grad():
            batch = self._prepare_batch(inputs)
            action = self._policy.select_action(batch)
            return action.cpu().numpy()

    def _prepare_batch(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        state = inputs["observation.state"]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self._config.device)
        batch: Dict[str, Any] = {"observation.state": state_tensor}

        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            if key in inputs:
                img_tensor = transform(inputs[key]).unsqueeze(0).to(self._config.device)
                batch[key] = img_tensor

        task_text = inputs.get("task", "")
        processor = getattr(self._policy, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None) if processor is not None else None
        if tokenizer is not None:
            tokens = tokenizer(
                task_text,
                return_tensors="pt",
                padding="max_length",
                max_length=48,
                truncation=True,
            )
            batch["observation.language.tokens"] = tokens["input_ids"].to(self._config.device)
            if "attention_mask" in tokens:
                batch["observation.language.attention_mask"] = tokens["attention_mask"].to(self._config.device)

        return batch

    @property
    def model_name(self) -> str:
        return f"Pi0({self._config.model_path})"

