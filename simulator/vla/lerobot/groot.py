"""GR00T wrapper."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from ..interface import VLAObservation, VLAAction
from .base import BaseLeRobotVLA, LeRobotConfig, VLAControlMode

logger = logging.getLogger(__name__)


class GR00TModel(BaseLeRobotVLA):
    DEFAULT_MODEL = "nvidia/GR00T-N1.5-3B"

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        device: str = "cuda:0",
        latency_budget_ms: float = 150.0,
        control_mode: str = VLAControlMode.LOW_LEVEL,
        eagle_processor_path: str | None = None,
        **_: Any,
    ):
        super().__init__(
            LeRobotConfig(
                model_type="groot",
                model_path=model_path,
                device=device,
                latency_budget_ms=latency_budget_ms,
                control_mode=control_mode,
            )
        )
        self._preprocess = None
        self._postprocess = None
        self._eagle_processor = None
        self._eagle_processor_path = (
            eagle_processor_path
            or os.getenv("GROOT_EAGLE_PROCESSOR_PATH")
            or "/home/modfi/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
        )

    def _load_model(self) -> None:
        import torch

        try:
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors
        except ImportError:
            from lerobot.common.policies.groot.modeling_groot import GrootPolicy
            from lerobot.common.policies.groot.processor_groot import make_groot_pre_post_processors

        logger.info("Loading GR00T: %s", self._config.model_path)

        self._policy = GrootPolicy.from_pretrained(self._config.model_path).to(self._config.device).eval()

        try:
            self._policy.config.device = self._config.device
            self._preprocess, self._postprocess = make_groot_pre_post_processors(self._policy.config)
        except Exception:
            logger.exception("GR00T processor init failed; will use Eagle processor fallback")
            self._preprocess = None
            self._postprocess = None

        self._loaded = True

    def _run_inference(self, inputs: Dict[str, Any]) -> np.ndarray:
        import torch

        with torch.no_grad():
            if self._preprocess is not None:
                frame = self._build_frame(inputs)
                batch = self._preprocess(frame)
                action = self._policy.select_action(batch)
                if self._postprocess is not None:
                    action = self._postprocess(action)
            else:
                batch = self._prepare_batch_eagle(inputs)
                action = self._policy.select_action(batch)
            return action.cpu().numpy()

    def _build_frame(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        frame: Dict[str, Any] = {
            "observation.state": torch.tensor(inputs["observation.state"], dtype=torch.float32),
            "task": inputs.get("task", ""),
        }

        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            if key in inputs:
                frame[key] = transform(inputs[key])
        return frame

    def _load_eagle_processor(self):
        from transformers import AutoProcessor

        path = self._eagle_processor_path
        if not os.path.exists(path) and "/" not in path:
            raise FileNotFoundError(f"Eagle processor path not found: {path}")
        self._eagle_processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

    def _prepare_batch_eagle(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        import numpy as np_module

        if self._eagle_processor is None:
            self._load_eagle_processor()

        device = self._config.device

        images: List[Image.Image] = []
        for key in (
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ):
            if key not in inputs:
                continue
            img = inputs[key]
            if isinstance(img, Image.Image):
                images.append(img)
            elif isinstance(img, np_module.ndarray):
                images.append(Image.fromarray(img.astype(np_module.uint8)))

        if not images:
            img = inputs.get("observation.images.camera1")
            if img is None:
                img = np_module.zeros((224, 224, 3), dtype=np_module.uint8)
            images = [Image.fromarray(img.astype(np_module.uint8)) if isinstance(img, np_module.ndarray) else img]

        task = inputs.get("task", "pick up the object")

        content = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": task})
        conversation = [{"role": "user", "content": content}]

        text = self._eagle_processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        img_inputs, _ = self._eagle_processor.process_vision_info(conversation)
        eagle_out = self._eagle_processor(text=text, images=img_inputs, return_tensors="pt", padding=True)

        batch: Dict[str, Any] = {
            f"eagle_{k}": (v.to(device) if hasattr(v, "to") else v) for k, v in eagle_out.items()
        }

        state = inputs["observation.state"]
        state_array = np_module.array(state, dtype=np_module.float32)
        if len(state_array) < 64:
            padded = np_module.zeros(64, dtype=np_module.float32)
            padded[: len(state_array)] = state_array
            state_array = padded

        state_tensor = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        batch["state"] = state_tensor
        batch["state_mask"] = torch.ones(state_tensor.shape[:2], dtype=torch.bool, device=device)
        batch["embodiment_id"] = torch.tensor([31], dtype=torch.long, device=device)
        return batch

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        return [self.predict(obs) for obs in observations]

    @property
    def model_name(self) -> str:
        return f"GR00T({self._config.model_path})"

