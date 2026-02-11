"""CogACT wrapper."""

from typing import List, Optional
import os
import time
import numpy as np
import torch
from PIL import Image

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class CogACTVLA(VLAInterface):
    """CogACT wrapper with batch inference support."""

    def __init__(self,
                 model_id: str = "CogACT/CogACT-Small",
                 action_model_type: str = "DiT-S",
                 future_action_window_size: int = 15,
                 latency_budget_ms: float = 100.0,
                 device: str = "cuda:0"):
        self._model_id = model_id
        self._action_model_type = action_model_type
        self._future_action_window_size = future_action_window_size
        self._latency_budget_ms = latency_budget_ms
        self._device = device
        self._model = None
        self._loaded = False
        self._last_metrics = VLAMetrics()

    def load(self) -> bool:
        if self._loaded:
            return True
        from vla import load_vla
        self._model = load_vla(
            self._model_id,
            load_for_training=False,
            action_model_type=self._action_model_type,
            future_action_window_size=self._future_action_window_size,
        )
        if hasattr(self._model, "vlm"):
            self._model.vlm = self._model.vlm.to(torch.bfloat16)
        self._model.to(self._device).eval()
        self._loaded = True
        return True

    def _to_pil(self, rgb: np.ndarray) -> Image.Image:
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        return Image.fromarray(rgb)

    def _action_from_chunk(self, action_chunk: np.ndarray) -> VLAAction:
        # action_chunk shape: [16, 7] (x,y,z,roll,pitch,yaw,grip)
        step0 = action_chunk[0]
        return VLAAction(
            base_velocity=(float(step0[0]), float(step0[1])),
            gripper_action=float(step0[6]),
            confidence=1.0
        )

    def predict(self, observation: VLAObservation) -> VLAAction:
        actions = self.predict_batch([observation])
        return actions[0]

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        if not observations:
            return []
        if not self._loaded:
            if not self.load():
                return [VLAAction(done=True, confidence=0.0) for _ in observations]

        start = time.perf_counter()
        images = [self._to_pil(obs.rgb_image) for obs in observations]
        prompts = [obs.instruction for obs in observations]

        try:
            # Prefer native batch API if available
            if hasattr(self._model, "predict_action_batch"):
                action_chunks, _ = self._model.predict_action_batch(
                    images,
                    prompts,
                    unnorm_key="fractal20220817_data",
                    cfg_scale=1.5,
                    use_ddim=True,
                    num_ddim_steps=10,
                )
            else:
                action_chunks = []
                for img, prompt in zip(images, prompts):
                    chunk, _ = self._model.predict_action(
                        img,
                        prompt,
                        unnorm_key="fractal20220817_data",
                        cfg_scale=1.5,
                        use_ddim=True,
                        num_ddim_steps=10,
                    )
                    action_chunks.append(chunk)
        except Exception:
            self._last_metrics = VLAMetrics(latency_ms=(time.perf_counter() - start) * 1000)
            return [VLAAction(done=True, confidence=0.0) for _ in observations]

        self._last_metrics = VLAMetrics(
            latency_ms=(time.perf_counter() - start) * 1000,
            flops=0.0,
            memory_bytes=0,
            tokens_processed=0
        )
        return [self._action_from_chunk(np.array(chunk)) for chunk in action_chunks]

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return f"CogACT ({self._model_id})"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
