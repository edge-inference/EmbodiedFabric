"""NoMaD navigation wrapper."""

from typing import List, Optional, Tuple
import os
import time
import logging
import numpy as np
from PIL import Image

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics

logger = logging.getLogger(__name__)


class NoMaDNavigator(VLAInterface):
    """Diffusion-based goal-conditioned navigation expert."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 config_path: Optional[str] = None,
                 latency_budget_ms: float = 100.0,
                 device: str = "cuda:0",
                 exploration_mode: bool = False):
        self._model_path = model_path or os.getenv(
            "NOMAD_MODEL_PATH",
            "extern/visualnav-transformer/deployment/model_weights/checkpoints/nomad.pth"
        )
        self._config_path = config_path
        self._latency_budget_ms = latency_budget_ms
        self._device = device
        self._exploration_mode = exploration_mode
        
        self._model = None
        self._loaded = False
        self._last_metrics = VLAMetrics()
        
        self._goal_image: Optional[np.ndarray] = None
        self._context_queue: List[np.ndarray] = []
        self._context_size = 5

    def load(self) -> bool:
        if self._loaded:
            return True
        
        try:
            import torch
            import sys
            
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            nomad_train_path = os.path.join(base_path, "extern/visualnav-transformer/train")
            
            if os.path.exists(nomad_train_path) and nomad_train_path not in sys.path:
                sys.path.insert(0, nomad_train_path)
            
            from vint_train.models.nomad.nomad_vint import NoMaD_ViNT
            from vint_train.models.vint.vint import ViNT
            
            self._model = NoMaD_ViNT(
                context_size=self._context_size,
                len_traj_pred=8,
                learn_angle=True,
                obs_encoder="efficientnet-b0",
                obs_encoding_size=512,
                goal_encoding_size=512,
                mha_num_attention_heads=4,
                mha_num_attention_layers=4,
                mha_ff_dim_factor=4,
            )
            
            if os.path.exists(self._model_path):
                checkpoint = torch.load(self._model_path, map_location=self._device)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self._model.load_state_dict(state_dict, strict=False)
                logger.info("Loaded NoMaD weights: %s", self._model_path)
            else:
                logger.warning("NoMaD weights not found: %s", self._model_path)
            
            self._model.to(self._device).eval()
            self._loaded = True
            return True
            
        except ImportError as e:
            logger.warning("NoMaD import failed: %s", e)
            return False
        except Exception:
            logger.exception("Failed to load NoMaD")
            return False

    def set_goal(self, goal_image: np.ndarray) -> None:
        self._goal_image = goal_image

    def _preprocess_image(self, rgb: np.ndarray) -> 'torch.Tensor':
        import torch
        from torchvision import transforms
        
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)
        
        pil_img = Image.fromarray(rgb)
        
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(pil_img).unsqueeze(0).to(self._device)

    def predict(self, observation: VLAObservation) -> VLAAction:
        if not self._loaded:
            if not self.load():
                return VLAAction(done=True, confidence=0.0)
        
        start = time.perf_counter()
        
        self._context_queue.append(observation.rgb_image)
        if len(self._context_queue) > self._context_size:
            self._context_queue.pop(0)
        
        try:
            import torch
            
            obs_images = torch.cat([
                self._preprocess_image(img) for img in self._context_queue
            ], dim=0).unsqueeze(0)
            
            if self._goal_image is not None:
                goal_tensor = self._preprocess_image(self._goal_image)
            else:
                goal_tensor = obs_images[:, -1]
            
            with torch.no_grad():
                if self._exploration_mode:
                    action = self._model.explore(obs_images)
                else:
                    action = self._model(obs_images, goal_tensor)
            
            action_np = action.cpu().numpy().flatten()
            
            linear_vel = float(action_np[0]) if len(action_np) > 0 else 0.0
            angular_vel = float(action_np[1]) if len(action_np) > 1 else 0.0
            
            linear_vel = np.clip(linear_vel, -0.5, 0.5)
            angular_vel = np.clip(angular_vel, -1.0, 1.0)
            
        except Exception:
            logger.exception("NoMaD inference failed")
            linear_vel, angular_vel = 0.0, 0.0
        
        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = VLAMetrics(latency_ms=latency_ms)
        
        return VLAAction(
            base_velocity=(linear_vel, angular_vel),
            gripper_action=0.5,
            confidence=1.0
        )

    def predict_batch(self, observations: List[VLAObservation]) -> List[VLAAction]:
        """Batch prediction for multiple robots."""
        return [self.predict(obs) for obs in observations]

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        """Reset navigation state."""
        self._context_queue.clear()
        self._goal_image = None

    @property
    def model_name(self) -> str:
        return "NoMaD"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms


class MockNoMaDNavigator(VLAInterface):
    """Simple heuristic navigator used when NoMaD isn't available."""

    def __init__(self, latency_budget_ms: float = 100.0):
        self._latency_budget_ms = latency_budget_ms
        self._last_metrics = VLAMetrics()
        self._step = 0

    def predict(self, observation: VLAObservation) -> VLAAction:
        start = time.perf_counter()
        
        self._step += 1
        phase = (self._step // 50) % 4
        
        if phase == 0:
            linear, angular = 0.3, 0.0
        elif phase == 1:
            linear, angular = 0.1, 0.5
        elif phase == 2:
            linear, angular = 0.3, 0.0
        else:
            linear, angular = 0.1, -0.5
        
        linear += np.random.normal(0, 0.02)
        angular += np.random.normal(0, 0.05)
        
        latency_ms = (time.perf_counter() - start) * 1000
        self._last_metrics = VLAMetrics(latency_ms=latency_ms)
        
        return VLAAction(
            base_velocity=(linear, angular),
            confidence=0.9
        )

    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics

    def reset(self) -> None:
        self._step = 0

    @property
    def model_name(self) -> str:
        return "MockNoMaD"

    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
