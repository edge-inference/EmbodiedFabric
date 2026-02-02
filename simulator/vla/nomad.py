"""
NoMaD Navigation Expert

Diffusion-based goal-conditioned navigation policy.
Repo: https://github.com/robodhruv/visualnav-transformer

Outputs (v, omega) velocity commands at 5-10 Hz for mobile base navigation.
"""

from typing import List, Optional, Tuple
import os
import time
import numpy as np
from PIL import Image

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class NoMaDNavigator(VLAInterface):
    """
    NoMaD navigation expert wrapper.
    
    Uses diffusion policy to generate multimodal action distributions
    for goal-directed navigation and exploration.
    """

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
        """Load NoMaD model weights."""
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
                print(f"Loaded NoMaD weights from {self._model_path}")
            else:
                print(f"NoMaD weights not found at {self._model_path}")
                print("Download from: https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg")
            
            self._model.to(self._device).eval()
            self._loaded = True
            return True
            
        except ImportError as e:
            print(f"NoMaD import failed: {e}")
            print("Ensure extern/visualnav-transformer is cloned and dependencies installed")
            return False
        except Exception as e:
            print(f"Failed to load NoMaD: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_goal(self, goal_image: np.ndarray) -> None:
        """Set navigation goal as an image."""
        self._goal_image = goal_image

    def _preprocess_image(self, rgb: np.ndarray) -> 'torch.Tensor':
        """Convert RGB image to model input format."""
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
        """Generate navigation action using diffusion policy."""
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
            
        except Exception as e:
            print(f"NoMaD inference failed: {e}")
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
    """
    Mock NoMaD for testing without actual model weights.
    Generates plausible navigation commands based on simple heuristics.
    """

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
