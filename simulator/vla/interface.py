"""VLA model interface + factory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


@dataclass
class VLAObservation:
    """Input to a VLA model."""
    rgb_image: np.ndarray                        # (H, W, 3) uint8
    depth_image: np.ndarray                      # (H, W) float32, meters
    instruction: str                             # Natural language task
    proprioception: np.ndarray                   # Robot state (joints, gripper)
    
    context: Optional[Dict[str, Any]] = None     # DSM context (R1 integration)
    history: Optional[List['VLAObservation']] = None  # Previous observations


@dataclass
class VLAAction:
    """Output from a VLA model."""
    base_velocity: Tuple[float, float] = (0.0, 0.0)    # (linear m/s, angular rad/s)
    arm_action: Optional[np.ndarray] = None             # Joint deltas or EE target
    gripper_action: float = 0.5                         # 0=open, 1=closed
    
    done: bool = False                                  # Task complete signal
    confidence: float = 1.0                             # Action confidence
    
    reasoning: Optional[str] = None                     # Explanation (for debugging)
    
    control_mode: str = "high_level"                    # high_level or low_level
    joint_velocities: Optional[np.ndarray] = None       # For low-level: raw joint vels
    joint_positions: Optional[np.ndarray] = None        # For low-level: raw joint pos


@dataclass
class VLAMetrics:
    """Metrics from VLA inference (for profiling)"""
    latency_ms: float = 0.0
    flops: float = 0.0
    memory_bytes: int = 0
    tokens_processed: int = 0


class VLAInterface(ABC):
    """Common VLA API used by the simulator."""
    
    @abstractmethod
    def predict(self, observation: VLAObservation) -> VLAAction:
        pass
    
    @abstractmethod
    def get_metrics(self) -> VLAMetrics:
        """Get metrics from last inference"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def latency_budget_ms(self) -> float:
        """Target latency for this model"""
        pass


_VLA_CACHE: Dict[str, VLAInterface] = {}


def _cache_key(model_name: str, kwargs: Dict[str, Any]) -> str:
    """cache key for VLA models."""
    if model_name in ("smolvla", "pi0"):
        model_path = kwargs.get("model_path", "default")
        device = kwargs.get("device", "cuda:0")
        control_mode = kwargs.get("control_mode", "low_level")
        return f"{model_name}|{model_path}|{device}|{control_mode}"
    return f"{model_name}|default"


def create_vla(model_name: str, **kwargs) -> VLAInterface:
    """Factory function to create VLA instances (cached for shared models)."""
    key = _cache_key(model_name, kwargs)
    if key in _VLA_CACHE:
        return _VLA_CACHE[key]
    
    if model_name == "smolvla":
        from .lerobot_vla import SmolVLAModel
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = SmolVLAModel(**kwargs)
    elif model_name == "pi0":
        from .lerobot_vla import Pi0Model
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = Pi0Model(**kwargs)
    elif model_name == "lerobot_server":
        from .lerobot_vla import LeRobotServerClient
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = LeRobotServerClient(**kwargs)
    elif model_name == "profiled":
        from .profiled_vla import ProfiledVLA
        _VLA_CACHE[key] = ProfiledVLA(**kwargs)
    else:
        raise ValueError(f"Unknown VLA model: {model_name}. "
                        f"Options: smolvla, pi0, lerobot_server, profiled")
    
    return _VLA_CACHE[key]
