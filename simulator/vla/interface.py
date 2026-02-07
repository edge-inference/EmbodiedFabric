"""
VLA Interface

Abstract interface for Vision-Language-Action models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any
import numpy as np


@dataclass
class VLAObservation:
    """
    Input to VLA model.
    
    Combines:
    - Visual input (RGB, depth)
    - Language instruction
    - Proprioceptive state
    - Context from DSM (neighbor states, shared knowledge)
    """
    rgb_image: np.ndarray                        # (H, W, 3) uint8
    depth_image: np.ndarray                      # (H, W) float32, meters
    instruction: str                             # Natural language task
    proprioception: np.ndarray                   # Robot state (joints, gripper)
    
    context: Optional[Dict[str, Any]] = None     # DSM context (R1 integration)
    history: Optional[List['VLAObservation']] = None  # Previous observations


@dataclass
class VLAAction:
    """
    Output from VLA model.
    
    Actions can be:
    - Continuous (velocity commands)
    - Discrete (grasp/release)
    - Hybrid (move to position, then grasp)
    
    Control modes:
    - high_level: Abstract commands (move_by, grasp)
    - low_level: Direct joint control
    """
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
    """
    Abstract VLA interface.
    
    All VLA implementations must provide:
    - predict(): Run inference
    - get_metrics(): Return profiling data
    """
    
    @abstractmethod
    def predict(self, observation: VLAObservation) -> VLAAction:
        """
        Run VLA inference.
        
        Args:
            observation: Sensor + language + context input
            
        Returns:
            Action to execute
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> VLAMetrics:
        """Get metrics from last inference"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (for new episode)"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name of the VLA model"""
        pass
    
    @property
    @abstractmethod
    def latency_budget_ms(self) -> float:
        """Target latency for this model"""
        pass


_VLA_CACHE: Dict[str, VLAInterface] = {}


def _cache_key(model_name: str, kwargs: Dict[str, Any]) -> str:
    """Create a stable cache key for VLA models."""
    if model_name == "openvla":
        quant = kwargs.get("quantization", "4bit")
        latency = kwargs.get("latency_budget_ms", 100.0)
        device = kwargs.get("device", "cuda")
        return f"{model_name}|{quant}|{latency}|{device}"
    if model_name == "cogact":
        model_id = kwargs.get("model_id", "CogACT/CogACT-Small")
        action_model_type = kwargs.get("action_model_type", "DiT-S")
        return f"{model_name}|{model_id}|{action_model_type}"
    if model_name == "cogact_server":
        url = kwargs.get("url", "http://127.0.0.1:5500/act_batch")
        return f"{model_name}|{url}"
    if model_name == "nomad":
        device = kwargs.get("device", "cuda:0")
        return f"{model_name}|{device}"
    if model_name == "hierarchical":
        device = kwargs.get("device", "cuda:0")
        return f"{model_name}|{device}"
    if model_name in ("smolvla", "pi0", "groot"):
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
    
    if model_name == "openvla":
        from .openvla import OpenVLAModel
        _VLA_CACHE[key] = OpenVLAModel(**kwargs)
    elif model_name == "cogact":
        from .cogact import CogACTVLA
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = CogACTVLA(**kwargs)
    elif model_name == "cogact_server":
        from .cogact_server import CogACTServerClient
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = CogACTServerClient(**kwargs)
    elif model_name == "trt_openvla":
        from .trt_openvla import TRTOpenVLAClient
        _VLA_CACHE[key] = TRTOpenVLAClient(**kwargs)
    elif model_name == "profiled":
        from .profiled_vla import ProfiledVLA
        _VLA_CACHE[key] = ProfiledVLA(**kwargs)
    elif model_name == "nomad":
        from .nomad import NoMaDNavigator
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = NoMaDNavigator(**kwargs)
    elif model_name == "nomad_mock":
        from .nomad import MockNoMaDNavigator
        _VLA_CACHE[key] = MockNoMaDNavigator(**kwargs)
    elif model_name == "hierarchical":
        from .hierarchical import HierarchicalPlanner
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = HierarchicalPlanner(**kwargs)
    elif model_name == "smolvla":
        from .lerobot_vla import SmolVLAModel
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = SmolVLAModel(**kwargs)
    elif model_name == "pi0":
        from .lerobot_vla import Pi0Model
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = Pi0Model(**kwargs)
    elif model_name == "groot":
        from .lerobot_vla import GR00TModel
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = GR00TModel(**kwargs)
    elif model_name == "lerobot_server":
        from .lerobot_vla import LeRobotServerClient
        kwargs.pop("quantization", None)
        _VLA_CACHE[key] = LeRobotServerClient(**kwargs)
    else:
        raise ValueError(f"Unknown VLA model: {model_name}. "
                        f"Options: openvla, cogact, cogact_server, nomad, hierarchical, "
                        f"smolvla, pi0, groot, lerobot_server, profiled")
    
    return _VLA_CACHE[key]