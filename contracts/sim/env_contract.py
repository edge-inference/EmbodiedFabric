"""
Environment Contract

Interface for physics simulation backends.
Current backends: Isaac Sim (Arena), MuJoCo (GR00T WBC), OmniGibson (BEHAVIOR).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class PhysicsBackend(Enum):
    ISAAC_SIM = "isaac_sim"       
    MUJOCO = "mujoco"             
    OMNIGIBSON = "omnigibson"     


@dataclass
class SceneConfig:
    scene_id: str
    backend: PhysicsBackend = PhysicsBackend.ISAAC_SIM
    objects: List[Dict[str, Any]] = field(default_factory=list)
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    time_step: float = 0.02       # 50Hz default physics step
    render_enabled: bool = True


@dataclass
class ObjectState:
    object_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # quaternion (w,x,y,z)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class EnvironmentState:
    timestamp: float
    step_count: int
    objects: Dict[str, ObjectState] = field(default_factory=dict)
    robot_states: Dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """All sim backends implement this to be swappable."""

    @abstractmethod
    def initialize(self, config: SceneConfig) -> bool: ...

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvironmentState: ...

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[EnvironmentState, Dict[str, Any]]: ...

    @abstractmethod
    def get_observation(self, robot_id: str, sensor_id: str) -> Any: ...

    @abstractmethod
    def spawn_robot(self, robot_config: Any, position: Tuple[float, float, float]) -> str: ...

    @abstractmethod
    def close(self) -> None: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...
