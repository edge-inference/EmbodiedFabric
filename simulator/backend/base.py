"""Physics backend interface used by the simulator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class RobotObservation:
    robot_id: str
    timestamp: float
    rgb: np.ndarray                              # (H, W, 3) uint8
    depth: np.ndarray                            # (H, W) float32, meters
    position: Tuple[float, float, float]         # (x, y, z)
    rotation: Tuple[float, float, float, float]  # quaternion
    velocity: Tuple[float, float, float]
    gripper_state: float                         # 0=open, 1=closed
    joint_positions: Optional[np.ndarray] = None


class ControlMode:
    HIGH_LEVEL = "high_level"   # Abstract: move_by, turn_by, grasp
    LOW_LEVEL = "low_level"     # Direct: joint velocities/positions


@dataclass
class RobotCommand:
    robot_id: str
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_action: Optional[float] = None       # None=no change, 0=open, 1=close
    arm_target: Optional[np.ndarray] = None      # Joint targets or EE pose
    control_mode: str = ControlMode.HIGH_LEVEL   # high_level or low_level
    joint_velocities: Optional[np.ndarray] = None  # For low-level: per-joint velocities
    joint_positions: Optional[np.ndarray] = None   # For low-level: per-joint targets


class PhysicsBackend(ABC):
    """Backend API for stepping, observing, and commanding robots."""
    
    @abstractmethod
    def initialize(self) -> bool:
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def step(self) -> None:
        pass
    
    @abstractmethod
    def spawn_robot(self, 
                    robot_id: str,
                    position: Tuple[float, float, float],
                    robot_type: str = "amr") -> bool:
        pass
    
    @abstractmethod
    def get_observation(self, robot_id: str) -> RobotObservation:
        pass
    
    @abstractmethod
    def send_command(self, command: RobotCommand) -> bool:
        pass
    
    @abstractmethod
    def spawn_object(self,
                     object_id: str,
                     object_type: str,
                     position: Tuple[float, float, float]) -> bool:
        pass
    
    @abstractmethod
    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass
    
    @property
    @abstractmethod
    def sim_time(self) -> float:
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass
