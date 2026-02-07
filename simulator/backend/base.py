"""
Physics Backend Base Class

Abstract interface for physics engines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


@dataclass
class RobotObservation:
    """Sensor observation from a robot"""
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
    """Control mode for robot commands"""
    HIGH_LEVEL = "high_level"   # Abstract: move_by, turn_by, grasp
    LOW_LEVEL = "low_level"     # Direct: joint velocities/positions


@dataclass
class RobotCommand:
    """Command to send to a robot"""
    robot_id: str
    linear_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_action: Optional[float] = None       # None=no change, 0=open, 1=close
    arm_target: Optional[np.ndarray] = None      # Joint targets or EE pose
    control_mode: str = ControlMode.HIGH_LEVEL   # high_level or low_level
    joint_velocities: Optional[np.ndarray] = None  # For low-level: per-joint velocities
    joint_positions: Optional[np.ndarray] = None   # For low-level: per-joint targets


class PhysicsBackend(ABC):
    """
    Abstract physics backend interface.
    
    Implementations: TDW, Isaac Sim, MuJoCo, Mock
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the physics engine"""
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset the environment"""
        pass
    
    @abstractmethod
    def step(self) -> None:
        """Advance physics by one timestep"""
        pass
    
    @abstractmethod
    def spawn_robot(self, 
                    robot_id: str,
                    position: Tuple[float, float, float],
                    robot_type: str = "amr") -> bool:
        """Spawn a robot in the environment"""
        pass
    
    @abstractmethod
    def get_observation(self, robot_id: str) -> RobotObservation:
        """Get sensor observation for a robot"""
        pass
    
    @abstractmethod
    def send_command(self, command: RobotCommand) -> bool:
        """Send control command to a robot"""
        pass
    
    @abstractmethod
    def spawn_object(self,
                     object_id: str,
                     object_type: str,
                     position: Tuple[float, float, float]) -> bool:
        """Spawn an object in the environment"""
        pass
    
    @abstractmethod
    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        """Get object position"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cleanup and close the backend"""
        pass
    
    @property
    @abstractmethod
    def sim_time(self) -> float:
        """Current simulation time in seconds"""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend"""
        pass
