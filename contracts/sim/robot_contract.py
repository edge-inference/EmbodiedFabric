"""
Robot Contract

Interface for robotic agents in simulation.
Current robots: Unitree G1 (legged, 43 DoF), GR1 (static, upper body),
NVIDIA R1 Pro (wheeled + dual arm, 28 DoF).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class MobilityType(Enum):
    STATIC = "static"      
    LEGGED = "legged"       
    WHEELED = "wheeled"     


class ControlMode(Enum):
    POSITION = "position"   # Joint target positions (GR00T VLA output)
    VELOCITY = "velocity"   
    TORQUE = "torque"       


@dataclass
class JointConfig:
    name: str
    joint_type: str          
    limits: Tuple[float, float]
    max_velocity: float = 1.0
    max_torque: float = 100.0


@dataclass
class RobotConfig:
    robot_id: str
    urdf_path: Optional[str] = None
    mobility_type: MobilityType = MobilityType.LEGGED
    num_dof: int = 43                # G1: 43, GR1: ~32, R1 Pro: 28
    joints: List[JointConfig] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)
    control_mode: ControlMode = ControlMode.POSITION
    max_linear_velocity: float = 1.0   # m/s
    max_angular_velocity: float = 2.0  # rad/s


@dataclass
class RobotState:
    robot_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)


class Robot(ABC):
    """Interface for controlling robots in sim."""

    @abstractmethod
    def initialize(self, config: RobotConfig) -> bool: ...

    @abstractmethod
    def get_state(self) -> RobotState: ...

    @abstractmethod
    def set_joint_positions(self, joint_targets: Dict[str, float]) -> bool: ...

    @abstractmethod
    def set_velocity_command(self, linear: Tuple[float, float, float],
                             angular: float = 0.0) -> bool: ...

    @abstractmethod
    def get_end_effector_pose(self, effector: str) -> Tuple[
        Tuple[float, float, float], Tuple[float, float, float, float]]: ...

    @abstractmethod
    def grasp(self, effector: str, object_id: Optional[str] = None) -> bool: ...

    @abstractmethod
    def release(self, effector: str) -> bool: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def robot_id(self) -> str: ...

    @property
    @abstractmethod
    def config(self) -> RobotConfig: ...
