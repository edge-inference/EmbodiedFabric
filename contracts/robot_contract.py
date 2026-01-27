"""
Robot Contract

Defines the interface for robotic agents in the simulation.
Supports various morphologies: wheeled, legged, manipulator arms, humanoid.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class MobilityType(Enum):
    """Robot mobility types"""
    STATIC = "static"           # Fixed base (e.g., arm on table)
    WHEELED = "wheeled"         # Differential drive, omnidirectional
    LEGGED = "legged"           # Walking robots
    FLYING = "flying"           # Drones


class ControlMode(Enum):
    """Joint/actuator control modes"""
    POSITION = "position"       # Target position control
    VELOCITY = "velocity"       # Velocity control
    TORQUE = "torque"           # Direct torque control


@dataclass
class JointConfig:
    """Configuration for a single joint"""
    name: str
    joint_type: str             # "revolute", "prismatic", "fixed"
    limits: Tuple[float, float] # (lower, upper) limits
    max_velocity: float = 1.0
    max_torque: float = 100.0
    default_position: float = 0.0


@dataclass
class EndEffectorConfig:
    """End effector configuration"""
    name: str
    effector_type: str          # "gripper", "suction", "magnet", "hand"
    max_grip_force: float = 50.0
    grip_width: Tuple[float, float] = (0.0, 0.1)  # (closed, open) in meters


@dataclass
class RobotConfig:
    """Complete robot configuration"""
    robot_id: str
    urdf_path: Optional[str] = None              # Path to URDF file
    mobility_type: MobilityType = MobilityType.WHEELED
    base_height: float = 0.5                      # Height of robot base
    joints: List[JointConfig] = field(default_factory=list)
    end_effectors: List[EndEffectorConfig] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)  # Sensor config IDs
    control_mode: ControlMode = ControlMode.POSITION
    max_linear_velocity: float = 1.0             # m/s
    max_angular_velocity: float = 2.0            # rad/s


@dataclass
class RobotState:
    """Current state of a robot"""
    robot_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]  # Quaternion
    linear_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    joint_positions: Dict[str, float]
    joint_velocities: Dict[str, float]
    end_effector_states: Dict[str, Any]          # e.g., gripper open/closed


@dataclass
class EndEffector:
    """End effector state and control"""
    name: str
    is_grasping: bool = False
    grasped_object_id: Optional[str] = None
    grip_force: float = 0.0


class Robot(ABC):
    """
    Abstract robot contract.
    
    Defines the interface for controlling robots in simulation.
    """
    
    @abstractmethod
    def initialize(self, config: RobotConfig) -> bool:
        """Initialize robot with configuration"""
        pass
    
    @abstractmethod
    def get_state(self) -> RobotState:
        """Get current robot state"""
        pass
    
    @abstractmethod
    def move_to(self, 
                target_position: Tuple[float, float, float],
                target_rotation: Optional[Tuple[float, float, float, float]] = None,
                speed: float = 1.0) -> bool:
        """
        Move robot base to target pose.
        
        Args:
            target_position: Target (x, y, z) position
            target_rotation: Target orientation quaternion
            speed: Movement speed multiplier (0-1)
            
        Returns:
            True if movement command accepted
        """
        pass
    
    @abstractmethod
    def move_by(self,
                delta_position: Tuple[float, float, float],
                delta_rotation: float = 0.0) -> bool:
        """
        Move robot by relative offset.
        
        Args:
            delta_position: Relative (dx, dy, dz) movement
            delta_rotation: Relative yaw rotation in radians
            
        Returns:
            True if movement command accepted
        """
        pass
    
    @abstractmethod
    def set_joint_positions(self, joint_targets: Dict[str, float]) -> bool:
        """
        Set target positions for specified joints.
        
        Args:
            joint_targets: Dict mapping joint_name to target position
            
        Returns:
            True if command accepted
        """
        pass
    
    @abstractmethod
    def get_end_effector_pose(self, effector_name: str) -> Tuple[Tuple[float, float, float], 
                                                                   Tuple[float, float, float, float]]:
        """
        Get end effector pose in world coordinates.
        
        Returns:
            (position, rotation_quaternion) tuple
        """
        pass
    
    @abstractmethod
    def grasp(self, effector_name: str, target_object_id: Optional[str] = None) -> bool:
        """
        Attempt to grasp with specified end effector.
        
        Args:
            effector_name: End effector to use
            target_object_id: Optional specific object to grasp
            
        Returns:
            True if grasp successful
        """
        pass
    
    @abstractmethod
    def release(self, effector_name: str) -> bool:
        """
        Release grasped object.
        
        Returns:
            True if release successful
        """
        pass
    
    @abstractmethod
    def get_grasped_objects(self) -> Dict[str, Optional[str]]:
        """
        Get currently grasped objects per end effector.
        
        Returns:
            Dict mapping effector_name to grasped object_id (or None)
        """
        pass
    
    @abstractmethod
    def reach_for(self, 
                  target_position: Tuple[float, float, float],
                  effector_name: str,
                  orientation: Optional[Tuple[float, float, float, float]] = None) -> bool:
        """
        Move end effector to target position using IK.
        
        Args:
            target_position: Target position for end effector
            effector_name: Which end effector to move
            orientation: Optional target orientation
            
        Returns:
            True if IK solution found and motion started
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop all robot motion immediately"""
        pass
    
    @abstractmethod
    def is_moving(self) -> bool:
        """Check if robot is currently in motion"""
        pass
    
    @abstractmethod
    def get_reachable_objects(self, effector_name: str, max_distance: float = 1.0) -> List[str]:
        """
        Get list of objects within reach of end effector.
        
        Args:
            effector_name: End effector to check from
            max_distance: Maximum reaching distance
            
        Returns:
            List of reachable object IDs
        """
        pass
    
    @property
    @abstractmethod
    def robot_id(self) -> str:
        """Unique robot identifier"""
        pass
    
    @property
    @abstractmethod
    def config(self) -> RobotConfig:
        """Robot configuration"""
        pass
