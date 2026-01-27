"""
Environment Contract

Defines the interface for physics simulation environments.
Backend implementations (TDW, AI2THOR, PyBullet) must implement this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class PhysicsMode(Enum):
    """Physics simulation fidelity levels"""
    KINEMATIC = "kinematic"     # No physics, teleport movements
    RIGID_BODY = "rigid_body"   # Rigid body dynamics
    SOFT_BODY = "soft_body"     # Deformable objects
    FLUID = "fluid"             # Fluid simulation


@dataclass
class SceneConfig:
    """Scene initialization configuration"""
    scene_id: str                                    # Scene identifier (e.g., "warehouse_001")
    layout_file: Optional[str] = None               # Path to layout definition
    objects: List[Dict[str, Any]] = field(default_factory=list)  # Initial objects
    physics_mode: PhysicsMode = PhysicsMode.RIGID_BODY
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    time_step: float = 0.02                          # Physics timestep (50Hz default)
    render_quality: str = "medium"                   # low/medium/high/ultra


@dataclass
class ObjectState:
    """State of an object in the environment"""
    object_id: str
    position: Tuple[float, float, float]            # (x, y, z) world coords
    rotation: Tuple[float, float, float, float]     # Quaternion (w, x, y, z)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_kinematic: bool = False


@dataclass
class EnvironmentState:
    """Full environment state snapshot"""
    timestamp: float                    # Simulation time in seconds
    step_count: int
    objects: Dict[str, ObjectState]     # object_id -> state
    robot_states: Dict[str, Any]        # robot_id -> robot state
    collisions: List[Tuple[str, str]]   # List of (obj_a, obj_b) collision pairs


class Environment(ABC):
    """
    Abstract environment contract.
    
    All physics backends must implement this interface to be swappable.
    """
    
    @abstractmethod
    def initialize(self, config: SceneConfig) -> bool:
        """
        Initialize the environment with a scene configuration.
        
        Args:
            config: Scene configuration specifying layout, objects, physics params
            
        Returns:
            True if initialization succeeded
        """
        pass
    
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        """
        Reset environment to initial state.
        
        Args:
            seed: Optional random seed for reproducibility
            
        Returns:
            Initial environment state
        """
        pass
    
    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[EnvironmentState, Dict[str, Any]]:
        """
        Advance simulation by one timestep with given actions.
        
        Args:
            actions: Dict mapping robot_id to action commands
            
        Returns:
            (new_state, info_dict) tuple
        """
        pass
    
    @abstractmethod
    def get_observation(self, robot_id: str, sensor_id: str) -> Any:
        """
        Get observation from a specific sensor on a robot.
        
        Args:
            robot_id: Robot identifier
            sensor_id: Sensor identifier on that robot
            
        Returns:
            Sensor-specific observation data
        """
        pass
    
    @abstractmethod
    def spawn_object(self, 
                     object_type: str,
                     position: Tuple[float, float, float],
                     rotation: Optional[Tuple[float, float, float, float]] = None,
                     scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                     properties: Optional[Dict[str, Any]] = None) -> str:
        """
        Spawn a new object in the environment.
        
        Args:
            object_type: Type/model identifier
            position: Spawn position (x, y, z)
            rotation: Orientation quaternion
            scale: Scale factors
            properties: Additional object properties (mass, friction, etc.)
            
        Returns:
            Unique object_id
        """
        pass
    
    @abstractmethod
    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the environment"""
        pass
    
    @abstractmethod
    def get_object_state(self, object_id: str) -> Optional[ObjectState]:
        """Get current state of an object"""
        pass
    
    @abstractmethod
    def spawn_robot(self, 
                    robot_config: 'RobotConfig',
                    position: Tuple[float, float, float],
                    rotation: Optional[Tuple[float, float, float, float]] = None) -> str:
        """
        Spawn a robot in the environment.
        
        Args:
            robot_config: Robot configuration (URDF, sensors, etc.)
            position: Spawn position
            rotation: Initial orientation
            
        Returns:
            Unique robot_id
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = "rgb_array") -> Any:
        """
        Render the environment.
        
        Args:
            mode: Render mode ("rgb_array", "human", "depth", etc.)
            
        Returns:
            Rendered output (numpy array for rgb_array, None for human)
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Cleanup and close the environment"""
        pass
    
    @property
    @abstractmethod
    def simulation_time(self) -> float:
        """Current simulation time in seconds"""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the backend implementation (e.g., 'TDW', 'AI2THOR')"""
        pass


class RobotConfig:
    """Forward declaration - defined in robot_contract.py"""
    pass
