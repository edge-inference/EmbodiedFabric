"""
Task Contract

Defines high-level tasks and actions for robot manipulation.
Tasks decompose into primitive actions that the robot executes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from enum import Enum


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ActionType(Enum):
    """Primitive action types"""
    NAVIGATE = "navigate"           # Move to location
    PICK = "pick"                   # Grasp object
    PLACE = "place"                 # Release object at location
    PUSH = "push"                   # Push object
    PULL = "pull"                   # Pull object
    OPEN = "open"                   # Open door/drawer
    CLOSE = "close"                 # Close door/drawer
    WAIT = "wait"                   # Wait for condition/time
    LOOK_AT = "look_at"             # Orient sensors toward target


@dataclass
class ActionResult:
    """Result of a primitive action"""
    success: bool
    action_type: ActionType
    duration_seconds: float
    energy_estimate: float = 0.0               # Joules (for power modeling)
    distance_traveled: float = 0.0             # Meters
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a complete task"""
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float
    action_results: List[ActionResult]
    success: bool = False
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def total_distance(self) -> float:
        return sum(a.distance_traveled for a in self.action_results)
    
    @property
    def total_energy(self) -> float:
        return sum(a.energy_estimate for a in self.action_results)


@dataclass
class NavigateAction:
    """Navigate to a position"""
    target_position: Tuple[float, float, float]
    target_rotation: Optional[Tuple[float, float, float, float]] = None
    speed: float = 1.0
    avoid_objects: List[str] = field(default_factory=list)


@dataclass
class PickAction:
    """Pick up an object"""
    object_id: str
    effector_name: str = "gripper"
    approach_direction: Optional[Tuple[float, float, float]] = None
    grip_force: float = 0.5                    # Normalized 0-1


@dataclass
class PlaceAction:
    """Place an object at location"""
    target_position: Tuple[float, float, float]
    effector_name: str = "gripper"
    target_rotation: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ActionSpace:
    """Defines available actions for a robot"""
    robot_id: str
    supported_actions: List[ActionType]
    max_payload: float = 10.0                  # kg
    max_reach: float = 1.0                     # meters
    gripper_width: Tuple[float, float] = (0.0, 0.1)  # (min, max) in meters


class Task(ABC):
    """
    Abstract task contract.
    
    High-level tasks that decompose into primitive actions.
    """
    
    @abstractmethod
    def initialize(self, 
                   task_id: str,
                   robot_id: str,
                   parameters: Dict[str, Any]) -> bool:
        """
        Initialize task with parameters.
        
        Args:
            task_id: Unique task identifier
            robot_id: Robot that will execute task
            parameters: Task-specific parameters
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def plan(self, env_state: Any) -> List[Any]:
        """
        Plan action sequence for this task.
        
        Args:
            env_state: Current environment state
            
        Returns:
            List of primitive actions to execute
        """
        pass
    
    @abstractmethod
    def execute_step(self) -> Tuple[ActionResult, bool]:
        """
        Execute next action in the task.
        
        Returns:
            (action_result, is_complete) tuple
        """
        pass
    
    @abstractmethod
    def get_status(self) -> TaskStatus:
        """Get current task status"""
        pass
    
    @abstractmethod
    def cancel(self) -> None:
        """Cancel task execution"""
        pass
    
    @abstractmethod
    def get_result(self) -> Optional[TaskResult]:
        """Get task result (None if not complete)"""
        pass
    
    @property
    @abstractmethod
    def task_id(self) -> str:
        """Unique task identifier"""
        pass
    
    @property
    @abstractmethod
    def task_type(self) -> str:
        """Type of task (e.g., 'transport', 'pick_place')"""
        pass
    
    @property
    @abstractmethod
    def progress(self) -> float:
        """Task progress 0.0 to 1.0"""
        pass


class TransportTask(Task):
    """
    Transport task: Pick object from A, move to B, place.
    
    This is the primary task type for warehouse workload modeling.
    """
    
    @abstractmethod
    def set_source(self, object_id: str, pickup_position: Tuple[float, float, float]) -> None:
        """Set pickup location"""
        pass
    
    @abstractmethod
    def set_destination(self, 
                        place_position: Tuple[float, float, float],
                        container_id: Optional[str] = None) -> None:
        """Set delivery location, optionally into a container"""
        pass
    
    @abstractmethod
    def get_waypoints(self) -> List[Tuple[float, float, float]]:
        """Get planned navigation waypoints"""
        pass


class TaskFactory(ABC):
    """Factory for creating task instances"""
    
    @abstractmethod
    def create_task(self, 
                    task_type: str,
                    task_id: str,
                    robot_id: str,
                    parameters: Dict[str, Any]) -> Task:
        """
        Create a task instance.
        
        Args:
            task_type: Type of task to create
            task_id: Unique identifier
            robot_id: Executing robot
            parameters: Task parameters
            
        Returns:
            Task instance
        """
        pass
    
    @abstractmethod
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported task types"""
        pass
