"""Robot agent base types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from enum import Enum


class RobotState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    WAITING = "waiting"
    ERROR = "error"


@dataclass
class RobotStatus:
    robot_id: str
    state: RobotState
    position: Tuple[float, float, float]
    current_task_id: Optional[str] = None
    battery_level: float = 1.0
    error_message: Optional[str] = None


class RobotAgent(ABC):
    """Abstract robot agent."""
    
    def __init__(self, robot_id: str, backend):
        self.robot_id = robot_id
        self._backend = backend
        self._state = RobotState.IDLE
        self._current_task = None
        self._metrics = {
            'tasks_completed': 0,
            'distance_traveled': 0.0,
            'total_steps': 0
        }
    
    @abstractmethod
    def step(self, coordinator, dsm, profiler) -> None:
        pass
    
    @abstractmethod
    def assign_task(self, task: Dict[str, Any]) -> bool:
        pass
    
    @abstractmethod
    def get_status(self) -> RobotStatus:
        pass
    
    @property
    def state(self) -> RobotState:
        return self._state
    
    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics.copy()
