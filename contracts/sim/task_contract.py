"""
Task Contract

High-level task primitives for loco-manipulation.
Decomposition: Task -> ActionSequence -> Robot commands.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionType(Enum):
    NAVIGATE = "navigate"     # Walk/drive to location
    PICK = "pick"             # Grasp object
    PLACE = "place"           # Release at target
    OPEN = "open"             # Open door/drawer/microwave
    CLOSE = "close"
    WAIT = "wait"


@dataclass
class ActionResult:
    success: bool
    action_type: ActionType
    duration_seconds: float
    distance_traveled: float = 0.0
    energy_joules: float = 0.0
    error: Optional[str] = None


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    start_time: float
    end_time: float
    actions: List[ActionResult] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_distance(self) -> float:
        return sum(a.distance_traveled for a in self.actions)

    @property
    def total_energy(self) -> float:
        return sum(a.energy_joules for a in self.actions)


class Task(ABC):
    """A task that decomposes into primitive actions."""

    @abstractmethod
    def plan(self, env_state: Any) -> List[ActionType]: ...

    @abstractmethod
    def execute_step(self) -> Tuple[ActionResult, bool]: ...

    @abstractmethod
    def get_status(self) -> TaskStatus: ...

    @abstractmethod
    def get_result(self) -> Optional[TaskResult]: ...

    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def progress(self) -> float: ...


class LocoManipTask(Task):
    """Navigate to object, pick, carry, place. The core G1 task."""

    @abstractmethod
    def set_source(self, object_id: str, position: Tuple[float, float, float]) -> None: ...

    @abstractmethod
    def set_destination(self, position: Tuple[float, float, float]) -> None: ...

    @abstractmethod
    def get_waypoints(self) -> List[Tuple[float, float, float]]: ...
