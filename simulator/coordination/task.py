"""Task definitions and lifecycle."""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
from enum import Enum
import time


class TaskStatus(Enum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


class TaskType(Enum):
    TRANSPORT = "transport"
    PICK = "pick"
    PLACE = "place"
    INSPECT = "inspect"
    PATROL = "patrol"


@dataclass
class Task:
    """Task definition."""
    task_id: str
    task_type: TaskType
    location: Tuple[float, float, float]
    instruction: str
    
    status: TaskStatus = TaskStatus.AVAILABLE
    priority: int = 1
    
    created_time: float = field(default_factory=time.time)
    claimed_time: Optional[float] = None
    completed_time: Optional[float] = None
    
    claimed_by: Optional[str] = None
    lease_ttl_ms: int = 30000
    
    destination: Optional[Tuple[float, float, float]] = None
    object_id: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def claim(self, agent_id: str) -> bool:
        """Attempt to claim this task"""
        if self.status != TaskStatus.AVAILABLE:
            return False
        
        self.status = TaskStatus.CLAIMED
        self.claimed_by = agent_id
        self.claimed_time = time.time()
        return True
    
    def release(self) -> None:
        """Release claim on this task"""
        self.status = TaskStatus.AVAILABLE
        self.claimed_by = None
        self.claimed_time = None
    
    def complete(self) -> None:
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_time = time.time()
    
    def fail(self) -> None:
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_time = time.time()
    
    def is_lease_expired(self) -> bool:
        """Check if lease has expired"""
        if self.claimed_time is None:
            return False
        elapsed_ms = (time.time() - self.claimed_time) * 1000
        return elapsed_ms > self.lease_ttl_ms
    
    @property
    def latency(self) -> Optional[float]:
        """Time from creation to completion"""
        if self.completed_time:
            return self.completed_time - self.created_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for VLA context"""
        return {
            'id': self.task_id,
            'type': self.task_type.value,
            'location': self.location,
            'instruction': self.instruction,
            'destination': self.destination,
            'priority': self.priority
        }


def create_transport_task(task_id: str,
                         pickup_location: Tuple[float, float, float],
                         dropoff_location: Tuple[float, float, float],
                         object_type: str = "box") -> Task:
    """Factory for transport tasks"""
    return Task(
        task_id=task_id,
        task_type=TaskType.TRANSPORT,
        location=pickup_location,
        destination=dropoff_location,
        instruction=f"Pick up the {object_type} at the pickup location and transport it to the destination",
        priority=1
    )
