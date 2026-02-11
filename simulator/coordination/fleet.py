"""Task allocation and coordination across robots."""

from typing import Dict, List, Any, Optional, Tuple
import time
import logging
import numpy as np

from .task import Task, TaskStatus, TaskType, create_transport_task
from .dsm import DistributedSharedMemory

logger = logging.getLogger(__name__)


class FleetCoordinator:
    """Fleet-level task coordinator."""
    
    def __init__(self,
                 n_robots: int,
                 dsm: Optional[DistributedSharedMemory] = None,
                 enable_lf: bool = False,
                 task_arrival_rate: float = 0.1):
        self.n_robots = n_robots
        self.dsm = dsm
        self.enable_lf = enable_lf
        self.task_arrival_rate = task_arrival_rate
        
        self._tasks: Dict[str, Task] = {}
        self._task_counter = 0
        self._completed_count = 0
        self._failed_count = 0
        
        self._agent_assignments: Dict[str, Optional[str]] = {}
        
        self._step_count = 0
        self._start_time = time.time()
        
        self._lf_bridge = None
        if enable_lf:
            self._init_lf_bridge()
    
    def _init_lf_bridge(self) -> None:
        logger.info("LF coordination enabled (placeholder)")
    
    def step(self) -> None:
        self._step_count += 1
        
        self._generate_tasks()
        
        self._check_lease_expirations()
        
        self._cleanup_completed_tasks()
    
    def _generate_tasks(self) -> None:
        """Generate new tasks based on arrival rate"""
        if np.random.random() < self.task_arrival_rate * 0.02:
            self._create_random_task()
    
    def _create_random_task(self) -> Task:
        """Create a random transport task"""
        self._task_counter += 1
        task_id = f"task_{self._task_counter}"
        
        pickup = (
            np.random.uniform(-8, 8),
            np.random.uniform(-8, 8),
            0.0
        )
        dropoff = (
            np.random.uniform(-8, 8),
            np.random.uniform(-8, 8),
            0.0
        )
        
        task = create_transport_task(task_id, pickup, dropoff)
        self._tasks[task_id] = task
        
        logger.debug(f"Created task {task_id} at {pickup}")
        return task
    
    def _check_lease_expirations(self) -> None:
        """Check and handle expired task leases"""
        for task in self._tasks.values():
            if task.status == TaskStatus.CLAIMED and task.is_lease_expired():
                logger.warning(f"Task {task.task_id} lease expired, releasing")
                agent_id = task.claimed_by
                task.release()
                if agent_id:
                    self._agent_assignments[agent_id] = None
    
    def _cleanup_completed_tasks(self) -> None:
        """Remove old completed/failed tasks"""
        to_remove = []
        current_time = time.time()
        
        for task_id, task in self._tasks.items():
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if task.completed_time and current_time - task.completed_time > 60:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
    
    def get_available_tasks(self) -> List[Task]:
        """Get all available tasks"""
        return [t for t in self._tasks.values() if t.status == TaskStatus.AVAILABLE]
    
    def try_claim_task(self, 
                       agent_id: str,
                       agent_position: Tuple[float, float, float]) -> Optional[Dict[str, Any]]:
        """
        Try to claim a task for an agent.
        
        Uses locality preference: prefer nearby tasks.
        """
        available = self.get_available_tasks()
        if not available:
            return None
        
        def distance_to_task(task: Task) -> float:
            return np.sqrt(
                (task.location[0] - agent_position[0])**2 +
                (task.location[1] - agent_position[1])**2
            )
        
        available.sort(key=lambda t: (-t.priority, distance_to_task(t)))
        
        for task in available:
            if task.claim(agent_id):
                self._agent_assignments[agent_id] = task.task_id
                logger.info(f"Agent {agent_id} claimed task {task.task_id}")
                return task.to_dict()
        
        return None
    
    def complete_task(self, task_id: str, agent_id: str) -> bool:
        """Mark a task as completed"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.claimed_by != agent_id:
            logger.warning(f"Agent {agent_id} tried to complete task {task_id} "
                          f"owned by {task.claimed_by}")
            return False
        
        task.complete()
        self._completed_count += 1
        self._agent_assignments[agent_id] = None
        
        logger.info(f"Task {task_id} completed by {agent_id}, "
                   f"latency={task.latency:.2f}s")
        return True
    
    def fail_task(self, task_id: str, agent_id: str) -> bool:
        """Mark a task as failed"""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.claimed_by != agent_id:
            return False
        
        task.fail()
        self._failed_count += 1
        self._agent_assignments[agent_id] = None
        
        logger.info(f"Task {task_id} failed by {agent_id}")
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get coordinator metrics"""
        elapsed = time.time() - self._start_time
        
        return {
            'total_tasks_created': self._task_counter,
            'tasks_completed': self._completed_count,
            'tasks_failed': self._failed_count,
            'tasks_available': len(self.get_available_tasks()),
            'tasks_claimed': sum(1 for t in self._tasks.values() 
                                if t.status == TaskStatus.CLAIMED),
            'throughput_per_hour': self._completed_count * 3600 / max(elapsed, 1),
            'elapsed_seconds': elapsed
        }
    
    def inject_task(self, task: Task) -> None:
        """Inject a custom task (for testing)"""
        self._tasks[task.task_id] = task
