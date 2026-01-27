"""
Multi-Robot Coordination

Reimplemented from Mesa learnings with proper physics integration.

Components:
- DSM: Distributed Shared Memory with gossip protocol
- Fleet Coordinator: Task allocation with LF integration
- Pathfinder: Collision-aware navigation (real obstacles)
"""

from .dsm import DistributedSharedMemory
from .fleet import FleetCoordinator
from .task import Task, TaskStatus

__all__ = ['DistributedSharedMemory', 'FleetCoordinator', 'Task', 'TaskStatus']
