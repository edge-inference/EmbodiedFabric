"""
PhysicAI Simulator Contracts

Abstract interfaces defining the simulation layer contracts.
These contracts allow swapping backends (TDW, AI2THOR, PyBullet, etc.)
without changing higher-level code.

Contracts:
- Environment: Scene management, physics stepping, observation retrieval
- Robot: Kinematics, actuation, sensor mounting
- Sensor: Raw observation capture, noise models
- Task: High-level actions (pick, place, navigate) decomposed into primitives
- Workload: Scenario definition and metric collection
"""

from .env_contract import Environment, SceneConfig
from .robot_contract import Robot, RobotConfig, EndEffector
from .sensor_contract import Sensor, SensorConfig, Observation
from .task_contract import Task, TaskResult, ActionSpace
from .workload_contract import Workload, WorkloadMetrics, Scenario

__all__ = [
    'Environment', 'SceneConfig',
    'Robot', 'RobotConfig', 'EndEffector',
    'Sensor', 'SensorConfig', 'Observation',
    'Task', 'TaskResult', 'ActionSpace',
    'Workload', 'WorkloadMetrics', 'Scenario',
]
