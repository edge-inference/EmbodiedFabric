"""
VLA (Vision-Language-Action) Interface

Hierarchical robot control with specialized experts:
- High-Level Planner (VLM @ 1-3 Hz) - Task decomposition and mode routing
- Navigation Expert (NoMaD @ 5-10 Hz) - Goal-conditioned navigation
- Manipulation Expert (CogACT @ 2-5 Hz) - Object manipulation

LeRobot Models (Flow Matching / DiT):
- SmolVLA (450M) - Fast, consumer GPU friendly
- Pi0 (3B) - General purpose robot foundation model
- GR00T (3B) - NVIDIA humanoid model with DiT
"""

from .interface import (
    VLAInterface, VLAObservation, VLAAction, VLAMetrics, create_vla
)
from .profiled_vla import ProfiledVLA
from .openvla import OpenVLAModel
from .nomad import NoMaDNavigator, MockNoMaDNavigator
from .hierarchical import HierarchicalPlanner, TaskMode, Subgoal
from .lerobot_vla import (
    SmolVLAModel, Pi0Model, GR00TModel, VLAControlMode, LeRobotConfig
)

__all__ = [
    'VLAInterface', 'VLAObservation', 'VLAAction', 'VLAMetrics', 'create_vla',
    'ProfiledVLA', 'OpenVLAModel',
    'NoMaDNavigator', 'MockNoMaDNavigator',
    'HierarchicalPlanner', 'TaskMode', 'Subgoal',
    'SmolVLAModel', 'Pi0Model', 'GR00TModel', 'VLAControlMode', 'LeRobotConfig',
]
