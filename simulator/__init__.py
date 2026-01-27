"""
PhysicAI Unified Simulator

Realistic physics simulation with VLA integration and multi-robot coordination.
Designed for hardware-software co-design (R3: μAgent tapeout).

Architecture:
- backend/: Physics engines (TDW, Isaac Sim, MuJoCo)
- robot/: Robot models with sensors and VLA integration
- coordination/: Multi-robot coordination (reimplemented from Mesa learnings)
- vla/: Vision-Language-Action model interface
- scenarios/: Task scenarios for workload profiling
"""

from .core import Simulator, SimulatorConfig
from .robot.vla_agent import VLAAgent
from .coordination.fleet import FleetCoordinator

__all__ = ['Simulator', 'SimulatorConfig', 'VLAAgent', 'FleetCoordinator']
