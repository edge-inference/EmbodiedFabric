"""
Physics Backends

- TDW (ThreeDWorld) - CPU-based physics, works everywhere
- Isaac Sim - GPU-accelerated physics/rendering (requires NVIDIA GPU)
"""

from .base import PhysicsBackend, RobotObservation, RobotCommand
from .tdw_backend import TDWBackend

__all__ = ['PhysicsBackend', 'RobotObservation', 'RobotCommand', 'TDWBackend']

# Isaac Sim is optional (requires separate install)
try:
    from .isaac_backend import IsaacSimBackend
    __all__.append('IsaacSimBackend')
except ImportError:
    pass
