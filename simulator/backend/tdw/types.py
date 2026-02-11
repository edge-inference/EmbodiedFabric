"""TDW backend types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


COLLISION_RECOVERY_THRESHOLD = 2
RECOVERY_TURN_RANGE = (30, 120)


@dataclass
class RobotState:
    magnebot: Any = None
    pending_action: bool = False
    action_type: str = ""
    consecutive_collisions: int = 0
    recovering: bool = False

