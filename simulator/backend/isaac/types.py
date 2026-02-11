"""Isaac backend types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class IsaacRobotState:
    prim_path: str
    robot_type: str
    articulation: Any = None

