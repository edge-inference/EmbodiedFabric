"""IsaacSimBackend modules."""

from __future__ import annotations

import numpy as np
from typing import Dict

from ..base import PhysicsBackend
from .assets import IsaacAssetsMixin
from .lifecycle import IsaacLifecycleMixin
from .objects import IsaacObjectMixin
from .recording import IsaacRecordingMixin
from .robots import IsaacRobotMixin
from .types import IsaacRobotState


class IsaacSimBackend(
    IsaacAssetsMixin,
    IsaacRecordingMixin,
    IsaacLifecycleMixin,
    IsaacRobotMixin,
    IsaacObjectMixin,
    PhysicsBackend,
):
    def __init__(
        self,
        config,
        enable_recording: bool = False,
        recording_path: str | None = None,
        headless: bool = True,
    ):
        self.config = config
        self._headless = headless
        self._sim = None
        self._world = None
        self._stage = None
        self._robots: Dict[str, IsaacRobotState] = {}
        self._objects: Dict[str, str] = {}
        self._sim_time = 0.0
        self._initialized = False
        self._time_step = getattr(config, "time_step", 1 / 60.0)

        self._recording_enabled = enable_recording
        self._recording_path = recording_path or "recordings"
        self._camera = None
        self._frame_count = 0

