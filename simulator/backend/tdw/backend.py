"""TDW backend (ThreeDWorld + Magnebot)."""

from __future__ import annotations

from typing import Dict

from ..base import PhysicsBackend
from .commands import TDWCommandMixin
from .lifecycle import TDWLifecycleMixin
from .objects import TDWObjectMixin
from .recording import TDWRecordingMixin
from .robots import TDWRobotMixin
from .types import RobotState


class TDWBackend(
    TDWRecordingMixin,
    TDWLifecycleMixin,
    TDWRobotMixin,
    TDWCommandMixin,
    TDWObjectMixin,
    PhysicsBackend,
):
    def __init__(
        self,
        config,
        enable_recording: bool = False,
        recording_path: str | None = None,
        launch_build: bool = True,
        tdw_address: str = "localhost",
        tdw_port: int = 1071,
    ):
        self.config = config
        self._controller = None
        self._robots: Dict[str, RobotState] = {}
        self._objects: Dict[str, int] = {}
        self._sim_time = 0.0
        self._initialized = False
        self._robot_counter = 0

        self._launch_build = launch_build
        self._tdw_address = tdw_address
        self._tdw_port = tdw_port

        self._recording_enabled = enable_recording
        self._recording_path = recording_path or "recordings"
        self._third_person_camera = None
        self._image_capture = None
        self._frame_count = 0
        self._capture_interval = 10

