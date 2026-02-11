"""
Sensor Contract

Interface for simulated sensors. Bandwidth, resolution, and data types
feed directly into SoC input pipeline sizing (DMA, buffer, bus width).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
import numpy as np


class SensorType(Enum):
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    JOINT_ENCODER = "joint_encoder"


@dataclass
class CameraConfig:
    width: int = 512             # N1.6 uses 512x512, N1.5 uses 224x224
    height: int = 512
    fov: float = 90.0
    fps: float = 30.0
    channels: int = 3

    @property
    def bandwidth_bytes_per_sec(self) -> float:
        """Raw pixel throughput -- SoC input bus sizing."""
        return self.width * self.height * self.channels * self.fps

    @property
    def frame_size_bytes(self) -> int:
        return self.width * self.height * self.channels


@dataclass
class SensorConfig:
    sensor_id: str
    sensor_type: SensorType
    parent_link: str = "head_link"
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    update_rate_hz: float = 30.0
    camera: Optional[CameraConfig] = None


# -- Typed observations , SoC buffer layout dependencies --

@dataclass
class Observation:
    sensor_id: str
    timestamp: float
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        return None


@dataclass
class RGBObservation(Observation):
    """(H, W, 3) uint8 -- camera input to VLA vision encoder."""
    data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint8))

    @property
    def image(self) -> np.ndarray:
        return self.data


@dataclass
class DepthObservation(Observation):
    """(H, W) float32 meters -- optional depth channel."""
    data: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))


@dataclass
class IMUObservation(Observation):
    """6-axis inertial measurement for WBC balance."""
    linear_acceleration: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_quat: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ProprioceptionObservation(Observation):
    """Joint positions + velocities -- state input to VLA/WBC."""
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)


class Sensor(ABC):
    """Interface for sim sensors."""

    @abstractmethod
    def initialize(self, config: SensorConfig) -> bool: ...

    @abstractmethod
    def get_observation(self) -> Observation: ...

    @abstractmethod
    def set_enabled(self, enabled: bool) -> None: ...

    @property
    @abstractmethod
    def sensor_type(self) -> SensorType: ...

    @property
    @abstractmethod
    def update_rate(self) -> float: ...

    @property
    @abstractmethod
    def data_bandwidth_bytes_per_sec(self) -> float:
        """Throughput for SoC bus sizing."""
        ...
