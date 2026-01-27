"""
Sensor Contract

Defines interfaces for simulated sensors.
Supports vision (RGB, depth, segmentation), proprioception, tactile, audio.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np


class SensorType(Enum):
    """Types of sensors"""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    RGBD_CAMERA = "rgbd_camera"
    LIDAR = "lidar"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    AUDIO = "audio"
    JOINT_ENCODER = "joint_encoder"


@dataclass
class CameraConfig:
    """Camera sensor configuration"""
    width: int = 640
    height: int = 480
    fov: float = 90.0                              # Field of view in degrees
    near_clip: float = 0.1
    far_clip: float = 100.0
    fps: float = 30.0


@dataclass
class LidarConfig:
    """LiDAR sensor configuration"""
    num_rays: int = 360
    min_range: float = 0.1
    max_range: float = 10.0
    fov_horizontal: float = 360.0                  # Degrees
    fov_vertical: float = 30.0
    fps: float = 10.0


@dataclass
class NoiseConfig:
    """Sensor noise model configuration"""
    enabled: bool = True
    gaussian_std: float = 0.01                     # Gaussian noise std dev
    dropout_rate: float = 0.0                      # Probability of dropped reading
    bias: float = 0.0                              # Constant bias
    quantization: Optional[float] = None           # Quantization step


@dataclass
class SensorConfig:
    """Generic sensor configuration"""
    sensor_id: str
    sensor_type: SensorType
    parent_link: str = "base_link"                 # Robot link to attach to
    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_offset: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    update_rate: float = 30.0                      # Hz
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    type_config: Optional[Union[CameraConfig, LidarConfig]] = None


@dataclass
class Observation:
    """Sensor observation data"""
    sensor_id: str
    timestamp: float                               # Simulation time
    data: Any                                      # Sensor-specific data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of observation data if numpy array"""
        if isinstance(self.data, np.ndarray):
            return self.data.shape
        return None


@dataclass
class RGBObservation(Observation):
    """RGB camera observation"""
    data: np.ndarray                               # (H, W, 3) uint8
    
    @property
    def image(self) -> np.ndarray:
        return self.data


@dataclass
class DepthObservation(Observation):
    """Depth camera observation"""
    data: np.ndarray                               # (H, W) float32 in meters
    
    @property
    def depth_map(self) -> np.ndarray:
        return self.data


@dataclass
class SegmentationObservation(Observation):
    """Semantic segmentation observation"""
    data: np.ndarray                               # (H, W) int32 class IDs
    id_to_class: Dict[int, str] = field(default_factory=dict)


@dataclass
class LidarObservation(Observation):
    """LiDAR point cloud observation"""
    data: np.ndarray                               # (N, 3) or (N, 4) with intensity
    
    @property
    def points(self) -> np.ndarray:
        return self.data[:, :3]
    
    @property
    def intensities(self) -> Optional[np.ndarray]:
        if self.data.shape[1] > 3:
            return self.data[:, 3]
        return None


@dataclass
class IMUObservation(Observation):
    """IMU observation"""
    linear_acceleration: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    orientation: Optional[Tuple[float, float, float, float]] = None


class Sensor(ABC):
    """
    Abstract sensor contract.
    
    All sensor implementations must implement this interface.
    """
    
    @abstractmethod
    def initialize(self, config: SensorConfig) -> bool:
        """Initialize sensor with configuration"""
        pass
    
    @abstractmethod
    def get_observation(self) -> Observation:
        """
        Get current sensor observation.
        
        Returns:
            Observation with sensor-specific data
        """
        pass
    
    @abstractmethod
    def get_latest_observations(self, count: int = 1) -> List[Observation]:
        """
        Get latest N observations (for temporal processing).
        
        Args:
            count: Number of observations to return
            
        Returns:
            List of observations, most recent first
        """
        pass
    
    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable sensor"""
        pass
    
    @abstractmethod
    def apply_noise(self, data: Any) -> Any:
        """
        Apply noise model to raw sensor data.
        
        Args:
            data: Raw sensor data
            
        Returns:
            Noisy sensor data
        """
        pass
    
    @property
    @abstractmethod
    def sensor_id(self) -> str:
        """Unique sensor identifier"""
        pass
    
    @property
    @abstractmethod
    def sensor_type(self) -> SensorType:
        """Type of sensor"""
        pass
    
    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Whether sensor is currently active"""
        pass
    
    @property
    @abstractmethod
    def update_rate(self) -> float:
        """Sensor update rate in Hz"""
        pass
    
    @property
    @abstractmethod
    def data_bandwidth(self) -> float:
        """
        Estimated data bandwidth in bytes/second.
        
        Useful for communication modeling.
        """
        pass
