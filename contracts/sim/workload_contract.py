"""
Workload Contract

Profiling metrics that feed SoC design. This is the bridge between
simulation measurements and hardware spec sheets.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json


@dataclass
class ComputeMetrics:
    """What the SoC compute block needs to handle."""
    vla_inference_flops: float = 0.0         # GR00T/SmolVLA forward pass
    vla_latency_ms: float = 0.0              # End-to-end VLA inference
    wbc_inference_flops: float = 0.0         # ONNX WBC forward pass
    wbc_latency_ms: float = 0.0              # WBC at 200Hz target
    preprocessing_ms: float = 0.0            # Image resize, normalize
    postprocessing_ms: float = 0.0           # Action denormalization
    peak_memory_mb: float = 0.0              # GPU VRAM high watermark


@dataclass
class SensorMetrics:
    """Input pipeline bandwidth requirements."""
    camera_bandwidth_mbps: float = 0.0       # Raw pixel throughput
    camera_resolution: str = "512x512"
    camera_fps: float = 30.0
    proprioception_bandwidth_kbps: float = 0.0  # Joint encoders, IMU


@dataclass
class PowerMetrics:
    """Per-subsystem energy breakdown."""
    vla_joules_per_step: float = 0.0
    wbc_joules_per_step: float = 0.0
    locomotion_joules_per_meter: float = 0.0
    sensor_watts: float = 0.0
    communication_joules: float = 0.0

    @property
    def total_inference_joules(self) -> float:
        return self.vla_joules_per_step + self.wbc_joules_per_step


@dataclass
class TaskMetrics:
    task_id: str
    task_type: str
    start_time: float
    end_time: float
    success: bool
    vla_calls: int = 0
    wbc_calls: int = 0
    distance_traveled: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class WorkloadMetrics:
    """Aggregated metrics for one eval run. Feeds SoC sizing."""
    scenario_id: str
    duration_seconds: float

    # Task stats
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_task_latency_s: float = 0.0
    p99_task_latency_s: float = 0.0

    # Subsystem metrics
    compute: ComputeMetrics = field(default_factory=ComputeMetrics)
    sensors: SensorMetrics = field(default_factory=SensorMetrics)
    power: PowerMetrics = field(default_factory=PowerMetrics)
    per_task: List[TaskMetrics] = field(default_factory=list)

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)


class WorkloadProfiler(ABC):
    """Collects metrics during eval runs for hardware sizing."""

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def record_vla_inference(self, latency_ms: float, memory_mb: float) -> None: ...

    @abstractmethod
    def record_wbc_inference(self, latency_ms: float) -> None: ...

    @abstractmethod
    def record_task(self, metrics: TaskMetrics) -> None: ...

    @abstractmethod
    def get_metrics(self) -> WorkloadMetrics: ...

    @abstractmethod
    def get_soc_requirements(self) -> Dict[str, Any]:
        """
        Derive SoC spec from measured workload:
        - Required TOPS for VLA at target Hz
        - Required TOPS for WBC at 200Hz
        - Memory bandwidth for sensor input pipeline
        - On-chip SRAM for action chunk buffer
        - Power budget breakdown
        """
        ...
