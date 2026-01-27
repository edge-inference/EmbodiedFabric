"""
Workload Contract

Defines interfaces for scenario specification and workload profiling.
This is the bridge between simulation and hardware/sensor/comms requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Iterator
from enum import Enum
import json


class ScenarioDistribution(Enum):
    """Task generation distributions"""
    UNIFORM = "uniform"             # Uniform random
    POISSON = "poisson"             # Poisson arrivals
    BURST = "burst"                 # Bursty arrivals
    SEQUENTIAL = "sequential"       # Fixed sequence


@dataclass
class TaskSpec:
    """Specification for a single task in a scenario"""
    task_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    deadline_seconds: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)  # Task IDs this depends on


@dataclass
class ScenarioConfig:
    """Scenario configuration"""
    scenario_id: str
    scene_config_path: str                          # Path to scene config
    robot_configs: List[str]                        # Paths to robot configs
    task_distribution: ScenarioDistribution = ScenarioDistribution.POISSON
    task_rate: float = 0.1                          # Tasks per second
    total_tasks: int = 100
    task_specs: List[TaskSpec] = field(default_factory=list)
    duration_seconds: float = 300.0                 # Max scenario duration
    random_seed: Optional[int] = None


@dataclass
class Scenario:
    """A simulation scenario with tasks and robots"""
    config: ScenarioConfig
    task_queue: List[TaskSpec] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    
    def add_task(self, task: TaskSpec) -> None:
        self.task_queue.append(task)
    
    def get_next_task(self) -> Optional[TaskSpec]:
        if self.task_queue:
            return self.task_queue.pop(0)
        return None


@dataclass
class ComputeMetrics:
    """Compute workload metrics"""
    perception_flops: float = 0.0               # FLOPs for perception pipeline
    planning_flops: float = 0.0                 # FLOPs for path planning
    inference_flops: float = 0.0                # FLOPs for ML inference (VLA)
    total_compute_time_ms: float = 0.0


@dataclass
class CommunicationMetrics:
    """Communication workload metrics"""
    sensor_bandwidth_mbps: float = 0.0          # Sensor data bandwidth
    coordination_bandwidth_kbps: float = 0.0    # Control plane messages
    gossip_bandwidth_kbps: float = 0.0          # DSM gossip overhead
    message_count: int = 0
    average_latency_ms: float = 0.0


@dataclass
class PowerMetrics:
    """Power consumption metrics"""
    locomotion_joules: float = 0.0
    manipulation_joules: float = 0.0
    compute_joules: float = 0.0
    sensor_joules: float = 0.0
    communication_joules: float = 0.0
    
    @property
    def total_joules(self) -> float:
        return (self.locomotion_joules + self.manipulation_joules + 
                self.compute_joules + self.sensor_joules + self.communication_joules)


@dataclass
class TaskMetrics:
    """Per-task metrics"""
    task_id: str
    task_type: str
    start_time: float
    end_time: float
    success: bool
    distance_traveled: float
    objects_manipulated: int
    actions_executed: int
    retries: int = 0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class WorkloadMetrics:
    """Aggregated workload metrics for hardware/sensor/comms sizing"""
    scenario_id: str
    duration_seconds: float
    
    # Task performance
    tasks_completed: int = 0
    tasks_failed: int = 0
    throughput_tasks_per_hour: float = 0.0
    average_task_latency_seconds: float = 0.0
    p99_task_latency_seconds: float = 0.0
    
    # Robot utilization
    robot_utilization: Dict[str, float] = field(default_factory=dict)  # robot_id -> 0-1
    total_distance_meters: float = 0.0
    total_objects_moved: int = 0
    
    # Resource consumption
    compute: ComputeMetrics = field(default_factory=ComputeMetrics)
    communication: CommunicationMetrics = field(default_factory=CommunicationMetrics)
    power: PowerMetrics = field(default_factory=PowerMetrics)
    
    # Per-task breakdown
    task_metrics: List[TaskMetrics] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'scenario_id': self.scenario_id,
            'duration_seconds': self.duration_seconds,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'throughput_tasks_per_hour': self.throughput_tasks_per_hour,
            'average_task_latency_seconds': self.average_task_latency_seconds,
            'p99_task_latency_seconds': self.p99_task_latency_seconds,
            'robot_utilization': self.robot_utilization,
            'total_distance_meters': self.total_distance_meters,
            'total_objects_moved': self.total_objects_moved,
            'compute': {
                'perception_flops': self.compute.perception_flops,
                'planning_flops': self.compute.planning_flops,
                'inference_flops': self.compute.inference_flops,
            },
            'communication': {
                'sensor_bandwidth_mbps': self.communication.sensor_bandwidth_mbps,
                'coordination_bandwidth_kbps': self.communication.coordination_bandwidth_kbps,
                'gossip_bandwidth_kbps': self.communication.gossip_bandwidth_kbps,
            },
            'power': {
                'total_joules': self.power.total_joules,
                'locomotion_joules': self.power.locomotion_joules,
                'manipulation_joules': self.power.manipulation_joules,
            }
        }
    
    def to_json(self, path: str) -> None:
        """Save metrics to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class Workload(ABC):
    """
    Abstract workload profiler contract.
    
    Collects metrics from simulation for hardware sizing.
    """
    
    @abstractmethod
    def initialize(self, scenario: Scenario) -> bool:
        """Initialize workload profiler with scenario"""
        pass
    
    @abstractmethod
    def start_recording(self) -> None:
        """Start recording metrics"""
        pass
    
    @abstractmethod
    def stop_recording(self) -> None:
        """Stop recording metrics"""
        pass
    
    @abstractmethod
    def record_task_start(self, task_id: str, task_type: str) -> None:
        """Record task start event"""
        pass
    
    @abstractmethod
    def record_task_end(self, task_id: str, success: bool, 
                        distance: float, objects: int, actions: int) -> None:
        """Record task completion event"""
        pass
    
    @abstractmethod
    def record_sensor_data(self, sensor_id: str, data_size_bytes: int) -> None:
        """Record sensor data generation"""
        pass
    
    @abstractmethod
    def record_compute_event(self, 
                             event_type: str, 
                             flops: float, 
                             duration_ms: float) -> None:
        """Record compute event (perception, planning, inference)"""
        pass
    
    @abstractmethod
    def record_communication_event(self,
                                   message_type: str,
                                   size_bytes: int,
                                   latency_ms: float) -> None:
        """Record communication event"""
        pass
    
    @abstractmethod
    def record_power_event(self,
                          subsystem: str,
                          joules: float) -> None:
        """Record power consumption event"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> WorkloadMetrics:
        """Get aggregated workload metrics"""
        pass
    
    @abstractmethod
    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """
        Generate hardware recommendations based on workload.
        
        Returns:
            Dict with recommendations for:
            - compute: Required FLOPS, memory
            - sensors: Resolution, framerate requirements
            - communication: Bandwidth, latency requirements
            - power: Battery capacity, power budget
        """
        pass
    
    @abstractmethod
    def export_metrics(self, path: str, format: str = "json") -> None:
        """Export metrics to file"""
        pass


class ScenarioRunner(ABC):
    """Runs scenarios and collects workload data"""
    
    @abstractmethod
    def load_scenario(self, config: ScenarioConfig) -> Scenario:
        """Load scenario from configuration"""
        pass
    
    @abstractmethod
    def run(self, 
            scenario: Scenario,
            workload_profiler: Optional[Workload] = None,
            callback: Optional[callable] = None) -> WorkloadMetrics:
        """
        Run scenario and collect metrics.
        
        Args:
            scenario: Scenario to run
            workload_profiler: Optional profiler for detailed metrics
            callback: Optional callback(step, state) for monitoring
            
        Returns:
            Workload metrics from the run
        """
        pass
    
    @abstractmethod
    def run_batch(self, 
                  scenarios: List[Scenario],
                  parallel: bool = False) -> List[WorkloadMetrics]:
        """Run multiple scenarios and collect metrics"""
        pass
