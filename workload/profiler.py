"""
Workload Profiler

Collects metrics during simulation for hardware sizing (R2/R3).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ComputeEvent:
    """Single compute event"""
    event_type: str
    flops: float
    duration_ms: float
    timestamp: float


@dataclass
class ContractViolation:
    """Contract violation record"""
    contract: str
    actual: float
    budget: float
    timestamp: float


@dataclass
class WorkloadMetrics:
    """Aggregated workload metrics"""
    duration_seconds: float = 0.0
    total_steps: int = 0
    
    vla_inferences: int = 0
    vla_total_flops: float = 0.0
    vla_avg_latency_ms: float = 0.0
    vla_max_latency_ms: float = 0.0
    vla_violations: int = 0
    
    perception_flops: float = 0.0
    planning_flops: float = 0.0
    
    gossip_rounds: int = 0
    gossip_updates: int = 0
    
    tasks_completed: int = 0
    tasks_failed: int = 0
    throughput_per_hour: float = 0.0
    
    sensor_bytes: int = 0
    sensor_bandwidth_mbps: float = 0.0


class WorkloadProfiler:
    """
    Collects workload metrics during simulation.
    
    Feeds into hardware sizing for R3.
    """
    
    def __init__(self):
        self._recording = False
        self._start_time = 0.0
        self._step_count = 0
        
        self._compute_events: List[ComputeEvent] = []
        self._violations: List[ContractViolation] = []
        
        self._vla_latencies: List[float] = []
        self._sensor_bytes = 0
    
    def start_recording(self) -> None:
        """Start recording metrics"""
        self._recording = True
        self._start_time = time.time()
        self._reset()
        logger.info("Workload profiler started")
    
    def stop_recording(self) -> None:
        """Stop recording metrics"""
        self._recording = False
        logger.info("Workload profiler stopped")
    
    def _reset(self) -> None:
        """Reset all counters"""
        self._step_count = 0
        self._compute_events.clear()
        self._violations.clear()
        self._vla_latencies.clear()
        self._sensor_bytes = 0
    
    def record_step(self, state: Any) -> None:
        """Record a simulation step"""
        if not self._recording:
            return
        self._step_count += 1
    
    def record_compute_event(self,
                             event_type: str,
                             flops: float,
                             duration_ms: float) -> None:
        """Record a compute event"""
        if not self._recording:
            return
        
        self._compute_events.append(ComputeEvent(
            event_type=event_type,
            flops=flops,
            duration_ms=duration_ms,
            timestamp=time.time()
        ))
        
        if event_type == "vla_inference":
            self._vla_latencies.append(duration_ms)
    
    def record_contract_violation(self,
                                  contract: str,
                                  actual: float,
                                  budget: float) -> None:
        """Record a contract violation"""
        if not self._recording:
            return
        
        self._violations.append(ContractViolation(
            contract=contract,
            actual=actual,
            budget=budget,
            timestamp=time.time()
        ))
    
    def record_sensor_data(self, sensor_id: str, bytes_count: int) -> None:
        """Record sensor data generation"""
        if not self._recording:
            return
        self._sensor_bytes += bytes_count
    
    def get_metrics(self) -> WorkloadMetrics:
        """Get aggregated metrics"""
        duration = time.time() - self._start_time if self._start_time > 0 else 1.0
        
        vla_events = [e for e in self._compute_events if e.event_type == "vla_inference"]
        vla_flops = sum(e.flops for e in vla_events)
        vla_avg = sum(self._vla_latencies) / len(self._vla_latencies) if self._vla_latencies else 0
        vla_max = max(self._vla_latencies) if self._vla_latencies else 0
        vla_violations = sum(1 for v in self._violations if v.contract == "vla_timing")
        
        perception_flops = sum(e.flops for e in self._compute_events 
                              if e.event_type == "perception")
        planning_flops = sum(e.flops for e in self._compute_events 
                            if e.event_type == "planning")
        
        return WorkloadMetrics(
            duration_seconds=duration,
            total_steps=self._step_count,
            vla_inferences=len(vla_events),
            vla_total_flops=vla_flops,
            vla_avg_latency_ms=vla_avg,
            vla_max_latency_ms=vla_max,
            vla_violations=vla_violations,
            perception_flops=perception_flops,
            planning_flops=planning_flops,
            sensor_bytes=self._sensor_bytes,
            sensor_bandwidth_mbps=(self._sensor_bytes * 8) / (duration * 1e6) if duration > 0 else 0
        )
    
    def get_hardware_recommendations(self) -> Dict[str, Any]:
        """Generate hardware recommendations from workload"""
        metrics = self.get_metrics()
        
        vla_tflops = metrics.vla_total_flops / 1e12
        vla_inference_rate = metrics.vla_inferences / max(metrics.duration_seconds, 1)
        required_tops = vla_tflops / max(metrics.duration_seconds, 1) * 1000 * 1.5
        
        return {
            'compute': {
                'npu_tops': max(20, required_tops),
                'vla_inference_rate_hz': vla_inference_rate,
                'vla_avg_latency_ms': metrics.vla_avg_latency_ms,
                'vla_violations': metrics.vla_violations,
                'recommendation': 'Jetson Orin NX' if required_tops < 50 else 'Jetson AGX Orin'
            },
            'memory': {
                'vla_weights_gb': 5.0,
                'context_buffer_mb': 256,
                'sensor_buffer_mb': 50
            },
            'communication': {
                'sensor_bandwidth_mbps': metrics.sensor_bandwidth_mbps,
                'gossip_bandwidth_kbps': 50
            },
            'contracts_met': metrics.vla_violations == 0
        }
    
    def export(self, path: str) -> None:
        """Export metrics to file"""
        metrics = self.get_metrics()
        recommendations = self.get_hardware_recommendations()
        
        data = {
            'metrics': {
                'duration_seconds': metrics.duration_seconds,
                'total_steps': metrics.total_steps,
                'vla_inferences': metrics.vla_inferences,
                'vla_avg_latency_ms': metrics.vla_avg_latency_ms,
                'vla_violations': metrics.vla_violations,
                'sensor_bandwidth_mbps': metrics.sensor_bandwidth_mbps
            },
            'recommendations': recommendations,
            'violations': [
                {'contract': v.contract, 'actual': v.actual, 'budget': v.budget}
                for v in self._violations
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Workload profile exported to {path}")
