"""
Profiled VLA Wrapper

Wraps a real VLA model and profiles its performance.
Used with TDW to get actual compute requirements for hardware sizing.
"""

import time

from .interface import VLAInterface, VLAObservation, VLAAction, VLAMetrics


class ProfiledVLA(VLAInterface):
    """
    Profiling wrapper for VLA models.
    
    Measures:
    - Inference latency
    - Memory usage
    - FLOPS (estimated)
    
    Use this to get real workload data for R3 hardware sizing.
    """
    
    def __init__(self, 
                 model: VLAInterface,
                 latency_budget_ms: float = 100.0):
        self._model = model
        self._latency_budget_ms = latency_budget_ms
        
        self._total_inferences = 0
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0
        self._violations = 0
        
        self._last_metrics = VLAMetrics()
    
    def predict(self, observation: VLAObservation) -> VLAAction:
        """Run inference with profiling"""
        start = time.perf_counter()
        
        action = self._model.predict(observation)
        inner_metrics = self._model.get_metrics()
        
        latency_ms = (time.perf_counter() - start) * 1000
        
        self._total_inferences += 1
        self._total_latency_ms += latency_ms
        self._max_latency_ms = max(self._max_latency_ms, latency_ms)
        
        if latency_ms > self._latency_budget_ms:
            self._violations += 1
        
        self._last_metrics = VLAMetrics(
            latency_ms=latency_ms,
            flops=inner_metrics.flops,
            memory_bytes=inner_metrics.memory_bytes,
            tokens_processed=inner_metrics.tokens_processed
        )
        
        return action
    
    def get_metrics(self) -> VLAMetrics:
        return self._last_metrics
    
    def get_profiling_summary(self) -> dict:
        """Get summary of all profiled data"""
        avg_latency = self._total_latency_ms / max(self._total_inferences, 1)
        violation_rate = self._violations / max(self._total_inferences, 1)
        
        return {
            'total_inferences': self._total_inferences,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': self._max_latency_ms,
            'latency_budget_ms': self._latency_budget_ms,
            'violations': self._violations,
            'violation_rate': violation_rate,
            'meets_contract': violation_rate < 0.01
        }
    
    def reset(self) -> None:
        self._model.reset()
    
    def reset_profiling(self) -> None:
        """Reset profiling counters"""
        self._total_inferences = 0
        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0
        self._violations = 0
    
    @property
    def model_name(self) -> str:
        return f"Profiled({self._model.model_name})"
    
    @property
    def latency_budget_ms(self) -> float:
        return self._latency_budget_ms
