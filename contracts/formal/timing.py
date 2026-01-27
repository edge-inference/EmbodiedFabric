"""
Timing Contracts for Real-Time CPS

Based on:
- Real-time scheduling theory
- Lingua Franca timing semantics
- ROS2 DDS QoS timing constraints
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

from .contract import Contract, Assumption, Guarantee, Predicate, LogicType


class TimeUnit(Enum):
    NANOSECONDS = "ns"
    MICROSECONDS = "us"
    MILLISECONDS = "ms"
    SECONDS = "s"


@dataclass
class LatencyBound:
    """
    Latency bound specification.
    
    Used in assumptions: "sensor latency <= 10ms"
    Used in guarantees: "end-to-end response <= 50ms"
    """
    min_value: float = 0.0
    max_value: float = float('inf')
    unit: TimeUnit = TimeUnit.MILLISECONDS
    percentile: float = 1.0                      # 1.0 = worst case, 0.99 = p99
    
    def to_seconds(self) -> tuple:
        """Convert bounds to seconds"""
        multipliers = {
            TimeUnit.NANOSECONDS: 1e-9,
            TimeUnit.MICROSECONDS: 1e-6,
            TimeUnit.MILLISECONDS: 1e-3,
            TimeUnit.SECONDS: 1.0
        }
        m = multipliers[self.unit]
        return (self.min_value * m, self.max_value * m)
    
    def check(self, value_seconds: float) -> bool:
        """Check if value satisfies bound"""
        min_s, max_s = self.to_seconds()
        return min_s <= value_seconds <= max_s


@dataclass
class PeriodContract:
    """
    Periodic execution contract.
    
    Specifies timing for periodic components (sensors, controllers).
    """
    period: float                                # Nominal period
    unit: TimeUnit = TimeUnit.MILLISECONDS
    jitter_bound: float = 0.0                    # Max deviation from period
    deadline: Optional[float] = None             # Must complete within deadline
    
    def to_seconds(self) -> Dict[str, float]:
        multipliers = {
            TimeUnit.NANOSECONDS: 1e-9,
            TimeUnit.MICROSECONDS: 1e-6,
            TimeUnit.MILLISECONDS: 1e-3,
            TimeUnit.SECONDS: 1.0
        }
        m = multipliers[self.unit]
        return {
            'period': self.period * m,
            'jitter': self.jitter_bound * m,
            'deadline': self.deadline * m if self.deadline else self.period * m
        }


@dataclass
class TimingContract(Contract):
    """
    Timing-specific contract with real-time guarantees.
    
    Extends base Contract with timing-specific predicates.
    
    Example:
        Assumption: sensor_period = 33ms (30Hz), jitter <= 1ms
        Guarantee: perception_latency <= 10ms (p99)
    """
    
    input_timing: List[PeriodContract] = field(default_factory=list)
    output_timing: List[LatencyBound] = field(default_factory=list)
    
    @classmethod
    def sensor_contract(cls,
                        name: str,
                        sensor_period_ms: float,
                        sensor_jitter_ms: float,
                        processing_latency_ms: float,
                        output_period_ms: float) -> 'TimingContract':
        """
        Create a standard sensor timing contract.
        
        Assumption: Raw sensor data arrives at sensor_period with bounded jitter
        Guarantee: Processed data available within processing_latency, output at output_period
        """
        assumptions = Assumption(
            name=f"{name}_timing_assumptions",
            predicates=[
                Predicate(
                    name="sensor_period",
                    expression=f"sensor.period == {sensor_period_ms}ms",
                    logic_type=LogicType.INTERVAL,
                    evaluator=lambda s: abs(s.get('sensor_period_ms', 0) - sensor_period_ms) <= sensor_jitter_ms
                ),
                Predicate(
                    name="sensor_jitter",
                    expression=f"sensor.jitter <= {sensor_jitter_ms}ms",
                    logic_type=LogicType.INTERVAL,
                    evaluator=lambda s: s.get('sensor_jitter_ms', 0) <= sensor_jitter_ms
                )
            ]
        )
        
        guarantees = Guarantee(
            name=f"{name}_timing_guarantees",
            predicates=[
                Predicate(
                    name="processing_latency",
                    expression=f"processing.latency <= {processing_latency_ms}ms",
                    logic_type=LogicType.INTERVAL,
                    evaluator=lambda s: s.get('processing_latency_ms', 0) <= processing_latency_ms
                ),
                Predicate(
                    name="output_period",
                    expression=f"output.period == {output_period_ms}ms",
                    logic_type=LogicType.INTERVAL,
                    evaluator=lambda s: abs(s.get('output_period_ms', 0) - output_period_ms) <= sensor_jitter_ms
                )
            ]
        )
        
        contract = cls(
            name=name,
            assumptions=assumptions,
            guarantees=guarantees,
            description=f"Sensor timing contract: {sensor_period_ms}ms input -> {processing_latency_ms}ms processing -> {output_period_ms}ms output"
        )
        contract.input_timing = [PeriodContract(period=sensor_period_ms, jitter_bound=sensor_jitter_ms)]
        contract.output_timing = [LatencyBound(max_value=processing_latency_ms)]
        
        return contract
    
    @classmethod
    def control_loop_contract(cls,
                              name: str,
                              sensing_latency_ms: float,
                              computation_latency_ms: float,
                              actuation_latency_ms: float,
                              loop_period_ms: float) -> 'TimingContract':
        """
        Create a control loop timing contract.
        
        Total latency = sensing + computation + actuation must fit within period.
        """
        total_latency = sensing_latency_ms + computation_latency_ms + actuation_latency_ms
        
        assumptions = Assumption(
            name=f"{name}_control_assumptions",
            predicates=[
                Predicate(
                    name="sensing_latency",
                    expression=f"sensing.latency <= {sensing_latency_ms}ms",
                    evaluator=lambda s: s.get('sensing_latency_ms', 0) <= sensing_latency_ms
                ),
                Predicate(
                    name="actuation_latency",
                    expression=f"actuation.latency <= {actuation_latency_ms}ms",
                    evaluator=lambda s: s.get('actuation_latency_ms', 0) <= actuation_latency_ms
                )
            ]
        )
        
        guarantees = Guarantee(
            name=f"{name}_control_guarantees",
            predicates=[
                Predicate(
                    name="loop_deadline",
                    expression=f"total_latency <= {loop_period_ms}ms",
                    evaluator=lambda s: (s.get('sensing_latency_ms', 0) + 
                                        s.get('computation_latency_ms', 0) + 
                                        s.get('actuation_latency_ms', 0)) <= loop_period_ms
                ),
                Predicate(
                    name="computation_bound",
                    expression=f"computation.latency <= {computation_latency_ms}ms",
                    evaluator=lambda s: s.get('computation_latency_ms', 0) <= computation_latency_ms
                )
            ]
        )
        
        return cls(
            name=name,
            assumptions=assumptions,
            guarantees=guarantees,
            description=f"Control loop: {total_latency}ms total latency within {loop_period_ms}ms period"
        )


@dataclass
class CommunicationTimingContract(TimingContract):
    """
    Communication timing contract for distributed systems.
    
    Models:
    - DSM gossip propagation delays
    - ROS2 DDS QoS latency/reliability
    - LF logical time vs physical time
    """
    
    propagation_delay_ms: float = 0.0
    bandwidth_mbps: float = 100.0
    reliability: float = 1.0                      # 0-1, probability of delivery
    ordering: str = "fifo"                        # "fifo", "causal", "total"
    
    @classmethod
    def gossip_contract(cls,
                        name: str,
                        gossip_period_ms: float,
                        propagation_hops: int,
                        per_hop_delay_ms: float,
                        convergence_probability: float = 0.99) -> 'CommunicationTimingContract':
        """
        Create gossip protocol timing contract.
        
        Models eventual consistency timing for DSM.
        """
        max_propagation = propagation_hops * per_hop_delay_ms
        
        assumptions = Assumption(
            name=f"{name}_gossip_assumptions",
            predicates=[
                Predicate(
                    name="network_available",
                    expression="network.connected == true",
                    evaluator=lambda s: s.get('network_connected', True)
                ),
                Predicate(
                    name="gossip_period",
                    expression=f"gossip.period == {gossip_period_ms}ms",
                    evaluator=lambda s: s.get('gossip_period_ms', gossip_period_ms) == gossip_period_ms
                )
            ]
        )
        
        guarantees = Guarantee(
            name=f"{name}_gossip_guarantees",
            predicates=[
                Predicate(
                    name="propagation_bound",
                    expression=f"propagation.delay <= {max_propagation}ms (p{convergence_probability*100:.0f})",
                    evaluator=lambda s: s.get('propagation_delay_ms', 0) <= max_propagation
                ),
                Predicate(
                    name="eventual_consistency",
                    expression=f"all_replicas_converge within {max_propagation * 2}ms",
                    evaluator=lambda s: s.get('convergence_time_ms', 0) <= max_propagation * 2
                )
            ]
        )
        
        contract = cls(
            name=name,
            assumptions=assumptions,
            guarantees=guarantees,
            propagation_delay_ms=max_propagation,
            description=f"Gossip protocol: {propagation_hops} hops, {max_propagation}ms max propagation"
        )
        return contract
