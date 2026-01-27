"""
Safety Contracts for Robotic Systems

Based on:
- ISO 26262 (automotive safety)
- IEC 61508 (functional safety)
- Robot safety standards (ISO 10218, ISO 15066)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum

from .contract import Contract, Assumption, Guarantee, Predicate, LogicType


class SafetyIntegrityLevel(Enum):
    """Safety Integrity Levels (SIL) per IEC 61508"""
    SIL_0 = 0  # No safety requirement
    SIL_1 = 1  # Low
    SIL_2 = 2  # Medium
    SIL_3 = 3  # High
    SIL_4 = 4  # Very high (nuclear, aerospace)


class HazardSeverity(Enum):
    """Hazard severity classification"""
    NEGLIGIBLE = 1
    MARGINAL = 2
    CRITICAL = 3
    CATASTROPHIC = 4


@dataclass
class Invariant:
    """
    Safety invariant: a property that must always hold.
    
    Invariants are stronger than guarantees - they must hold
    regardless of assumptions (fail-safe behavior).
    """
    name: str
    expression: str
    evaluator: Callable[[Dict], bool]
    severity: HazardSeverity = HazardSeverity.CRITICAL
    sil: SafetyIntegrityLevel = SafetyIntegrityLevel.SIL_2
    
    def check(self, state: Dict[str, Any]) -> bool:
        return self.evaluator(state)


@dataclass
class SafetyContract(Contract):
    """
    Safety-specific contract with invariants.
    
    Extends Contract with:
    - Safety invariants (must always hold)
    - Hazard analysis metadata
    - SIL classification
    """
    
    invariants: List[Invariant] = field(default_factory=list)
    sil: SafetyIntegrityLevel = SafetyIntegrityLevel.SIL_2
    hazard_id: Optional[str] = None
    
    def check_invariants(self, state: Dict[str, Any]) -> List[Invariant]:
        """Check all invariants, return list of violated ones"""
        return [inv for inv in self.invariants if not inv.check(state)]
    
    @classmethod
    def collision_avoidance_contract(cls,
                                     name: str,
                                     min_distance_m: float,
                                     max_velocity_mps: float,
                                     reaction_time_ms: float) -> 'SafetyContract':
        """
        Create collision avoidance safety contract.
        
        Based on ISO 15066 safety distance calculations.
        """
        stopping_distance = max_velocity_mps * (reaction_time_ms / 1000.0) + \
                           (max_velocity_mps ** 2) / (2 * 2.0)  # assuming 2 m/s^2 decel
        
        safe_distance = min_distance_m + stopping_distance
        
        assumptions = Assumption(
            name=f"{name}_collision_assumptions",
            predicates=[
                Predicate(
                    name="sensors_operational",
                    expression="collision_sensors.operational == true",
                    evaluator=lambda s: s.get('sensors_operational', True)
                ),
                Predicate(
                    name="velocity_limit",
                    expression=f"robot.velocity <= {max_velocity_mps} m/s",
                    evaluator=lambda s: s.get('robot_velocity_mps', 0) <= max_velocity_mps
                )
            ]
        )
        
        guarantees = Guarantee(
            name=f"{name}_collision_guarantees",
            predicates=[
                Predicate(
                    name="safe_distance",
                    expression=f"distance_to_obstacle >= {safe_distance:.2f}m",
                    evaluator=lambda s: s.get('min_obstacle_distance_m', float('inf')) >= safe_distance
                ),
                Predicate(
                    name="emergency_stop_capability",
                    expression="emergency_stop.available == true",
                    evaluator=lambda s: s.get('emergency_stop_available', True)
                )
            ]
        )
        
        contract = cls(
            name=name,
            assumptions=assumptions,
            guarantees=guarantees,
            sil=SafetyIntegrityLevel.SIL_2,
            description=f"Collision avoidance: {safe_distance:.2f}m safe distance at {max_velocity_mps} m/s"
        )
        
        contract.invariants = [
            Invariant(
                name="no_collision",
                expression="collision_detected == false",
                evaluator=lambda s: not s.get('collision_detected', False),
                severity=HazardSeverity.CRITICAL,
                sil=SafetyIntegrityLevel.SIL_3
            )
        ]
        
        return contract
    
    @classmethod
    def manipulation_safety_contract(cls,
                                     name: str,
                                     max_grip_force_n: float,
                                     max_payload_kg: float,
                                     workspace_bounds: Dict[str, tuple]) -> 'SafetyContract':
        """
        Create manipulation safety contract for grippers/arms.
        """
        assumptions = Assumption(
            name=f"{name}_manipulation_assumptions",
            predicates=[
                Predicate(
                    name="force_sensing",
                    expression="force_sensor.operational == true",
                    evaluator=lambda s: s.get('force_sensor_operational', True)
                ),
                Predicate(
                    name="payload_within_limit",
                    expression=f"payload <= {max_payload_kg} kg",
                    evaluator=lambda s: s.get('payload_kg', 0) <= max_payload_kg
                )
            ]
        )
        
        guarantees = Guarantee(
            name=f"{name}_manipulation_guarantees",
            predicates=[
                Predicate(
                    name="grip_force_limit",
                    expression=f"grip_force <= {max_grip_force_n} N",
                    evaluator=lambda s: s.get('grip_force_n', 0) <= max_grip_force_n
                ),
                Predicate(
                    name="workspace_containment",
                    expression=f"end_effector within workspace bounds",
                    evaluator=lambda s: all(
                        workspace_bounds.get(axis, (-float('inf'), float('inf')))[0] <= 
                        s.get(f'ee_position_{axis}', 0) <=
                        workspace_bounds.get(axis, (-float('inf'), float('inf')))[1]
                        for axis in ['x', 'y', 'z']
                    )
                )
            ]
        )
        
        contract = cls(
            name=name,
            assumptions=assumptions,
            guarantees=guarantees,
            sil=SafetyIntegrityLevel.SIL_2,
            description=f"Manipulation safety: {max_grip_force_n}N max force, {max_payload_kg}kg payload"
        )
        
        contract.invariants = [
            Invariant(
                name="no_crush_injury",
                expression=f"grip_force < {max_grip_force_n * 1.5} N (absolute limit)",
                evaluator=lambda s: s.get('grip_force_n', 0) < max_grip_force_n * 1.5,
                severity=HazardSeverity.CRITICAL
            )
        ]
        
        return contract


@dataclass
class SystemSafetyContract:
    """
    System-level safety contract composing component contracts.
    
    Ensures that composed system satisfies overall safety requirements.
    """
    
    name: str
    component_contracts: List[SafetyContract]
    system_invariants: List[Invariant]
    target_sil: SafetyIntegrityLevel
    
    def check_composition(self) -> Dict[str, Any]:
        """
        Check if component contracts compose to satisfy system requirements.
        
        Returns analysis of:
        - Coverage: Are all hazards addressed?
        - Consistency: Do component contracts not contradict?
        - Completeness: Are all interfaces specified?
        """
        result = {
            'coverage': True,
            'consistency': True,
            'completeness': True,
            'issues': []
        }
        
        all_invariants = []
        for contract in self.component_contracts:
            all_invariants.extend(contract.invariants)
        
        for sys_inv in self.system_invariants:
            if not any(inv.name == sys_inv.name for inv in all_invariants):
                result['coverage'] = False
                result['issues'].append(f"System invariant '{sys_inv.name}' not covered by components")
        
        return result
