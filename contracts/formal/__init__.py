"""
Formal Contracts for PhysicAI Simulator

Based on Assume-Guarantee (A/G) contract paradigm from:
- CHASE (Contract-Based Heterogeneous Analysis and Systems Engineering)
- Ptolemy/Berkeley CPS design methodology
- ISO 26262 / DO-178C safety standards

A contract C = (A, G) where:
- A: Assumptions about the environment/inputs
- G: Guarantees about outputs (if A holds)

Contract composition rules:
- Parallel: C1 || C2 = (A1 ∧ A2, G1 ∧ G2)
- Series: C1 ; C2 requires G1 ⊆ A2 (refinement)
"""

from .contract import Contract, Assumption, Guarantee
from .timing import TimingContract, LatencyBound, PeriodContract
from .safety import SafetyContract, Invariant
from .component import ComponentSpec, Interface, Port
from .composition import compose_parallel, compose_series, check_refinement

__all__ = [
    'Contract', 'Assumption', 'Guarantee',
    'TimingContract', 'LatencyBound', 'PeriodContract',
    'SafetyContract', 'Invariant',
    'ComponentSpec', 'Interface', 'Port',
    'compose_parallel', 'compose_series', 'check_refinement',
]
