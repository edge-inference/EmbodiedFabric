"""
Formal Contract Definitions

Based on Assume-Guarantee reasoning for CPS.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Set
from enum import Enum
from abc import ABC, abstractmethod


class LogicType(Enum):
    """Supported logic types for contract specifications"""
    BOOLEAN = "boolean"           # Propositional logic
    LTL = "ltl"                   # Linear Temporal Logic
    STL = "stl"                   # Signal Temporal Logic (for continuous)
    INTERVAL = "interval"         # Interval arithmetic bounds


@dataclass
class Predicate:
    """
    A predicate in formal logic.
    Can be evaluated against a system state/trace.
    """
    name: str
    expression: str                              # Human-readable expression
    logic_type: LogicType = LogicType.BOOLEAN
    variables: List[str] = field(default_factory=list)
    evaluator: Optional[Callable[[Dict], bool]] = None
    
    def evaluate(self, state: Dict[str, Any]) -> bool:
        """Evaluate predicate against state"""
        if self.evaluator:
            return self.evaluator(state)
        raise NotImplementedError(f"No evaluator for predicate: {self.name}")
    
    def __str__(self) -> str:
        return f"{self.name}: {self.expression}"


@dataclass
class Assumption:
    """
    Contract assumption: what the component expects from environment.
    
    A = {a1, a2, ...} where each ai is a predicate.
    The assumption holds iff all predicates hold.
    """
    name: str
    predicates: List[Predicate] = field(default_factory=list)
    description: str = ""
    
    def holds(self, state: Dict[str, Any]) -> bool:
        """Check if assumption holds for given state"""
        return all(p.evaluate(state) for p in self.predicates)
    
    def add_predicate(self, predicate: Predicate) -> None:
        self.predicates.append(predicate)
    
    def __and__(self, other: 'Assumption') -> 'Assumption':
        """Conjunction of assumptions"""
        return Assumption(
            name=f"({self.name} AND {other.name})",
            predicates=self.predicates + other.predicates,
            description=f"Conjunction of {self.name} and {other.name}"
        )


@dataclass
class Guarantee:
    """
    Contract guarantee: what the component promises if assumptions hold.
    
    G = {g1, g2, ...} where each gi is a predicate.
    The guarantee must hold iff all predicates hold (when A holds).
    """
    name: str
    predicates: List[Predicate] = field(default_factory=list)
    description: str = ""
    
    def holds(self, state: Dict[str, Any]) -> bool:
        """Check if guarantee holds for given state"""
        return all(p.evaluate(state) for p in self.predicates)
    
    def add_predicate(self, predicate: Predicate) -> None:
        self.predicates.append(predicate)
    
    def __and__(self, other: 'Guarantee') -> 'Guarantee':
        """Conjunction of guarantees"""
        return Guarantee(
            name=f"({self.name} AND {other.name})",
            predicates=self.predicates + other.predicates,
            description=f"Conjunction of {self.name} and {other.name}"
        )


@dataclass
class Contract:
    """
    Formal Contract C = (A, G)
    
    Semantics: If A holds, then G must hold.
    Equivalently: A → G (implication)
    
    Contract satisfaction: Implementation I satisfies C iff
        for all traces t: A(t) → G(t)
    """
    name: str
    assumptions: Assumption
    guarantees: Guarantee
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check_satisfaction(self, trace: List[Dict[str, Any]]) -> 'ContractCheckResult':
        """
        Check if a trace satisfies this contract.
        
        For each state in trace:
        - If assumption holds, guarantee must hold
        - If assumption doesn't hold, contract is vacuously satisfied
        """
        violations = []
        vacuous_count = 0
        
        for i, state in enumerate(trace):
            a_holds = self.assumptions.holds(state)
            g_holds = self.guarantees.holds(state)
            
            if a_holds and not g_holds:
                violations.append(ContractViolation(
                    step=i,
                    state=state,
                    assumption_held=True,
                    guarantee_held=False,
                    message=f"Step {i}: Assumption held but guarantee violated"
                ))
            elif not a_holds:
                vacuous_count += 1
        
        return ContractCheckResult(
            contract_name=self.name,
            satisfied=len(violations) == 0,
            violations=violations,
            vacuous_steps=vacuous_count,
            total_steps=len(trace)
        )
    
    def __rshift__(self, other: 'Contract') -> bool:
        """
        Refinement check: self >> other means self refines other.
        
        C1 refines C2 iff:
        - A2 ⊆ A1 (weaker assumptions)
        - G1 ⊆ G2 (stronger guarantees)
        """
        pass
    
    def __or__(self, other: 'Contract') -> 'Contract':
        """Parallel composition of contracts"""
        return Contract(
            name=f"({self.name} || {other.name})",
            assumptions=self.assumptions & other.assumptions,
            guarantees=self.guarantees & other.guarantees,
            description=f"Parallel composition"
        )


@dataclass
class ContractViolation:
    """Record of a contract violation"""
    step: int
    state: Dict[str, Any]
    assumption_held: bool
    guarantee_held: bool
    message: str


@dataclass
class ContractCheckResult:
    """Result of contract satisfaction check"""
    contract_name: str
    satisfied: bool
    violations: List[ContractViolation]
    vacuous_steps: int
    total_steps: int
    
    @property
    def violation_rate(self) -> float:
        effective_steps = self.total_steps - self.vacuous_steps
        if effective_steps == 0:
            return 0.0
        return len(self.violations) / effective_steps


class ContractBuilder:
    """Fluent builder for constructing contracts"""
    
    def __init__(self, name: str):
        self._name = name
        self._assumptions: List[Predicate] = []
        self._guarantees: List[Predicate] = []
        self._description = ""
    
    def assume(self, name: str, expression: str, 
               evaluator: Optional[Callable] = None) -> 'ContractBuilder':
        """Add an assumption predicate"""
        self._assumptions.append(Predicate(
            name=name,
            expression=expression,
            evaluator=evaluator
        ))
        return self
    
    def guarantee(self, name: str, expression: str,
                  evaluator: Optional[Callable] = None) -> 'ContractBuilder':
        """Add a guarantee predicate"""
        self._guarantees.append(Predicate(
            name=name,
            expression=expression,
            evaluator=evaluator
        ))
        return self
    
    def describe(self, description: str) -> 'ContractBuilder':
        self._description = description
        return self
    
    def build(self) -> Contract:
        return Contract(
            name=self._name,
            assumptions=Assumption(
                name=f"{self._name}_assumptions",
                predicates=self._assumptions
            ),
            guarantees=Guarantee(
                name=f"{self._name}_guarantees",
                predicates=self._guarantees
            ),
            description=self._description
        )
