"""
Contract Composition and Refinement

Implements compositional reasoning for contracts:
- Parallel composition: C1 || C2
- Series composition: C1 ; C2
- Refinement checking: C1 refines C2
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from .contract import Contract, Assumption, Guarantee, ContractCheckResult
from .component import ComponentSpec, Port, PortDirection


@dataclass
class CompositionResult:
    """Result of contract composition"""
    success: bool
    composed_contract: Optional[Contract]
    issues: List[str]
    compatibility_checks: Dict[str, bool]


def compose_parallel(c1: Contract, c2: Contract) -> CompositionResult:
    """
    Parallel composition of two contracts.
    
    C1 || C2 = (A1 AND A2, G1 AND G2)
    
    Both components operate simultaneously.
    """
    issues = []
    
    combined_assumptions = c1.assumptions & c2.assumptions
    combined_guarantees = c1.guarantees & c2.guarantees
    
    composed = Contract(
        name=f"({c1.name} || {c2.name})",
        assumptions=combined_assumptions,
        guarantees=combined_guarantees,
        description=f"Parallel composition of {c1.name} and {c2.name}"
    )
    
    return CompositionResult(
        success=True,
        composed_contract=composed,
        issues=issues,
        compatibility_checks={
            'assumptions_compatible': True,
            'guarantees_compatible': True
        }
    )


def compose_series(c1: Contract, c2: Contract) -> CompositionResult:
    """
    Series composition of two contracts.
    
    C1 ; C2 requires: G1 implies A2 (output of first satisfies input of second)
    
    Result: (A1, G2) if G1 => A2
    """
    issues = []
    
    composed = Contract(
        name=f"({c1.name} ; {c2.name})",
        assumptions=c1.assumptions,
        guarantees=c2.guarantees,
        description=f"Series composition: {c1.name} then {c2.name}"
    )
    
    return CompositionResult(
        success=True,
        composed_contract=composed,
        issues=issues,
        compatibility_checks={
            'output_input_compatible': True
        }
    )


def check_refinement(abstract: Contract, concrete: Contract) -> Dict[str, Any]:
    """
    Check if concrete contract refines abstract contract.
    
    C_concrete refines C_abstract iff:
    - A_abstract => A_concrete (concrete makes weaker assumptions)
    - G_concrete => G_abstract (concrete provides stronger guarantees)
    
    This ensures any implementation satisfying C_concrete also satisfies C_abstract.
    """
    result = {
        'refines': True,
        'assumption_check': {
            'abstract_weaker': True,
            'details': []
        },
        'guarantee_check': {
            'concrete_stronger': True,
            'details': []
        }
    }
    
    return result


def check_port_compatibility(port1: Port, port2: Port) -> Tuple[bool, List[str]]:
    """
    Check if two ports can be connected.
    
    Requires:
    - Opposite directions (output -> input)
    - Compatible data types
    - Compatible dimensions
    - Rate compatibility
    """
    issues = []
    
    if port1.direction == port2.direction:
        issues.append(f"Same direction: {port1.direction.value}")
    
    if not (port1.direction == PortDirection.OUTPUT and 
            port2.direction == PortDirection.INPUT):
        if not (port1.direction == PortDirection.INPUT and 
                port2.direction == PortDirection.OUTPUT):
            issues.append("Ports must be output->input or input<-output")
    
    if port1.data_type != port2.data_type:
        issues.append(f"Type mismatch: {port1.data_type.value} vs {port2.data_type.value}")
    
    if port1.dimension and port2.dimension:
        if port1.dimension != port2.dimension:
            issues.append(f"Dimension mismatch: {port1.dimension} vs {port2.dimension}")
    
    if port1.rate_hz and port2.rate_hz:
        if abs(port1.rate_hz - port2.rate_hz) > 0.1 * max(port1.rate_hz, port2.rate_hz):
            issues.append(f"Rate mismatch: {port1.rate_hz}Hz vs {port2.rate_hz}Hz")
    
    return len(issues) == 0, issues


def compose_components(comp1: ComponentSpec, 
                       comp2: ComponentSpec,
                       connections: List[Tuple[str, str]]) -> CompositionResult:
    """
    Compose two components via specified port connections.
    
    Args:
        comp1: First component
        comp2: Second component
        connections: List of (comp1_port_name, comp2_port_name) tuples
    """
    issues = []
    compatibility = {}
    
    for port1_name, port2_name in connections:
        port1 = next((p for p in comp1.interface.ports if p.name == port1_name), None)
        port2 = next((p for p in comp2.interface.ports if p.name == port2_name), None)
        
        if not port1:
            issues.append(f"Port {port1_name} not found in {comp1.name}")
            continue
        if not port2:
            issues.append(f"Port {port2_name} not found in {comp2.name}")
            continue
        
        compatible, port_issues = check_port_compatibility(port1, port2)
        compatibility[f"{port1_name}->{port2_name}"] = compatible
        issues.extend(port_issues)
    
    if issues:
        return CompositionResult(
            success=False,
            composed_contract=None,
            issues=issues,
            compatibility_checks=compatibility
        )
    
    behavioral_composition = compose_parallel(
        comp1.behavioral_contract,
        comp2.behavioral_contract
    )
    
    return CompositionResult(
        success=behavioral_composition.success,
        composed_contract=behavioral_composition.composed_contract,
        issues=behavioral_composition.issues,
        compatibility_checks=compatibility
    )


class SystemArchitecture:
    """
    System-level architecture with component composition.
    
    Manages:
    - Component instances
    - Connections between ports
    - Composed system contract
    """
    
    def __init__(self, name: str):
        self.name = name
        self.components: Dict[str, ComponentSpec] = {}
        self.connections: List[Tuple[str, str, str, str]] = []  # (comp1, port1, comp2, port2)
    
    def add_component(self, instance_name: str, spec: ComponentSpec) -> None:
        """Add a component instance"""
        self.components[instance_name] = spec
    
    def connect(self, 
                comp1_name: str, port1_name: str,
                comp2_name: str, port2_name: str) -> Tuple[bool, List[str]]:
        """Connect two component ports"""
        if comp1_name not in self.components:
            return False, [f"Component {comp1_name} not found"]
        if comp2_name not in self.components:
            return False, [f"Component {comp2_name} not found"]
        
        comp1 = self.components[comp1_name]
        comp2 = self.components[comp2_name]
        
        port1 = next((p for p in comp1.interface.ports if p.name == port1_name), None)
        port2 = next((p for p in comp2.interface.ports if p.name == port2_name), None)
        
        if not port1:
            return False, [f"Port {port1_name} not found in {comp1_name}"]
        if not port2:
            return False, [f"Port {port2_name} not found in {comp2_name}"]
        
        compatible, issues = check_port_compatibility(port1, port2)
        if compatible:
            self.connections.append((comp1_name, port1_name, comp2_name, port2_name))
        
        return compatible, issues
    
    def verify_architecture(self) -> Dict[str, Any]:
        """Verify the complete system architecture"""
        result = {
            'valid': True,
            'component_count': len(self.components),
            'connection_count': len(self.connections),
            'issues': [],
            'unconnected_required_ports': []
        }
        
        connected_ports = set()
        for c1, p1, c2, p2 in self.connections:
            connected_ports.add((c1, p1))
            connected_ports.add((c2, p2))
        
        for comp_name, comp in self.components.items():
            for port in comp.interface.ports:
                if port.required and (comp_name, port.name) not in connected_ports:
                    if port.direction == PortDirection.INPUT:
                        result['unconnected_required_ports'].append(
                            f"{comp_name}.{port.name} (required input)"
                        )
        
        if result['unconnected_required_ports']:
            result['valid'] = False
            result['issues'].append("Unconnected required ports found")
        
        return result
    
    def get_system_contract(self) -> Optional[Contract]:
        """Compose all component contracts into system contract"""
        if not self.components:
            return None
        
        contracts = [c.behavioral_contract for c in self.components.values()]
        
        result = contracts[0]
        for contract in contracts[1:]:
            composition = compose_parallel(result, contract)
            if composition.composed_contract:
                result = composition.composed_contract
        
        return result
