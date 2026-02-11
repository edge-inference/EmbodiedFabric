"""
Co-design contracts -- assume-guarantee specs for SoC timing and safety.
"""

from .contract import Contract, ContractBuilder, Assumption, Guarantee, Predicate
from .timing import TimingContract, CommunicationTimingContract, LatencyBound, PeriodContract
from .safety import SafetyContract, SystemSafetyContract, SafetyIntegrityLevel
from .component import ComponentSpec, Port, Interface, SensorComponentSpec, ActuatorComponentSpec
from .composition import compose_parallel, compose_series, check_port_compatibility, SystemArchitecture
