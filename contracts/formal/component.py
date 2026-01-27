"""
Component Specifications with Formal Contracts

Based on:
- Ptolemy actor semantics
- AADL (Architecture Analysis and Design Language)
- SysML component specifications
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum

from .contract import Contract
from .timing import TimingContract
from .safety import SafetyContract


class PortDirection(Enum):
    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class DataType(Enum):
    """Standard data types for port specifications"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    REAL = "real"
    ARRAY = "array"
    IMAGE = "image"           # RGB/Depth image
    POINTCLOUD = "pointcloud" # LiDAR data
    POSE = "pose"             # Position + Orientation
    TWIST = "twist"           # Linear + Angular velocity
    WRENCH = "wrench"         # Force + Torque
    CUSTOM = "custom"


@dataclass
class Port:
    """
    Component port specification.
    
    Ports define the interface through which components communicate.
    Each port has associated contracts (timing, data validity, etc.)
    """
    name: str
    direction: PortDirection
    data_type: DataType
    dimension: Optional[tuple] = None            # e.g., (640, 480, 3) for image
    unit: Optional[str] = None                   # e.g., "m/s", "rad"
    range: Optional[tuple] = None                # e.g., (-1.0, 1.0)
    rate_hz: Optional[float] = None              # Expected data rate
    required: bool = True
    
    def compatible_with(self, other: 'Port') -> bool:
        """Check if this port can connect to another"""
        if self.direction == other.direction:
            return False
        if self.data_type != other.data_type:
            return False
        if self.dimension and other.dimension and self.dimension != other.dimension:
            return False
        return True


@dataclass
class Interface:
    """
    Component interface: collection of ports.
    
    Defines what a component exposes to other components.
    """
    name: str
    ports: List[Port] = field(default_factory=list)
    
    def get_input_ports(self) -> List[Port]:
        return [p for p in self.ports if p.direction in (PortDirection.INPUT, PortDirection.INOUT)]
    
    def get_output_ports(self) -> List[Port]:
        return [p for p in self.ports if p.direction in (PortDirection.OUTPUT, PortDirection.INOUT)]
    
    def add_port(self, port: Port) -> None:
        self.ports.append(port)


@dataclass
class ComponentSpec:
    """
    Full component specification with contracts.
    
    A component specification includes:
    - Interface (ports)
    - Behavioral contract (A/G)
    - Timing contract
    - Safety contract (if applicable)
    - Implementation requirements
    """
    name: str
    interface: Interface
    behavioral_contract: Contract
    timing_contract: Optional[TimingContract] = None
    safety_contract: Optional[SafetyContract] = None
    
    implementation_constraints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_contracts(self) -> List[Contract]:
        """Get all contracts associated with this component"""
        contracts = [self.behavioral_contract]
        if self.timing_contract:
            contracts.append(self.timing_contract)
        if self.safety_contract:
            contracts.append(self.safety_contract)
        return contracts
    
    def verify_against_trace(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify all contracts against a simulation trace"""
        results = {}
        for contract in self.get_all_contracts():
            results[contract.name] = contract.check_satisfaction(trace)
        return results


class SensorComponentSpec(ComponentSpec):
    """Specialized component spec for sensors"""
    
    @classmethod
    def rgb_camera(cls,
                   name: str,
                   resolution: tuple,
                   fps: float,
                   fov_degrees: float,
                   latency_ms: float) -> 'SensorComponentSpec':
        """Create RGB camera component specification"""
        from .contract import ContractBuilder
        
        interface = Interface(
            name=f"{name}_interface",
            ports=[
                Port(
                    name="rgb_image",
                    direction=PortDirection.OUTPUT,
                    data_type=DataType.IMAGE,
                    dimension=(*resolution, 3),
                    rate_hz=fps
                ),
                Port(
                    name="camera_info",
                    direction=PortDirection.OUTPUT,
                    data_type=DataType.CUSTOM,
                    rate_hz=fps
                )
            ]
        )
        
        behavioral = ContractBuilder(f"{name}_behavioral") \
            .assume("power_on", "camera.powered == true",
                   evaluator=lambda s: s.get('camera_powered', True)) \
            .assume("lighting_adequate", "ambient_light > 10 lux",
                   evaluator=lambda s: s.get('ambient_light_lux', 100) > 10) \
            .guarantee("image_valid", "image.shape == resolution",
                      evaluator=lambda s: s.get('image_shape') == resolution) \
            .guarantee("image_rate", f"output_rate >= {fps * 0.95} Hz",
                      evaluator=lambda s: s.get('output_rate_hz', fps) >= fps * 0.95) \
            .build()
        
        timing = TimingContract.sensor_contract(
            name=f"{name}_timing",
            sensor_period_ms=1000.0 / fps,
            sensor_jitter_ms=1.0,
            processing_latency_ms=latency_ms,
            output_period_ms=1000.0 / fps
        )
        
        return cls(
            name=name,
            interface=interface,
            behavioral_contract=behavioral,
            timing_contract=timing,
            metadata={
                'type': 'sensor',
                'subtype': 'rgb_camera',
                'resolution': resolution,
                'fps': fps,
                'fov': fov_degrees
            }
        )
    
    @classmethod
    def depth_camera(cls,
                     name: str,
                     resolution: tuple,
                     fps: float,
                     min_range_m: float,
                     max_range_m: float,
                     latency_ms: float) -> 'SensorComponentSpec':
        """Create depth camera component specification"""
        from .contract import ContractBuilder
        
        interface = Interface(
            name=f"{name}_interface",
            ports=[
                Port(
                    name="depth_image",
                    direction=PortDirection.OUTPUT,
                    data_type=DataType.IMAGE,
                    dimension=resolution,
                    unit="m",
                    range=(min_range_m, max_range_m),
                    rate_hz=fps
                )
            ]
        )
        
        behavioral = ContractBuilder(f"{name}_behavioral") \
            .assume("depth_sensor_operational", "depth_sensor.status == ok",
                   evaluator=lambda s: s.get('depth_sensor_status') == 'ok') \
            .guarantee("depth_range_valid", f"depth in [{min_range_m}, {max_range_m}]m",
                      evaluator=lambda s: min_range_m <= s.get('depth_value', 0) <= max_range_m) \
            .build()
        
        timing = TimingContract.sensor_contract(
            name=f"{name}_timing",
            sensor_period_ms=1000.0 / fps,
            sensor_jitter_ms=2.0,
            processing_latency_ms=latency_ms,
            output_period_ms=1000.0 / fps
        )
        
        return cls(
            name=name,
            interface=interface,
            behavioral_contract=behavioral,
            timing_contract=timing,
            metadata={
                'type': 'sensor',
                'subtype': 'depth_camera',
                'resolution': resolution,
                'fps': fps,
                'range': (min_range_m, max_range_m)
            }
        )


class ActuatorComponentSpec(ComponentSpec):
    """Specialized component spec for actuators"""
    
    @classmethod
    def wheeled_base(cls,
                     name: str,
                     max_linear_velocity_mps: float,
                     max_angular_velocity_rps: float,
                     control_rate_hz: float) -> 'ActuatorComponentSpec':
        """Create wheeled mobile base component specification"""
        from .contract import ContractBuilder
        from .safety import SafetyContract
        
        interface = Interface(
            name=f"{name}_interface",
            ports=[
                Port(
                    name="cmd_vel",
                    direction=PortDirection.INPUT,
                    data_type=DataType.TWIST,
                    range=((-max_linear_velocity_mps, max_linear_velocity_mps),
                           (-max_angular_velocity_rps, max_angular_velocity_rps)),
                    rate_hz=control_rate_hz
                ),
                Port(
                    name="odom",
                    direction=PortDirection.OUTPUT,
                    data_type=DataType.POSE,
                    rate_hz=control_rate_hz
                ),
                Port(
                    name="actual_vel",
                    direction=PortDirection.OUTPUT,
                    data_type=DataType.TWIST,
                    rate_hz=control_rate_hz
                )
            ]
        )
        
        behavioral = ContractBuilder(f"{name}_behavioral") \
            .assume("motors_enabled", "motors.enabled == true") \
            .assume("battery_sufficient", "battery.level > 10%") \
            .assume("cmd_within_limits", 
                   f"|cmd_vel.linear| <= {max_linear_velocity_mps} m/s") \
            .guarantee("velocity_tracking", 
                      "actual_vel tracks cmd_vel with < 5% error") \
            .guarantee("odom_accurate",
                      "odom drift < 2% of distance traveled") \
            .build()
        
        safety = SafetyContract.collision_avoidance_contract(
            name=f"{name}_safety",
            min_distance_m=0.3,
            max_velocity_mps=max_linear_velocity_mps,
            reaction_time_ms=100.0
        )
        
        return cls(
            name=name,
            interface=interface,
            behavioral_contract=behavioral,
            safety_contract=safety,
            metadata={
                'type': 'actuator',
                'subtype': 'wheeled_base',
                'max_velocity': max_linear_velocity_mps
            }
        )
