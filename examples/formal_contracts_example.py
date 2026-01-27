#!/usr/bin/env python3
"""
Example: Formal Contract-Based System Design

Demonstrates the formal A/G contract approach for:
1. Defining component contracts with assumptions and guarantees
2. Composing contracts to verify system-level properties
3. Checking traces against contracts
4. Mapping to hardware requirements through verified contracts

This is how NVIDIA/DeepMind/serious CPS projects approach simulator design.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.formal.contract import Contract, ContractBuilder
from contracts.formal.timing import TimingContract, CommunicationTimingContract
from contracts.formal.safety import SafetyContract, SafetyIntegrityLevel
from contracts.formal.component import (
    ComponentSpec, Interface, Port, PortDirection, DataType,
    SensorComponentSpec, ActuatorComponentSpec
)
from contracts.formal.composition import (
    compose_parallel, compose_series, compose_components, 
    SystemArchitecture
)


def example_1_basic_contracts():
    """Example 1: Define basic A/G contracts"""
    print("=" * 60)
    print("Example 1: Basic Assume-Guarantee Contracts")
    print("=" * 60)
    
    perception_contract = ContractBuilder("perception_pipeline") \
        .assume(
            "camera_operational",
            "camera.status == OPERATIONAL",
            evaluator=lambda s: s.get('camera_status') == 'OPERATIONAL'
        ) \
        .assume(
            "lighting_adequate",
            "ambient_light >= 50 lux",
            evaluator=lambda s: s.get('ambient_light_lux', 0) >= 50
        ) \
        .assume(
            "frame_rate",
            "camera.fps >= 28",
            evaluator=lambda s: s.get('camera_fps', 0) >= 28
        ) \
        .guarantee(
            "object_detection",
            "detection_confidence >= 0.8 for objects within 5m",
            evaluator=lambda s: s.get('detection_confidence', 0) >= 0.8
        ) \
        .guarantee(
            "detection_latency",
            "perception_latency <= 50ms",
            evaluator=lambda s: s.get('perception_latency_ms', 999) <= 50
        ) \
        .describe("Perception pipeline: camera input to object detection output") \
        .build()
    
    print(f"\nContract: {perception_contract.name}")
    print(f"Description: {perception_contract.description}")
    print("\nAssumptions:")
    for pred in perception_contract.assumptions.predicates:
        print(f"  - {pred}")
    print("\nGuarantees:")
    for pred in perception_contract.guarantees.predicates:
        print(f"  - {pred}")
    
    print("\n--- Checking against trace ---")
    
    trace = [
        {'camera_status': 'OPERATIONAL', 'ambient_light_lux': 100, 'camera_fps': 30,
         'detection_confidence': 0.9, 'perception_latency_ms': 30},
        
        {'camera_status': 'OPERATIONAL', 'ambient_light_lux': 100, 'camera_fps': 30,
         'detection_confidence': 0.7, 'perception_latency_ms': 30},  # Violation!
        
        {'camera_status': 'DEGRADED', 'ambient_light_lux': 20, 'camera_fps': 15,
         'detection_confidence': 0.5, 'perception_latency_ms': 100},  # Assumption violated
    ]
    
    result = perception_contract.check_satisfaction(trace)
    print(f"Contract satisfied: {result.satisfied}")
    print(f"Violations: {len(result.violations)}")
    print(f"Vacuous steps (assumption not met): {result.vacuous_steps}")
    
    for v in result.violations:
        print(f"  - {v.message}")


def example_2_timing_contracts():
    """Example 2: Timing contracts for real-time systems"""
    print("\n" + "=" * 60)
    print("Example 2: Timing Contracts")
    print("=" * 60)
    
    sensor_timing = TimingContract.sensor_contract(
        name="rgb_camera_timing",
        sensor_period_ms=33.33,     # 30 Hz
        sensor_jitter_ms=1.0,
        processing_latency_ms=10.0,
        output_period_ms=33.33
    )
    
    control_timing = TimingContract.control_loop_contract(
        name="motion_control_timing",
        sensing_latency_ms=15.0,
        computation_latency_ms=5.0,
        actuation_latency_ms=10.0,
        loop_period_ms=50.0         # 20 Hz control loop
    )
    
    gossip_timing = CommunicationTimingContract.gossip_contract(
        name="dsm_gossip_timing",
        gossip_period_ms=50.0,
        propagation_hops=3,
        per_hop_delay_ms=5.0,
        convergence_probability=0.99
    )
    
    print(f"\nSensor Contract: {sensor_timing.name}")
    print(f"  {sensor_timing.description}")
    
    print(f"\nControl Contract: {control_timing.name}")
    print(f"  {control_timing.description}")
    
    print(f"\nGossip Contract: {gossip_timing.name}")
    print(f"  {gossip_timing.description}")
    
    total_sensing = 33.33 + 10.0   # sensor period + processing
    total_control = 15.0 + 5.0 + 10.0  # sensing + compute + actuate
    total_gossip = 3 * 5.0 * 2     # propagation * 2 for convergence
    
    print(f"\n--- End-to-end timing budget ---")
    print(f"Sensing path: {total_sensing:.1f}ms")
    print(f"Control loop: {total_control:.1f}ms (within {control_timing.description.split(':')[1]})")
    print(f"DSM convergence: {total_gossip:.1f}ms (eventual consistency)")


def example_3_safety_contracts():
    """Example 3: Safety contracts with invariants"""
    print("\n" + "=" * 60)
    print("Example 3: Safety Contracts (ISO 15066 compliant)")
    print("=" * 60)
    
    collision_contract = SafetyContract.collision_avoidance_contract(
        name="amr_collision_safety",
        min_distance_m=0.5,
        max_velocity_mps=1.0,
        reaction_time_ms=100.0
    )
    
    manipulation_contract = SafetyContract.manipulation_safety_contract(
        name="gripper_safety",
        max_grip_force_n=50.0,
        max_payload_kg=5.0,
        workspace_bounds={'x': (-1.0, 1.0), 'y': (-1.0, 1.0), 'z': (0.0, 1.5)}
    )
    
    print(f"\nCollision Contract: {collision_contract.name}")
    print(f"  SIL Level: {collision_contract.sil.name}")
    print(f"  {collision_contract.description}")
    print(f"  Invariants:")
    for inv in collision_contract.invariants:
        print(f"    - {inv.name}: {inv.expression} (Severity: {inv.severity.name})")
    
    print(f"\nManipulation Contract: {manipulation_contract.name}")
    print(f"  SIL Level: {manipulation_contract.sil.name}")
    print(f"  {manipulation_contract.description}")


def example_4_component_composition():
    """Example 4: Component composition with contract verification"""
    print("\n" + "=" * 60)
    print("Example 4: Component Composition")
    print("=" * 60)
    
    rgb_camera = SensorComponentSpec.rgb_camera(
        name="front_camera",
        resolution=(640, 480),
        fps=30.0,
        fov_degrees=90.0,
        latency_ms=10.0
    )
    
    depth_camera = SensorComponentSpec.depth_camera(
        name="depth_sensor",
        resolution=(640, 480),
        fps=30.0,
        min_range_m=0.3,
        max_range_m=10.0,
        latency_ms=15.0
    )
    
    mobile_base = ActuatorComponentSpec.wheeled_base(
        name="amr_base",
        max_linear_velocity_mps=1.0,
        max_angular_velocity_rps=2.0,
        control_rate_hz=20.0
    )
    
    print(f"\nComponents defined:")
    print(f"  - {rgb_camera.name}: {len(rgb_camera.interface.ports)} ports")
    print(f"  - {depth_camera.name}: {len(depth_camera.interface.ports)} ports")
    print(f"  - {mobile_base.name}: {len(mobile_base.interface.ports)} ports")
    
    arch = SystemArchitecture("warehouse_robot")
    arch.add_component("camera", rgb_camera)
    arch.add_component("depth", depth_camera)
    arch.add_component("base", mobile_base)
    
    verification = arch.verify_architecture()
    print(f"\n--- Architecture Verification ---")
    print(f"Valid: {verification['valid']}")
    print(f"Components: {verification['component_count']}")
    print(f"Connections: {verification['connection_count']}")
    if verification['unconnected_required_ports']:
        print(f"Unconnected required ports:")
        for port in verification['unconnected_required_ports']:
            print(f"  - {port}")
    
    system_contract = arch.get_system_contract()
    if system_contract:
        print(f"\n--- Composed System Contract ---")
        print(f"Name: {system_contract.name}")
        print(f"Total assumptions: {len(system_contract.assumptions.predicates)}")
        print(f"Total guarantees: {len(system_contract.guarantees.predicates)}")


def example_5_hardware_mapping():
    """Example 5: Mapping contracts to hardware requirements"""
    print("\n" + "=" * 60)
    print("Example 5: Contract-to-Hardware Mapping")
    print("=" * 60)
    
    print("""
    Contract-Based Hardware Sizing:
    
    1. TIMING CONTRACTS -> Compute Requirements
       - Perception latency <= 10ms @ 30fps
         -> Need: 30 GFLOPS for CNN inference
         -> Platform: Jetson Orin Nano (20 TOPS NPU)
       
    2. TIMING CONTRACTS -> Communication Requirements  
       - Gossip convergence <= 30ms for 3 hops
         -> Need: WiFi 6 with < 5ms per-hop latency
         -> Protocol: ROS2 DDS with QoS deadline policy
       
    3. SAFETY CONTRACTS -> Sensor Requirements
       - Collision avoidance at 1m/s with 100ms reaction
         -> Need: Depth sensor with 30Hz, < 15ms latency
         -> Range: 0.3m - 10m for stopping distance
       
    4. BEHAVIORAL CONTRACTS -> Software Architecture
       - Detection confidence >= 0.8
         -> Need: Pre-trained model with mAP >= 0.85
         -> Fallback: Conservative speed reduction if degraded
    
    This is how R3 (Hardware-Software Co-Design) connects to simulation!
    """)
    
    print("Contract Refinement Chain:")
    print("  System Contract")
    print("    ├── Perception Contract (A: camera, G: detection)")
    print("    │     └── Camera HW Spec: 640x480 @ 30Hz, < 10ms")
    print("    │     └── Compute Spec: 30 GFLOPS, 4GB RAM")
    print("    ├── Control Contract (A: sensing, G: actuation)")  
    print("    │     └── IMU Spec: 200Hz, < 1ms")
    print("    │     └── Motor Driver: 20Hz control loop")
    print("    └── Communication Contract (A: network, G: convergence)")
    print("          └── WiFi 6: 100Mbps, < 5ms latency")
    print("          └── Protocol: LF (control) + DDS (data)")


def main():
    """Run all examples"""
    example_1_basic_contracts()
    example_2_timing_contracts()
    example_3_safety_contracts()
    example_4_component_composition()
    example_5_hardware_mapping()
    
    print("\n" + "=" * 60)
    print("Summary: Formal Contracts vs Software Interfaces")
    print("=" * 60)
    print("""
    SOFTWARE INTERFACES:
    - Python ABCs with type hints
    - Useful for: Code modularity, IDE support
    - Cannot verify: Timing, safety, behavioral correctness
    
    FORMAL CONTRACTS:
    - Assume-Guarantee specifications
    - Timing contracts with bounds
    - Safety contracts with invariants (SIL levels)
    - Composable with verification
    - Useful for: Hardware sizing, safety certification, 
                  simulation validation, system integration
    
    Formal contracts for specification/verification
    - For R3 (hardware co-design): Formal contracts are essential
    - For safety-critical: Required by ISO 26262, DO-178C, etc.
    """)


if __name__ == "__main__":
    main()
