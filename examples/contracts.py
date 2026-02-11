#!/usr/bin/env python3
"""Formal contract-based system design."""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Any, Dict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.codesign.contract import Contract, ContractBuilder
from contracts.codesign.timing import TimingContract, CommunicationTimingContract
from contracts.codesign.safety import SafetyContract, SafetyIntegrityLevel
from contracts.codesign.component import (
    ComponentSpec, Interface, Port, PortDirection, DataType,
    SensorComponentSpec, ActuatorComponentSpec
)
from contracts.codesign.composition import (
    compose_parallel, compose_series, compose_components,
    SystemArchitecture
)


def _load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if p.is_absolute():
        return json.loads(p.read_text(encoding="utf-8"))

    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / p,
        Path(__file__).resolve().parent / p,
    ]
    resolved = next((c for c in candidates if c.exists()), candidates[0])
    return json.loads(resolved.read_text(encoding="utf-8"))


def example_1_basic_contracts(cfg: Dict[str, Any]):
    print("=" * 60)
    print("Example 1: Basic Assume-Guarantee Contracts")
    print("=" * 60)

    ex1 = cfg.get("example_1", {})
    ambient_min = float(ex1.get("ambient_light_lux_min", 50))
    fps_min = float(ex1.get("camera_fps_min", 28))
    conf_min = float(ex1.get("detection_confidence_min", 0.8))
    latency_max = float(ex1.get("perception_latency_ms_max", 50))

    perception_contract = ContractBuilder("perception_pipeline") \
        .assume(
            "camera_operational",
            "camera.status == OPERATIONAL",
            evaluator=lambda s: s.get('camera_status') == 'OPERATIONAL'
        ) \
        .assume(
            "lighting_adequate",
            f"ambient_light >= {ambient_min} lux",
            evaluator=lambda s: s.get('ambient_light_lux', 0) >= ambient_min
        ) \
        .assume(
            "frame_rate",
            f"camera.fps >= {fps_min}",
            evaluator=lambda s: s.get('camera_fps', 0) >= fps_min
        ) \
        .guarantee(
            "object_detection",
            f"detection_confidence >= {conf_min} for objects within 5m",
            evaluator=lambda s: s.get('detection_confidence', 0) >= conf_min
        ) \
        .guarantee(
            "detection_latency",
            f"perception_latency <= {latency_max}ms",
            evaluator=lambda s: s.get('perception_latency_ms', 999) <= latency_max
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

    trace = ex1.get("trace") or []
    
    result = perception_contract.check_satisfaction(trace)
    print(f"Contract satisfied: {result.satisfied}")
    print(f"Violations: {len(result.violations)}")
    print(f"Vacuous steps (assumption not met): {result.vacuous_steps}")
    
    for v in result.violations:
        print(f"  - {v.message}")


def example_2_timing_contracts(cfg: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("Example 2: Timing Contracts")
    print("=" * 60)

    ex2 = cfg.get("example_2", {})
    sensor_cfg = ex2.get("sensor", {})
    control_cfg = ex2.get("control_loop", {})
    gossip_cfg = ex2.get("gossip", {})

    sensor_timing = TimingContract.sensor_contract(
        name=str(sensor_cfg.get("name", "rgb_camera_timing")),
        sensor_period_ms=float(sensor_cfg.get("sensor_period_ms", 33.33)),
        sensor_jitter_ms=float(sensor_cfg.get("sensor_jitter_ms", 1.0)),
        processing_latency_ms=float(sensor_cfg.get("processing_latency_ms", 10.0)),
        output_period_ms=float(sensor_cfg.get("output_period_ms", 33.33)),
    )
    
    control_timing = TimingContract.control_loop_contract(
        name=str(control_cfg.get("name", "motion_control_timing")),
        sensing_latency_ms=float(control_cfg.get("sensing_latency_ms", 15.0)),
        computation_latency_ms=float(control_cfg.get("computation_latency_ms", 5.0)),
        actuation_latency_ms=float(control_cfg.get("actuation_latency_ms", 10.0)),
        loop_period_ms=float(control_cfg.get("loop_period_ms", 50.0)),
    )
    
    gossip_timing = CommunicationTimingContract.gossip_contract(
        name=str(gossip_cfg.get("name", "dsm_gossip_timing")),
        gossip_period_ms=float(gossip_cfg.get("gossip_period_ms", 50.0)),
        propagation_hops=int(gossip_cfg.get("propagation_hops", 3)),
        per_hop_delay_ms=float(gossip_cfg.get("per_hop_delay_ms", 5.0)),
        convergence_probability=float(gossip_cfg.get("convergence_probability", 0.99)),
    )
    
    print(f"\nSensor Contract: {sensor_timing.name}")
    print(f"  {sensor_timing.description}")
    
    print(f"\nControl Contract: {control_timing.name}")
    print(f"  {control_timing.description}")
    
    print(f"\nGossip Contract: {gossip_timing.name}")
    print(f"  {gossip_timing.description}")
    
    total_sensing = sensor_timing.input_timing[0].period + sensor_timing.output_timing[0].max_value
    total_control = (
        float(control_cfg.get("sensing_latency_ms", 15.0))
        + float(control_cfg.get("computation_latency_ms", 5.0))
        + float(control_cfg.get("actuation_latency_ms", 10.0))
    )
    total_gossip = int(gossip_cfg.get("propagation_hops", 3)) * float(gossip_cfg.get("per_hop_delay_ms", 5.0)) * 2
    
    print(f"\n--- End-to-end timing budget ---")
    print(f"Sensing path: {total_sensing:.1f}ms")
    print(f"Control loop: {total_control:.1f}ms (within {control_timing.description.split(':')[1]})")
    print(f"DSM convergence: {total_gossip:.1f}ms (eventual consistency)")


def example_3_safety_contracts(cfg: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("Example 3: Safety Contracts (ISO 15066 compliant)")
    print("=" * 60)

    ex3 = cfg.get("example_3", {})
    collision_cfg = ex3.get("collision", {})
    manip_cfg = ex3.get("manipulation", {})

    collision_contract = SafetyContract.collision_avoidance_contract(
        name=str(collision_cfg.get("name", "amr_collision_safety")),
        min_distance_m=float(collision_cfg.get("min_distance_m", 0.5)),
        max_velocity_mps=float(collision_cfg.get("max_velocity_mps", 1.0)),
        reaction_time_ms=float(collision_cfg.get("reaction_time_ms", 100.0)),
    )
    
    manipulation_contract = SafetyContract.manipulation_safety_contract(
        name=str(manip_cfg.get("name", "gripper_safety")),
        max_grip_force_n=float(manip_cfg.get("max_grip_force_n", 50.0)),
        max_payload_kg=float(manip_cfg.get("max_payload_kg", 5.0)),
        workspace_bounds=manip_cfg.get("workspace_bounds", {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.5)}),
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


def example_4_component_composition(cfg: Dict[str, Any]):
    print("\n" + "=" * 60)
    print("Example 4: Component Composition")
    print("=" * 60)

    ex4 = cfg.get("example_4", {})
    rgb_cfg = ex4.get("rgb_camera", {})
    depth_cfg = ex4.get("depth_camera", {})
    base_cfg = ex4.get("mobile_base", {})

    rgb_camera = SensorComponentSpec.rgb_camera(
        name=str(rgb_cfg.get("name", "front_camera")),
        resolution=tuple(rgb_cfg.get("resolution", (640, 480))),
        fps=float(rgb_cfg.get("fps", 30.0)),
        fov_degrees=float(rgb_cfg.get("fov_degrees", 90.0)),
        latency_ms=float(rgb_cfg.get("latency_ms", 10.0)),
    )
    
    depth_camera = SensorComponentSpec.depth_camera(
        name=str(depth_cfg.get("name", "depth_sensor")),
        resolution=tuple(depth_cfg.get("resolution", (640, 480))),
        fps=float(depth_cfg.get("fps", 30.0)),
        min_range_m=float(depth_cfg.get("min_range_m", 0.3)),
        max_range_m=float(depth_cfg.get("max_range_m", 10.0)),
        latency_ms=float(depth_cfg.get("latency_ms", 15.0)),
    )
    
    mobile_base = ActuatorComponentSpec.wheeled_base(
        name=str(base_cfg.get("name", "amr_base")),
        max_linear_velocity_mps=float(base_cfg.get("max_linear_velocity_mps", 1.0)),
        max_angular_velocity_rps=float(base_cfg.get("max_angular_velocity_rps", 2.0)),
        control_rate_hz=float(base_cfg.get("control_rate_hz", 20.0)),
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
    print("\n" + "=" * 60)
    print("Example 5: Contract-to-Hardware Mapping")
    print("=" * 60)
    
    print("Contract-Based Hardware Sizing:")
    print("  Timing -> compute/latency budgets")
    print("  Communication -> network latency budgets")
    print("  Safety -> sensor range/latency requirements")
    print("  Behavior -> model accuracy and fallback policy")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/contracts.json",
        help="Path relative to repo root or absolute",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    example_1_basic_contracts(cfg)
    example_2_timing_contracts(cfg)
    example_3_safety_contracts(cfg)
    example_4_component_composition(cfg)
    example_5_hardware_mapping()
    
    print("\n" + "=" * 60)
    print("Summary: Formal Contracts vs Software Interfaces")
    print("=" * 60)
    print("SOFTWARE INTERFACES: type hints, API boundaries")
    print("FORMAL CONTRACTS: assumptions/guarantees, timing, safety")


if __name__ == "__main__":
    main()
