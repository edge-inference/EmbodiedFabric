#!/usr/bin/env python3
"""Run multi-robot TDW floorplans with the Simulator."""

import sys
import os
import argparse
import json
import re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="4-Zone Floorplan Simulation")
parser.add_argument("--robots", type=int, default=6, help="Number of robots")
parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
parser.add_argument("--vla", type=str, default="hierarchical",
                    choices=["hierarchical", "cogact", "nomad", "cogact_server",
                             "smolvla", "pi0", "groot", "lerobot_server"],
                    help="VLA model")
parser.add_argument("--control-mode", type=str, default="auto",
                    choices=["auto", "high_level", "low_level"],
                    help="Control mode for VLA (auto selects based on model)")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--no-record", action="store_true")
parser.add_argument("--scene", type=str, default="floorplan_5b",
                    help="TDW scene name (e.g., 'tdw_room', 'floorplan_1a')")
parser.add_argument("--layout", type=int, default=1,
                    help="Floorplan layout index (0, 1, 2)")
parser.add_argument("--instruction", type=str, default=None,
                    help="Custom instruction for all robots (e.g., 'move forward')")
parser.add_argument("--tdw-address", type=str, default=None,
                    help="Connect to existing TDW build at this address (skip launch)")
parser.add_argument("--tdw-port", type=int, default=1071,
                    help="TDW port (default 1071)")
args = parser.parse_args()

import torch
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1).cuda()

import logging

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
log_dir = os.path.join(project_root, "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "simulation.log")

logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='w')
    ]
)
logging.info(f"Logging to {log_file}")

from simulator.core import Simulator, SimulatorConfig


def _get_floorplan_spawn_positions(scene_name: str, n_robots: int):
    try:
        import tdw.add_ons.floorplan as floorplan  # type: ignore[import-not-found]
    except Exception:
        return []
    if scene_name.startswith("floorplan_"):
        scene_index = scene_name[-2]
    else:
        scene_index = scene_name[0] if scene_name else ""
    if scene_index not in ("1", "2", "4", "5"):
        return []
    flood_path = Path(floorplan.__file__).with_name("floorplan_floods.json")
    floods = json.loads(flood_path.read_text(encoding="utf-8"))
    floors = floods.get(scene_index, {})
    tiles = [
        (float(v["position"]["x"]), float(v["position"]["z"]))
        for _, v in sorted(floors.items(), key=lambda kv: int(kv[0]))
    ]
    if not tiles:
        return []
    if len(tiles) <= n_robots:
        return [(x, 0.0, z) for x, z in tiles[:n_robots]]

    k = min(n_robots, len(tiles))
    centroids = []
    avg_x = sum(t[0] for t in tiles) / len(tiles)
    avg_z = sum(t[1] for t in tiles) / len(tiles)
    centroids.append((avg_x, avg_z))
    while len(centroids) < k:
        best = None
        best_dist = -1.0
        for x, z in tiles:
            dist = min((x - cx) ** 2 + (z - cz) ** 2 for cx, cz in centroids)
            if dist > best_dist:
                best_dist = dist
                best = (x, z)
        centroids.append(best)

    for _ in range(10):
        clusters = [[] for _ in range(k)]
        for x, z in tiles:
            idx = min(range(k), key=lambda i: (x - centroids[i][0]) ** 2 + (z - centroids[i][1]) ** 2)
            clusters[idx].append((x, z))
        new_centroids = []
        for i, cluster in enumerate(clusters):
            if not cluster:
                new_centroids.append(centroids[i])
                continue
            cx = sum(p[0] for p in cluster) / len(cluster)
            cz = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append((cx, cz))
        centroids = new_centroids

    spawn_positions = []
    for cx, cz in centroids:
        x, z = min(tiles, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cz) ** 2)
        spawn_positions.append((x, 0.0, z))
    return spawn_positions


def assign_tasks(sim: Simulator, custom_instruction: str = None):
    house_instructions = [
        "pick up the box",
        "grab the box and move it forward",
        "navigate to the box and grasp it",
        "move toward the sofa",
        "explore the room",
        "navigate around the furniture",
    ]
    
    for i, robot_id in enumerate(sorted(sim.robots.keys())):
        robot = sim.robots[robot_id]
        if custom_instruction:
            robot._current_instruction = custom_instruction
        else:
            robot._current_instruction = house_instructions[i % len(house_instructions)]
        print(f"  {robot_id}: {robot._current_instruction}")


def run_simulation():
    print("=" * 60)
    print("4-Zone Floorplan Simulation")
    print("=" * 60)

    scene_name = args.scene
    layout_index = args.layout
    if scene_name:
        match = re.match(r"^(?:floorplan_)?([1245][abc])_(\d)$", scene_name)
        if match:
            scene_name = f"floorplan_{match.group(1)}"
            layout_index = int(match.group(2))

    # Determine if we should launch TDW or connect to existing
    launch_build = args.tdw_address is None
    tdw_address = args.tdw_address or "localhost"
    
    backend = os.environ.get("BACKEND", "tdw")
    if backend != "tdw":
        raise SystemExit("floorplan_4zone.py is TDW floorplan-only. Use examples/isaac_humanoid_test.py for Isaac.")

    spawn_positions = _get_floorplan_spawn_positions(scene_name, args.robots)
    
    config = SimulatorConfig(
        n_robots=args.robots,
        scene_name=scene_name,
        floorplan_layout=layout_index,
        scene_size=(20, 20),
        spawn_positions=spawn_positions or None,
        vla_model=args.vla,
        vla_latency_budget_ms=150.0,
        vla_quantization="none",
        enable_dsm=False,
        gossip_period_ms=50.0,
        enable_lf_coordination=True,
        max_steps=args.steps,
        profiling_enabled=True,
        enable_recording=not args.no_record,
        recording_path="recordings/floorplan_4zone",
        recording_resolution=(1280, 720),
        inference_interval=5,
        verbose=args.verbose,
        backend=backend,
        tdw_launch_build=launch_build,
        tdw_address=tdw_address,
        tdw_port=args.tdw_port
    )

    print(f"\nConfiguration:")
    print(f"  Robots: {config.n_robots}")
    print(f"  VLA: {config.vla_model}")
    print(f"  Scene: {config.scene_size[0]}m x {config.scene_size[1]}m")
    if launch_build:
        print(f"  TDW: launching new build")
    else:
        print(f"  TDW: connecting to {tdw_address}:{args.tdw_port}")

    sim = Simulator(config)

    print("\nInitializing...")
    if not sim.initialize():
        print("ERROR: Failed to initialize")
        return

    print("\nAssigning tasks:")
    assign_tasks(sim, args.instruction)

    def progress_callback(state):
        if state.step_count % 100 == 0:
            rtf = state.sim_time / max(state.real_time, 0.001)
            print(f"\n[Step {state.step_count}] sim={state.sim_time:.2f}s, RTF={rtf:.2f}x")

    print("\nRunning simulation...")
    sim.run(callback=progress_callback)

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    coord_metrics = sim.coordinator.get_metrics()
    print(f"\nCoordination:")
    print(f"  Tasks completed: {coord_metrics['tasks_completed']}")
    print(f"  Throughput: {coord_metrics['throughput_per_hour']:.1f}/hr")

    workload = sim.get_metrics()
    if workload:
        print(f"\nVLA Performance:")
        print(f"  Inferences: {workload.vla_inferences}")
        print(f"  Avg latency: {workload.vla_avg_latency_ms:.1f}ms")
        print(f"  Violations: {workload.vla_violations}")

    sim.close()
    print("\nSimulation complete.")


if __name__ == "__main__":
    run_simulation()
