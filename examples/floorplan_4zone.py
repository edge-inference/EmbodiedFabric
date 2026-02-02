#!/usr/bin/env python3
"""
4-Zone Floorplan Scenario

Usage:
    python examples/floorplan_4zone.py --robots 6 --steps 500 --scene floorplan_1a
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

parser = argparse.ArgumentParser(description="4-Zone Floorplan Simulation")
parser.add_argument("--robots", type=int, default=6, help="Number of robots")
parser.add_argument("--steps", type=int, default=500, help="Simulation steps")
parser.add_argument("--vla", type=str, default="hierarchical",
                    choices=["hierarchical", "cogact", "nomad", "cogact_server"],
                    help="VLA model (default: hierarchical)")
parser.add_argument("--verbose", "-v", action="store_true")
parser.add_argument("--no-record", action="store_true")
parser.add_argument("--scene", type=str, default="floorplan_5b",
                    help="TDW scene name (e.g., 'tdw_room', 'floorplan_1a')")
parser.add_argument("--layout", type=int, default=1,
                    help="Floorplan layout index (0, 1, 2)")
args = parser.parse_args()

import torch
if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1).cuda()

import logging
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from simulator.core import Simulator, SimulatorConfig
from simulator.scenarios.warehouse import WarehouseLayout, ZoneID


class Floorplan4ZoneSimulator(Simulator):
    """Extended simulator with 4-zone layout."""

    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        self.layout = WarehouseLayout(
            width=config.scene_size[0],
            depth=config.scene_size[1]
        )

    def _get_floorplan_spawn_positions(self, scene_name: str, n_robots: int):
        try:
            import tdw.add_ons.floorplan as floorplan
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
                idx = min(
                    range(k),
                    key=lambda i: (x - centroids[i][0]) ** 2 + (z - centroids[i][1]) ** 2
                )
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

    def initialize(self) -> bool:
        if not super().initialize():
            return False
        self._spawn_warehouse_objects()
        self._position_robots()
        return True

    def _spawn_warehouse_objects(self):
        """Spawn objects if not using pre-built scene."""
        objects = self.layout.get_objects_to_spawn()
        if not objects:
            print("\nUsing pre-built scene")
            return
        print(f"\nSpawning {len(objects)} objects...")
        for obj in objects:
            try:
                self._backend.spawn_object(
                    obj["object_id"], obj["object_type"], obj["position"]
                )
            except Exception as e:
                print(f"  Warning: {obj['object_id']}: {e}")

    def _position_robots(self):
        scene_name = getattr(self.config, "scene_name", "") or ""
        if scene_name.startswith("floorplan"):
            spawn_positions = self._get_floorplan_spawn_positions(scene_name, len(self._robots))
            if not spawn_positions:
                spawn_positions = [
                    (-3.5, 0.0, 3.5),
                    (3.0, 0.0, 3.5),
                    (-3.5, 0.0, -2.5),
                    (3.0, 0.0, -2.5),
                ]
        else:
            spawn_positions = self.layout.get_spawn_positions(len(self._robots))
        for i, (robot_id, robot) in enumerate(self._robots.items()):
            if i < len(spawn_positions):
                pos = spawn_positions[i]
                self.layout.update_robot_position(robot_id, pos)
                print(f"  {robot_id} assigned to zone {self.layout.get_zone_for_position(pos[0], pos[2])}")

    def step(self):
        state = super().step()
        for robot_id, robot in self._robots.items():
            pos = robot._position
            self.layout.update_robot_position(robot_id, pos)

        # DSM disabled for now

        return state

    def _share_layout_context(self):
        return

    def get_layout_status(self) -> dict:
        return {
            "zones": {
                z.value: {
                    "robots": list(self.layout.zones[z].robot_ids),
                    "shelves": len(self.layout.zones[z].shelves),
                }
                for z in ZoneID
            },
            "waypoints": {
                wid: self.layout.get_waypoint_status(wid)
                for wid in self.layout.waypoints
            },
            "targets": {
                tid: {
                    "congested": t.is_congested,
                    "robots": list(t.current_robots)
                }
                for tid, t in self.layout.targets.items()
            }
        }


def assign_tasks(sim: Floorplan4ZoneSimulator):
    """Assign pickup/dropoff tasks to robots based on their zones."""
    tasks = [
        ("robot_0", "T1", "T5", "pickup at T1, drop off at T5"),
        ("robot_1", "T2", "T5", "pickup at T2, drop off at T5"),
        ("robot_2", "T3", "T5", "pickup at T3, drop off at T5"),
        ("robot_3", "T4", "T5", "pickup at T4, drop off at T5"),
        ("robot_4", "T1", "T5", "go to T1, pick box, return to dock"),
        ("robot_5", "T2", "T5", "go to T2, pick box, return to dock"),
    ]
    for robot_id, pickup, dropoff, instruction in tasks:
        if robot_id in sim.robots:
            robot = sim.robots[robot_id]
            robot._instruction = instruction
            print(f"  {robot_id}: {instruction}")


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

    config = SimulatorConfig(
        n_robots=args.robots,
        scene_name=scene_name,
        floorplan_layout=layout_index,
        scene_size=(20, 20),
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
        verbose=args.verbose
    )

    print(f"\nConfiguration:")
    print(f"  Robots: {config.n_robots}")
    print(f"  VLA: {config.vla_model}")
    print(f"  Scene: {config.scene_size[0]}m x {config.scene_size[1]}m")

    sim = Floorplan4ZoneSimulator(config)

    print("\nInitializing...")
    if not sim.initialize():
        print("ERROR: Failed to initialize")
        return

    print("\nAssigning tasks:")
    assign_tasks(sim)

    def progress_callback(state):
        if state.step_count % 100 == 0:
            rtf = state.sim_time / max(state.real_time, 0.001)
            print(f"\n[Step {state.step_count}] sim={state.sim_time:.2f}s, RTF={rtf:.2f}x")

            status = sim.get_layout_status()
            for wid, ws in status["waypoints"].items():
                if ws.get("blocked"):
                    print(f"  BLOCKED: {wid} (robots: {ws['robots']})")
                elif ws.get("congestion", 0) > 0.5:
                    print(f"  CONGESTED: {wid} ({ws['congestion']:.0%})")

            for tid, ts in status["targets"].items():
                if ts.get("congested"):
                    print(f"  TARGET {tid} congested: {ts['robots']}")

    print("\nRunning simulation...")
    sim.run(callback=progress_callback)

    print("\n" + "=" * 60)
    print("FINAL STATUS")
    print("=" * 60)

    status = sim.get_layout_status()
    print("\nZone occupancy:")
    for zone_name, zone_data in status["zones"].items():
        robots = zone_data["robots"]
        print(f"  {zone_name}: {len(robots)} robots {robots}")

    print("\nWaypoint status:")
    for wid, ws in status["waypoints"].items():
        state = "BLOCKED" if ws.get("blocked") else f"{ws.get('congestion', 0):.0%}"
        print(f"  {wid}: {state}")

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
