"""
Warehouse Layout

4-zone layout with waypoints and targets for coordination testing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Tuple, Optional, Set
import math


class ZoneID(Enum):
    NW = "northwest"
    NE = "northeast"
    SW = "southwest"
    SE = "southeast"


@dataclass
class Shelf:
    shelf_id: str
    zone: ZoneID
    position: Tuple[float, float, float]
    size: Tuple[float, float, float] = (2.0, 2.0, 0.5)
    capacity: int = 10
    current_items: int = 0


@dataclass
class Waypoint:
    waypoint_id: str
    position: Tuple[float, float, float]
    connects: List[str] = field(default_factory=list)
    max_robots: int = 2
    current_robots: Set[str] = field(default_factory=set)

    @property
    def is_blocked(self) -> bool:
        return len(self.current_robots) >= self.max_robots

    @property
    def congestion_level(self) -> float:
        return len(self.current_robots) / max(self.max_robots, 1)


@dataclass
class TargetLocation:
    target_id: str
    zone: ZoneID
    position: Tuple[float, float, float]
    location_type: str = "pickup"
    current_robots: Set[str] = field(default_factory=set)

    @property
    def is_congested(self) -> bool:
        return len(self.current_robots) >= 2


@dataclass
class Zone:
    zone_id: ZoneID
    bounds: Tuple[float, float, float, float]  # (min_x, min_z, max_x, max_z)
    shelves: List[Shelf] = field(default_factory=list)
    targets: List[TargetLocation] = field(default_factory=list)
    robot_ids: Set[str] = field(default_factory=set)

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bounds[0] + self.bounds[2]) / 2,
            (self.bounds[1] + self.bounds[3]) / 2
        )


class WarehouseLayout:
    """
    4-region warehouse with shelves, waypoints, and task locations.
    
    Layout:
        +------------------+------------------+
        |       NW         |       NE         |
        |   [S1] [S2]      |   [S5] [S6]      |
        |      T1    W1----+----W2    T2      |
        +--------+---------+---------+--------+
        |        |         |         |        |
        |   SW   |    Central Aisle  |   SE   |
        |   [S3] [S4]      |   [S7] [S8]      |
        |      T3    W3----+----W4    T4      |
        +------------------+------------------+
                           T5 (dock)
    """

    def __init__(self, width: float = 20.0, depth: float = 20.0):
        self.width = width
        self.depth = depth
        self.half_w = width / 2
        self.half_d = depth / 2

        self.zones: Dict[ZoneID, Zone] = {}
        self.shelves: Dict[str, Shelf] = {}
        self.waypoints: Dict[str, Waypoint] = {}
        self.targets: Dict[str, TargetLocation] = {}
        self.robot_positions: Dict[str, Tuple[float, float, float]] = {}

        self._build_layout()

    def _build_layout(self):
        self._create_zones()
        self._create_shelves()
        self._create_waypoints()
        self._create_targets()

    def _create_zones(self):
        hw, hd = self.half_w, self.half_d
        self.zones = {
            ZoneID.NW: Zone(ZoneID.NW, (-hw, 0, 0, hd)),
            ZoneID.NE: Zone(ZoneID.NE, (0, 0, hw, hd)),
            ZoneID.SW: Zone(ZoneID.SW, (-hw, -hd, 0, 0)),
            ZoneID.SE: Zone(ZoneID.SE, (0, -hd, hw, 0)),
        }

    def _create_shelves(self):
        shelf_configs = [
            ("S1", ZoneID.NW, (-7.0, 0.0, 5.0)),
            ("S2", ZoneID.NW, (-3.0, 0.0, 5.0)),
            ("S3", ZoneID.SW, (-7.0, 0.0, -5.0)),
            ("S4", ZoneID.SW, (-3.0, 0.0, -5.0)),
            ("S5", ZoneID.NE, (3.0, 0.0, 5.0)),
            ("S6", ZoneID.NE, (7.0, 0.0, 5.0)),
            ("S7", ZoneID.SE, (3.0, 0.0, -5.0)),
            ("S8", ZoneID.SE, (7.0, 0.0, -5.0)),
        ]
        for sid, zone, pos in shelf_configs:
            shelf = Shelf(shelf_id=sid, zone=zone, position=pos)
            self.shelves[sid] = shelf
            self.zones[zone].shelves.append(shelf)

    def _create_waypoints(self):
        waypoint_configs = [
            ("W1", (-2.0, 0.0, 2.0), ["W2", "W3"]),
            ("W2", (2.0, 0.0, 2.0), ["W1", "W4"]),
            ("W3", (-2.0, 0.0, -2.0), ["W1", "W4", "DOCK"]),
            ("W4", (2.0, 0.0, -2.0), ["W2", "W3", "DOCK"]),
            ("DOCK", (0.0, 0.0, -8.0), ["W3", "W4"]),
        ]
        for wid, pos, connects in waypoint_configs:
            self.waypoints[wid] = Waypoint(
                waypoint_id=wid, position=pos, connects=connects
            )

    def _create_targets(self):
        target_configs = [
            ("T1", ZoneID.NW, (-5.0, 0.0, 3.0), "pickup"),
            ("T2", ZoneID.NE, (5.0, 0.0, 3.0), "pickup"),
            ("T3", ZoneID.SW, (-5.0, 0.0, -3.0), "pickup"),
            ("T4", ZoneID.SE, (5.0, 0.0, -3.0), "pickup"),
            ("T5", ZoneID.SE, (0.0, 0.0, -9.0), "dropoff"),
        ]
        for tid, zone, pos, loc_type in target_configs:
            target = TargetLocation(
                target_id=tid, zone=zone, position=pos, location_type=loc_type
            )
            self.targets[tid] = target
            self.zones[zone].targets.append(target)

    def get_zone_for_position(self, x: float, z: float) -> Optional[ZoneID]:
        for zone_id, zone in self.zones.items():
            b = zone.bounds
            if b[0] <= x <= b[2] and b[1] <= z <= b[3]:
                return zone_id
        return None

    def update_robot_position(self, robot_id: str, position: Tuple[float, float, float]):
        old_pos = self.robot_positions.get(robot_id)
        if old_pos:
            old_zone = self.get_zone_for_position(old_pos[0], old_pos[2])
            if old_zone:
                self.zones[old_zone].robot_ids.discard(robot_id)

        self.robot_positions[robot_id] = position
        new_zone = self.get_zone_for_position(position[0], position[2])
        if new_zone:
            self.zones[new_zone].robot_ids.add(robot_id)

        for wp in self.waypoints.values():
            dist = self._distance_2d(position, wp.position)
            if dist < 1.5:
                wp.current_robots.add(robot_id)
            else:
                wp.current_robots.discard(robot_id)

        for target in self.targets.values():
            dist = self._distance_2d(position, target.position)
            if dist < 2.0:
                target.current_robots.add(robot_id)
            else:
                target.current_robots.discard(robot_id)

    def get_waypoint_status(self, waypoint_id: str) -> Dict:
        wp = self.waypoints.get(waypoint_id)
        if not wp:
            return {"exists": False}
        return {
            "exists": True,
            "blocked": wp.is_blocked,
            "congestion": wp.congestion_level,
            "robots": list(wp.current_robots),
        }

    def get_clear_path(self, from_pos: Tuple[float, float, float],
                       to_target: str) -> Optional[List[str]]:
        target = self.targets.get(to_target)
        if not target:
            return None

        from_zone = self.get_zone_for_position(from_pos[0], from_pos[2])
        to_zone = target.zone

        if from_zone == to_zone:
            return [to_target]

        path = []
        if from_zone in (ZoneID.NW, ZoneID.NE) and to_zone in (ZoneID.SW, ZoneID.SE):
            w1 = "W1" if from_zone == ZoneID.NW else "W2"
            w2 = "W3" if to_zone == ZoneID.SW else "W4"
            if not self.waypoints[w1].is_blocked:
                path.append(w1)
            if not self.waypoints[w2].is_blocked:
                path.append(w2)
            path.append(to_target)
        elif from_zone in (ZoneID.SW, ZoneID.SE) and to_zone in (ZoneID.NW, ZoneID.NE):
            w1 = "W3" if from_zone == ZoneID.SW else "W4"
            w2 = "W1" if to_zone == ZoneID.NW else "W2"
            if not self.waypoints[w1].is_blocked:
                path.append(w1)
            if not self.waypoints[w2].is_blocked:
                path.append(w2)
            path.append(to_target)
        else:
            path.append(to_target)

        return path if path else None

    def get_context_for_robot(self, robot_id: str) -> Dict:
        pos = self.robot_positions.get(robot_id)
        if not pos:
            return {}

        zone = self.get_zone_for_position(pos[0], pos[2])
        nearby_waypoints = {}
        for wid, wp in self.waypoints.items():
            if self._distance_2d(pos, wp.position) < 5.0:
                nearby_waypoints[wid] = {
                    "blocked": wp.is_blocked,
                    "congestion": wp.congestion_level
                }

        nearby_targets = {}
        for tid, t in self.targets.items():
            if self._distance_2d(pos, t.position) < 8.0:
                nearby_targets[tid] = {
                    "congested": t.is_congested,
                    "type": t.location_type
                }

        return {
            "zone": zone.value if zone else None,
            "position": pos,
            "nearby_waypoints": nearby_waypoints,
            "nearby_targets": nearby_targets,
        }

    def get_spawn_positions(self, n_robots: int) -> List[Tuple[float, float, float]]:
        positions = []
        zone_list = list(self.zones.values())
        for i in range(n_robots):
            zone = zone_list[i % len(zone_list)]
            cx, cz = zone.center
            offset_x = (i // 4) * 1.5
            offset_z = ((i // 4) % 2) * 1.5
            positions.append((cx + offset_x, 0.0, cz + offset_z))
        return positions

    def get_objects_to_spawn(self) -> List[Dict]:
        """Return objects to spawn. Empty when using pre-built scenes."""
        return []

    @staticmethod
    def _distance_2d(p1: Tuple[float, float, float],
                     p2: Tuple[float, float, float]) -> float:
        return math.sqrt((p1[0] - p2[0])**2 + (p1[2] - p2[2])**2)
