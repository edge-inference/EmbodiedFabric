"""Asset lookup helpers for IsaacSimBackend."""

from __future__ import annotations

import os
from typing import Optional


class IsaacAssetsMixin:
    def _resolve_assets_root(self) -> Optional[str]:
        assets_path = os.getenv("ISAAC_ASSETS_PATH") or os.getenv("OMNI_ISAAC_ASSETS_PATH")
        if assets_path:
            candidates = [
                os.path.join(assets_path, "Assets", "Isaac", "5.1"),
                os.path.join(assets_path, "Assets"),
                assets_path,
            ]
            for candidate in candidates:
                if os.path.isdir(os.path.join(candidate, "Isaac", "Robots")):
                    return candidate
                if os.path.isdir(os.path.join(candidate, "Isaac")):
                    return candidate

        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
        except Exception:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path
            except Exception:
                return None
        return get_assets_root_path()

    def _get_robot_usd_path(self, assets_root: str, robot_type: str) -> Optional[str]:
        robot_key = (robot_type or "franka").lower()
        robot_map = {
            "franka": "Isaac/Robots/FrankaRobotics/FrankaEmika/panda_instanceable.usd",
            "factory_franka": "Isaac/Robots/FrankaRobotics/FactoryFranka/factory_franka.usd",
            "carter": "Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
            "nova_carter": "Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd",
            "ur10": "Isaac/Robots/UniversalRobots/ur10/ur10.usd",
            "turtlebot": "Isaac/Robots/Turtlebot/turtlebot3_burger.usd",
            "g1": "Isaac/Robots/Unitree/G1_23dof/g1.usd",
            "g1_full": "Isaac/Robots/Unitree/G1/g1.usd",
            "h1": "Isaac/Robots/Unitree/H1/h1.usd",
            "digit": "Isaac/Robots/Agility/Digit/digit_v4.usd",
            "gr1": "Isaac/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd",
            "neo": "Isaac/Robots/1X/Neo/Neo.usd",
            "spot": "Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd",
        }
        rel_path = robot_map.get(robot_key)
        if not rel_path:
            return None

        full = os.path.join(assets_root, rel_path)
        if os.path.isfile(full):
            return full

        robot_dir = os.path.join(assets_root, os.path.dirname(rel_path))
        if os.path.isdir(robot_dir):
            for f in os.listdir(robot_dir):
                if f.endswith(".usd") and not f.startswith("."):
                    return os.path.join(robot_dir, f)

        return full

