"""Robot spawn/observe/command for IsaacSimBackend."""

from __future__ import annotations

import logging
import os
from typing import Tuple

import numpy as np

from ..base import ControlMode, RobotCommand, RobotObservation
from .types import IsaacRobotState

logger = logging.getLogger(__name__)


class IsaacRobotMixin:
    def spawn_robot(self, robot_id: str, position: Tuple[float, float, float], robot_type: str = "fetch") -> bool:
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.robots import Robot

            prim_path = f"/World/Robots/{robot_id}"
            assets_root = self._resolve_assets_root()

            if assets_root is None:
                return self._spawn_proxy_robot(robot_id, position)

            robot_usd = self._get_robot_usd_path(assets_root, robot_type)
            if not robot_usd or not os.path.isfile(robot_usd):
                return self._spawn_proxy_robot(robot_id, position)

            add_reference_to_stage(usd_path=robot_usd, prim_path=prim_path)
            robot = Robot(
                prim_path=prim_path,
                name=robot_id,
                position=np.array(position),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
            self._world.scene.add(robot)
            self._robots[robot_id] = IsaacRobotState(prim_path=prim_path, robot_type=robot_type, articulation=robot)
            logger.info("Spawned %s robot '%s' at %s", robot_type, robot_id, position)
            return True
        except Exception as e:
            logger.warning("Robot spawn failed (%s): %s; using proxy", robot_id, e)
            return self._spawn_proxy_robot(robot_id, position)

    def _spawn_proxy_robot(self, robot_id: str, position: Tuple[float, float, float]) -> bool:
        try:
            from omni.isaac.core.objects import DynamicCuboid
            from omni.isaac.core.utils.prims import delete_prim

            prim_path = f"/World/Robots/{robot_id}"
            delete_prim(prim_path)
            robot = DynamicCuboid(
                prim_path=prim_path,
                name=robot_id,
                position=np.array(position),
                size=0.5,
                color=np.array([0.2, 0.6, 1.0]),
                mass=10.0,
            )
            self._world.scene.add(robot)
            self._robots[robot_id] = IsaacRobotState(prim_path=prim_path, robot_type="proxy", articulation=robot)
            logger.info("Spawned proxy robot '%s' at %s", robot_id, position)
            return True
        except Exception as e:
            logger.error("Failed to spawn proxy robot '%s': %s", robot_id, e)
            return False

    def get_observation(self, robot_id: str) -> RobotObservation:
        state = self._robots.get(robot_id)
        if state is None:
            raise ValueError(f"Robot '{robot_id}' not found")

        robot = state.articulation

        try:
            pose = robot.get_world_pose()
            position = pose[0] if pose[0] is not None else np.zeros(3)
            orientation = pose[1] if pose[1] is not None else np.array([1, 0, 0, 0])
        except Exception:
            position = np.zeros(3)
            orientation = np.array([1, 0, 0, 0])

        try:
            velocity = robot.get_linear_velocity()
            if velocity is None:
                velocity = np.zeros(3)
        except Exception:
            velocity = np.zeros(3)

        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        depth = np.zeros((256, 256), dtype=np.float32)

        try:
            from omni.isaac.sensor import Camera
            from PIL import Image

            camera_paths = [
                f"{state.prim_path}/head_camera",
                f"{state.prim_path}/fetch/head_camera_link/head_camera",
                f"{state.prim_path}/camera",
            ]
            for cam_path in camera_paths:
                try:
                    cam = Camera(prim_path=cam_path)
                    cam.initialize()
                    rgba = cam.get_rgba()
                    if rgba is None or rgba.size == 0:
                        continue
                    img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
                    img = img.resize((256, 256))
                    rgb = np.array(img)
                    break
                except Exception:
                    continue
        except Exception:
            pass

        joint_positions = None
        gripper_state = 0.0
        try:
            joint_positions = robot.get_joint_positions()
            if joint_positions is not None and len(joint_positions) >= 2:
                gripper_state = 1.0 - (joint_positions[-1] / 0.05)
                gripper_state = np.clip(gripper_state, 0.0, 1.0)
        except Exception:
            pass

        return RobotObservation(
            robot_id=robot_id,
            timestamp=self._sim_time,
            rgb=rgb,
            depth=depth,
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(float(orientation[0]), float(orientation[1]), float(orientation[2]), float(orientation[3])),
            velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
            gripper_state=float(gripper_state),
            joint_positions=joint_positions,
        )

    def send_command(self, command: RobotCommand) -> bool:
        state = self._robots.get(command.robot_id)
        if state is None:
            return False

        robot = state.articulation

        try:
            if command.control_mode == ControlMode.HIGH_LEVEL:
                linear = command.linear_velocity[0]
                angular = command.angular_velocity[2]

                wheel_base = 0.37
                wheel_radius = 0.06
                left_vel = (linear - angular * wheel_base / 2) / wheel_radius
                right_vel = (linear + angular * wheel_base / 2) / wheel_radius

                try:
                    current_vels = robot.get_joint_velocities()
                    if current_vels is not None and len(current_vels) >= 2:
                        current_vels[0] = left_vel
                        current_vels[1] = right_vel
                        robot.set_joint_velocities(current_vels)
                except Exception:
                    pass

                if command.gripper_action is not None:
                    self._set_gripper(robot, command.gripper_action)

            elif command.control_mode == ControlMode.LOW_LEVEL:
                if command.joint_velocities is not None:
                    robot.set_joint_velocities(command.joint_velocities)
                elif command.joint_positions is not None:
                    robot.set_joint_positions(command.joint_positions)

                if command.gripper_action is not None:
                    self._set_gripper(robot, command.gripper_action)

            return True
        except Exception as e:
            logger.error("Failed to send command to '%s': %s", command.robot_id, e)
            return False

    def _set_gripper(self, robot, action: float) -> None:
        try:
            joint_positions = robot.get_joint_positions()
            if joint_positions is None:
                return
            gripper_pos = 0.05 * (1.0 - action)
            joint_positions[-1] = gripper_pos
            joint_positions[-2] = gripper_pos
            robot.set_joint_positions(joint_positions)
        except Exception:
            return

    def is_robot_idle(self, robot_id: str) -> bool:
        return True

