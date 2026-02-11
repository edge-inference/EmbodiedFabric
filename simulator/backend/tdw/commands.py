"""Command helpers for TDWBackend."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..base import ControlMode, RobotCommand
from .types import COLLISION_RECOVERY_THRESHOLD, RECOVERY_TURN_RANGE, RobotState

logger = logging.getLogger(__name__)


class TDWCommandMixin:
    def send_command(self, command: RobotCommand) -> bool:
        state = self._robots.get(command.robot_id)
        if state is None or state.magnebot is None:
            raise ValueError(f"Robot '{command.robot_id}' not found")

        magnebot = state.magnebot

        if state.pending_action and command.control_mode == ControlMode.HIGH_LEVEL:
            return False

        try:
            if command.control_mode == ControlMode.LOW_LEVEL:
                return self._send_low_level_command(command, state, magnebot)
            return self._send_high_level_command(command, state, magnebot)
        except Exception as e:
            logger.error("Command failed for %s: %s", command.robot_id, e)
            return False

    def _send_low_level_command(self, command: RobotCommand, state: RobotState, magnebot) -> bool:
        if command.joint_velocities is not None:
            joint_vels = command.joint_velocities

            forward = command.linear_velocity[0]
            turn = command.angular_velocity[2]

            if state.consecutive_collisions >= COLLISION_RECOVERY_THRESHOLD:
                import random

                angle = random.uniform(*RECOVERY_TURN_RANGE)
                if random.random() < 0.5:
                    angle = -angle
                magnebot.turn_by(angle=angle)
                state.pending_action = True
                state.recovering = True
                state.action_type = f"recovery_turn({angle:.0f})"
                logger.info(
                    "[%s] recovery turn %.0fdeg after %d collisions",
                    command.robot_id,
                    angle,
                    state.consecutive_collisions,
                )
                state.consecutive_collisions = 0
                return True

            if abs(forward) > 0.01:
                distance = forward * 3.0
                magnebot.move_by(distance=distance, arrived_at=0.05)
                state.pending_action = True
                state.action_type = f"low_level_move({forward:.3f})"
            elif abs(turn) > 0.01:
                angle = turn * 30.0
                magnebot.turn_by(angle=angle)
                state.pending_action = True
                state.action_type = f"low_level_turn({turn:.3f})"

            if len(joint_vels) > 6:
                gripper_vel = float(joint_vels[6])
                if gripper_vel > 0.3:
                    self._try_grasp(magnebot, state, command.robot_id)
                elif gripper_vel < -0.3:
                    self._try_drop(magnebot, state, command.robot_id)

            return True

        if command.joint_positions is not None:
            return True

        return False

    def _send_high_level_command(self, command: RobotCommand, state: RobotState, magnebot) -> bool:
        vx, _, vz = command.linear_velocity
        if vx != 0.0 or vz != 0.0:
            magnitude = float(np.hypot(vx, vz))
            distance = magnitude * 5.0
            if vx < 0:
                distance = -distance
            magnebot.move_by(distance=distance, arrived_at=0.3)
            state.pending_action = True
            state.action_type = f"move_by({distance:.2f})"
            return True

        if command.angular_velocity[2] != 0.0:
            angle = command.angular_velocity[2] * 15.0
            magnebot.turn_by(angle=angle)
            state.pending_action = True
            state.action_type = f"turn_by({angle:.1f})"
            return True

        if command.gripper_action is not None:
            if command.gripper_action > 0.5:
                if command.arm_target is not None:
                    self._try_grasp(magnebot, state, command.robot_id, command.arm_target)
            else:
                self._try_drop(magnebot, state, command.robot_id)
            return True

        return True

    def _try_grasp(self, magnebot, state: RobotState, robot_id: str, target=None) -> None:
        from magnebot import Arm

        if target is not None:
            magnebot.grasp(target=target, arm=Arm.right)
            state.pending_action = True
            state.action_type = f"grasp({target})"
            return

        nearest_obj = self._find_nearest_object(magnebot)
        if nearest_obj is None:
            return
        magnebot.grasp(target=nearest_obj, arm=Arm.right)
        state.pending_action = True
        state.action_type = f"grasp({nearest_obj})"

    def _try_drop(self, magnebot, state: RobotState, robot_id: str) -> None:
        from magnebot import Arm

        held_right = magnebot.dynamic.held.get(Arm.right, [])
        if not held_right:
            return
        target_obj = int(held_right[0])
        magnebot.drop(target=target_obj, arm=Arm.right)
        state.pending_action = True
        state.action_type = f"drop({target_obj})"

    def _find_nearest_object(self, magnebot) -> Optional[int]:
        if not self._objects or not self._controller:
            return None

        try:
            from tdw.output_data import OutputData, Transforms

            robot_pos = magnebot.dynamic.transform.position
            resp = self._controller.communicate([{"$type": "send_transforms", "frequency": "once"}])

            obj_positions = {}
            for i in range(len(resp) - 1):
                if OutputData.get_data_type_id(resp[i]) != "tran":
                    continue
                transforms = Transforms(resp[i])
                for j in range(transforms.get_num()):
                    obj_positions[transforms.get_id(j)] = transforms.get_position(j)

            nearest_dist = float("inf")
            nearest_obj = None
            for _, obj_id in self._objects.items():
                if obj_id not in obj_positions:
                    continue
                pos = obj_positions[obj_id]
                dist = float(np.sqrt((robot_pos[0] - pos[0]) ** 2 + (robot_pos[2] - pos[2]) ** 2))
                if dist < nearest_dist and dist < 2.0:
                    nearest_dist = dist
                    nearest_obj = obj_id
            return nearest_obj
        except Exception as e:
            logger.debug("_find_nearest_object error: %s", e)
            return None

