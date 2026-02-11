"""Robot spawn/observe utilities for TDWBackend."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

from ..base import RobotObservation
from .types import RobotState

logger = logging.getLogger(__name__)


class TDWRobotMixin:
    def spawn_robot(
        self,
        robot_id: str,
        position: Tuple[float, float, float],
        robot_type: str = "magnebot",
    ) -> bool:
        if not self._controller:
            raise RuntimeError("TDW controller not initialized")

        try:
            from magnebot import Magnebot
            from magnebot.image_frequency import ImageFrequency

            magnebot = Magnebot(
                robot_id=self._robot_counter,
                position={"x": position[0], "y": position[1], "z": position[2]},
                image_frequency=ImageFrequency.always,
            )

            self._controller.add_ons.append(magnebot)

            magnebot.collision_detection.previous_was_same = False

            self._robots[robot_id] = RobotState(magnebot=magnebot)
            self._controller.communicate([])

            if self._image_capture is not None and magnebot.static is not None:
                avatar_id = magnebot.static.avatar_id
                if avatar_id not in self._image_capture.avatar_ids:
                    self._image_capture.avatar_ids.append(avatar_id)
                    logger.info("Registered avatar '%s' with ImageCapture", avatar_id)

            self._robot_counter += 1
            logger.info("Spawned Magnebot '%s' at %s", robot_id, position)
            return True
        except ImportError:
            logger.error("Magnebot not installed. Install with: pip install magnebot")
            return False
        except Exception as e:
            logger.error("Failed to spawn robot '%s': %s", robot_id, e)
            return False

    def get_observation(self, robot_id: str) -> RobotObservation:
        state = self._robots.get(robot_id)
        if state is None or state.magnebot is None:
            raise ValueError(f"Robot '{robot_id}' not found")

        magnebot = state.magnebot

        rgb = None
        try:
            pil_images = magnebot.dynamic.get_pil_images()
            if pil_images:
                for key in ("img", "_img"):
                    if key in pil_images and pil_images[key] is not None:
                        rgb = np.array(pil_images[key])
                        break
                if rgb is None:
                    first_key = next(iter(pil_images))
                    if pil_images[first_key] is not None:
                        rgb = np.array(pil_images[first_key])
        except Exception as e:
            logger.debug("[%s] get_pil_images error: %s", robot_id, e)

        if rgb is None:
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            if self._sim_time < 1.0 or int(self._sim_time * 10) % 50 == 0:
                logger.warning("No RGB image available for %s", robot_id)

        try:
            depth = magnebot.dynamic.get_depth_values()
        except Exception:
            depth = np.zeros((480, 640), dtype=np.float32)

        transform = magnebot.dynamic.transform
        position = (
            float(transform.position[0]),
            float(transform.position[1]),
            float(transform.position[2]),
        )
        rotation = (
            float(transform.rotation[0]),
            float(transform.rotation[1]),
            float(transform.rotation[2]),
            float(transform.rotation[3]),
        )

        holding = False
        for _, held_objects in magnebot.dynamic.held.items():
            if held_objects:
                holding = True
                break

        return RobotObservation(
            robot_id=robot_id,
            timestamp=self._sim_time,
            rgb=rgb,
            depth=depth,
            position=position,
            rotation=rotation,
            velocity=(0.0, 0.0, 0.0),
            gripper_state=holding,
        )

    def is_robot_idle(self, robot_id: str) -> bool:
        state = self._robots.get(robot_id)
        if state is None:
            return True
        return not state.pending_action

