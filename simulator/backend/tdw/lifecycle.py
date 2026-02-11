"""Lifecycle methods for TDWBackend."""

from __future__ import annotations

import logging
from typing import Optional

from .json_patch import patch_json_encoder_for_numpy

logger = logging.getLogger(__name__)


class TDWLifecycleMixin:
    def initialize(self) -> bool:
        patch_json_encoder_for_numpy()

        try:
            from tdw.controller import Controller
            from tdw.tdw_utils import TDWUtils

            if self._launch_build:
                logger.info("Starting TDW controller (launching build)")
                self._controller = Controller(launch_build=True)
            else:
                logger.info("Connecting to TDW build at %s:%s", self._tdw_address, self._tdw_port)
                self._controller = Controller(
                    launch_build=False,
                    address=self._tdw_address,
                    port=self._tdw_port,
                )

            self._controller.communicate(
                [
                    {"$type": "set_target_framerate", "framerate": -1},
                    {"$type": "destroy_all_objects"},
                ]
            )
            self._controller.add_ons.clear()

            scene_name = getattr(self.config, "scene_name", None)
            floorplan_layout = getattr(self.config, "floorplan_layout", None)

            if scene_name and str(scene_name).startswith("floorplan") and floorplan_layout is not None:
                from tdw.add_ons.floorplan import Floorplan

                floorplan = Floorplan()
                floorplan.init_scene(scene=str(scene_name), layout=int(floorplan_layout))
                self._controller.add_ons.append(floorplan)
                commands = [
                    {"$type": "set_time_step", "time_step": self.config.time_step},
                    {"$type": "set_floorplan_roof", "show": False},
                ]
            elif scene_name:
                commands = [
                    self._controller.get_add_scene(scene_name=scene_name),
                    {"$type": "set_time_step", "time_step": self.config.time_step},
                ]
                if str(scene_name).startswith("floorplan"):
                    commands.append({"$type": "set_floorplan_roof", "show": False})
            else:
                w = int(self.config.scene_size[0])
                h = int(self.config.scene_size[1])
                commands = [
                    {"$type": "load_scene", "scene_name": "ProcGenScene"},
                    TDWUtils.create_empty_room(width=w, length=h),
                    {"$type": "set_time_step", "time_step": self.config.time_step},
                ]

            self._controller.communicate(commands)

            if self._recording_enabled:
                self._setup_recording()

            self._initialized = True
            return True
        except ImportError as e:
            logger.error("TDW not installed: %s", e)
            return False
        except Exception as e:
            logger.error("TDW initialization failed: %s", e)
            return False

    def reset(self, seed: Optional[int] = None) -> None:
        if not self._controller:
            return

        self._controller.add_ons.clear()
        self._robots.clear()
        self._objects.clear()
        self._robot_counter = 0

        from tdw.tdw_utils import TDWUtils

        self._controller.communicate(
            [
                {"$type": "destroy_all_objects"},
                TDWUtils.create_empty_room(
                    width=int(self.config.scene_size[0]),
                    length=int(self.config.scene_size[1]),
                ),
            ]
        )
        self._sim_time = 0.0

    def step(self) -> None:
        if not self._controller:
            return

        commands = []
        for state in self._robots.values():
            if state.magnebot and state.magnebot.static:
                commands.append(
                    {
                        "$type": "enable_image_sensor",
                        "enable": True,
                        "avatar_id": state.magnebot.static.avatar_id,
                    }
                )

        self._controller.communicate(commands)

        if self._recording_enabled:
            self._frame_count += 1
        self._sim_time += self.config.time_step

        self._update_action_states()

    def _update_action_states(self) -> None:
        from magnebot import ActionStatus

        for robot_id, state in self._robots.items():
            if not state.pending_action or not state.magnebot:
                continue
            status = state.magnebot.action.status
            if status == ActionStatus.ongoing:
                continue

            state.pending_action = False
            if status in (ActionStatus.collision, ActionStatus.failed_to_move):
                state.consecutive_collisions += 1
                logger.warning(
                    "Robot %s action '%s' %s #%d",
                    robot_id,
                    state.action_type,
                    status.name,
                    state.consecutive_collisions,
                )
            elif status == ActionStatus.success:
                state.consecutive_collisions = 0

    def close(self) -> None:
        if self._recording_enabled and self._frame_count > 0:
            self._compile_video()

        if self._controller:
            try:
                if self._launch_build:
                    self._controller.communicate([{"$type": "terminate"}])
                else:
                    logger.info("Disconnecting from TDW (build stays running)")
            except Exception as e:
                logger.warning("Error during TDW shutdown: %s", e)
            finally:
                self._controller = None

        self._robots.clear()
        self._objects.clear()
        self._initialized = False

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def backend_name(self) -> str:
        return "TDW"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

