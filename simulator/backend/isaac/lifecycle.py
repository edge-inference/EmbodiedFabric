"""Lifecycle methods for IsaacSimBackend."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class IsaacLifecycleMixin:
    def initialize(self) -> bool:
        try:
            from isaacsim import SimulationApp

            renderer = getattr(self.config, "renderer", "RayTracedLighting")
            self._sim = SimulationApp(
                {
                    "headless": self._headless,
                    "width": 1280,
                    "height": 720,
                    "anti_aliasing": 0,
                    "renderer": renderer,
                }
            )

            from omni.isaac.core import World
            from omni.isaac.core.utils.stage import create_new_stage

            create_new_stage()
            self._world = World(stage_units_in_meters=1.0)
            self._world.scene.add_default_ground_plane()

            assets_path = os.getenv("ISAAC_ASSETS_PATH") or os.getenv("OMNI_ISAAC_ASSETS_PATH")
            if assets_path and os.path.isdir(assets_path):
                os.environ.setdefault("OMNI_ISAAC_ASSETS_PATH", assets_path)
                os.environ.setdefault("OMNI_USD_RESOLVER_AR_DEFAULT_SEARCH_PATH", assets_path)
                try:
                    try:
                        from isaacsim.core.utils.nucleus import set_assets_root_path
                    except Exception:
                        set_assets_root_path = None
                    if set_assets_root_path:
                        set_assets_root_path(assets_path)
                except Exception:
                    pass

            physics_dt = self._time_step
            self._world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=physics_dt)

            self._setup_scene()

            if self._recording_enabled:
                self._setup_recording()

            self._world.reset()
            self._initialized = True
            logger.info("Isaac Sim initialized")
            return True
        except ImportError as e:
            logger.error("Isaac Sim not installed: %s", e)
            return False
        except Exception as e:
            logger.error("Isaac Sim initialization failed: %s", e)
            return False

    def _setup_scene(self) -> None:
        scene_name = getattr(self.config, "scene_name", None)
        scene_size = getattr(self.config, "scene_size", (20, 20))
        assets_root = self._resolve_assets_root()

        if scene_name and "warehouse" in str(scene_name).lower():
            try:
                if not assets_root:
                    raise FileNotFoundError("assets_root is not set")
                warehouse_usd = f"{assets_root}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
                if not os.path.isfile(warehouse_usd):
                    raise FileNotFoundError(warehouse_usd)
                from omni.isaac.core.utils.stage import add_reference_to_stage

                add_reference_to_stage(usd_path=warehouse_usd, prim_path="/World/Warehouse")
                logger.info("Loaded warehouse environment")
                return
            except Exception as e:
                logger.warning("Could not load warehouse: %s; using empty scene", e)

        logger.info("Isaac Sim scene: %sx%sm (empty)", scene_size[0], scene_size[1])

    def reset(self, seed: Optional[int] = None) -> None:
        if self._world:
            self._world.reset()
            self._sim_time = 0.0
            self._robots.clear()
            self._objects.clear()

    def step(self) -> None:
        if not self._world:
            return

        self._world.step(render=True)
        self._sim_time += self._time_step

        if self._recording_enabled and self._camera:
            if self._frame_count % 2 == 0:
                self._capture_frame()
            self._frame_count += 1

    def close(self) -> None:
        if self._recording_enabled:
            self._compile_video()

        if self._sim:
            self._sim.close()
            self._sim = None

        self._world = None
        self._robots.clear()
        self._objects.clear()
        self._initialized = False
        logger.info("Isaac Sim closed")

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def backend_name(self) -> str:
        return "isaac_sim"

