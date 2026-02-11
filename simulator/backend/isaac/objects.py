"""Object spawn/query for IsaacSimBackend."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class IsaacObjectMixin:
    def spawn_object(self, object_id: str, object_type: str, position: Tuple[float, float, float]) -> bool:
        try:
            from omni.isaac.core.objects import DynamicCuboid

            prim_path = f"/World/Objects/{object_id}"
            obj = DynamicCuboid(
                prim_path=prim_path,
                name=object_id,
                position=np.array(position),
                size=0.15,
                color=np.array([0.8, 0.2, 0.2]),
            )
            self._world.scene.add(obj)
            self._objects[object_id] = prim_path
            logger.info("Spawned object '%s' at %s", object_id, position)
            return True
        except Exception as e:
            logger.error("Failed to spawn object '%s': %s", object_id, e)
            return False

    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        prim_path = self._objects.get(object_id)
        if not prim_path:
            return None

        try:
            from omni.isaac.core.utils.prims import get_prim_at_path
            from pxr import UsdGeom

            prim = get_prim_at_path(prim_path)
            xform = UsdGeom.Xformable(prim)
            transform = xform.ComputeLocalToWorldTransform(0)
            pos = transform.ExtractTranslation()
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        except Exception:
            return None

