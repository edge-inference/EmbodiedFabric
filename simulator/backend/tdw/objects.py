"""Object spawn/query helpers for TDWBackend."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class TDWObjectMixin:
    def spawn_object(self, object_id: str, object_type: str, position: Tuple[float, float, float]) -> bool:
        if not self._controller:
            return False

        try:
            obj_id = self._controller.get_unique_id()
            spawn_pos = {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}

            model_name = object_type or "iron_box"
            cmd = self._controller.get_add_object(
                model_name=model_name,
                object_id=obj_id,
                position=spawn_pos,
                library="models_core.json",
            )
            self._controller.communicate([cmd])
            self._objects[object_id] = obj_id
            logger.info("Spawned '%s' (%s) at %s", object_id, model_name, spawn_pos)
            return True
        except Exception as e:
            logger.error("Failed to spawn object '%s': %s", object_id, e)
            return False

    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        if not self._controller or object_id not in self._objects:
            return None

        try:
            from tdw.output_data import OutputData, Transforms

            obj_id = self._objects[object_id]
            resp = self._controller.communicate([{"$type": "send_transforms", "frequency": "once"}])

            for i in range(len(resp) - 1):
                if OutputData.get_data_type_id(resp[i]) != "tran":
                    continue
                transforms = Transforms(resp[i])
                for j in range(transforms.get_num()):
                    if transforms.get_id(j) == obj_id:
                        pos = transforms.get_position(j)
                        return (float(pos[0]), float(pos[1]), float(pos[2]))
            return None
        except Exception as e:
            logger.debug("Failed to get position for '%s': %s", object_id, e)
            return None

