"""TDW 1.13 JSON encoder patch for numpy types."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


def patch_json_encoder_for_numpy() -> None:
    if getattr(json.JSONEncoder, "_vla_simu_numpy_patch", False):
        return

    original_default = json.JSONEncoder.default

    def patched_default(self: json.JSONEncoder, obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return original_default(self, obj)

    json.JSONEncoder.default = patched_default  # type: ignore[assignment]
    json.JSONEncoder._vla_simu_numpy_patch = True  # type: ignore[attr-defined]

