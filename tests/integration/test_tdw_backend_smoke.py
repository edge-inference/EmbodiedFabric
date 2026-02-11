import os
from dataclasses import dataclass

import pytest


@pytest.mark.integration
@pytest.mark.tdw
def test_tdw_backend_spawn_observe_step():
    pytest.importorskip("tdw")
    pytest.importorskip("magnebot")

    from simulator.backend.tdw import TDWBackend

    @dataclass
    class Config:
        scene_size: tuple = (12, 12)
        time_step: float = 1 / 60.0

    backend = TDWBackend(config=Config(), launch_build=True, enable_recording=False)
    try:
        assert backend.initialize() is True
        assert backend.spawn_robot("r0", position=(0.0, 0.0, 0.0)) is True
        backend.step()
        obs = backend.get_observation("r0")
        assert obs.rgb.ndim == 3
        assert obs.rgb.shape[-1] == 3
    finally:
        backend.close()

