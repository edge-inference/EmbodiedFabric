import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.isaac
@pytest.mark.parametrize("robot_type", ["g1", "h1"])
def test_isaac_humanoid_spawn_and_observe(robot_type: str):
    if importlib.util.find_spec("isaacsim") is None:
        pytest.skip("isaacsim not installed")

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    settle_steps = int(os.getenv("ISAAC_HUMANOID_SETTLE_STEPS", "300"))
    hold_steps = int(os.getenv("ISAAC_HUMANOID_HOLD_STEPS", "1800"))

    code = textwrap.dedent(
        f"""
        import numpy as np
        from dataclasses import dataclass

        from simulator.backend.isaac_backend import IsaacSimBackend

        @dataclass
        class Config:
            scene_name: str = "warehouse"
            scene_size: tuple = (20, 20)
            time_step: float = 1 / 60.0

        backend = IsaacSimBackend(config=Config(), headless=True, enable_recording=False)
        try:
            if not backend.initialize():
                raise SystemExit(2)

            if not backend.spawn_robot("humanoid", position=(0.0, 0.0, 1.2), robot_type={robot_type!r}):
                raise SystemExit(3)

            if not backend.spawn_object("box_1", "cube", position=(1.5, 0.0, 0.5)):
                raise SystemExit(4)
            if not backend.spawn_object("box_2", "cube", position=(-1.0, 1.0, 0.5)):
                raise SystemExit(5)

            for i in range({settle_steps}):
                backend.step()
                if i % 60 == 0:
                    obs = backend.get_observation("humanoid")
                    if not (isinstance(obs.position, tuple) and len(obs.position) == 3):
                        raise SystemExit(6)

            for _ in range({hold_steps}):
                backend.step()

            obs = backend.get_observation("humanoid")
            ok = (
                isinstance(obs.position, tuple)
                and len(obs.position) == 3
                and obs.rgb.shape == (256, 256, 3)
                and np.isfinite(np.asarray(obs.position, dtype=float)).all()
                and (obs.joint_positions is None or len(obs.joint_positions) > 0)
            )
        finally:
            backend.close()

        raise SystemExit(0 if ok else 7)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, (
        "Isaac humanoid smoke test subprocess failed.\n"
        f"exit_code={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )
