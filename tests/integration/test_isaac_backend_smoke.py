import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.integration
@pytest.mark.isaac
def test_isaac_backend_initialize_and_close():
    if importlib.util.find_spec("isaacsim") is None:
        pytest.skip("isaacsim not installed")

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    code = textwrap.dedent(
        """
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        from dataclasses import dataclass

        from simulator.backend.isaac import IsaacSimBackend

        @dataclass
        class Config:
            scene_name: str = "warehouse"
            scene_size: tuple = (20, 20)
            time_step: float = 1 / 60.0

        backend = IsaacSimBackend(config=Config(), headless=True, enable_recording=False)
        try:
            ok = backend.initialize()
            backend.step()
        finally:
            backend.close()

        raise SystemExit(0 if ok else 2)
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        "Isaac backend subprocess failed.\n"
        f"exit_code={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}\n"
    )

