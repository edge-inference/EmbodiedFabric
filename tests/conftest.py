import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests (or export RUN_INTEGRATION=1).",
    )


def _integration_enabled(config) -> bool:
    if config.getoption("--run-integration"):
        return True
    return os.getenv("RUN_INTEGRATION", "0") == "1"


def pytest_collection_modifyitems(config, items):
    if _integration_enabled(config):
        return
    skip = pytest.mark.skip(reason="Pass --run-integration or export RUN_INTEGRATION=1")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)

