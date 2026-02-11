#!/usr/bin/env python3

import sys
import os
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.backend.isaac_backend import IsaacSimBackend
from dataclasses import dataclass

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "humanoid_test.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    scene_name: str = "warehouse"
    scene_size: tuple = (20, 20)
    time_step: float = 1/60.0


def run_humanoid_test(robot_type: str):
    logger.info("=" * 60)
    logger.info(f"Humanoid Test: {robot_type}")
    logger.info("=" * 60)

    config = TestConfig()
    backend = IsaacSimBackend(config=config, headless=False)

    logger.info("[1] Starting Isaac Sim...")
    if not backend.initialize():
        logger.error("Init failed")
        return False

    logger.info(f"[2] Spawning {robot_type}...")
    backend.spawn_robot("humanoid", position=(0.0, 0.0, 1.2), robot_type=robot_type)

    logger.info("[3] Spawning objects...")
    backend.spawn_object("box_1", "cube", position=(1.5, 0.0, 0.5))
    backend.spawn_object("box_2", "cube", position=(-1.0, 1.0, 0.5))

    logger.info("[4] Settling physics (5 seconds)...")
    for i in range(300):
        backend.step()
        if i % 60 == 0:
            obs = backend.get_observation("humanoid")
            logger.info(
                f"  t={i/60:.1f}s pos=({obs.position[0]:.2f}, "
                f"{obs.position[1]:.2f}, {obs.position[2]:.2f})"
            )

    obs = backend.get_observation("humanoid")
    logger.info(f"[5] Final state:")
    logger.info(f"  Position: ({obs.position[0]:.2f}, {obs.position[1]:.2f}, {obs.position[2]:.2f})")
    logger.info(f"  RGB shape: {obs.rgb.shape}")
    if obs.joint_positions is not None:
        logger.info(f"  Joint count: {len(obs.joint_positions)}")
        logger.info(f"  Joint positions: {np.round(obs.joint_positions, 3)}")

    logger.info("[6] Holding scene for 30 seconds...")
    for i in range(1800):
        backend.step()

    logger.info("[7] Done.")
    backend.close()
    return True


if __name__ == "__main__":
    robot_type = sys.argv[1] if len(sys.argv) > 1 else "g1"
    try:
        run_humanoid_test(robot_type)
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
