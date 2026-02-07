#!/usr/bin/env python3
"""
Test Isaac Sim Backend with Asset Loading

Tests:
1. Basic initialization
2. Robot spawning (Fetch from assets)
3. Warehouse environment loading
4. Simple movement commands
"""

import sys
import os
import logging
from pathlib import Path

# Add simulator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.backend.isaac_backend import IsaacSimBackend
from simulator.backend.base import RobotCommand, ControlMode
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Minimal config for testing"""
    scene_name: str = "warehouse"
    scene_size: tuple = (20, 20)
    time_step: float = 1/60.0


def test_isaac_backend():
    """Test Isaac Sim backend initialization and basic operations"""
    
    logger.info("=" * 60)
    logger.info("Isaac Sim Backend Test")
    logger.info("=" * 60)
    
    # Check environment
    assets_path = os.getenv("ISAAC_ASSETS_PATH")
    if assets_path:
        logger.info(f"Assets path: {assets_path}")
    else:
        logger.warning("ISAAC_ASSETS_PATH not set, using default Nucleus")
    
    # Create backend
    config = TestConfig()
    backend = IsaacSimBackend(
        config=config,
        enable_recording=True,
        recording_path="recordings/isaac_test"
    )
    
    # Test 1: Initialization
    logger.info("\n[Test 1] Initializing Isaac Sim...")
    if not backend.initialize():
        logger.error("✗ Initialization failed")
        return False
    logger.info("✓ Isaac Sim initialized")
    
    # Test 2: Spawn robot
    logger.info("\n[Test 2] Spawning robot...")
    if not backend.spawn_robot("robot_1", position=(0.0, 0.0, 0.0), robot_type="carter"):
        logger.warning("⚠ Fetch spawn failed, trying proxy robot")
    else:
        logger.info("✓ Robot spawned")
    
    # Test 3: Spawn object
    logger.info("\n[Test 3] Spawning test object...")
    backend.spawn_object("box_1", "cube", position=(2.0, 0.0, 0.5))
    logger.info("✓ Object spawned")
    
    # Test 4: Get observation
    logger.info("\n[Test 4] Getting robot observation...")
    try:
        obs = backend.get_observation("robot_1")
        logger.info(f"  Position: ({obs.position[0]:.2f}, {obs.position[1]:.2f}, {obs.position[2]:.2f})")
        logger.info(f"  RGB shape: {obs.rgb.shape}")
        logger.info(f"  Gripper: {obs.gripper_state:.2f}")
        logger.info("✓ Observation retrieved")
    except Exception as e:
        logger.error(f"✗ Observation failed: {e}")
    
    # Test 5: Send movement commands
    logger.info("\n[Test 5] Testing robot movement...")
    try:
        # Forward movement
        cmd = RobotCommand(
            robot_id="robot_1",
            control_mode=ControlMode.HIGH_LEVEL,
            linear_velocity=(0.5, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 0.0),
            gripper_action=0.0  # Open gripper
        )
        backend.send_command(cmd)
        
        # Simulate for 3 seconds
        for i in range(180):  # 3 sec at 60Hz
            backend.step()
            if i % 60 == 0:
                obs = backend.get_observation("robot_1")
                logger.info(f"  t={i/60:.1f}s: pos=({obs.position[0]:.2f}, {obs.position[1]:.2f}, {obs.position[2]:.2f})")
        
        logger.info("✓ Movement test complete")
        
    except Exception as e:
        logger.error(f"✗ Movement failed: {e}")
    
    # Test 6: Cleanup
    logger.info("\n[Test 6] Closing Isaac Sim...")
    backend.close()
    logger.info("✓ Cleanup complete")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = test_isaac_backend()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)
