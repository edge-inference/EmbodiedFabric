"""
Isaac Sim Backend

GPU-accelerated physics and rendering via NVIDIA Isaac Sim.
Requires: pip install isaacsim-rl isaacsim-robot
"""

import logging
import os
import numpy as np
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .base import PhysicsBackend, RobotObservation, RobotCommand, ControlMode

logger = logging.getLogger(__name__)


@dataclass
class IsaacRobotState:
    """Track robot state in Isaac Sim"""
    prim_path: str
    robot_type: str
    articulation: Any = None


class IsaacSimBackend(PhysicsBackend):
    """
    Isaac Sim backend with GPU-accelerated physics and rendering.
    
    Uses NVIDIA Isaac Sim (Omniverse) for:
    - GPU PhysX physics
    - RTX ray-traced rendering
    - Native robot articulation support
    """
    
    def __init__(self, config, enable_recording: bool = False, recording_path: str = None):
        self.config = config
        self._sim = None
        self._world = None
        self._stage = None
        self._robots: Dict[str, IsaacRobotState] = {}
        self._objects: Dict[str, str] = {}
        self._sim_time = 0.0
        self._initialized = False
        self._time_step = getattr(config, 'time_step', 1/60.0)
        
        self._recording_enabled = enable_recording
        self._recording_path = recording_path or "recordings"
        self._camera = None
        self._frame_count = 0

    def _resolve_assets_root(self) -> Optional[str]:
        assets_path = os.getenv("ISAAC_ASSETS_PATH") or os.getenv("OMNI_ISAAC_ASSETS_PATH")
        if assets_path:
            candidates = [
                os.path.join(assets_path, "Assets", "Isaac", "5.1"),
                os.path.join(assets_path, "Assets"),
                assets_path,
            ]
            for candidate in candidates:
                if os.path.isdir(os.path.join(candidate, "Isaac", "Robots")):
                    return candidate
                if os.path.isdir(os.path.join(candidate, "Isaac")):
                    return candidate
        try:
            from isaacsim.core.utils.nucleus import get_assets_root_path
        except Exception:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path
            except Exception:
                return None
        return get_assets_root_path()

    def _get_robot_usd_path(self, assets_root: str, robot_type: str) -> Optional[str]:
        robot_key = (robot_type or "fetch").lower()
        robot_map = {
            "fetch": "Isaac/Robots/Fetch/fetch.usd",
            "carter": "Isaac/Robots/NVIDIA/Carter/carter_v1.usd",
        }
        rel_path = robot_map.get(robot_key)
        if not rel_path:
            return None
        return os.path.join(assets_root, rel_path)

    def initialize(self) -> bool:
        """Initialize Isaac Sim"""
        try:
            from isaacsim import SimulationApp
            
            logger.info("Starting Isaac Sim...")
            
            # Launch headless Isaac Sim
            renderer = getattr(self.config, "renderer", "Rasterizer")
            self._sim = SimulationApp({
                "headless": True,
                "width": 1280,
                "height": 720,
                "anti_aliasing": 0,
                "renderer": renderer
            })
            
            # Import after SimulationApp is created
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
                    logger.info(f"Using local Isaac assets: {assets_path}")
                except Exception:
                    logger.info(f"Using local Isaac assets: {assets_path}")
            
            # Set physics parameters
            from omni.isaac.core.utils.prims import define_prim
            physics_dt = self._time_step
            rendering_dt = self._time_step
            self._world.set_simulation_dt(physics_dt=physics_dt, rendering_dt=rendering_dt)
            
            # Create scene based on config
            self._setup_scene()
            
            if self._recording_enabled:
                self._setup_recording()
            
            self._world.reset()
            self._initialized = True
            
            logger.info("Isaac Sim initialized successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Isaac Sim not installed: {e}")
            logger.error("Install with: pip install isaacsim-rl isaacsim-robot")
            return False
        except Exception as e:
            logger.error(f"Isaac Sim initialization failed: {e}")
            return False

    def _setup_scene(self):
        """Setup simulation scene - warehouse or custom"""
        scene_name = getattr(self.config, 'scene_name', None)
        scene_size = getattr(self.config, 'scene_size', (20, 20))
        assets_root = self._resolve_assets_root()
        
        # Try to load warehouse environment
        if scene_name and "warehouse" in scene_name.lower():
            try:
                warehouse_usd = f"{assets_root}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
                if not os.path.isfile(warehouse_usd):
                    raise FileNotFoundError(warehouse_usd)
                from omni.isaac.core.utils.stage import add_reference_to_stage
                add_reference_to_stage(usd_path=warehouse_usd, prim_path="/World/Warehouse")
                logger.info("Loaded Isaac Sim warehouse environment")
                return
            except Exception as e:
                logger.warning(f"Could not load warehouse: {e}, using empty scene")
        
        # Fallback: empty scene with ground
        logger.info(f"Isaac Sim scene: {scene_size[0]}x{scene_size[1]}m (empty)")

    def _setup_recording(self):
        """Setup camera for recording"""
        try:
            from omni.isaac.sensor import Camera
            import os
            
            os.makedirs(self._recording_path, exist_ok=True)
            
            # Create overhead camera
            self._camera = Camera(
                prim_path="/World/RecordingCamera",
                position=np.array([-4.0, 18.0, -4.0]),
                frequency=30,
                resolution=(1280, 720),
                orientation=None  # Will look at origin
            )
            self._world.scene.add(self._camera)
            
            logger.info(f"Recording enabled: {self._recording_path}")
            
        except Exception as e:
            logger.warning(f"Failed to setup recording: {e}")
            self._recording_enabled = False

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset Isaac Sim environment"""
        if self._world:
            self._world.reset()
            self._sim_time = 0.0
            self._robots.clear()
            self._objects.clear()

    def step(self) -> None:
        """Advance physics by one timestep"""
        if not self._world:
            return
        
        self._world.step(render=True)
        self._sim_time += self._time_step
        
        # Capture frame if recording
        if self._recording_enabled and self._camera:
            if self._frame_count % 2 == 0:  # Every other frame
                self._capture_frame()
            self._frame_count += 1

    def _capture_frame(self):
        """Capture frame from recording camera"""
        try:
            import os
            from PIL import Image
            
            rgba = self._camera.get_rgba()
            if rgba is not None:
                img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
                frame_path = os.path.join(
                    self._recording_path, 
                    f"frame_{self._frame_count:06d}.jpg"
                )
                img.save(frame_path, quality=90)
        except Exception:
            pass

    def spawn_robot(self, robot_id: str, position: Tuple[float, float, float],
                    robot_type: str = "fetch") -> bool:
        """Spawn a robot in Isaac Sim (tries Fetch, falls back to cube proxy)"""
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
            from omni.isaac.core.robots import Robot
            
            prim_path = f"/World/Robots/{robot_id}"
            assets_root = self._resolve_assets_root()
            
            if assets_root is not None:
                robot_usd = self._get_robot_usd_path(assets_root, robot_type)
                if not robot_usd:
                    logger.warning(f"Unknown robot_type '{robot_type}', using proxy")
                    return self._spawn_proxy_robot(robot_id, position)
                if not os.path.isfile(robot_usd):
                    logger.warning(f"Robot asset not found: {robot_usd}")
                    return self._spawn_proxy_robot(robot_id, position)
                add_reference_to_stage(usd_path=robot_usd, prim_path=prim_path)
                robot = Robot(
                    prim_path=prim_path,
                    name=robot_id,
                    position=np.array(position),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0])
                )
                self._world.scene.add(robot)
                self._robots[robot_id] = IsaacRobotState(
                    prim_path=prim_path, robot_type="fetch", articulation=robot
                )
                logger.info(f"Spawned {robot_type} robot '{robot_id}' at {position}")
                return True
            else:
                logger.warning("Nucleus not available, using cube proxy robot")
                return self._spawn_proxy_robot(robot_id, position)
            
        except Exception as e:
            logger.warning(f"Fetch spawn failed: {e}, using proxy")
            return self._spawn_proxy_robot(robot_id, position)
    
    def _spawn_proxy_robot(self, robot_id: str, position: Tuple[float, float, float]) -> bool:
        """Spawn a cube as robot proxy when Nucleus assets unavailable"""
        try:
            from omni.isaac.core.objects import DynamicCuboid
            from omni.isaac.core.utils.prims import delete_prim
            
            prim_path = f"/World/Robots/{robot_id}"
            delete_prim(prim_path)
            robot = DynamicCuboid(
                prim_path=prim_path,
                name=robot_id,
                position=np.array(position),
                size=0.5,
                color=np.array([0.2, 0.6, 1.0]),
                mass=10.0
            )
            self._world.scene.add(robot)
            self._robots[robot_id] = IsaacRobotState(
                prim_path=prim_path, robot_type="proxy", articulation=robot
            )
            logger.info(f"Spawned proxy robot '{robot_id}' at {position}")
            return True
        except Exception as e:
            logger.error(f"Failed to spawn robot '{robot_id}': {e}")
            return False

    def get_observation(self, robot_id: str) -> RobotObservation:
        """Get sensor observation from Fetch robot"""
        state = self._robots.get(robot_id)
        if state is None:
            raise ValueError(f"Robot '{robot_id}' not found")
        
        robot = state.articulation
        
        # Get position and orientation
        try:
            pose = robot.get_world_pose()
            position = pose[0] if pose[0] is not None else np.zeros(3)
            orientation = pose[1] if pose[1] is not None else np.array([1, 0, 0, 0])
        except Exception:
            position = np.zeros(3)
            orientation = np.array([1, 0, 0, 0])
        
        # Get velocity
        try:
            velocity = robot.get_linear_velocity()
            if velocity is None:
                velocity = np.zeros(3)
        except Exception:
            velocity = np.zeros(3)
        
        # Get camera image from Fetch head camera
        rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        depth = np.zeros((256, 256), dtype=np.float32)
        
        try:
            from omni.isaac.sensor import Camera
            # Fetch has head_camera
            camera_paths = [
                f"{state.prim_path}/head_camera",
                f"{state.prim_path}/fetch/head_camera_link/head_camera",
                f"{state.prim_path}/camera"
            ]
            for cam_path in camera_paths:
                try:
                    cam = Camera(prim_path=cam_path)
                    cam.initialize()
                    rgba = cam.get_rgba()
                    if rgba is not None and rgba.size > 0:
                        from PIL import Image
                        img = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
                        img = img.resize((256, 256))
                        rgb = np.array(img)
                        break
                except Exception:
                    continue
        except Exception:
            pass
        
        # Get joint positions (Fetch has ~13 joints: 2 base + 7 arm + 2 gripper + head)
        joint_positions = None
        gripper_state = 0.0
        try:
            joint_positions = robot.get_joint_positions()
            if joint_positions is not None and len(joint_positions) >= 2:
                # Gripper state from last joints (0.05=open, 0=closed)
                gripper_state = 1.0 - (joint_positions[-1] / 0.05)
                gripper_state = np.clip(gripper_state, 0.0, 1.0)
        except Exception:
            pass
        
        return RobotObservation(
            robot_id=robot_id,
            timestamp=self._sim_time,
            rgb=rgb,
            depth=depth,
            position=(float(position[0]), float(position[1]), float(position[2])),
            rotation=(float(orientation[0]), float(orientation[1]), 
                     float(orientation[2]), float(orientation[3])),
            velocity=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
            gripper_state=float(gripper_state),
            joint_positions=joint_positions
        )

    def send_command(self, command: RobotCommand) -> bool:
        """Send control command to a Fetch robot (base + arm + gripper)"""
        state = self._robots.get(command.robot_id)
        if state is None:
            return False
        
        robot = state.articulation
        
        try:
            if command.control_mode == ControlMode.HIGH_LEVEL:
                # Base movement: differential drive
                linear = command.linear_velocity[0]
                angular = command.angular_velocity[2]
                
                # Fetch wheel parameters
                wheel_base = 0.37  # Fetch wheel base
                wheel_radius = 0.06
                
                left_vel = (linear - angular * wheel_base / 2) / wheel_radius
                right_vel = (linear + angular * wheel_base / 2) / wheel_radius
                
                # Apply to base wheels (first 2 joints typically)
                base_velocities = np.array([left_vel, right_vel])
                
                # Get current joint velocities and update base
                try:
                    current_vels = robot.get_joint_velocities()
                    if current_vels is not None and len(current_vels) >= 2:
                        current_vels[0] = left_vel
                        current_vels[1] = right_vel
                        robot.set_joint_velocities(current_vels)
                except Exception:
                    pass
                
                # Gripper action
                if command.gripper_action is not None:
                    self._set_gripper(robot, command.gripper_action)
                
            elif command.control_mode == ControlMode.LOW_LEVEL:
                # Direct joint control (arm + base)
                if command.joint_velocities is not None:
                    robot.set_joint_velocities(command.joint_velocities)
                elif command.joint_positions is not None:
                    robot.set_joint_positions(command.joint_positions)
                
                if command.gripper_action is not None:
                    self._set_gripper(robot, command.gripper_action)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send command to '{command.robot_id}': {e}")
            return False
    
    def _set_gripper(self, robot, action: float):
        """Set gripper state (0=open, 1=closed)"""
        try:
            # Fetch gripper joints are typically the last 2
            joint_positions = robot.get_joint_positions()
            if joint_positions is not None:
                # Gripper range: 0.0 (closed) to 0.05 (open)
                gripper_pos = 0.05 * (1.0 - action)
                # Last 2 joints are gripper fingers
                joint_positions[-1] = gripper_pos
                joint_positions[-2] = gripper_pos
                robot.set_joint_positions(joint_positions)
        except Exception:
            pass

    def spawn_object(self, object_id: str, object_type: str,
                     position: Tuple[float, float, float]) -> bool:
        """Spawn an object in Isaac Sim"""
        try:
            from omni.isaac.core.objects import DynamicCuboid
            
            prim_path = f"/World/Objects/{object_id}"
            
            obj = DynamicCuboid(
                prim_path=prim_path,
                name=object_id,
                position=np.array(position),
                size=0.15,
                color=np.array([0.8, 0.2, 0.2])
            )
            
            self._world.scene.add(obj)
            self._objects[object_id] = prim_path
            
            logger.info(f"Spawned object '{object_id}' at {position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn object '{object_id}': {e}")
            return False

    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        """Get object position"""
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

    def is_robot_idle(self, robot_id: str) -> bool:
        """Check if robot has no pending actions"""
        return True  # Isaac Sim handles this internally

    def close(self) -> None:
        """Cleanup and close Isaac Sim"""
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

    def _compile_video(self):
        """Compile frames to video"""
        import subprocess
        import glob
        import os
        
        frames = glob.glob(os.path.join(self._recording_path, "frame_*.jpg"))
        if not frames:
            return
        
        video_path = os.path.join(self._recording_path, "simulation.mp4")
        
        try:
            cmd = [
                "ffmpeg", "-y", "-framerate", "30",
                "-pattern_type", "glob",
                "-i", os.path.join(self._recording_path, "frame_*.jpg"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                video_path
            ]
            subprocess.run(cmd, capture_output=True)
            logger.info(f"Video saved: {video_path}")
        except Exception as e:
            logger.warning(f"Failed to compile video: {e}")

    @property
    def sim_time(self) -> float:
        return self._sim_time

    @property
    def backend_name(self) -> str:
        return "isaac_sim"
