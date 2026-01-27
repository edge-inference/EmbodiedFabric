"""
TDW (ThreeDWorld) Backend

High-fidelity physics simulation using MIT's ThreeDWorld.
https://threedworld.org/

Features:
- Rigid body physics
- Photorealistic rendering
- Sensor simulation (RGB, depth)
- Robot articulation (Magnebot)

Usage:
    Magnebot is added as a TDW add-on. Actions are non-blocking;
    you must call step() until the action completes.
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging
import json

from .base import PhysicsBackend, RobotObservation, RobotCommand

logger = logging.getLogger(__name__)


def _patch_json_for_numpy():
    """
    Patch JSON encoder to handle numpy types.
    
    Magnebot uses numpy float32/int32 internally, but TDW 1.13.0
    can't serialize them. This patches json.JSONEncoder globally.
    """
    _original_default = json.JSONEncoder.default
    
    def _patched_default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return _original_default(self, obj)
    
    json.JSONEncoder.default = _patched_default
    logger.debug("Patched JSON encoder for numpy types")

# Apply patch on module load
_patch_json_for_numpy()


@dataclass
class RobotState:
    """Track robot's pending action state"""
    magnebot: Any = None
    pending_action: bool = False
    action_type: str = ""


class TDWBackend(PhysicsBackend):
    """
    TDW physics backend with proper Magnebot integration.
    
    Magnebot is used as a TDW add-on for multi-robot support.
    Actions are non-blocking - step() advances until actions complete.
    """
    
    def __init__(self, config, enable_recording: bool = False, recording_path: str = None):
        self.config = config
        self._controller = None
        self._robots: Dict[str, RobotState] = {}
        self._objects: Dict[str, int] = {}
        self._sim_time = 0.0
        self._initialized = False
        self._robot_counter = 0
        
        # Video recording
        self._recording_enabled = enable_recording
        self._recording_path = recording_path or "recordings"
        self._third_person_camera = None
        self._image_capture = None
        self._frame_count = 0
    
    def initialize(self) -> bool:
        """Initialize TDW controller and scene"""
        try:
            from tdw.controller import Controller
            from tdw.tdw_utils import TDWUtils
            
            logger.info("Starting TDW controller...")
            self._controller = Controller(launch_build=True)
            
            # Create warehouse-like room
            commands = [
                {"$type": "load_scene", "scene_name": "ProcGenScene"},
                TDWUtils.create_empty_room(
                    width=int(self.config.scene_size[0]),
                    length=int(self.config.scene_size[1])
                ),
                {"$type": "set_time_step", "time_step": self.config.time_step}
            ]
            
            self._controller.communicate(commands)
            
            # Setup video recording if enabled
            if self._recording_enabled:
                self._setup_recording()
            
            self._initialized = True
            logger.info(f"TDW backend initialized: {self.config.scene_size[0]}x{self.config.scene_size[1]}m room")
            return True
            
        except ImportError as e:
            logger.error(f"TDW not installed: {e}")
            logger.error("Install with: pip install tdw magnebot")
            return False
        except Exception as e:
            logger.error(f"TDW initialization failed: {e}")
            return False
    
    def _setup_recording(self) -> None:
        """Setup third-person camera and image capture for video recording"""
        try:
            from tdw.add_ons.third_person_camera import ThirdPersonCamera
            from tdw.add_ons.image_capture import ImageCapture
            import os
            
            # Create a numbered run directory to avoid mixing recordings
            def _next_run_id(base_dir: str) -> int:
                try:
                    existing = [
                        int(name.split("_")[1])
                        for name in os.listdir(base_dir)
                        if name.startswith("run_") and name.split("_")[1].isdigit()
                    ]
                    return max(existing, default=0) + 1
                except FileNotFoundError:
                    return 1
            
            run_id = _next_run_id(self._recording_path)
            self._recording_path = os.path.join(self._recording_path, f"run_{run_id}")
            os.makedirs(self._recording_path, exist_ok=True)
            
            # Get resolution from config (default 1280x720)
            width, height = getattr(self.config, 'recording_resolution', (1280, 720))
            
            # Set render quality for better visuals
            self._controller.communicate([
                {"$type": "set_screen_size", "width": width, "height": height},
                {"$type": "set_render_quality", "render_quality": 5}  # Max quality
            ])
            
            # Add overhead camera for scene view
            self._third_person_camera = ThirdPersonCamera(
                position={"x": 0, "y": 18, "z": 0},  # Overhead view (slightly higher)
                look_at={"x": 0, "y": 0, "z": 0},
                avatar_id="overhead_cam",
                field_of_view=60  # Wider FOV to see more of the scene
            )
            
            # Capture images from the camera at full resolution
            self._image_capture = ImageCapture(
                avatar_ids=["overhead_cam"],
                path=self._recording_path,
                png=True
            )
            
            logger.info(f"Recording at {width}x{height} resolution")
            
            self._controller.add_ons.extend([self._third_person_camera, self._image_capture])
            self._controller.communicate([])  # Initialize add-ons
            
            logger.info(f"Video recording enabled: {self._recording_path}")
            
        except Exception as e:
            logger.warning(f"Failed to setup recording: {e}")
            self._recording_enabled = False
    
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset TDW environment"""
        if not self._controller:
            return
            
        # Clear add-ons (robots)
        self._controller.add_ons.clear()
        self._robots.clear()
        self._objects.clear()
        self._robot_counter = 0
        
        # Reset scene
        from tdw.tdw_utils import TDWUtils
        self._controller.communicate([
            {"$type": "destroy_all_objects"},
            TDWUtils.create_empty_room(
                width=int(self.config.scene_size[0]),
                length=int(self.config.scene_size[1])
            )
        ])
        self._sim_time = 0.0
        logger.info("TDW scene reset")
    
    def step(self) -> None:
        """
        Advance TDW physics by one step.
        
        Also processes any pending robot actions.
        """
        if not self._controller:
            return
        
        # Communicate advances the simulation (also captures frames if recording)
        self._controller.communicate([])
        
        if self._recording_enabled:
            self._frame_count += 1
        self._sim_time += self.config.time_step
        
        # Check for completed actions
        self._update_action_states()
    
    def _update_action_states(self) -> None:
        """Update pending action states for all robots"""
        from magnebot import ActionStatus
        
        for robot_id, state in self._robots.items():
            if state.pending_action and state.magnebot:
                status = state.magnebot.action.status
                if status != ActionStatus.ongoing:
                    state.pending_action = False
                    if status != ActionStatus.success:
                        logger.warning(f"Robot {robot_id} action '{state.action_type}' "
                                     f"completed with status: {status}")
    
    def spawn_robot(self,
                    robot_id: str,
                    position: Tuple[float, float, float],
                    robot_type: str = "magnebot") -> bool:
        """
        Spawn a Magnebot robot as a TDW add-on.
        
        Args:
            robot_id: Unique identifier for the robot
            position: (x, y, z) spawn position in meters
            robot_type: Robot type (only "magnebot" supported)
        
        Returns:
            True if successful
        """
        if not self._controller:
            raise RuntimeError("TDW controller not initialized")
        
        try:
            from magnebot import Magnebot
            from magnebot.image_frequency import ImageFrequency
            
            # Create Magnebot add-on
            magnebot = Magnebot(
                robot_id=self._robot_counter,
                position={"x": position[0], "y": position[1], "z": position[2]},
                image_frequency=ImageFrequency.always  # Capture images every frame
            )
            
            # Add to controller
            self._controller.add_ons.append(magnebot)
            
            # Store state
            self._robots[robot_id] = RobotState(
                magnebot=magnebot,
                pending_action=False,
                action_type=""
            )
            
            # Initialize the robot in the scene
            self._controller.communicate([])
            
            self._robot_counter += 1
            logger.info(f"Spawned Magnebot '{robot_id}' at {position}")
            return True
            
        except ImportError:
            logger.error("Magnebot not installed. Install with: pip install magnebot")
            return False
        except Exception as e:
            logger.error(f"Failed to spawn robot: {e}")
            return False
    
    def get_observation(self, robot_id: str) -> RobotObservation:
        """
        Get sensor observation from a Magnebot.
        
        Returns RGB image, depth map, position, rotation, and gripper state.
        """
        state = self._robots.get(robot_id)
        if state is None or state.magnebot is None:
            raise ValueError(f"Robot '{robot_id}' not found")
        
        magnebot = state.magnebot
        
        # Get images from Magnebot's camera
        # Images are in magnebot.dynamic.images after communicate()
        pil_images = magnebot.dynamic.get_pil_images()
        
        # If no images yet, advance one frame and retry once
        if not pil_images and self._controller:
            self._controller.communicate([])
            pil_images = magnebot.dynamic.get_pil_images()
        
        # If still no images, force a Wait action to trigger image capture
        if not pil_images and self._controller:
            try:
                from magnebot.actions.wait import Wait
                magnebot.action = Wait()
                self._controller.communicate([])
                pil_images = magnebot.dynamic.get_pil_images()
            except Exception:
                pass
        
        # RGB image (key can be "img" or "_img" depending on TDW version)
        if "img" in pil_images:
            rgb = np.array(pil_images["img"])
        elif "_img" in pil_images:
            rgb = np.array(pil_images["_img"])
        else:
            # Fallback: empty image (might happen on first frame)
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            if pil_images:
                logger.debug(f"Available image keys for {robot_id}: {list(pil_images.keys())}")
            else:
                logger.warning(f"No RGB image available for {robot_id}")
        
        # Depth values
        try:
            depth = magnebot.dynamic.get_depth_values()
        except Exception:
            depth = np.zeros((480, 640), dtype=np.float32)
        
        # Transform (position and rotation)
        transform = magnebot.dynamic.transform
        position = (
            float(transform.position[0]),
            float(transform.position[1]),
            float(transform.position[2])
        )
        rotation = (
            float(transform.rotation[0]),
            float(transform.rotation[1]),
            float(transform.rotation[2]),
            float(transform.rotation[3])
        )
        
        # Gripper state: check if holding objects
        # magnebot.dynamic.held is a dict: Arm -> list of object IDs
        holding_objects = False
        for arm, held_objects in magnebot.dynamic.held.items():
            if len(held_objects) > 0:
                holding_objects = True
                break
        
        return RobotObservation(
            robot_id=robot_id,
            timestamp=self._sim_time,
            rgb=rgb,
            depth=depth,
            position=position,
            rotation=rotation,
            velocity=(0.0, 0.0, 0.0),  # TODO: compute from position delta
            gripper_state=holding_objects
        )
    
    def send_command(self, command: RobotCommand) -> bool:
        """
        Send command to a Magnebot.
        
        Commands are non-blocking. Call step() to advance the action.
        Check robot state to see if action is complete.
        """
        state = self._robots.get(command.robot_id)
        if state is None or state.magnebot is None:
            raise ValueError(f"Robot '{command.robot_id}' not found")
        
        magnebot = state.magnebot
        
        # Don't send new commands while one is pending
        if state.pending_action:
            logger.debug(f"Robot {command.robot_id} has pending action, skipping")
            return False
        
        try:
            # Movement command
            vx, _, vz = command.linear_velocity
            if vx != 0.0 or vz != 0.0:
                # Map velocity-like action to visible movement distance.
                # VLA outputs ~0.1-0.2, scale up for 20x20m room visibility
                magnitude = float(np.hypot(vx, vz))
                distance = magnitude * 5.0  # 5m scale factor for visibility
                if vx < 0:
                    distance = -distance
                magnebot.move_by(distance=distance, arrived_at=0.3)
                state.pending_action = True
                state.action_type = f"move_by({distance:.2f})"
                logger.debug(f"[{command.robot_id}] MOVE: vx={vx:.3f} vz={vz:.3f} -> distance={distance:.3f}m")
                return True
            
            # Rotation command
            if command.angular_velocity[2] != 0.0:
                angle = command.angular_velocity[2] * 15.0  # Turn 15 deg per command
                magnebot.turn_by(angle=angle)
                state.pending_action = True
                state.action_type = f"turn_by({angle:.1f})"
                logger.debug(f"[{command.robot_id}] TURN: angle={angle:.1f}deg")
                return True
            
            # Gripper command
            if command.gripper_action is not None:
                from magnebot import Arm
                
                if command.gripper_action > 0.5:
                    # Grasp - need a target object
                    if command.arm_target is not None:
                        magnebot.grasp(target=command.arm_target, arm=Arm.right)
                        state.pending_action = True
                        state.action_type = f"grasp({command.arm_target})"
                        logger.debug(f"[{command.robot_id}] GRASP: target={command.arm_target}")
                else:
                    # Drop - only if holding something
                    held_right = magnebot.dynamic.held.get(Arm.right, [])
                    if len(held_right) > 0:
                        # Drop the first held object
                        target_obj = int(held_right[0])
                        magnebot.drop(target=target_obj, arm=Arm.right)
                        state.pending_action = True
                        state.action_type = f"drop({target_obj})"
                        logger.debug(f"[{command.robot_id}] DROP: target={target_obj}")
                return True
            
            logger.debug(f"[{command.robot_id}] NO-OP: zero velocity command")
            return True  # No-op command
            
        except Exception as e:
            logger.error(f"Command failed for {command.robot_id}: {e}")
            return False
    
    def is_robot_idle(self, robot_id: str) -> bool:
        """Check if robot has no pending actions"""
        state = self._robots.get(robot_id)
        if state is None:
            return True
        return not state.pending_action
    
    def spawn_object(self,
                     object_id: str,
                     object_type: str,
                     position: Tuple[float, float, float]) -> bool:
        """Spawn a physics object in TDW"""
        if not self._controller:
            return False
        
        try:
            from tdw.controller import Controller
            
            obj_id = self._controller.get_unique_id()
            
            # Map common names to TDW model names
            model_map = {
                "box": "iron_box",
                "crate": "wood_crate",
                "pallet": "pallet_plastic_rectangular",
                "bin": "basket_18inx18inx12iin_plastic_lattice"
            }
            model_name = model_map.get(object_type, object_type)
            
            self._controller.communicate([
                self._controller.get_add_object(
                    model_name=model_name,
                    object_id=obj_id,
                    position={"x": position[0], "y": position[1], "z": position[2]}
                )
            ])
            
            self._objects[object_id] = obj_id
            logger.info(f"Spawned object '{object_id}' ({model_name}) at {position}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn object '{object_id}': {e}")
            return False
    
    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        """Get object position from TDW"""
        # TODO: Query TDW for actual object transforms
        # For now, return None (position tracking not implemented)
        return None
    
    def close(self) -> None:
        """Close TDW connection and cleanup"""
        # Compile video if recording was enabled
        if self._recording_enabled and self._frame_count > 0:
            self._compile_video()
        
        if self._controller:
            try:
                self._controller.communicate([{"$type": "terminate"}])
            except Exception as e:
                logger.warning(f"Error during TDW shutdown: {e}")
            finally:
                self._controller = None
        
        self._robots.clear()
    
    def _compile_video(self, fps: int = 30) -> Optional[str]:
        """
        Compile captured frames into a video file.
        
        Requires ffmpeg to be installed.
        """
        import os
        import subprocess
        import glob
        
        video_path = os.path.join(self._recording_path, "simulation.mp4")
        frames_dir = os.path.join(self._recording_path, "overhead_cam")
        
        # Try both .png and .jpg patterns (TDW may save as either)
        jpg_frames = glob.glob(os.path.join(frames_dir, "img_*.jpg"))
        png_frames = glob.glob(os.path.join(frames_dir, "img_*.png"))
        
        if len(jpg_frames) > len(png_frames):
            frames_pattern = os.path.join(frames_dir, "img_%04d.jpg")
        else:
            frames_pattern = os.path.join(frames_dir, "img_%04d.png")
        
        # Check if frames exist
        if not os.path.exists(frames_dir):
            logger.warning(f"No frames found in {frames_dir}")
            return None
        
        total_frames = len(jpg_frames) + len(png_frames)
        if total_frames == 0:
            logger.warning(f"No image frames found in {frames_dir}")
            return None
        
        try:
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frames_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "23",
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Video saved: {video_path} ({self._frame_count} frames)")
                return video_path
            else:
                logger.warning(f"ffmpeg failed: {result.stderr}")
                logger.info(f"Frames saved in: {frames_dir}")
                return None
                
        except FileNotFoundError:
            logger.warning("ffmpeg not found. Install with: sudo apt install ffmpeg")
            logger.info(f"Frames saved in: {frames_dir}")
            return None
        self._objects.clear()
        self._initialized = False
        logger.info("TDW backend closed")
    
    @property
    def sim_time(self) -> float:
        return self._sim_time
    
    @property
    def backend_name(self) -> str:
        return "TDW"
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
