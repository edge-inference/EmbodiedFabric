"""
TDW Backend

Physics simulation via ThreeDWorld + Magnebot.
https://threedworld.org/
"""

from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import logging
import json

from .base import PhysicsBackend, RobotObservation, RobotCommand

logger = logging.getLogger(__name__)


def _patch_json_for_numpy():
    """Patch JSON encoder for numpy types (TDW 1.13 compat)."""
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


COLLISION_RECOVERY_THRESHOLD = 2
RECOVERY_TURN_RANGE = (30, 120)


@dataclass
class RobotState:
    """Track robot's pending action state"""
    magnebot: Any = None
    pending_action: bool = False
    action_type: str = ""
    consecutive_collisions: int = 0
    recovering: bool = False


class TDWBackend(PhysicsBackend):
    """TDW physics backend with Magnebot."""
    
    def __init__(self, config, enable_recording: bool = False, recording_path: str = None,
                 launch_build: bool = True, tdw_address: str = "localhost", tdw_port: int = 1071):
        self.config = config
        self._controller = None
        self._robots: Dict[str, RobotState] = {}
        self._objects: Dict[str, int] = {}
        self._sim_time = 0.0
        self._initialized = False
        self._robot_counter = 0
        
        # TDW connection settings
        self._launch_build = launch_build
        self._tdw_address = tdw_address
        self._tdw_port = tdw_port
        
        # Video recording
        self._recording_enabled = enable_recording
        self._recording_path = recording_path or "recordings"
        self._third_person_camera = None
        self._image_capture = None
        self._frame_count = 0
        self._capture_interval = 10  # Only capture every N steps
    
    def initialize(self) -> bool:
        """Initialize TDW controller and scene"""
        try:
            from tdw.controller import Controller
            from tdw.tdw_utils import TDWUtils
            
            if self._launch_build:
                logger.info("Starting TDW controller (launching build)...")
                self._controller = Controller(launch_build=True)
            else:
                logger.info(f"Connecting to existing TDW build at {self._tdw_address}:{self._tdw_port}...")
                self._controller = Controller(
                    launch_build=False,
                    address=self._tdw_address,
                    port=self._tdw_port
                )
            
            # Reset scene state (clear from previous run)
            self._controller.communicate([
                {"$type": "set_target_framerate", "framerate": -1},
                {"$type": "destroy_all_objects"}
            ])
            self._controller.add_ons.clear()
            
            scene_name = getattr(self.config, 'scene_name', None)
            floorplan_layout = getattr(self.config, "floorplan_layout", None)
            
            if scene_name and str(scene_name).startswith("floorplan") and floorplan_layout is not None:
                from tdw.add_ons.floorplan import Floorplan
                floorplan = Floorplan()
                floorplan.init_scene(scene=str(scene_name), layout=int(floorplan_layout))
                self._controller.add_ons.append(floorplan)
                commands = [
                    {"$type": "set_time_step", "time_step": self.config.time_step},
                    {"$type": "set_floorplan_roof", "show": False}
                ]
            elif scene_name:
                logger.info(f"Loading pre-built scene: {scene_name}")
                commands = [
                    self._controller.get_add_scene(scene_name=scene_name),
                    {"$type": "set_time_step", "time_step": self.config.time_step}
                ]
                if str(scene_name).startswith("floorplan"):
                    commands.append({"$type": "set_floorplan_roof", "show": False})
            else:
                w = int(self.config.scene_size[0])
                h = int(self.config.scene_size[1])
                commands = [
                    {"$type": "load_scene", "scene_name": "ProcGenScene"},
                    TDWUtils.create_empty_room(width=w, length=h),
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
            
            # Create a numbered run directory
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
            
            self._controller.communicate([
                {"$type": "set_screen_size", "width": width, "height": height},
                {"$type": "set_render_quality", "render_quality": 2},
                {"$type": "set_post_process", "value": False},
                {"$type": "set_shadow_strength", "strength": 0.5}
            ])
            
            scene_name = getattr(self.config, "scene_name", "") or ""
            if scene_name.startswith("floorplan"):
                # Angled view: rotated left, tilted back, same height
                camera_pos = {"x": -8, "y": 12, "z": -12}
                camera_look = {"x": 1, "y": 0, "z": 2}
                fov = 68
            else:
                camera_pos = {"x": -5, "y": 25, "z": -8}
                camera_look = {"x": 0, "y": 1.5, "z": 0}
                fov = 60

            self._third_person_camera = ThirdPersonCamera(
                position=camera_pos,
                look_at=camera_look,
                avatar_id="overhead_cam",
                field_of_view=fov
            )
            
            self._image_capture = ImageCapture(
                avatar_ids=["overhead_cam"],
                path=self._recording_path,
                png=False
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
        
        # Ensure robot camera sensors stay enabled (safety against overrides)
        commands = []
        for state in self._robots.values():
            if state.magnebot and state.magnebot.static:
                commands.append({
                    "$type": "enable_image_sensor",
                    "enable": True,
                    "avatar_id": state.magnebot.static.avatar_id
                })
        
        self._controller.communicate(commands)
        
        if self._recording_enabled:
            self._frame_count += 1
        self._sim_time += self.config.time_step
        
        # Check for completed actions
        self._update_action_states()
    
    def _update_action_states(self) -> None:
        """Update pending action states and track collisions for recovery."""
        from magnebot import ActionStatus
        
        for robot_id, state in self._robots.items():
            if state.pending_action and state.magnebot:
                status = state.magnebot.action.status
                if status != ActionStatus.ongoing:
                    state.pending_action = False
                    if status in (ActionStatus.collision, ActionStatus.failed_to_move):
                        state.consecutive_collisions += 1
                        logger.warning(
                            f"Robot {robot_id} action '{state.action_type}' "
                            f"{status.name} #{state.consecutive_collisions}")
                    elif status == ActionStatus.success:
                        state.consecutive_collisions = 0
                    else:
                        logger.warning(
                            f"Robot {robot_id} action '{state.action_type}' "
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
            
            # Disable Magnebot's built-in "don't repeat after collision" check
            # so we can handle recovery ourselves with forced turns
            magnebot.collision_detection.previous_was_same = False
            
            self._robots[robot_id] = RobotState(
                magnebot=magnebot,
                pending_action=False,
                action_type=""
            )
            
            # Initialize the robot in the scene
            self._controller.communicate([])
            
            # Register Magnebot camera with ImageCapture so images are
            # requested every frame (ImageCapture overrides send_images globally)
            if self._image_capture is not None and magnebot.static is not None:
                avatar_id = magnebot.static.avatar_id
                if avatar_id not in self._image_capture.avatar_ids:
                    self._image_capture.avatar_ids.append(avatar_id)
                    logger.info(f"Registered avatar '{avatar_id}' with ImageCapture")
            
            self._robot_counter += 1
            logger.info(f"Spawned Magnebot '{robot_id}' at {position}")
            logger.debug(f"[{robot_id}] image_frequency={magnebot.image_frequency}")
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
        rgb = None
        
        try:
            pil_images = magnebot.dynamic.get_pil_images()
            if pil_images:
                # Standard key is "img" (pass mask "_img" -> stored as "img")
                for key in ["img", "_img"]:
                    if key in pil_images and pil_images[key] is not None:
                        rgb = np.array(pil_images[key])
                        break
                if rgb is None:
                    first_key = next(iter(pil_images))
                    if pil_images[first_key] is not None:
                        rgb = np.array(pil_images[first_key])
            else:
                logger.debug(f"[{robot_id}] pil_images empty, raw keys: "
                             f"{list(magnebot.dynamic.images.keys()) if magnebot.dynamic.images else 'None'}")
        except Exception as e:
            logger.debug(f"[{robot_id}] get_pil_images error: {e}")
        
        if rgb is None:
            rgb = np.zeros((256, 256, 3), dtype=np.uint8)
            if self._sim_time < 1.0 or int(self._sim_time * 10) % 50 == 0:
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
        
        Supports two control modes:
        - HIGH_LEVEL: Abstract commands (move_by, turn_by, grasp)
        - LOW_LEVEL: Direct joint velocity/position control
        
        Commands are non-blocking. Call step() to advance the action.
        """
        from .base import ControlMode
        
        state = self._robots.get(command.robot_id)
        if state is None or state.magnebot is None:
            raise ValueError(f"Robot '{command.robot_id}' not found")
        
        magnebot = state.magnebot
        
        # Don't send new commands while one is pending (high-level mode)
        if state.pending_action and command.control_mode == ControlMode.HIGH_LEVEL:
            logger.debug(f"Robot {command.robot_id} has pending action, skipping")
            return False
        
        try:
            if command.control_mode == ControlMode.LOW_LEVEL:
                return self._send_low_level_command(command, state, magnebot)
            else:
                return self._send_high_level_command(command, state, magnebot)
            
        except Exception as e:
            logger.error(f"Command failed for {command.robot_id}: {e}")
            return False

    def _send_low_level_command(self, command: RobotCommand, state: RobotState, magnebot) -> bool:
        """Send low-level joint control commands (for LeRobot VLAs)."""
        if command.joint_velocities is not None:
            joint_vels = command.joint_velocities
            
            forward = command.linear_velocity[0]
            turn = command.angular_velocity[2]
            
            # Collision recovery: after repeated collisions, force a random turn
            # to give the VLA a new viewpoint (AFI-inspired proprioception check)
            if state.consecutive_collisions >= COLLISION_RECOVERY_THRESHOLD:
                import random
                angle = random.uniform(*RECOVERY_TURN_RANGE)
                if random.random() < 0.5:
                    angle = -angle
                magnebot.turn_by(angle=angle)
                state.pending_action = True
                state.recovering = True
                state.action_type = f"recovery_turn({angle:.0f})"
                logger.info(f"[{command.robot_id}] COLLISION RECOVERY: "
                            f"turning {angle:.0f}deg after {state.consecutive_collisions} collisions")
                state.consecutive_collisions = 0
                return True
            
            if abs(forward) > 0.01:
                distance = forward * 3.0
                magnebot.move_by(distance=distance, arrived_at=0.05)
                state.pending_action = True
                state.action_type = f"low_level_move({forward:.3f})"
                logger.debug(f"[{command.robot_id}] LOW_LEVEL move: forward={forward:.3f} -> dist={distance:.3f}")
            elif abs(turn) > 0.01:
                angle = turn * 30.0
                magnebot.turn_by(angle=angle)
                state.pending_action = True
                state.action_type = f"low_level_turn({turn:.3f})"
                logger.debug(f"[{command.robot_id}] LOW_LEVEL turn: turn={turn:.3f} -> angle={angle:.1f}")
            
            # Arm joint control (if available)
            if len(joint_vels) > 2:
                arm_vels = joint_vels[2:6] if len(joint_vels) >= 6 else joint_vels[2:]
                # Store for potential IK-based control
                logger.debug(f"[{command.robot_id}] LOW_LEVEL arm: {arm_vels}")
            
            # Gripper control
            if len(joint_vels) > 6:
                gripper_vel = float(joint_vels[6])
                if gripper_vel > 0.3:
                    self._try_grasp(magnebot, state, command.robot_id)
                elif gripper_vel < -0.3:
                    self._try_drop(magnebot, state, command.robot_id)
            
            return True
        
        elif command.joint_positions is not None:
            # Position control
            joint_pos = command.joint_positions
            logger.debug(f"[{command.robot_id}] LOW_LEVEL joint positions: {joint_pos[:4]}...")
            
            # For now, convert to velocity-like increments
            # Full IK control would require custom TDW commands
            return True
        
        return False

    def _send_high_level_command(self, command: RobotCommand, state: RobotState, magnebot) -> bool:
        """Send high-level abstract commands (original behavior)."""
        vx, _, vz = command.linear_velocity
        if vx != 0.0 or vz != 0.0:
            magnitude = float(np.hypot(vx, vz))
            distance = magnitude * 5.0
            if vx < 0:
                distance = -distance
            magnebot.move_by(distance=distance, arrived_at=0.3)
            state.pending_action = True
            state.action_type = f"move_by({distance:.2f})"
            logger.debug(f"[{command.robot_id}] MOVE: vx={vx:.3f} vz={vz:.3f} -> distance={distance:.3f}m")
            return True
        
        if command.angular_velocity[2] != 0.0:
            angle = command.angular_velocity[2] * 15.0
            magnebot.turn_by(angle=angle)
            state.pending_action = True
            state.action_type = f"turn_by({angle:.1f})"
            logger.debug(f"[{command.robot_id}] TURN: angle={angle:.1f}deg")
            return True
        
        if command.gripper_action is not None:
            if command.gripper_action > 0.5:
                if command.arm_target is not None:
                    self._try_grasp(magnebot, state, command.robot_id, command.arm_target)
            else:
                self._try_drop(magnebot, state, command.robot_id)
            return True
        
        logger.debug(f"[{command.robot_id}] NO-OP: zero velocity command")
        return True

    def _try_grasp(self, magnebot, state: RobotState, robot_id: str, target=None) -> None:
        """Attempt to grasp nearest object or specified target."""
        from magnebot import Arm
        
        if target is not None:
            magnebot.grasp(target=target, arm=Arm.right)
            state.pending_action = True
            state.action_type = f"grasp({target})"
            logger.debug(f"[{robot_id}] GRASP: target={target}")
        else:
            # Find nearest graspable object
            nearest_obj = self._find_nearest_object(magnebot)
            if nearest_obj is not None:
                magnebot.grasp(target=nearest_obj, arm=Arm.right)
                state.pending_action = True
                state.action_type = f"grasp({nearest_obj})"
                logger.debug(f"[{robot_id}] GRASP: nearest={nearest_obj}")

    def _try_drop(self, magnebot, state: RobotState, robot_id: str) -> None:
        """Drop held object if any."""
        from magnebot import Arm
        
        held_right = magnebot.dynamic.held.get(Arm.right, [])
        if len(held_right) > 0:
            target_obj = int(held_right[0])
            magnebot.drop(target=target_obj, arm=Arm.right)
            state.pending_action = True
            state.action_type = f"drop({target_obj})"
            logger.debug(f"[{robot_id}] DROP: target={target_obj}")

    def _find_nearest_object(self, magnebot) -> Optional[int]:
        """Find nearest graspable object using a single communicate call."""
        if not self._objects or not self._controller:
            return None
        
        try:
            from tdw.output_data import OutputData, Transforms
            
            robot_pos = magnebot.dynamic.transform.position
            resp = self._controller.communicate([
                {"$type": "send_transforms", "frequency": "once"}
            ])
            
            obj_positions = {}
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                if r_id == "tran":
                    transforms = Transforms(resp[i])
                    for j in range(transforms.get_num()):
                        obj_positions[transforms.get_id(j)] = transforms.get_position(j)
            
            nearest_dist = float('inf')
            nearest_obj = None
            for obj_name, obj_id in self._objects.items():
                if obj_id in obj_positions:
                    pos = obj_positions[obj_id]
                    dist = np.sqrt(
                        (robot_pos[0] - pos[0])**2 +
                        (robot_pos[2] - pos[2])**2
                    )
                    if dist < nearest_dist and dist < 2.0:
                        nearest_dist = dist
                        nearest_obj = obj_id
            
            return nearest_obj
        except Exception as e:
            logger.debug(f"_find_nearest_object error: {e}")
            return None
    
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
            obj_id = self._controller.get_unique_id()
            spawn_pos = {
                "x": float(position[0]), 
                "y": float(position[1]), 
                "z": float(position[2])
            }
            
            cmd = self._controller.get_add_object(
                model_name="iron_box",
                object_id=obj_id,
                position=spawn_pos,
                library="models_core.json"
            )
            self._controller.communicate([cmd])
            self._objects[object_id] = obj_id
            logger.info(f"Spawned '{object_id}' at {spawn_pos}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to spawn object '{object_id}': {e}")
            return False
    
    def get_object_position(self, object_id: str) -> Optional[Tuple[float, float, float]]:
        """Get object position from TDW"""
        if not self._controller or object_id not in self._objects:
            return None
        
        try:
            from tdw.output_data import OutputData, Transforms
            
            obj_id = self._objects[object_id]
            resp = self._controller.communicate([
                {"$type": "send_transforms", "frequency": "once"}
            ])
            
            for i in range(len(resp) - 1):
                r_id = OutputData.get_data_type_id(resp[i])
                if r_id == "tran":
                    transforms = Transforms(resp[i])
                    for j in range(transforms.get_num()):
                        if transforms.get_id(j) == obj_id:
                            pos = transforms.get_position(j)
                            return (float(pos[0]), float(pos[1]), float(pos[2]))
            return None
            
        except Exception as e:
            logger.debug(f"Failed to get position for '{object_id}': {e}")
            return None
    
    def close(self) -> None:
        """Close TDW connection and cleanup"""
        # Compile video if recording was enabled
        if self._recording_enabled and self._frame_count > 0:
            self._compile_video()
        
        if self._controller:
            try:
                if self._launch_build:
                    # Only terminate if we launched the build
                    self._controller.communicate([{"$type": "terminate"}])
                else:
                    # Just disconnect, leave build running
                    logger.info("Disconnecting from TDW (build stays running)")
            except Exception as e:
                logger.warning(f"Error during TDW shutdown: {e}")
            finally:
                self._controller = None
        
        self._robots.clear()
        self._objects.clear()
        self._initialized = False
        logger.info("TDW backend closed")
    
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
    
    @property
    def sim_time(self) -> float:
        return self._sim_time
    
    @property
    def backend_name(self) -> str:
        return "TDW"
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
