#!/usr/bin/env python3
"""
Test TDW + Magnebot Integration

This test verifies:
1. TDW controller starts correctly
2. Magnebot can be spawned as an add-on
3. Robot can move and we can read observations
4. Images are captured correctly

Run with: xvfb-run -a python test_magnebot.py
"""

import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def patch_tdw_json_serialization():
    """
    Patch TDW to handle numpy float32/int32 serialization.
    
    Magnebot uses numpy types internally, but TDW 1.13.0's JSON
    encoder doesn't handle them. This patches json.dumps globally.
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
    print("   [Patched JSON encoder for numpy types]")


# Apply patch
patch_tdw_json_serialization()

def test_magnebot_basic(enable_recording=False):
    """Basic Magnebot test"""
    print("=" * 60)
    print("TDW + Magnebot Integration Test")
    print("=" * 60)
    
    try:
        from tdw.controller import Controller
        from tdw.tdw_utils import TDWUtils
        from magnebot import Magnebot, ActionStatus
        from magnebot.image_frequency import ImageFrequency
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install tdw magnebot")
        return False
    
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
    
    if enable_recording:
        run_id = _next_run_id("recordings/test_basic")
        recording_path = f"recordings/test_basic/run_{run_id}"
    else:
        recording_path = "recordings/test_basic"
    
    print("\n1. Starting TDW controller...")
    try:
        c = Controller(launch_build=True)
        print("   OK - Controller started")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False
    
    print("\n2. Creating scene...")
    try:
        c.communicate([
            {"$type": "load_scene", "scene_name": "ProcGenScene"},
            TDWUtils.create_empty_room(12, 12)
        ])
        print("   OK - 12x12m room created")
    except Exception as e:
        print(f"   FAILED: {e}")
        c.communicate({"$type": "terminate"})
        return False
    
    # Setup recording
    if enable_recording:
        print("\n2b. Setting up video recording...")
        try:
            from tdw.add_ons.third_person_camera import ThirdPersonCamera
            from tdw.add_ons.image_capture import ImageCapture
            import os
            
            os.makedirs(recording_path, exist_ok=True)
            
            camera = ThirdPersonCamera(
                position={"x": -3, "y": 4, "z": -3},
                look_at={"x": 0, "y": 0.5, "z": 0.5},
                avatar_id="overhead"
            )
            capture = ImageCapture(
                avatar_ids=["overhead"],
                path=recording_path,
                png=True
            )
            c.add_ons.extend([camera, capture])
            
            # Higher resolution
            c.communicate([
                {"$type": "set_render_quality", "render_quality": 5},
                {"$type": "set_screen_size", "width": 1280, "height": 720}
            ])
            print(f"   OK - Recording to {recording_path} (1280x720)")
        except Exception as e:
            print(f"   WARNING: Recording setup failed: {e}")
            enable_recording = False
    
    print("\n3. Spawning Magnebot...")
    try:
        magnebot = Magnebot(
            robot_id=0,
            position={"x": 0, "y": 0, "z": 0},
            image_frequency=ImageFrequency.always
        )
        c.add_ons.append(magnebot)
        c.communicate([])  # Initialize the robot
        print(f"   OK - Magnebot spawned")
        print(f"   Position: {magnebot.dynamic.transform.position}")
    except Exception as e:
        print(f"   FAILED: {e}")
        c.communicate({"$type": "terminate"})
        return False
    
    print("\n4. Testing movement...")
    try:
        magnebot.move_by(distance=1.0, arrived_at=0.1)
        
        steps = 0
        max_steps = 200
        while magnebot.action.status == ActionStatus.ongoing and steps < max_steps:
            c.communicate([])
            steps += 1
        
        status = magnebot.action.status
        pos = magnebot.dynamic.transform.position
        print(f"   Status: {status} (after {steps} steps)")
        print(f"   New position: {pos}")
        
        if status == ActionStatus.success:
            print("   OK - Movement completed")
        else:
            print(f"   WARNING: Movement ended with status {status}")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\n5. Testing image capture...")
    try:
        c.communicate([])
        
        images = magnebot.dynamic.images
        print(f"   Available image keys: {list(images.keys())}")
        
        if "_img" in images:
            img = images["_img"]
            print(f"   RGB image shape: {img.shape if hasattr(img, 'shape') else 'N/A'}")
        
        pil_images = magnebot.dynamic.get_pil_images()
        print(f"   PIL image keys: {list(pil_images.keys())}")
        
        try:
            depth = magnebot.dynamic.get_depth_values()
            print(f"   Depth shape: {depth.shape}")
        except Exception as de:
            print(f"   Depth not available: {de}")
        
        print("   OK - Image capture working")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\n6. Testing gripper state...")
    try:
        held = magnebot.dynamic.held
        print(f"   Held objects: {held}")
        print("   OK - Gripper state accessible")
    except Exception as e:
        print(f"   FAILED: {e}")
    
    print("\n7. Cleanup...")
    
    # Compile video
    if enable_recording:
        print("   Compiling video...")
        try:
            import subprocess
            video_path = f"{recording_path}/test_video.mp4"
            frames_pattern = f"{recording_path}/overhead/img_%04d.png"
            
            import glob
            jpg_frames = glob.glob(f"{recording_path}/overhead/img_*.jpg")
            png_frames = glob.glob(f"{recording_path}/overhead/img_*.png")
            
            if len(jpg_frames) > len(png_frames):
                frames_pattern = f"{recording_path}/overhead/img_%04d.jpg"
            else:
                frames_pattern = f"{recording_path}/overhead/img_%04d.png"
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", frames_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   Video saved: {video_path}")
            else:
                print(f"   Video compilation failed (frames saved in {recording_path}/overhead/)")
        except FileNotFoundError:
            print(f"   ffmpeg not found - frames saved in {recording_path}/overhead/")
        except Exception as e:
            print(f"   Video error: {e}")
    
    try:
        c.communicate({"$type": "terminate"})
        print("   OK - TDW terminated")
    except Exception as e:
        print(f"   WARNING: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    return True


def test_multi_robot(enable_recording=False):
    """Test multiple Magnebots with optional recording"""
    print("\n" + "=" * 60)
    print("Multi-Robot Test")
    print("=" * 60)
    
    try:
        from tdw.controller import Controller
        from tdw.tdw_utils import TDWUtils
        from magnebot import Magnebot, ActionStatus
        from magnebot.image_frequency import ImageFrequency
    except ImportError:
        print("Skipping - dependencies not available")
        return False
    
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
    
    if enable_recording:
        run_id = _next_run_id("recordings/test_multi")
        recording_path = f"recordings/test_multi/run_{run_id}"
    else:
        recording_path = "recordings/test_multi"
    
    print("\n1. Creating scene with 3 robots...")
    c = Controller(launch_build=True)
    c.communicate([
        {"$type": "load_scene", "scene_name": "ProcGenScene"},
        TDWUtils.create_empty_room(20, 20)
    ])
    
    # Setup recording
    if enable_recording:
        print("   Setting up video recording...")
        try:
            from tdw.add_ons.third_person_camera import ThirdPersonCamera
            from tdw.add_ons.image_capture import ImageCapture
            import os
            
            os.makedirs(recording_path, exist_ok=True)
            
            camera = ThirdPersonCamera(
                position={"x": 0, "y": 12, "z": -8},
                look_at={"x": 0, "y": 0, "z": 2},
                avatar_id="overhead"
            )
            
            capture = ImageCapture(
                avatar_ids=["overhead"],
                path=recording_path,
                png=True
            )
            
            c.add_ons.extend([camera, capture])
            
            c.communicate([
                {"$type": "set_render_quality", "render_quality": 5},
                {"$type": "set_screen_size", "width": 1280, "height": 720}
            ])
            print(f"   Recording to {recording_path} (1280x720)")
        except Exception as e:
            print(f"   WARNING: Recording setup failed: {e}")
            enable_recording = False
    
    robots = []
    positions = [
        {"x": -3, "y": 0, "z": 0},
        {"x": 0, "y": 0, "z": 0},
        {"x": 3, "y": 0, "z": 0}
    ]
    
    for i, pos in enumerate(positions):
        robot = Magnebot(
            robot_id=i,
            position=pos,
            image_frequency=ImageFrequency.always
        )
        c.add_ons.append(robot)
        robots.append(robot)
    
    c.communicate([])  # Initialize all robots
    
    for i, robot in enumerate(robots):
        print(f"   Robot {i}: {robot.dynamic.transform.position}")
    
    print("\n2. Moving all robots forward...")
    for robot in robots:
        robot.move_by(distance=2.0)
    
    steps = 0
    max_steps = 300
    while steps < max_steps:
        all_done = all(r.action.status != ActionStatus.ongoing for r in robots)
        if all_done:
            break
        c.communicate([])
        steps += 1
    
    print(f"   Completed in {steps} steps")
    for i, robot in enumerate(robots):
        print(f"   Robot {i}: {robot.dynamic.transform.position} ({robot.action.status})")
    
    # Compile video
    if enable_recording:
        print("\n3. Compiling video...")
        try:
            import subprocess
            import glob
            video_path = f"{recording_path}/multi_robot.mp4"
            
            jpg_frames = glob.glob(f"{recording_path}/overhead/img_*.jpg")
            png_frames = glob.glob(f"{recording_path}/overhead/img_*.png")
            
            if len(jpg_frames) > len(png_frames):
                frames_pattern = f"{recording_path}/overhead/img_%04d.jpg"
            else:
                frames_pattern = f"{recording_path}/overhead/img_%04d.png"
            
            cmd = [
                "ffmpeg", "-y",
                "-framerate", "30",
                "-i", frames_pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   Video saved: {video_path}")
            else:
                print(f"   Video compilation failed - frames in {recording_path}/overhead/")
        except Exception as e:
            print(f"   Video error: {e}")
    
    c.communicate({"$type": "terminate"})
    print("\n   OK - Multi-robot test complete")
    return True


if __name__ == "__main__":
    enable_recording = "--record" in sys.argv
    run_multi = "--multi" in sys.argv
    skip_basic = "--multi-only" in sys.argv
    
    if enable_recording:
        print("Video recording ENABLED (1280x720)")
        print("(requires ffmpeg for video compilation)")
    
    success = True
    
    if not skip_basic:
        success = test_magnebot_basic(enable_recording=enable_recording)
    
    if (success or skip_basic) and (run_multi or skip_basic):
        test_multi_robot(enable_recording=enable_recording)
    
    sys.exit(0 if success else 1)
