#!/usr/bin/env python3
"""
TDW Installation Test

From: https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/pc.md

This will:
1. Download TDW build (~2GB on first run)
2. Launch windowed Unity application
3. Create a room with an object
4. Capture an image
5. Save to disk and terminate
"""

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.backend.paths import EXAMPLE_CONTROLLER_OUTPUT_PATH

camera = ThirdPersonCamera(
    position={"x": 2, "y": 1.6, "z": -0.6},
    avatar_id="a",
    look_at={"x": 0, "y": 0, "z": 0}
)

path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_capture")
print(f"Images will be saved to: {path}")

capture = ImageCapture(avatar_ids=["a"], path=path)

print("Starting TDW controller (may download build on first run)...")
c = Controller()
c.add_ons.extend([camera, capture])

object_id = c.get_unique_id()
commands = [TDWUtils.create_empty_room(12, 12)]
commands.extend(c.get_add_physics_object(
    model_name="iron_box",
    position={"x": 0, "y": 0, "z": 0},
    object_id=object_id
))

print("Creating scene with iron_box...")
c.communicate(commands)

print("Terminating...")
c.communicate({"$type": "terminate"})

print(f"\nSuccess! Check image at: {path}")
