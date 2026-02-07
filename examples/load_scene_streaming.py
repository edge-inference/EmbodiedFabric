#!/usr/bin/env python3
"""
Simple script to load a scene into the running Isaac Sim streaming server
"""

from isaacsim import SimulationApp

# Connect to the running streaming instance
simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# Create a world with simple scene
world = World()
world.scene.add_default_ground_plane()

# Add a simple cube
cube = DynamicCuboid(
    prim_path="/World/Cube",
    name="cube",
    position=(0, 0, 1.0),
    size=0.5,
    color=(1.0, 0.0, 0.0)
)

world.reset()

print("Scene loaded! You should see a red cube in the streaming window.")
print("Press Ctrl+C to exit...")

# Keep running
try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    pass

simulation_app.close()
