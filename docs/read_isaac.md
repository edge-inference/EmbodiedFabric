# Isaac Sim Backend Setup

GPU-accelerated physics simulation backend using NVIDIA Isaac Sim.

## Prerequisites

- NVIDIA GPU (RTX recommended)
- CUDA 12.x
- Ubuntu 20.04/22.04
- Driver 525+

## Installation

### 1. Install Isaac Sim (Python)

```bash
# Create virtual environment
python3.11 -m venv venv_isaac
source venv_isaac/bin/activate

# Install Isaac Sim packages
pip install isaacsim-rl isaacsim-robot isaacsim
```

### 2. Download Assets

Choose asset pack based on your needs:

```bash
# Option A: Minimal (Robots only, ~5GB)
bash scripts/download/isaac_assets.sh minimal

# Option B: Standard (Robots + Environments, ~15GB) - Recommended
bash scripts/download/isaac_assets.sh standard

# Option C: Complete (Everything, ~50GB+)
bash scripts/download/isaac_assets.sh complete
```

Assets are installed to: `~/.local/share/ov/pkg/isaac-sim-5.1.0/isaac-sim-assets`

### 3. Set Environment Variables

```bash
# Add to ~/.bashrc or .env
export ISAAC_ASSETS_PATH="$HOME/.local/share/ov/pkg/isaac-sim-5.1.0/isaac-sim-assets"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/venv_isaac/lib/python3.11/site-packages/isaacsim/exts"
```

## Usage

### Test Installation

```bash
source venv_isaac/bin/activate
python examples/isaac_sim_test.py
```

### Run with Isaac Backend

```python
from simulator.core import PhysicsSimulator

config = SimulatorConfig(
    backend="isaac_sim",
    scene_name="warehouse",
    enable_recording=True
)

sim = PhysicsSimulator(config)
sim.initialize()

# Spawn robot
sim.spawn_robot("robot_1", position=(0, 0, 0), robot_type="fetch")

# Run simulation
for step in range(1000):
    obs = sim.get_observation("robot_1")
    cmd = agent.act(obs)
    sim.send_command(cmd)
    sim.step()
```

## Asset Packs Overview

| Pack | Size | Contains | Use Case |
|------|------|----------|----------|
| **Minimal** | ~5GB | Robots (Fetch, Franka, etc.) | Robot manipulation only |
| **Standard** | ~15GB | + Environments, Props | General warehouse/indoor |
| **Complete** | ~50GB+ | Everything | Production/full scenes |

## Available Assets (Standard Pack)

### Robots
- Fetch Mobile Manipulator
- Franka Emika Panda
- UR5/UR10
- Jetbot
- Carter warehouse robot

### Environments
- Simple Warehouse
- Office environments
- Hospital
- Full warehouse scenes

### Props & Objects
- Bins, pallets, shelves
- Common manipulation objects
- Materials library

## Troubleshooting

### Import Error: `isaacsim` not found

```bash
# Ensure correct venv is active
source venv_isaac/bin/activate
which python  # Should point to venv_isaac

# Reinstall if needed
pip install --force-reinstall isaacsim-rl
```

### Assets not loading

```bash
# Check environment variable
echo $ISAAC_ASSETS_PATH

# Verify assets exist
ls $ISAAC_ASSETS_PATH/Isaac/Robots/Fetch/

# If missing, rerun download script
bash scripts/download/isaac_assets.sh standard
```

### GPU/CUDA Issues

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Required: Driver 525+, CUDA 12.x
```

### Headless Mode Issues

If running headless (no display):

```bash
# Use Xvfb virtual display
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99

python examples/isaac_sim_test.py
```

## Performance Notes

- **GPU Memory**: Fetch robot + simple scene ~2-3GB VRAM
- **Rendering**: RTX GPUs get ray-traced lighting
- **Physics**: GPU PhysX for faster-than-realtime if needed
- **Headless**: 60+ FPS typical on RTX 3090/4090

## Next Steps

1. Download assets (start with `standard`)
2. Run test script to verify setup
3. Integrate with your VLA agents
4. Enable recording for training data collection

## References

- [Isaac Sim Docs](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [Asset Downloads](https://developer.nvidia.com/isaac-sim)
- [Python API](https://docs.omniverse.nvidia.com/py/isaacsim/)
