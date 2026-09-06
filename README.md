# EmbodiedFabric:

Simulation framework for hardware-software co-design of embodied AI systems.

## Research Problems

- **R1**: Context-Memory Fabric - How should context be structured and shared for multi-robot VLA coordination?
- **R2**: Sensor Edge Processing - When should embodied agents reuse, communicate, or recompute context across sensors, robots, and edge accelerators to meet latency, energy, and reliability constraints?
- **R3**: Hardware Co-Design - What runtime and architectural mechanisms can enforce these guarantees efficiently, and which functions should be implemented or accelerated in hardware?

## Architecture

```
physicai/
├── config/
├── contracts/
│   ├── sim/
│   └── codesign/
├── scripts/
├── simulator/
│   ├── core.py
│   ├── backend/  (tdw/, isaac/)
│   ├── robot/
│   ├── vla/
│   ├── coordination/
│   └── scenarios/
├── tests/
├── workload/
├── examples/
└── extern/  (submodules)
```

## Download / submodules

```bash
git submodule update --init --recursive
```

Submodules:

| Path | Repo |
|------|------|
| `extern/CogACT` | microsoft/CogACT |
| `extern/visualnav-transformer` | robodhruv/visualnav-transformer |
| `extern/CoELA` | UMass-Embodied-AGI/CoELA |
| `extern/Isaac-GR00T` | NVIDIA/Isaac-GR00T (n1.5 tag) |
| `extern/Isaac-GR00T-n1.6` | NVIDIA/Isaac-GR00T (main / n1.6) |
| `extern/IsaacLab` | isaac-sim/IsaacLab |
| `extern/IsaacLab-Arena` | isaac-sim/IsaacLab-Arena (release/0.1.1) |
| `extern/lerobot` | huggingface/lerobot |

# Install deps
pip install -r requirements.txt -r requirements-dev.txt

## Quick Start

```bash
pip install tdw magnebot
# Then change backend="tdw" in config

# Unit tests
python -m pytest -q

cd physicai
CUDA_VISIBLE_DEVICES=1 xvfb-run -a python examples/floorplan_4zone.py --robots 2


# Contract demo (uses config/contracts.json)
python examples/contracts.py
```

## Isaac Sim backend

Install Isaac Sim from NVIDIA docs: [Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html).


```bash
python -c "import isaacsim; print('isaacsim OK')"
```

## Robot eval scripts (run in ubuntu desktop with an NVIDIA GPU having RT cores)

See `scripts/setup/` and `scripts/eval/` for GR00T / Isaac Lab Arena runs.

## Key Concepts

### 1. VLA Integration

Every robot runs a VLA model:

```python
# VLA pipeline
observation = robot.get_sensors()        # RGB, depth, proprioception
context = dsm.get_shared_context()       # From other robots (R1)
instruction = task.get_instruction()     # Natural language

action = vla.predict(observation, context, instruction)
robot.execute(action)
```

### 2. Formal Contracts

Contracts specify requirements that flow to hardware:

```python
# Example Timing contract
TimingContract.control_loop_contract(
    sensing_latency_ms=15.0,
    computation_latency_ms=25.0,  # VLA inference budget
    actuation_latency_ms=10.0,
    loop_period_ms=50.0           # 20Hz control
)
# → NPU must do VLA inference in (time)
# → Need (#) FLOPS for (#) param model at 4-bit
```

### 3. DSM Coordination (R1)

Gossip-based state sharing for eventual consistency:

```python
# Each robot writes its state
dsm.write_agent_state(robot_id, position, state, task)

# Gossip propagates to neighbors
dsm.gossip_round(agents)

# VLA reads context for decisions
context = dsm.get_nearby_agents(position, radius=5.0)
```

### 4. Workload Profiling (R2/R3)

Simulation metrics feed hardware sizing:

```python
profiler.start_recording()
sim.run(steps=1000)
profiler.stop_recording()

metrics = profiler.get_metrics()
# → VLA FLOPS, latency, sensor bandwidth

recommendations = profiler.get_hardware_recommendations()
# → NPU TOPS, memory, platform recommendation
```

## Scaling Modes

| Mode | Physics Robots | Use Case |
|------|---------------|----------|
| FULL_PHYSICS | All | workload profiling (5-10 robots) |

## Hardware Output (R3)

Simulation produces specifications for μAgent SoC:

Example
```
Compute:
  NPU: 5 TOPS (VLA inference in 25ms)
  Memory: 8GB LPDDR5 (5GB weights + buffers)

Sensors:
  Camera: 640x480 @ 30Hz, < x latency
  Depth: 640x480 @ 30Hz

Communication:
  WiFi 6: 100 Mbps for sensors
  Gossip: 50 Kbps per robot pair

Power:
  Compute: xW TDP
  Total robot: xW
```

## References

- [ThreeDWorld (TDW)](https://threedworld.org/) - Physics simulation
- [Lingua Franca](https://www.lf-lang.org/) - Deterministic coordination
- [CHASE](https://github.com/chase-cps/core-library) - Contract-based CPS design
- [CogACT](https://github.com/microsoft/CogACT) - DiT-based manipulation policy (action chunking)
- [NoMaD / ViNT](https://github.com/robodhruv/visualnav-transformer) - Diffusion-based navigation policy
