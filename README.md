# PhysicAI

Simulation framework for hardware-software co-design of embodied AI systems.

## Research Problems

- **R1**: Context-Memory Fabric - How should context be structured and shared for multi-robot VLA coordination? A venture for System 3: S3
- **R2**: Sensor Edge Processing - Can we distribute compute across sensors to lower E2E latency?
- **R3**: Hardware Co-Design - How to co-design hardware and software targeting uAgent SoC tapeout?

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
│   ├── backend/  (isaac/)
│   ├── robot/
│   ├── vla/
│   ├── coordination/
│   └── scenarios/
├── hardware/     (C++ preprocessing testbench)
├── tests/
├── workload/
├── examples/
├── patches/      (extern patches for repos not tracked as submodules)
└── extern/       (submodules + patched repos)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/edge-inference/physicai.git
cd physicai
```

### Submodules (tracked)

| Path | Repo |
|------|------|
| `extern/Isaac-GR00T` | NVIDIA/Isaac-GR00T (n1.5 tag) |
| `extern/IsaacLab` | isaac-sim/IsaacLab |
| `extern/lerobot` | huggingface/lerobot |
| `extern/leisaac` | bkubwimana/leisaac |

### Patched repos (not submodules)

See [patches/README.md](patches/README.md) for setup instructions.

| Path | Repo | Patch |
|------|------|-------|
| `extern/IsaacLab-Arena` | isaac-sim/IsaacLab-Arena @ `755e8cf` | G1 runner, teleop, WBC |

### Install deps

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

## Quick Start

```bash
# Unit tests
python -m pytest -q

# SmolVLA SO101 evaluation (requires Isaac Sim)
bash scripts/eval/leisaac_so101_smolvla.sh

# Contract demo (uses config/contracts.json)
python examples/contracts.py
```

## Isaac Sim backend

Install Isaac Sim from NVIDIA docs: [Installation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/index.html).

```bash
python -c "import isaacsim; print('isaacsim OK')"
```

## Robot eval scripts (run in ubuntu desktop with an NVIDIA GPU having RT cores)

See `scripts/setup/` and `scripts/eval/` for SmolVLA / Isaac Lab runs.

## Key Concepts

### 1. VLA Integration

Every robot runs a VLA model:

```python
observation = robot.get_sensors()        # RGB, depth, proprioception
context = dsm.get_shared_context()       # From other robots (R1)
instruction = task.get_instruction()     # Natural language

action = vla.predict(observation, context, instruction)
robot.execute(action)
```

### 2. Formal Contracts

Contracts specify requirements that flow to hardware:

```python
TimingContract.control_loop_contract(
    sensing_latency_ms=15.0,
    computation_latency_ms=25.0,  # VLA inference budget
    actuation_latency_ms=10.0,
    loop_period_ms=50.0           # 20Hz control
)
# -> NPU must do VLA inference in (time)
# -> Need (#) FLOPS for (#) param model at 4-bit
```

### 3. DSM Coordination (R1)

Gossip-based state sharing for eventual consistency:

```python
dsm.write_agent_state(robot_id, position, state, task)
dsm.gossip_round(agents)
context = dsm.get_nearby_agents(position, radius=5.0)
```

### 4. Workload Profiling (R2/R3)

Simulation metrics feed hardware sizing:

```python
profiler.start_recording()
sim.run(steps=1000)
profiler.stop_recording()

metrics = profiler.get_metrics()
recommendations = profiler.get_hardware_recommendations()
```

## Hardware Output (R3)

Simulation produces specifications for uAgent SoC:

```
Compute:
  NPU: 5 TOPS (VLA inference in 25ms)
  Memory: 8GB LPDDR5 (5GB weights + buffers)

Sensors:
  Camera: 640x480 @ 30Hz
  Depth: 640x480 @ 30Hz

Communication:
  WiFi 6: 100 Mbps for sensors
  Gossip: 50 Kbps per robot pair
```

## References

- [Lingua Franca](https://www.lf-lang.org/) - Deterministic coordination
- [CHASE](https://github.com/chase-cps/core-library) - Contract-based CPS design
