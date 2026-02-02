# PhysicAI:

Simulation framework for hardware-software co-design of embodied AI systems.

## Research Problems

- **R1**: Context-Memory Fabric - How should context be structured and shared for multi-robot VLA coordination?
- **R2**: Sensor Edge Processing - Can we distribute compute across sensors to lower E2E latency?
- **R3**: Hardware Co-Design - How to co-design hardware and software targeting μAgent SoC tapeout?

## Architecture

```
physicai/
├── contracts/                 # Formal A/G contracts for CPS design
│   ├── formal/               # Assume-Guarantee contracts
│   │   ├── contract.py       # Base contract definitions
│   │   ├── timing.py         # Real-time timing contracts
│   │   ├── safety.py         # Safety contracts
│   │   ├── component.py      # Component specifications
│   │   └── composition.py    # Contract composition rules
│   └── ...
│
├── simulator/                 # Unified simulator
│   ├── core.py               # Main simulator orchestration
│   │
│   ├── backend/              # Physics engines
│   │   ├── base.py           # Backend interface
│   │   ├── tdw_backend.py    # TDW (ThreeDWorld) integration
│   │
│   ├── robot/                # Robot implementations
│   │   ├── base.py           # Robot interface
│   │   ├── vla_agent.py      # VLA-driven robot (full pipeline)
│   │
│   ├── vla/                  # Vision-Language-Action models
│   │   ├── interface.py      # VLA contract
│   │   └── profiled_vla.py   # Profiling wrapper
│   │
│   ├── coordination/         # Multi-robot coordination (from R1)
│   │   ├── dsm.py            # Distributed Shared Memory + gossip
│   │   ├── fleet.py          # Task allocation
│   │   └── task.py           # Task definitions
│   │
│   └── scenarios/            # Test scenarios
│
├── workload/                  # Profiling for hardware sizing (R2/R3)
│   └── profiler.py           # Workload metrics collection
│
└── examples/                  # Usage examples
    └── floorplan_4zone.py  # demo
```

## Quick Start

```bash
# Run with TDW (requires install)
pip install tdw magnebot
# Then change backend="tdw" in config

cd physicai
CUDA_VISIBLE_DEVICES=1 xvfb-run -a python examples/floorplan_4zone.py --robots 4

```

## Key Concepts

### 1. VLA Integration

Every robot runs a VLA (Vision-Language-Action) model:

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
