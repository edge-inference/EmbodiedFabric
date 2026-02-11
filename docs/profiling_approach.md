# VLA Inference Profiling Approach

## Goal

Profile the VLA inference pipeline as it would run on a real robot's edge SoC.
Simulation overhead (physics, rendering) is excluded -- we only measure the
neural network forward pass and its surrounding data movement.

## Target Models

| Model | Params | Architecture | Action Generation |
|---|---|---|---|
| GR00T N1.5 | 3B | Eagle VL backbone + Flow Matching DiT | Iterative denoising (N steps) |
| SmolVLA | 450M | SmolVLM backbone + Flow Matching | Single-pass flow matching |

## Inference Pipeline (what runs on the SoC)

```
Camera RGB (512x512)
  |
  v
[Preprocessing] -- resize, normalize, tokenize language instruction
  |
  v
[Vision-Language Backbone] -- image encoding + language fusion
  |                           (Eagle for GR00T, SmolVLM for SmolVLA)
  |
  v
[Action Head] -- flow matching / DiT denoising
  |               produces action chunk (e.g., 16 future joint targets)
  |
  v
[Postprocessing] -- denormalize, remap joints
  |
  v
Joint targets --> WBC (runs at 500-1000 Hz, <1ms, separate from VLA)
```

The VLA runs once per action chunk (every ~1-2 seconds). The WBC interpolates
between chunks at high frequency. For SoC design, the VLA forward pass is the
target workload.

## Profiling Layers

### Layer 1: Pipeline Timing (torch.cuda.Event)

Instrument `get_action_chunk()` with CUDA event timers around each stage.
Gives per-call wall-clock latency with no framework overhead.

Metrics captured:
- Total inference latency (ms)
- Backbone latency (ms)
- Action head latency (ms)
- Preprocessing latency (ms)
- Action chunk frequency (Hz)

### Layer 2: PyTorch Profiler (torch.profiler)

Wraps the forward pass to capture operator-level breakdown.
Exports Chrome trace for visualization.

Metrics captured:
- Per-operator CUDA kernel time
- Memory allocation timeline
- GPU utilization per operator
- Attention vs MLP vs convolution split

### Layer 3: NVIDIA Nsight Systems (nsys)

Full system trace from command line. Gold standard for HW-SW co-design.

```bash
nsys profile --trace=cuda,nvtx --output=groot_profile \
  python isaaclab_arena/examples/policy_runner.py ...
```

Metrics captured:
- CUDA kernel occupancy and throughput
- Memory bandwidth utilization (HBM, L2)
- CPU-GPU transfer overhead
- Pipeline bubbles between stages
- SM utilization percentage

### Layer 4: Memory Profiling

```python
torch.cuda.max_memory_allocated()   # peak GPU memory
torch.cuda.memory_summary()         # detailed breakdown
```

For SoC SRAM/HBM budget sizing.

## Key Numbers for SoC Spec

| Metric | Why it matters | Tool |
|---|---|---|
| Backbone latency | Largest compute block, drives total latency | Layer 1 |
| Action head latency | Denoising iterations, parallelism opportunity | Layer 1 |
| Peak memory | On-chip memory budget | Layer 4 |
| FLOPs per forward pass | Compute unit sizing (TOPs requirement) | Layer 2 |
| Attention kernel time | Custom attention accelerator ROI | Layer 2/3 |
| Memory bandwidth | HBM spec, tiling strategy | Layer 3 |
| Denoising step count | Pipeline depth for action head | Layer 1 |

## What We Do NOT Profile

- Isaac Sim physics stepping (PhysX) -- not on the SoC
- Viewport rendering -- not on the SoC
- WBC (Whole Body Controller) -- tiny ONNX model, <1ms, separate concern
- ROS2 communication overhead -- deployment-specific, not model-inherent

## Implementation Plan

1. Add CUDA event timing to `gr00t_closedloop_policy.py::get_action_chunk()`
2. Add CUDA event timing to `gr00t_n1.py::get_action()` (backbone vs action head)
3. Log timings to CSV per episode for analysis
4. Run torch.profiler trace for one episode, export Chrome trace
5. Run nsys on headless mode for clean GPU traces
