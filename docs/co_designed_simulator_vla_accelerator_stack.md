# Co-Designed Simulator Stack and Design Goals

## Scope

This document summarizes the architecture and priorities for the co-designed simulator pipeline:

- Multi-robot relay task development in Isaac Sim / IsaacLab-Arena.
- Data pipeline from demonstrations to VLA-compatible training datasets.
- Profiling-driven acceleration loop (software + hardware decisions).
- Long-term target: collaboration-aware policy stack with explicit context sharing.

Related diagram:

- `docs/co_designed_simulator_vla_accelerator_stack.drawio`

## Architecture Summary

The stack is organized as a closed loop:

1. **Task and Scene Authoring**
   - Shared USD stage, transfer workspace, robot placements, task constraints.
2. **Runtime Simulation and Control**
   - Isaac runtime + control stack (WBC/Pink/closed-loop policy execution).
3. **Data and Policy Pipeline**
   - Demo collection, annotation, augmentation/generation, schema conversion, policy training/eval.
4. **Profiling and Accelerator Co-Design**
   - Runtime traces and bottlenecks feed kernel/runtime/precision decisions.
5. **Multi-Robot System Goal (S3)**
   - Context-sharing interfaces for ownership, handover readiness, and implicit coordination.

## Design Goals

- **G1:** Keep one authoritative shared-scene relay environment for all experiments.
- **G2:** Make dataset generation reproducible and auditable (inputs, labels, metrics).
- **G3:** Isolate control-interface assumptions (teleop, scripted, policy replay) from environment logic.
- **G4:** Use profiling evidence to prioritize acceleration work, not intuition.
- **G5:** Co-optimize policy quality and runtime efficiency under multi-robot constraints.
- **G6:** Progress from single-robot reliability to two-robot coordination without skipping validation gates.

## Key Challenges and Mitigations

### 1) Control-Interface Mismatch

**Challenge**
- Different control paths expose different action spaces (for example, SE3 teleop vs full humanoid action vectors).

**Mitigation**
- Add explicit adapter layers between device output and environment action space.
- Validate action shape and semantic fields at runtime before `env.step()`.

### 2) Scene Configuration Drift

**Challenge**
- Runtime scene may not match edited local USD stage (robot/table/bin overlaps, stale asset defaults).

**Mitigation**
- Always log resolved scene source and critical spawn coordinates at startup.
- Prefer explicit stage path override and a single source-of-truth scene config.

### 3) Handover Dynamics Are Contact-Sensitive

**Challenge**
- Scripted waypoint logic can look correct but fail physically (reaching loops, unstable transfer).

**Mitigation**
- Decompose relay into primitives with strict transition checks:
  - approach,
  - stable grasp,
  - transfer-ready pose,
  - release-on-receive.
- Gate transitions on contact/object state, not only timeouts.

### 4) Data Quality Bottleneck

**Challenge**
- High-DoF humanoid manipulation is difficult to collect with keyboard-only teleop.

**Mitigation**
- Use teleop for seed demonstrations where feasible, then scale with augmentation/generation.
- Track episode quality metrics (success windows, object stability, phase durations).

### 5) Profiling-to-Acceleration Gap

**Challenge**
- Profiling output exists but is not consistently translated into accelerator decisions.

**Mitigation**
- Standardize a short profiling report template per experiment:
  - top kernels/operators,
  - memory pressure,
  - frequency/latency budget.
- Tie every optimization task to one measured bottleneck.

## Near-Term Execution Plan

1. Stabilize single-robot pick-place data flow and metrics.
2. Promote relay primitives with explicit transition signals.
3. Build dual-robot dataset from validated primitives.
4. Run profiling baselines on simulation and policy inference paths.
5. Feed the hotspot map into accelerator/runtime design choices.

## Exit Criteria for This Phase

- Dual-robot relay episodes complete with consistent success conditions.
- Dataset export is schema-consistent and repeatable across runs.
- Profiling baselines identify top acceleration targets with measured impact.
- Architecture supports adding S3 context-sharing terms without rewriting the runtime stack.
