# Simulation Platform Comparison: Isaac Lab vs TDW/CoELA

## Context

We evaluated two simulation stacks for profiling VLA (Vision-Language-Action) workloads
targeting hardware-software co-design of a robot inference accelerator.

## Stack Overview

| | Isaac Lab (NVIDIA) | TDW + CoELA (UMass/MIT) |
|---|---|---|
| Physics | PhysX 5, GPU-accelerated | PhysX 4 via Unity, CPU only |
| Rendering | RTX raytracing, photorealistic | Unity renderer, decent quality |
| Parallel envs | 1000s simultaneously on GPU | Single environment |
| Robot models | Real robots (G1, GR1, Digit, Spot, Franka, etc.) | Generic humanoid avatar ("Replicant") |
| Sim-to-real | Direct -- same URDF/USD as hardware | None -- avatar has no real counterpart |
| VLA support | GR00T (3B), SmolVLA (450M) via LeRobot | None native, LLM planning only |
| Multi-agent | Not built-in, requires custom layer | Built-in, used in CoELA/ReCA |
| Action space | Per-joint torques/positions (43-54 DOF) | 7 discrete symbolic actions |

## Control Architecture Comparison

**Isaac Lab + VLA pipeline:**
Camera RGB (512x512) -> VLA model (GPU) -> joint angle targets (50 per chunk)
-> Whole Body Controller (balance + IK) -> per-actuator torques -> PhysX GPU sim

The VLA handles perception and action generation in one model. The robot physically
walks, reaches, and grasps with individual fingers. Every contact is simulated.

**CoELA + LLM pipeline:**
Camera RGB-D + segmentation -> Mask R-CNN detection -> spatial memory map
-> LLM planner (GPT-4) -> symbolic action ("go grasp apple")
-> TDW Replicant API -> internal IK -> CPU physics

Perception and planning are separate modules. The LLM reasons at room/object level.
Low-level control (IK, grasping) is handled internally by TDW, not learned.

## What Each Stack Does Well

**Isaac Lab strengths:**
- Accurate physics on real robot embodiments with real actuator models
- GPU acceleration makes large-scale training and evaluation practical
- VLA models output joint-level control -- the actual workload an edge SoC would run
- Direct path from simulation to deployment on physical robots

**CoELA/TDW strengths:**
- Multi-agent coordination with decentralized communication
- Spatial memory, opponent modeling, task delegation between agents
- Long-horizon planning across rooms with A* navigation on occupancy grids
- Robust handling of partial observability (agents explore, share findings)

## Gap Analysis

Neither platform alone covers the full stack:
- Isaac Lab lacks multi-agent support (Arena is single-robot only)
- CoELA lacks low-level VLA control and real robot models
- No existing benchmark combines multi-agent coordination with per-joint VLA inference

## Recommendation

Use Isaac Lab as the simulation backbone for hardware profiling. The VLA inference
pipeline (vision encoding, action prediction, whole body control) is the target
workload for the SoC, and Isaac Lab runs it on accurate robot models with GPU physics.

For multi-agent scenarios, build a coordination layer on top of Isaac Lab inspired
by CoELA's architecture (spatial memory, inter-agent communication, task allocation),
but replace the symbolic action execution with VLA-driven joint control.

## What We Validated

| Task | Model | Robot | Status |
|---|---|---|---|
| Open microwave | SmolVLA (450M) | GR1 | 100% success, ~9s/episode |
| Loco-manipulation pick & place | GR00T N1.5 (3B) | G1 | Working, walks + picks box |
| Warehouse pick & place | SmolVLA (450M) | GR1 | Scene works, needs task-specific policy |

Profiling these workloads gives us the compute, memory, and latency characteristics
needed to spec the inference accelerator.


| Dataset                                       | What's in it                                             | Scale                  | Robot       | Loco-manip?         |
|-----------------------------------------------|---------------------------------------------------------|------------------------|-------------|---------------------|
| PhysicalAI-Robotics-GR00T-X-Embodiment-Sim    | Multi-embodiment, multi-task trajectories for GR00T post-training | 1.1M+ downloads, massive | Multiple    | Likely yes          |
| PhysicalAI-GR00T-Tuned-Tasks                  | GR1 tabletop manipulation, multiple industrial tasks     | 100K–1M samples        | GR1         | No (tabletop)       |
| PhysicalAI-Robotics-GR00T-Teleop-G1           | 1000 teleoperation trajectories, G1 fruits pick & place  | 1000 demos             | G1          | Unclear             |
| ORCA-sim-push-cart-gr00t                      | Cart pushing (like WholeBodyVLA!)                       | Small                  | G1?         | Yes                 |
| Arena-G1-Loco-Manipulation-Task               | Walk + pick + place                                     | 50 demos               | G1          | Yes                 |
| PhysicalAI-Robotics-GR00T-Eval                | 123 initial frames for various tasks                     | Small                  | Mixed       | Eval only           |


| Embodiment/Dataset        | Tasks / Description                        | Trajectories / Size             | What it does / Notes                                                | Robot(s)             | Locomotion?           | Available?                                   |
|--------------------------|--------------------------------------------|----------------------------------|---------------------------------------------------------------------|----------------------|-----------------------|----------------------------------------------|
| unitree_g1               | 1 task (LMPnPAppleToPlateDC)               | 102                              | G1 loco-manipulation -- walk + pick apple + place on plate          | Unitree G1           | Yes                   |                                              |
| gr1_arms_waist           | 24 tasks                                   | 240k (10k each)                  | GR1 tabletop pick-and-place (no locomotion)                         | GR1                  | No                    |                                              |
| gr1_unified              | 24 tasks                                   | 24k (1k each)                    | Same as above, downsampled, with Fourier hands                      | GR1                  | No                    |                                              |
| gr1_arms_only            | 1 task (CanSort)                           | 1k                               | GR1 arms-only can sorting                                           | GR1                  | No                    |                                              |
| gr1_full_upper_body      | 2 tasks (Coffee, Pouring)                  | 2k                               | GR1 upper body manipulation                                         | GR1                  | No                    |                                              |
| bimanual_panda_*         | 6 tasks                                    | 6k                               | Dual Panda arm tasks                                                | Panda                | No                    |                                              |
| single_panda_gripper     | 24 tasks                                   | 72k                              | Single Panda arm kitchen tasks                                      | Panda                | No                    |                                              |
| sim_behavior_r1_pro      | 50 tasks                                   | ~50 each?                        | Diverse household tasks on R1 Pro robot                             | R1 Pro               | No                    |                                              |
| **OmniRetarget**         | -                                          | 4 hours of trajectories          | Walking, carrying, terrain                                          | Unitree G1           | Yes                   | HuggingFace, MIT license                    |
| **Humanoid Everyday**    | 260 tasks, 7 categories                    | 10.3k trajectories, 3M frames    | Humanoid household & daily activities                               | Unitree H1           | Yes                   | HuggingFace (USC-GVL/humanoid-everyday)     |
| **NVIDIA X-Embodiment**  | G1: 1 task, GR1, R1 Pro                    | 340k+ total trajectories         | G1 has 102 loco-manip demos                                         | G1, GR1, R1 Pro      | Mixed                 | HuggingFace, CC-BY-4.0                      |
| **NVIDIA G1 Teleop**     | -                                          | 1000 demos                       | Real teleop                                                         | Unitree G1           | Yes                   | HuggingFace                                 |
| **WholeBodyVLA**         | Pushing, loading, bimanual                 | Code released, data TBD          | Whole-body VLA tasks (AgiBot X2)                                    | AgiBot X2            | Yes                   | GitHub (MIT)                                |


Humanoid Everyday	G1 + H1	260	LeRobot v2.0 (ready for GR00T)
OmniRetarget	G1	~3 types	.npz (needs conversion)
NVIDIA X-Embodiment	G1/GR1/Panda/R1	132	LMDB (needs conversion)


the sim_behavior_r1_pro tasks (50 of them) also include locomotion-heavy tasks:
task-0013_loading_the_car
task-0014_carrying_in_groceries
task-0016_moving_boxes_to_storage
task-0017_bringing_water
But those are on the R1 Pro robot, not G1 -- different embodiment, can't directly finetune for G1.
Summary of what's actually usable for G1 loco-manip finetuning:
Data	Loco-manip?	Trajectories	Usable for G1?
unitree_g1.LMPnPAppleToPlateDC	Yes, confirmed	102	Yes, directly
sim_behavior_r1_pro (50 tasks)	Many yes	~50 each	No, wrong robot
OmniRetarget robot-object/	Yes	~3h	Yes, needs format conversion
Humanoid Everyday G1 loco-manip	Yes	Subset of 10.3k	Yes, already L