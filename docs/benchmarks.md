# Benchmarks & Datasets

Datasets and benchmarks we're using or plan to use for GR00T finetuning,
evaluation, and VLA profiling. Ordered by priority.

---

## Finetuning Data

### 1. X-Embodiment G1 (unitree_g1.LMPnPAppleToPlateDC)
- Source: `nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`
- 102 loco-manipulation trajectories: G1 walks to object, picks apple, places on plate
- LMDB/LeRobot format, minimal conversion
- NVIDIA already finetuned N1.6 on this and released the checkpoint
- **Robot: Unitree G1 | Embodiment tag: UNITREE_G1**

### 2. X-Embodiment R1 Pro (sim_behavior_r1_pro.*)
- Source: same repo as above, different subset
- 50 household tasks (cooking, cleaning, carrying, sorting, etc.)
- LMDB/LeRobot format, ready for finetuning via `finetune_BEHAVIOR.sh`
- Multi-task -- single checkpoint does all 50 tasks at 26.3% avg task progress
- **Robot: R1 Pro (not G1) | Embodiment tag: BEHAVIOR_R1_PRO**
- Cannot cross-finetune onto G1 directly (different action/state space)

### 3. OmniRetarget Dataset
- Source: `omniretarget/OmniRetarget_Dataset` on HuggingFace
- ~3 hours of G1 loco-manip trajectories retargeted from human motion capture
- Raw .npz format (qpos), needs conversion to LeRobot v2
- Good for diverse locomotion patterns (walking styles, carrying objects)
- **Robot: Unitree G1**

### 4. Humanoid Everyday G1 Subset
- Source: `USC-GVL/humanoid-everyday`
- 10.3k trajectories, 260 tasks across G1 and H1
- Already LeRobot v2, no conversion needed
- Filter for G1 loco-manipulation tasks
- Real-world teleoperation data (not sim)
- **Robot: Unitree G1 + H1**

### 5. LeVERB-Bench Dataset
- Source: `ember-lab-berkeley/LeVERB-Bench-Dataset`
- 3,696 episodes, 1,069 unique tasks (navigation, sitting, reaching, obstacle avoidance)
- LeRobot v2.1, 29-DOF, 1080x1920 video, 657 MB total
- Kinematic demonstrations from retargeted human MoCap in IsaacSim
- Code not released yet (ICLR 2026 submission), dataset is public
- **Robot: G1-class humanoid (29 DoF)**

---

## Eval Benchmarks

### G1 PnPAppleToPlate (MuJoCo-based)
- Checkpoint: `nvidia/GR00T-N1.6-G1-PnPAppleToPlate`
- 58% success rate (high variance, +/-15%)
- Script: `scripts/eval/g1_groot_n16.sh`
- Requires WBC env setup: `scripts/setup/g1_wbc_env.sh`

### BEHAVIOR-1K R1 Pro (OmniGibson-based)
- Checkpoint: `nvidia/GR00T-N1.6-BEHAVIOR1k`
- 50 tasks, avg 26.3% task progress (beats Pi0.5 at 11.3%)
- Script: `scripts/eval/behavior_r1.sh`
- Requires OmniGibson + RT-core GPU: `scripts/setup/behavior_env.sh`

### SmolVLA GR1 Microwave / Warehouse (Isaac Sim Arena)
- Already running, used for baseline profiling
- Scripts: `scripts/eval/smolvla_microwave.sh`, `scripts/eval/smolvla_warehouse.sh`

### LeVERB-Bench (IsaacSim, closed-loop)
- 150+ WBC tasks, 10 categories
- Eval code not yet released (marked "Coming Soon")
- Interesting for profiling: 104M param model (30x smaller than GR00T)

---

## Quick Reference

| Dataset | Robot | Tasks | Format | Use |
|---|---|---|---|---|
| X-Embodiment G1 | G1 | 1 (loco-manip PnP) | LMDB/LeRobot | Finetune + Eval |
| X-Embodiment R1 | R1 Pro | 50 household | LMDB/LeRobot | Finetune + Eval |
| OmniRetarget | G1 | Locomotion | .npz | Finetune |
| Humanoid Everyday | G1/H1 | 260 mixed | LeRobot v2 | Finetune |
| LeVERB-Bench | G1-class | 1,069 WBC | LeRobot v2.1 | Finetune + Eval (pending code) |
| SmolVLA Arena | GR1 | Microwave/Warehouse | Isaac Sim | Profiling baseline |
