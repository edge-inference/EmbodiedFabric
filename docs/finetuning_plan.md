# GR00T N1.6 Finetuning Plan -- G1 Loco-Manipulation

## What we're doing

We have a single working G1 loco-manipulation demo (walk to box, pick up, carry
to bin) running GR00T N1.5. We want to finetune N1.6 on more diverse G1 data
so the robot can handle varied pick-and-place tasks, different objects, and
different environments. The finetuned model will be profiled for SoC design.

## Current state

- Isaac-GR00T N1.5 at `extern/Isaac-GR00T` (don't touch, working sim)
- Isaac-GR00T N1.6 at `extern/Isaac-GR00T-n1.6` (git worktree, for finetuning)
- Hardware: 2x RTX A6000 (48GB each)
- Base model: `nvidia/GR00T-N1.6-3B` (will download from HuggingFace)
- Embodiment tag: `UNITREE_G1` (pre-registered in N1.6, no custom config needed)

## Training data candidates

| Dataset | Format | G1 loco-manip? | Size | Conversion needed? |
|---|---|---|---|---|
| NVIDIA X-Embodiment `unitree_g1.LMPnPAppleToPlateDC` | LMDB/LeRobot | Yes | 102 trajectories | Possibly minor |
| OmniRetarget `robot-object/` | .npz (qpos) | Yes (walking + carrying) | 3h | Yes, to LeRobot v2 |
| Humanoid Everyday G1 subset | LeRobot v2 | Yes (loco-manip category) | Part of 10.3k trajs | No |

Priority order: X-Embodiment first (smallest, already closest to right format,
from NVIDIA's own pipeline), then OmniRetarget if we need more data.

## Step-by-step

### Phase 1: Environment setup

1. Install N1.6 dependencies in the worktree:
   ```bash
   cd extern/Isaac-GR00T-n1.6
   pip install -e .
   ```
   Watch for numpy/packaging conflicts with Isaac Sim. Pin back if needed:
   ```bash
   pip install numpy==1.26.0
   ```

2. Verify the install:
   ```bash
   python -c "from gr00t.data.embodiment_tags import EmbodimentTag; print(EmbodimentTag.UNITREE_G1)"
   ```

### Phase 2: Get training data

3. Download the X-Embodiment G1 subset:
   ```bash
   huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
     --repo-type dataset \
     --include "unitree_g1.LMPnPAppleToPlateDC/**" \
     --local-dir data/x_embodiment_g1
   ```

4. Inspect the data structure -- check if it's already in LeRobot v2 or needs
   conversion. The X-Embodiment dataset ships in LMDB + SquashFS. NVIDIA's
   data_preparation.md documents the expected schema.

5. If conversion is needed, follow `getting_started/data_preparation.md` in
   the N1.6 worktree. Key fields the model expects:
   - `video.ego_view` -- camera frames
   - `state.{left_arm, right_arm, left_hand, right_hand, waist, left_leg, right_leg}`
   - `action.{left_arm, right_arm, left_hand, right_hand, waist, base_height_command, navigate_command}`
   - `annotation.human.action.task_description` -- language instruction

### Phase 3: Finetune

6. Run finetuning (single GPU first to verify, ~4h for 20k steps):
   ```bash
   cd extern/Isaac-GR00T-n1.6

   CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.6-3B \
     --dataset-path ../../data/x_embodiment_g1/unitree_g1.LMPnPAppleToPlateDC \
     --embodiment-tag UNITREE_G1 \
     --num-gpus 1 \
     --output-dir ../../checkpoints/n1.6-g1-finetuned \
     --save-steps 5000 \
     --max-steps 20000 \
     --global-batch-size 32 \
     --dataloader-num-workers 4
   ```

   If memory is tight at batch 32, drop to 16. A6000 at 48GB should handle it.

7. For dual-GPU (faster):
   ```bash
   export NUM_GPUS=2
   python gr00t/experiment/launch_finetune.py \
     --base-model-path nvidia/GR00T-N1.6-3B \
     --dataset-path ../../data/x_embodiment_g1/unitree_g1.LMPnPAppleToPlateDC \
     --embodiment-tag UNITREE_G1 \
     --num-gpus $NUM_GPUS \
     --output-dir ../../checkpoints/n1.6-g1-finetuned \
     --save-steps 5000 \
     --max-steps 20000 \
     --global-batch-size 32
   ```

### Phase 4: Evaluate

8. Open-loop eval (quick sanity check, no sim needed):
   ```bash
   python gr00t/eval/open_loop_eval.py \
     --dataset-path ../../data/x_embodiment_g1/unitree_g1.LMPnPAppleToPlateDC \
     --embodiment-tag UNITREE_G1 \
     --model-path ../../checkpoints/n1.6-g1-finetuned/checkpoint-20000 \
     --traj-ids 0 1 2 \
     --action-horizon 30 \
     --steps 400
   ```
   This plots predicted vs ground-truth actions. If they track, the model
   learned something.

9. Closed-loop eval in Arena (the real test). This requires updating the
   Arena GR00T config to point to the N1.6 checkpoint and ensuring the
   Arena integration works with N1.6's API. We will likely need to:
   - Update `g1_locomanip_gr00t_closedloop_config.yaml` with the new checkpoint path
   - Verify `gr00t_closedloop_policy.py` is compatible with N1.6's `get_action()` API
   - Test in the existing locomanip environment first, then the warehouse

### Phase 5: Scale up data (if Phase 4 works)

10. Add OmniRetarget data for more diverse locomotion:
    ```bash
    git lfs install
    git clone https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset data/omniretarget
    ```
    Convert .npz trajectories to LeRobot v2 format (write a conversion script
    mapping qpos fields to the G1 modality config).

11. Add Humanoid Everyday G1 loco-manip subset (already LeRobot v2):
    ```bash
    huggingface-cli download USC-GVL/Humanoid-Everyday-G1 --local-dir data/humanoid_everyday_g1
    ```
    Filter for loco-manipulation tasks only.

12. Combine all datasets and retrain with more steps (40-60k).

## What could go wrong

- **X-Embodiment data format mismatch**: The dataset may use LMDB format
  that needs extraction before GR00T's dataloader can read it. Check the
  README in the downloaded folder.
- **Numpy/package conflicts**: N1.6 install may pull newer numpy. Pin back
  to 1.26.0 after install.
- **OOM during training**: Reduce batch size or enable gradient checkpointing.
- **N1.6 API incompatibility with Arena**: The Arena GR00T integration was
  written for N1.5. The model class changed from `GR00T_N1_5` to presumably
  `GR00T_N1d6`. We'll need to bridge this for closed-loop eval.
- **OmniRetarget conversion**: The .npz files contain raw qpos, not the
  modality-keyed format GR00T expects. Need a mapping script from
  OmniRetarget's 36D qpos vector to the G1 joint groups.

## Timeline estimate

| Phase | Effort | Blocking? |
|---|---|---|
| 1. Install N1.6 | 30 min | No |
| 2. Download + inspect data | 1-2h | No |
| 3. Finetune (20k steps) | 4-6h (GPU time) | Yes, wait for training |
| 4. Open-loop eval | 30 min | No |
| 4b. Arena closed-loop eval | 2-4h (API bridging) | Depends on API diff |
| 5. Scale data + retrain | 1-2 days | Optional, after Phase 4 |
