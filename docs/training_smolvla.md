# SmolVLA Fine-tuning for LeIsaac Evaluation

## Prerequisites (B300 or any GPU machine)

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[smolvla]"
```

Everything else (base model, dataset, VLM backbone) auto-downloads from HuggingFace on first run.

## Available Training Tasks

| Task | Command | Dataset | Action Dims | Episodes | Source |
|------|---------|---------|:-:|----------|--------|
| SO101 PickOrange | `TASK=pick_orange` | `LightwheelAI/leisaac-pick-orange` | 6D (arm) | ~50 | Sim (LeIsaac) |
| LeKiwi block cleanup | `TASK=lekiwi_cleanup` | `pepijn223/lekiwi_block_cleanup2` | 9D (arm+wheels) | 56 | Real-world |
| LeKiwi ball into basket | `TASK=lekiwi_ball` | `francocipollone/lekiwi_ball_into_basket` | 9D (arm+wheels) | 10 | Real-world |

## Training

```bash
# LeKiwi cleanup (full 9D platform)
TASK=lekiwi_cleanup BATCH_SIZE=32 STEPS=30000 ./scripts/train/train_smolvla.sh

# SO101 pick orange (sim-trained, 6D arm only)
TASK=pick_orange BATCH_SIZE=16 STEPS=30000 ./scripts/train/train_smolvla.sh

# LeKiwi ball into basket (smaller dataset)
TASK=lekiwi_ball BATCH_SIZE=8 STEPS=30000 ./scripts/train/train_smolvla.sh
```

### Key parameters

| Parameter | Default | B300 recommended |
|-----------|---------|-----------------|
| `BATCH_SIZE` | 8 | 32 |
| `STEPS` | 30000 | 30000 |
| `LR` | 1e-4 | 1e-4 |
| `SAVE_FREQ` | 5000 | 5000 |

Training uses bf16 mixed precision (`--policy.use_amp=true`), freezes the vision encoder, and only trains the expert action head. ~450M params total, ~1GB model.

### Estimated training time

| GPU | Batch size | Time (30k steps) |
|-----|-----------|-------------------|
| A6000 (48GB) | 8 | ~2-3 hours |
| B300 (80GB+) | 32 | ~30-60 min |

## After Training

Checkpoint is saved at: `<output_dir>/checkpoints/last/pretrained_model`

### Evaluate on this machine

Copy the checkpoint back, then:

```bash
# For LeKiwi tasks
POLICY_CHECKPOINT=outputs/smolvla_lekiwi_cleanup/checkpoints/last/pretrained_model \
./scripts/eval/leisaac_lekiwi_smolvla.sh

# For SO101 tasks
POLICY_CHECKPOINT=outputs/smolvla_so101_pick_orange/checkpoints/last/pretrained_model \
TASK=LeIsaac-SO101-PickOrange-v0 \
./scripts/eval/leisaac_so101_smolvla.sh
```

## Notes

- The LeKiwi cleanup dataset is real-world data. There will be a real-to-sim visual domain gap when evaluating in LeIsaac's CleanupTrash env. For best results, collect teleop data directly in the sim env and train on that.
- The SO101 PickOrange dataset is sim-collected (from LeIsaac). Training on this and evaluating in the same sim should produce working behavior.
- SmolVLA base model (`lerobot/smolvla_base`) uses `camera1/camera2/camera3` naming. The fine-tuned model inherits camera naming from the dataset. Ensure eval script checkpoint matches the camera names in the target env.
