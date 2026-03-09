#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# SmolVLA Fine-tuning for LeIsaac Evaluation
# =============================================================================
#
# PREREQUISITES (on the training machine, e.g. B300):
#
#   1. Clone lerobot:
#        git clone https://github.com/huggingface/lerobot.git
#        cd lerobot && pip install -e ".[smolvla]"
#
#   2. Everything else (dataset, base model) is auto-downloaded from
#      HuggingFace on first run. No manual downloads needed.
#
# USAGE:
#   # SO101 PickOrange (6D arm, sim data):
#   TASK=pick_orange ./scripts/train/train_smolvla.sh
#
#   # LeKiwi block cleanup (9D arm+wheels, real data, 56 episodes):
#   TASK=lekiwi_cleanup ./scripts/train/train_smolvla.sh
#
#   # LeKiwi ball into basket (9D, real data, 10 episodes):
#   TASK=lekiwi_ball ./scripts/train/train_smolvla.sh
#
# After training, evaluate with:
#   POLICY_CHECKPOINT=<output_dir>/checkpoints/last/pretrained_model \
#   ./scripts/eval/leisaac_so101_smolvla.sh   # (or leisaac_lekiwi_smolvla.sh)
# =============================================================================

TASK="${TASK:-pick_orange}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

case "$TASK" in
  pick_orange)
    DATASET="LightwheelAI/leisaac-pick-orange"
    OUTPUT="outputs/smolvla_so101_pick_orange"
    ;;
  lekiwi_cleanup)
    DATASET="pepijn223/lekiwi_block_cleanup2"
    OUTPUT="outputs/smolvla_lekiwi_cleanup"
    ;;
  lekiwi_ball)
    DATASET="francocipollone/lekiwi_ball_into_basket"
    OUTPUT="outputs/smolvla_lekiwi_ball"
    ;;
  *)
    echo "Unknown TASK=$TASK. Use: pick_orange, lekiwi_cleanup, lekiwi_ball"
    exit 1
    ;;
esac

OUTPUT="${OUTPUT_DIR:-$OUTPUT}"

echo "=== Training SmolVLA ==="
echo "  Task:      $TASK"
echo "  Dataset:   $DATASET"
echo "  Output:    $OUTPUT"
echo "  Batch:     $BATCH_SIZE"
echo "  Steps:     $STEPS"
echo "  LR:        $LR"
echo ""

python -m lerobot.scripts.lerobot_train \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --dataset.repo_id="$DATASET" \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --output_dir="$OUTPUT" \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100 \
  --num_workers="$NUM_WORKERS" \
  --policy.train_expert_only=true \
  --policy.freeze_vision_encoder=true \
  --policy.optimizer_lr="$LR" \
  --policy.num_vlm_layers=16 \
  --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --policy.use_amp=true
