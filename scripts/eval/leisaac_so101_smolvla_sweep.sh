#!/usr/bin/env bash
# Sequential multi-seed sweep around `leisaac_so101_smolvla.sh`.
#
# Defaults: 10 episodes per seed, 3 seeds. Override via `SEEDS`, `EVAL_ROUNDS`,
# `RUN_TAG`, etc. Each seed lands in its own video / log directory so the runs
# are independent and easy to compare.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_RUN="$SCRIPT_DIR/leisaac_so101_smolvla.sh"
LOG_DIR="${LOG_DIR:-/home/modfi/models/physicai/logs}"

EVAL_ROUNDS="${EVAL_ROUNDS:-10}"
SEEDS="${SEEDS:-42 123 2026}"
RUN_TAG="${RUN_TAG:-v2-100-ep_sweep}"

POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-/home/modfi/models/physicai/checkpoints/smolvla-so101-pick-orange-v2-100-ep}"
POLICY_LANGUAGE_INSTRUCTION="${POLICY_LANGUAGE_INSTRUCTION:-pick up the orange and place it on the plate}"

SUMMARY_LOG="${SUMMARY_LOG:-$LOG_DIR/${RUN_TAG}_summary.log}"
mkdir -p "$LOG_DIR"

echo "=== Sweep config ===" | tee "$SUMMARY_LOG"
{
  echo "  seeds:              $SEEDS"
  echo "  eval_rounds/seed:   $EVAL_ROUNDS"
  echo "  run_tag:            $RUN_TAG"
  echo "  checkpoint:         $POLICY_CHECKPOINT"
  echo "  prompt:             $POLICY_LANGUAGE_INSTRUCTION"
  echo "  started_at:         $(date -Iseconds)"
} | tee -a "$SUMMARY_LOG"

for SEED in $SEEDS; do
  TAG="${RUN_TAG}_seed${SEED}"
  VIDEO_DIR="$LOG_DIR/videos/$TAG"
  SERVER_LOG="$LOG_DIR/leisaac_policy_server_${TAG}.log"
  EVAL_LOG="$LOG_DIR/leisaac_so101_eval_${TAG}.log"
  LAUNCH_LOG="$LOG_DIR/leisaac_launch_${TAG}.log"

  echo | tee -a "$SUMMARY_LOG"
  echo "=== seed $SEED -> $TAG ($(date -Iseconds)) ===" | tee -a "$SUMMARY_LOG"

  SEED="$SEED" \
  EVAL_ROUNDS="$EVAL_ROUNDS" \
  POLICY_CHECKPOINT="$POLICY_CHECKPOINT" \
  POLICY_LANGUAGE_INSTRUCTION="$POLICY_LANGUAGE_INSTRUCTION" \
  VIDEO_DIR="$VIDEO_DIR" \
  SERVER_LOG="$SERVER_LOG" \
  EVAL_LOG="$EVAL_LOG" \
  bash "$SINGLE_RUN" >"$LAUNCH_LOG" 2>&1 || {
    echo "  WARN: seed $SEED returned non-zero (see $LAUNCH_LOG)" | tee -a "$SUMMARY_LOG"
  }

  if FINAL=$(grep -E "Final success rate" "$EVAL_LOG" 2>/dev/null | tail -1); then
    echo "  $FINAL" | tee -a "$SUMMARY_LOG"
  fi
done

echo | tee -a "$SUMMARY_LOG"
echo "=== Sweep complete ($(date -Iseconds)) ===" | tee -a "$SUMMARY_LOG"
