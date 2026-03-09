#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/modfi/models/vla_simu"
LEISAAC_DIR="$ROOT_DIR/extern/leisaac"
GROOT_DIR="$ROOT_DIR/extern/Isaac-GR00T"
VENV_PYTHON="$ROOT_DIR/.venv_isaac/bin/python"
LOG_DIR="$ROOT_DIR/logs"

TASK="${TASK:-LeIsaac-SO101-PickOrange-v0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-3}"
STEP_HZ="${STEP_HZ:-30}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-1}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-$LOG_DIR/videos}"

GROOT_HOST="${GROOT_HOST:-localhost}"
GROOT_PORT="${GROOT_PORT:-5555}"
GROOT_CHECKPOINT="${GROOT_CHECKPOINT:-LightwheelAI/leisaac-pick-orange-v0}"
GROOT_DATA_CONFIG="${GROOT_DATA_CONFIG:-so100_dualcam}"
GROOT_EMBODIMENT="${GROOT_EMBODIMENT:-new_embodiment}"
GROOT_DEVICE="${GROOT_DEVICE:-cuda:1}"

POLICY_LANGUAGE="${POLICY_LANGUAGE:-Pick up the orange and place it on the plate}"

SERVER_LOG="${SERVER_LOG:-$LOG_DIR/groot_server.log}"
EVAL_LOG="${EVAL_LOG:-$LOG_DIR/leisaac_so101_groot_eval.log}"

mkdir -p "$LOG_DIR"

# Kill any existing GR00T server on the port
fuser -k "${GROOT_PORT}/tcp" >/dev/null 2>&1 || true

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
  fi
  fuser -k "${GROOT_PORT}/tcp" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "=== Starting GR00T N1.5 inference server ==="
echo "  Checkpoint: $GROOT_CHECKPOINT"
echo "  Data config: $GROOT_DATA_CONFIG"
echo "  Device: $GROOT_DEVICE"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="${GROOT_DEVICE##*:}" "$VENV_PYTHON" \
  "$GROOT_DIR/scripts/inference_service.py" \
  --server \
  --model_path="$GROOT_CHECKPOINT" \
  --data_config="$GROOT_DATA_CONFIG" \
  --embodiment_tag="$GROOT_EMBODIMENT" \
  --port="$GROOT_PORT" \
  --host="$GROOT_HOST" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "GR00T server PID: $SERVER_PID"

echo "Waiting for GR00T model to load (this takes ~30-60s for 7.6GB model)..."
sleep 30

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "GR00T server failed to start. See: $SERVER_LOG"
  cat "$SERVER_LOG"
  exit 1
fi
echo "GR00T server appears to be running."

cd "$LEISAAC_DIR"

cmd=(
  "$VENV_PYTHON" scripts/evaluation/policy_inference.py
  --task="$TASK"
  --eval_rounds="$EVAL_ROUNDS"
  --step_hz="$STEP_HZ"
  --policy_type=gr00tn1.5
  --policy_host="$GROOT_HOST"
  --policy_port="$GROOT_PORT"
  --policy_timeout_ms=30000
  --policy_action_horizon=16
  --policy_language_instruction="$POLICY_LANGUAGE"
  --device="$DEVICE"
)

if [[ "$HEADLESS" == "1" ]]; then
  cmd+=(--headless)
fi

if [[ "$ENABLE_CAMERAS" == "1" ]]; then
  cmd+=(--enable_cameras)
fi

if [[ "$SAVE_VIDEO" == "1" ]]; then
  cmd+=(--save_video --video_dir="$VIDEO_DIR")
fi

printf 'Running: %q ' "${cmd[@]}"
echo
PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$EVAL_LOG"
