#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/modfi/models/vla_simu"
LEISAAC_DIR="$ROOT_DIR/extern/leisaac"
VENV_PYTHON="$ROOT_DIR/.venv_isaac/bin/python"
LOG_DIR="$ROOT_DIR/logs"

TASK="${TASK:-LeIsaac-SO101-LiftCube-v0}"
EVAL_ROUNDS="${EVAL_ROUNDS:-1}"
STEP_HZ="${STEP_HZ:-30}"
DEVICE="${DEVICE:-cuda:0}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda:1}"
HEADLESS="${HEADLESS:-1}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VIDEO_DIR="${VIDEO_DIR:-$LOG_DIR/videos}"

POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
POLICY_PORT="${POLICY_PORT:-8080}"
POLICY_TIMEOUT_MS="${POLICY_TIMEOUT_MS:-15000}"
POLICY_ACTION_HORIZON="${POLICY_ACTION_HORIZON:-50}"
POLICY_TYPE="${POLICY_TYPE:-lerobot-smolvla}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-azazdeaz/smolvla_so101_leisaac_lift_cube}"
POLICY_LANGUAGE_INSTRUCTION="${POLICY_LANGUAGE_INSTRUCTION:-Lift the red cube up.}"
POLICY_CAMERA_REMAP="${POLICY_CAMERA_REMAP:-front=camera1,wrist=camera2}"

SERVER_LOG="${SERVER_LOG:-$LOG_DIR/leisaac_policy_server.log}"
EVAL_LOG="${EVAL_LOG:-$LOG_DIR/leisaac_so101_eval.log}"

mkdir -p "$LOG_DIR"
cd "$LEISAAC_DIR"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Missing python: $VENV_PYTHON"
  exit 1
fi

fuser -k "${POLICY_PORT}/tcp" >/dev/null 2>&1 || true

cleanup() {
  if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
  fi
  fuser -k "${POLICY_PORT}/tcp" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "=== Starting LeRobot async policy server on ${POLICY_HOST}:${POLICY_PORT} ==="
PYTHONUNBUFFERED=1 "$VENV_PYTHON" -m lerobot.async_inference.policy_server \
  --host="$POLICY_HOST" \
  --port="$POLICY_PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Policy server PID: $SERVER_PID"
sleep 3

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "Policy server failed to start. See: $SERVER_LOG"
  exit 1
fi

cmd=(
  "$VENV_PYTHON" scripts/evaluation/policy_inference.py
  --task="$TASK"
  --eval_rounds="$EVAL_ROUNDS"
  --step_hz="$STEP_HZ"
  --policy_type="$POLICY_TYPE"
  --policy_host="$POLICY_HOST"
  --policy_port="$POLICY_PORT"
  --policy_timeout_ms="$POLICY_TIMEOUT_MS"
  --policy_action_horizon="$POLICY_ACTION_HORIZON"
  --policy_language_instruction="$POLICY_LANGUAGE_INSTRUCTION"
  --policy_checkpoint_path="$POLICY_CHECKPOINT"
  --policy_device="$POLICY_DEVICE"
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

if [[ -n "$POLICY_CAMERA_REMAP" ]]; then
  cmd+=(--policy_camera_remap="$POLICY_CAMERA_REMAP")
fi

printf 'Running: %q ' "${cmd[@]}"
echo
PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$EVAL_LOG"
