#!/usr/bin/env bash
# BEHAVIOR R1 Pro eval with GR00T N1.6 -- 50 diverse household tasks.
# Client-server: both run in conda "behavior" (has torch, flash-attn, gr00t, OmniGibson).
#
# Prerequisites:
#   1. Run scripts/setup/behavior_env.sh
#   2. GPU with RT cores
#
# Usage: ./scripts/eval/behavior_r1.sh [task_name]
set -e

LOCK_FILE="/tmp/behavior_r1.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another behavior run is already active. Stop it first."
  exit 1
fi

export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v isaacsim | paste -sd:)

TASK="${1:-turning_on_radio}"
PORT=5555
GROOT_DIR="/home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6"
LOG_DIR="/home/modfi/models/vla_simu/logs"
mkdir -p "$LOG_DIR"
MODEL_PATH="${MODEL_PATH:-nvidia/GR00T-N1.6-BEHAVIOR1k}"

if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
  GR00T_BEHAVIOR_HEADLESS="${GR00T_BEHAVIOR_HEADLESS:-0}"
else
  GR00T_BEHAVIOR_HEADLESS="${GR00T_BEHAVIOR_HEADLESS:-1}"
fi
echo "GR00T_BEHAVIOR_HEADLESS=$GR00T_BEHAVIOR_HEADLESS"

if [ "$GR00T_BEHAVIOR_HEADLESS" = "0" ]; then
  N_EPISODES="${N_EPISODES:-1}"
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-1200}"
  N_ACTION_STEPS="${N_ACTION_STEPS:-4}"
else
  N_EPISODES="${N_EPISODES:-10}"
  MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-999999999}"
  N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
fi
N_ENVS="${N_ENVS:-1}"
echo "MODEL_PATH=$MODEL_PATH N_EPISODES=$N_EPISODES MAX_EPISODE_STEPS=$MAX_EPISODE_STEPS N_ACTION_STEPS=$N_ACTION_STEPS N_ENVS=$N_ENVS"

CONDA_BASE="${CONDA_PREFIX:-$([ -d "$HOME/miniconda3" ] && echo "$HOME/miniconda3" || echo "$HOME/anaconda3")}"
source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
if ! command -v conda &>/dev/null || ! conda env list 2>/dev/null | grep -q "^behavior "; then
  echo "ERROR: conda env 'behavior' not found. Run scripts/setup/behavior_env.sh first."
  exit 1
fi

cd "$GROOT_DIR"

cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  fuser "$PORT/tcp" 2>/dev/null | xargs -r kill -9
}
trap cleanup EXIT

fuser "$PORT/tcp" 2>/dev/null | xargs -r kill -9 && sleep 1

echo "=== Starting GR00T N1.6 server ==="
PYTHONUNBUFFERED=1 stdbuf -oL -eL conda run -n behavior python -u /home/modfi/models/vla_simu/scripts/eval/_cuda_init_wrapper.py gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag BEHAVIOR_R1_PRO \
  --use-sim-policy-wrapper \
  > >(tee "$LOG_DIR/behavior_server.log") 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

tail -f "$LOG_DIR/behavior_server.log" &
TAIL_PID=$!

echo "Waiting for server (may take several minutes for model init)..."
for i in $(seq 1 120); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server process died. Check $LOG_DIR/behavior_server.log"
    kill "$TAIL_PID" 2>/dev/null
    exit 1
  fi
  if fuser "$PORT/tcp" >/dev/null 2>&1; then
    echo "Server ready."
    break
  fi
  sleep 5
  if [ "$i" -eq 120 ]; then
    echo "Server failed to start after 10 min. Check $LOG_DIR/behavior_server.log"
    kill "$TAIL_PID" 2>/dev/null
    exit 1
  fi
done
kill "$TAIL_PID" 2>/dev/null

echo "=== Running BEHAVIOR task: $TASK ==="
CLIENT_LOG="$LOG_DIR/behavior_${TASK}.log"
CLIENT_CMD="GR00T_BEHAVIOR_HEADLESS=${GR00T_BEHAVIOR_HEADLESS} PYTHONUNBUFFERED=1 TQDM_DISABLE=1 stdbuf -oL -eL conda run -n behavior python -u gr00t/eval/rollout_policy.py --n_episodes ${N_EPISODES} --policy_client_host 127.0.0.1 --policy_client_port ${PORT} --max_episode_steps ${MAX_EPISODE_STEPS} --env_name sim_behavior_r1_pro/${TASK} --n_action_steps ${N_ACTION_STEPS} --n_envs ${N_ENVS}"
set +e
script -qefc "$CLIENT_CMD" /dev/null | tee "$CLIENT_LOG"
CLIENT_STATUS=${PIPESTATUS[0]}
set -e

if [ "$CLIENT_STATUS" -ne 0 ]; then
  # Isaac/OmniGibson sometimes segfaults during shutdown after evaluation has completed.
  if grep -q "^results:" "$CLIENT_LOG" && grep -q "^success rate:" "$CLIENT_LOG"; then
    echo "Client exited non-zero after writing results; treating as completed run."
  else
    exit "$CLIENT_STATUS"
  fi
fi

echo "Done. Logs at $LOG_DIR/behavior_*.log"
