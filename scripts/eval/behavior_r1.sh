#!/usr/bin/env bash
# BEHAVIOR R1 Pro eval with GR00T N1.6 -- 50 diverse household tasks.
# Uses client-server architecture: server loads model, client runs OmniGibson sim.
#
# Prerequisites:
#   1. Run scripts/setup/behavior_env.sh
#   2. GPU with some RT cores
#   3. Checkpoint from HuggingFace.
#
# Pass task name as first arg, default: turning_on_radio.
# Example: ./scripts/eval/behavior_r1.sh carrying_in_groceries
set -e

TASK="${1:-turning_on_radio}"
GROOT_DIR="/home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6"
LOG_DIR="/home/modfi/models/vla_simu/logs"
mkdir -p "$LOG_DIR"

cd "$GROOT_DIR"

echo "=== Starting GR00T N1.6 server (BEHAVIOR R1 Pro) ==="
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-BEHAVIOR1k \
  --embodiment-tag BEHAVIOR_R1_PRO \
  --use-sim-policy-wrapper \
  2>&1 | tee "$LOG_DIR/behavior_server.log" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to start..."
for i in $(seq 1 60); do
  if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',5555)); s.close()" 2>/dev/null; then
    echo "Server ready."
    break
  fi
  sleep 5
  if [ $i -eq 60 ]; then
    echo "Server failed to start after 5 min. Check $LOG_DIR/behavior_server.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
done

echo "=== Running BEHAVIOR task: $TASK ==="
uv run python gr00t/eval/rollout_policy.py \
  --n_episodes 10 \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  --max_episode_steps 999999999 \
  --env_name "sim_behavior_r1_pro/$TASK" \
  --n_action_steps 8 \
  --n_envs 1 \
  2>&1 | tee "$LOG_DIR/behavior_${TASK}.log"

echo "=== Shutting down server ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done. Logs at $LOG_DIR/behavior_*.log"
