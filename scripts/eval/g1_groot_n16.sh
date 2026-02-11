#!/usr/bin/env bash
# G1 loco-manipulation eval with GR00T N1.6 (PnPAppleToPlate).
# Uses client-server architecture: server loads model, client runs MuJoCo sim.
#
# Prerequisites:
#   1. Run scripts/setup/g1_wbc_env.sh
#   2. Checkpoint from HuggingFace.
#
# Usage: run this script, it launches server in background then client.
set -e

GROOT_DIR="/home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6"
LOG_DIR="/home/modfi/models/vla_simu/logs"
WBC_VENV="$GROOT_DIR/gr00t/eval/sim/GR00T-WholeBodyControl/GR00T-WholeBodyControl_uv/.venv/bin/python"
mkdir -p "$LOG_DIR"

cd "$GROOT_DIR"

echo "=== Starting GR00T N1.6 server (G1 PnPAppleToPlate) ==="
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
  --embodiment-tag UNITREE_G1 \
  --use-sim-policy-wrapper \
  2>&1 | tee "$LOG_DIR/g1_n16_server.log" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# polls port 5555
echo "Waiting for server to start..."
for i in $(seq 1 60); do
  if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1',5555)); s.close()" 2>/dev/null; then
    echo "Server ready."
    break
  fi
  sleep 5
  if [ $i -eq 60 ]; then
    echo "Server failed to start after 5 min. Check $LOG_DIR/g1_n16_server.log"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
done

echo "=== Starting G1 rollout client ==="
"$WBC_VENV" gr00t/eval/rollout_policy.py \
  --n_episodes 10 \
  --max_episode_steps 1440 \
  --env_name gr00tlocomanip_g1_sim/LMPnPAppleToPlateDC_G1_gear_wbc \
  --n_action_steps 20 \
  --n_envs 5 \
  --policy_client_host 127.0.0.1 \
  --policy_client_port 5555 \
  2>&1 | tee "$LOG_DIR/g1_n16_client.log"

echo "=== Shutting down server ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done. Logs at $LOG_DIR/g1_n16_{server,client}.log"
