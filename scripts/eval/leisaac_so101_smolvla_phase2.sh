#!/usr/bin/env bash
# Phase 2 wrapper: waits for the strict sweep to finish, applies a temporary
# wrist_roll slack patch (+-55deg instead of +-30deg) to the leisaac rest-pose
# range, runs a second 3-seed sweep, then restores the original tolerance.
# This gives us a side-by-side strict vs. slack comparison without touching the
# strict run that's already in flight.
set -euo pipefail

LEROBOT_PY="/home/modfi/models/physicai/extern/leisaac/source/leisaac/leisaac/assets/robots/lerobot.py"
LOG_DIR="/home/modfi/models/physicai/logs"
SWEEP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/leisaac_so101_smolvla_sweep.sh"
PHASE2_LOG="$LOG_DIR/v2-100-ep_sweep_phase2.log"

mkdir -p "$LOG_DIR"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$PHASE2_LOG"; }

wait_for_strict_sweep() {
  log "Waiting for strict sweep to exit (looking for leisaac_so101_smolvla_sweep.sh)..."
  while pgrep -f "leisaac_so101_smolvla_sweep.sh" >/dev/null; do
    sleep 30
  done
  log "Strict sweep exited."
  # also wait for any straggler children
  while pgrep -f "policy_inference.py" >/dev/null || pgrep -f "lerobot.async_inference.policy_server" >/dev/null; do
    sleep 5
  done
  fuser -k 8080/tcp >/dev/null 2>&1 || true
  sleep 3
}

apply_slack_patch() {
  log "Applying wrist_roll slack patch (+-55deg)..."
  cp "$LEROBOT_PY" "${LEROBOT_PY}.strict.bak"
  python3 - <<'PY'
from pathlib import Path
p = Path("/home/modfi/models/physicai/extern/leisaac/source/leisaac/leisaac/assets/robots/lerobot.py")
text = p.read_text()
old = '    "wrist_roll": (0.0 - 30.0, 0.0 + 30.0),  # 0 degree'
new = '    "wrist_roll": (0.0 - 55.0, 0.0 + 55.0),  # 0 degree (slacked from +-30 to +-55: v2-100-ep policy parks at +30..+52)'
assert old in text, "could not find target line for slack patch"
p.write_text(text.replace(old, new))
print("patch applied")
PY
}

restore_strict() {
  log "Restoring strict wrist_roll tolerance..."
  if [[ -f "${LEROBOT_PY}.strict.bak" ]]; then
    mv "${LEROBOT_PY}.strict.bak" "$LEROBOT_PY"
  fi
}

trap restore_strict EXIT

wait_for_strict_sweep
apply_slack_patch

log "Launching slack sweep (3 seeds x 10 episodes)..."
RUN_TAG="v2-100-ep_slack_sweep" \
SUMMARY_LOG="$LOG_DIR/v2-100-ep_slack_sweep_summary.log" \
bash "$SWEEP"

log "Slack sweep done."
