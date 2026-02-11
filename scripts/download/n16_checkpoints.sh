#!/usr/bin/env bash
# Download GR00T N1.6 checkpoints for G1 and BEHAVIOR R1 Pro tasks.

set -e
CKPT_DIR="/home/modfi/models/vla_simu/checkpoints"

echo "=== Downloading GR00T N1.6 G1 PnPAppleToPlate ==="
huggingface-cli download nvidia/GR00T-N1.6-G1-PnPAppleToPlate \
  --local-dir "$CKPT_DIR/n1.6-g1-pnp-apple"

echo ""
echo "=== Downloading GR00T N1.6 BEHAVIOR1k (R1 Pro) ==="
huggingface-cli download nvidia/GR00T-N1.6-BEHAVIOR1k \
  --local-dir "$CKPT_DIR/n1.6-behavior1k"

echo ""
echo "Done. Checkpoints at:"
echo "  G1:   $CKPT_DIR/n1.6-g1-pnp-apple"
echo "  R1:   $CKPT_DIR/n1.6-behavior1k"
