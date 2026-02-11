#!/usr/bin/env bash
# G1 loco-manipulation eval environment (MuJoCo-based).
# Installs gr00t_wbc, robosuite, robocasa inside the N1.6 worktree.
set -euxo pipefail

sudo apt-get update && sudo apt-get install -y libegl1-mesa-dev libglu1-mesa

cd /home/modfi/models/vla_simu/extern/Isaac-GR00T-n1.6
bash gr00t/eval/sim/GR00T-WholeBodyControl/setup_GR00T_WholeBodyControl.sh
