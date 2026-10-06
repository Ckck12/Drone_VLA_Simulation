#!/usr/bin/env bash
# Rollout GIFs for blog part 4: the v0.2 seed-0 policy and the expert on the same val pairs.
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate
for pair in 0 3; do
  python scripts/rollout_video.py --policy runs/bc_v0.2_s0 --data data/v0.2 --split val \
    --pair "$pair" --out "reports/videos/bc_v0.2_s0_val_pair${pair}.gif"
  python scripts/rollout_video.py --policy expert --data data/v0.2 --split val \
    --pair "$pair" --out "reports/videos/expert_v0.2_val_pair${pair}.gif"
done
ls -la reports/videos
