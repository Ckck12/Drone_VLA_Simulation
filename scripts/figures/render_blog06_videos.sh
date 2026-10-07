#!/usr/bin/env bash
# GIFs for blog part 6: val pair 0 (the first pair, not selected). The planner's CEM seed is the
# episode counter, so pair 0 (episodes 0 and 1) uses the same seeds as in the full evaluation.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/figures/blog06
mkdir -p "$OUT"
for dyn in integrator wm; do
  .venv/bin/python scripts/rollout_video.py --policy "plan:runs/wm_v1_s0:$dyn" --data data/v0.2 \
    --split val --pair 0 --out "$OUT/pair0_plan_$dyn.gif"
done
ls -la "$OUT"/*.gif
