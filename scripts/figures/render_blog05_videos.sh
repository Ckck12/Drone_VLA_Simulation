#!/usr/bin/env bash
# GIFs for blog part 5: val pair 7 (chosen because chunk 8 succeeds on both sentences there).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=reports/figures/blog05
mkdir -p "$OUT"
for run in bc_tokchunk8_v0.2_s0 bc_v0.2_s0 bc_tok256_v0.2_s0; do
  .venv/bin/python scripts/rollout_video.py --policy "runs/$run" --data data/v0.2 --split val \
    --pair 7 --out "$OUT/pair7_$run.gif"
done
ls -la "$OUT"/*.gif
