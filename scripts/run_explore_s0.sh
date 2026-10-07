#!/usr/bin/env bash
# Matching the world model's data: token+chunk BC trained on the demos PLUS the world model's
# exploration states (data/explore_v0.2/train, 19,896 states), each relabelled by the oracle
# expert toward one target chosen per episode -> 29,848 rows (the world model saw 30,448).
#   A  tokchunk8 + explore                     (img_dim 104, 97.9% params)
#   B  tokchunk8 + explore + aux offset head   (img_dim 100, 97.5% params)
# Sequential (each run peaks at ~3.2 GB RSS). Seed 0, v0.2, val only.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=16
LOG=reports/explore_s0.log
: > "$LOG"
train() {  # name flags...
  local name=$1; shift
  local out="runs/${name}_s0"
  if [ -f "$out/model.pt" ]; then echo "skip $out" >> "$LOG"; return; fi
  echo "train $out $*  $(date +%T)" >> "$LOG"
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed 0 \
      --out "$out" "$@" >> "$LOG" 2>&1
}
train bc_tokchunk8_explore_v0.2 --action-bins 256 --img-dim 104 --chunk 8 \
    --explore data/explore_v0.2/train
train bc_tokchunk8_explore_aux_v0.2 --action-bins 256 --img-dim 100 --chunk 8 --aux-offsets \
    --explore data/explore_v0.2/train
echo "evaluate $(date +%T)" >> "$LOG"
python -m dronevla.evaluate --data data/v0.2 --split val \
    --policy runs/bc_tokchunk8_v0.2_s0 --policy runs/bc_tokchunk8_explore_v0.2_s0 \
    --policy runs/bc_tokchunk8_explore_aux_v0.2_s0 \
    --out reports/eval_explore_s0_val.json >> "$LOG" 2>&1
python scripts/candidates_table.py --eval reports/eval_explore_s0_val.json \
    --out reports/explore_s0_val_summary.json | tee -a "$LOG"
python scripts/instruction_sensitivity.py --data data/v0.2 --split train \
    --runs runs/bc_tokchunk8_explore_v0.2_s0 runs/bc_tokchunk8_explore_aux_v0.2_s0 \
    --out reports/instruction_sensitivity_explore_train.json | tee -a "$LOG"
echo "done $(date +%T)" >> "$LOG"
