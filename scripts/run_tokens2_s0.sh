#!/usr/bin/env bash
# Action-token follow-ups, seed 0, v0.2, val only; every model within 80-100% of the baseline's
# parameters. Baselines: runs/bc_v0.2_s0 (continuous) and runs/bc_tok256_v0.2_s0 (256 bins).
#   tok64      64 bins                                        (img_dim 120)
#   tok1024    1024 bins                                      (img_dim 88, head_hidden 64)
#   tokchunk   256 bins, 8-step action chunk; evaluated executing 1 step and 8 steps per query
#   tokpaired  256 bins + paired goal loss (both sentences on the same frame -> named target offset)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/tokens2_s0.log
: > "$LOG"

train() {  # name flags...
  local name=$1; shift
  local out="runs/${name}_s0"
  if [ -f "$out/model.pt" ]; then echo "skip $out" >> "$LOG"; return; fi
  echo "train $out $*  $(date +%T)" >> "$LOG"
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed 0 \
      --out "$out" "$@" >> "$LOG" 2>&1
}

train bc_tok64_v0.2 --action-bins 64 --img-dim 120 &
train bc_tok1024_v0.2 --action-bins 1024 --img-dim 88 --head-hidden 64 &
wait
train bc_tokchunk8_v0.2 --action-bins 256 --img-dim 104 --chunk 8 &
train bc_tokpaired_v0.2 --action-bins 256 --img-dim 104 --paired-goal &
wait

echo "evaluate $(date +%T)" >> "$LOG"
python -m dronevla.evaluate --data data/v0.2 --split val \
    --policy runs/bc_v0.2_s0 --policy runs/bc_tok256_v0.2_s0 \
    --policy runs/bc_tok64_v0.2_s0 --policy runs/bc_tok1024_v0.2_s0 \
    --policy runs/bc_tokchunk8_v0.2_s0 --policy "runs/bc_tokchunk8_v0.2_s0#exec=8" \
    --policy runs/bc_tokpaired_v0.2_s0 \
    --out reports/eval_tokens2_s0_val.json >> "$LOG" 2>&1
python scripts/candidates_table.py --eval reports/eval_tokens2_s0_val.json \
    --out reports/tokens2_s0_val_summary.json | tee -a "$LOG"
python scripts/instruction_sensitivity.py --data data/v0.2 --split train \
    --runs runs/bc_tok64_v0.2_s0 runs/bc_tok1024_v0.2_s0 runs/bc_tokchunk8_v0.2_s0 runs/bc_tokpaired_v0.2_s0 \
    --out reports/instruction_sensitivity_tokens2_train.json | tee -a "$LOG"
echo "done $(date +%T)" >> "$LOG"
