#!/usr/bin/env bash
# Queued after run_tokens2_s0.sh: cross-attention + 256-bin action tokens, seed 0, v0.2, val only.
# img_dim 104 keeps the model at 99.2% of the baseline's parameters.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/attn_tok_s0.log
: > "$LOG"
while pgrep -f run_tokens2_s0.sh > /dev/null; do sleep 30; done
if [ ! -f runs/bc_attn_tok256_v0.2_s0/model.pt ]; then
  echo "train $(date +%T)" >> "$LOG"
  OMP_NUM_THREADS=16 python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed 0 \
      --attn --action-bins 256 --img-dim 104 --out runs/bc_attn_tok256_v0.2_s0 >> "$LOG" 2>&1
fi
python -m dronevla.evaluate --data data/v0.2 --split val \
    --policy runs/bc_attn_v0.2_s0 --policy runs/bc_attn_tok256_v0.2_s0 \
    --out reports/eval_attn_tok_s0_val.json >> "$LOG" 2>&1
python scripts/candidates_table.py --eval reports/eval_attn_tok_s0_val.json \
    --out reports/attn_tok_s0_val_summary.json | tee -a "$LOG"
python scripts/instruction_sensitivity.py --data data/v0.2 --split train \
    --runs runs/bc_attn_tok256_v0.2_s0 --out reports/instruction_sensitivity_attn_tok_train.json | tee -a "$LOG"
python scripts/attention_maps.py --run runs/bc_attn_tok256_v0.2_s0 --pairs 4 \
    --out reports/figures/attention/bc_attn_tok256_v0.2_s0.png >> "$LOG" 2>&1
echo "done $(date +%T)" >> "$LOG"
