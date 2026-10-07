#!/usr/bin/env bash
# OpenVLA-style action tokens (256 bins on vx, vy; cross-entropy), seed 0, v0.2, val only.
# Same inputs/backbone as the baseline except the head; img_dim 104 keeps parameters at 97.7%.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/tokens_s0.log
: > "$LOG"
if [ ! -f runs/bc_tok256_v0.2_s0/model.pt ]; then
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed 0 \
      --action-bins 256 --img-dim 104 --out runs/bc_tok256_v0.2_s0 >> "$LOG" 2>&1
fi
python -m dronevla.evaluate --data data/v0.2 --split val --policy runs/bc_v0.2_s0 \
    --policy runs/bc_tok256_v0.2_s0 --out reports/eval_tokens_s0_val.json >> "$LOG" 2>&1
python scripts/language_use.py reports/eval_tokens_s0_val.json | tee -a "$LOG"
python scripts/candidates_table.py --eval reports/eval_tokens_s0_val.json \
    --out reports/tokens_s0_val_summary.json | tee -a "$LOG"
python scripts/instruction_sensitivity.py --data data/v0.2 --split train \
    --runs runs/bc_tok256_v0.2_s0 --out reports/instruction_sensitivity_tok256_train.json | tee -a "$LOG"
echo "done $(date +%T)" >> "$LOG"
