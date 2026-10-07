#!/usr/bin/env bash
# Attention variants, same protocol as scripts/run_season2_val.sh (v0.2, val only, 3 seeds,
# 20 epochs, last epoch kept):
#   B1  BC with language-conditioned cross-attention pooling        (--attn)
#   B2  B1 + counterfactual relabelling                             (--attn --counterfactual)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/attn_val.log
: > "$LOG"

train() {  # name seed flags...
  local name=$1 seed=$2; shift 2
  local out="runs/${name}_s${seed}"
  if [ -f "$out/model.pt" ]; then echo "skip $out" >> "$LOG"; return; fi
  echo "train $out $*  $(date +%T)" >> "$LOG"
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed "$seed" \
      --out "$out" "$@" >> "$LOG" 2>&1
}

for seed in 0 1 2; do
  train bc_attn_v0.2 "$seed" --attn &
  train bc_attn_cf_v0.2 "$seed" --attn --counterfactual &
  wait
done

POL=()
for seed in 0 1 2; do
  POL+=(--policy "runs/bc_attn_v0.2_s$seed" --policy "runs/bc_attn_cf_v0.2_s$seed")
done
echo "evaluate $(date +%T)" >> "$LOG"
python -m dronevla.evaluate --data data/v0.2 --split val "${POL[@]}" \
    --out reports/eval_attn_val.json >> "$LOG" 2>&1
python scripts/language_use.py reports/eval_attn_val.json | tee -a "$LOG"
for seed in 0 1 2; do
  for v in bc_attn_v0.2 bc_attn_cf_v0.2; do
    python scripts/attention_maps.py --run "runs/${v}_s$seed" --pairs 4 \
        --out "reports/figures/attention/${v}_s${seed}.png" >> "$LOG" 2>&1
  done
done
echo "done $(date +%T)" >> "$LOG"
