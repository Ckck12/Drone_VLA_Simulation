#!/usr/bin/env bash
# Season 2 candidates, ONE training seed (seed 0) per variant -- a quick screen, decided on the
# v0.2 validation split only (test untouched). Same protocol otherwise: 20 epochs, last epoch
# kept, closest-approach target scoring. A promising variant must be confirmed with more seeds
# before it is reported as a result.
#   cf        BC + counterfactual relabelling
#   cf_film   BC + counterfactual relabelling + FiLM
#   aux       BC + auxiliary offset head (the world model's privileged supervision)
#   attn      BC with language-conditioned cross-attention (87.5% of the baseline's parameters)
#   attn_cf   attn + counterfactual relabelling
#   planner   world model v1 seed 0 + goal selector + CEM (integrator / WM dynamics), oracle ceiling
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/candidates_s0.log
: > "$LOG"

# the seed-0 cf / cf_film runs started by run_season2_val.sh may still be training
while pgrep -f "dronevla.train .*_s0" > /dev/null; do sleep 30; done

train() {  # name flags...
  local name=$1; shift
  local out="runs/${name}_s0"
  if [ -f "$out/model.pt" ]; then echo "skip $out" >> "$LOG"; return; fi
  echo "train $out $*  $(date +%T)" >> "$LOG"
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed 0 \
      --out "$out" "$@" >> "$LOG" 2>&1
}

train bc_cf_v0.2 --counterfactual & train bc_cf_film_v0.2 --counterfactual --film & wait
train bc_aux_v0.2 --aux-offsets & train bc_attn_v0.2 --attn & wait
train bc_attn_cf_v0.2 --attn --counterfactual

echo "evaluate $(date +%T)" >> "$LOG"
python -m dronevla.evaluate --data data/v0.2 --split val \
    --policy expert --policy plan-oracle \
    --policy runs/bc_v0.2_s0 --policy runs/bc_cf_v0.2_s0 --policy runs/bc_cf_film_v0.2_s0 \
    --policy runs/bc_aux_v0.2_s0 --policy runs/bc_attn_v0.2_s0 --policy runs/bc_attn_cf_v0.2_s0 \
    --policy plan:runs/wm_v1_s0:integrator --policy plan:runs/wm_v1_s0:wm \
    --out reports/eval_candidates_s0_val.json >> "$LOG" 2>&1
python scripts/language_use.py reports/eval_candidates_s0_val.json | tee -a "$LOG"
for v in bc_attn_v0.2 bc_attn_cf_v0.2; do
  python scripts/attention_maps.py --run "runs/${v}_s0" --pairs 4 \
      --out "reports/figures/attention/${v}_s0.png" >> "$LOG" 2>&1
done
echo "done $(date +%T)" >> "$LOG"
