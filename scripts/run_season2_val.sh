#!/usr/bin/env bash
# Season 2 candidates, decided on the v0.2 validation split only (test untouched).
# Fixed protocol: 3 training seeds, 20 epochs, last epoch kept, closest-approach target scoring.
#   A1  BC + counterfactual relabelling            (--counterfactual)
#   A2  BC + counterfactual relabelling + FiLM     (--counterfactual --film)
#   A3  BC + auxiliary offset head                 (--aux-offsets; same privileged supervision as the WM)
#   B   world-model planner, integrator / WM dynamics, 3 WM seeds; plus the oracle planner ceiling
# Baseline: runs/bc_v0.2_s{0,1,2} (already trained, same protocol).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=8
LOG=reports/season2_val.log
: > "$LOG"

train() {  # name seed flags...
  local name=$1 seed=$2; shift 2
  local out="runs/${name}_s${seed}"
  if [ -f "$out/model.pt" ]; then echo "skip $out" >> "$LOG"; return; fi
  echo "train $out $*  $(date +%T)" >> "$LOG"
  python -m dronevla.train --data data/v0.2 --epochs 20 --select last --seed "$seed" \
      --out "$out" "$@" >> "$LOG" 2>&1
}

# two trainings at a time (16 logical CPUs, 8 torch threads each)
for seed in 0 1 2; do
  train bc_cf_v0.2 "$seed" --counterfactual &
  train bc_cf_film_v0.2 "$seed" --counterfactual --film &
  wait
done
train bc_aux_v0.2 0 --aux-offsets & train bc_aux_v0.2 1 --aux-offsets & wait
train bc_aux_v0.2 2 --aux-offsets

POL=()
for seed in 0 1 2; do
  POL+=(--policy "runs/bc_v0.2_s$seed" --policy "runs/bc_cf_v0.2_s$seed"
        --policy "runs/bc_cf_film_v0.2_s$seed" --policy "runs/bc_aux_v0.2_s$seed"
        --policy "plan:runs/wm_v1_s$seed:integrator" --policy "plan:runs/wm_v1_s$seed:wm")
done
echo "evaluate $(date +%T)" >> "$LOG"
python -m dronevla.evaluate --data data/v0.2 --split val --policy expert --policy plan-oracle \
    "${POL[@]}" --out reports/eval_season2_val.json >> "$LOG" 2>&1
python scripts/language_use.py reports/eval_season2_val.json | tee -a "$LOG"
echo "done $(date +%T)" >> "$LOG"
