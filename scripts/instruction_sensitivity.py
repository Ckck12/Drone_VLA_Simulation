"""Open-loop instruction sensitivity of trained BC policies (dataset v0.2, fixed protocol).

For every recorded state, ask each policy for an action twice: with the episode's own
instruction, and with its pair partner's instruction (the *other* target). Do the same for
the oracle expert, whose action toward the other goal is recomputed from the logged true
state (dronevla.train.counterfactual_relabel, whose own-goal reconstruction is checked).

Reported per step bucket (t = 0 is the identical first frame of both pair members):
  * swap_gap: |vy(own) - vy(other)|, the lateral change caused by changing only the words
  * lateral_err: |vy(own) - expert vy(own)|, how well the own-instruction label is fit
  * direction_ok (t = 0 only): sign of vy(own) - vy(other) matches the expert's

    python scripts/instruction_sensitivity.py --data data/v0.2 --split train \
        --runs runs/bc_v0.2_s0 runs/bc_v0.2_s1 runs/bc_v0.2_s2 \
        --out reports/instruction_sensitivity_v0.2.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dronevla.dataset import load_episodes                     # noqa: E402
from dronevla.evaluate import LearnedPolicy                    # noqa: E402
from dronevla.model import pad_batch                           # noqa: E402
from dronevla.task import TaskConfig                           # noqa: E402
from dronevla.train import counterfactual_relabel, load_rows   # noqa: E402

BUCKETS = [("t = 0", 0, 0), ("t = 1-4", 1, 4), ("t = 5-9", 5, 9), ("t >= 10", 10, 10 ** 6)]


def step_index(episode_ids):
    out, last, k = [], None, 0
    for e in episode_ids:
        k = k + 1 if e == last else 0
        out.append(k)
        last = e
    return np.asarray(out)


def policy_vy(pol, data, instr, batch=256):
    vy = []
    with torch.no_grad():
        for s in range(0, len(instr), batch):
            idx = list(range(s, min(s + batch, len(instr))))
            ids = pad_batch([[1] if pol.no_language else pol.vocab.encode(instr[i]) for i in idx])
            prop = (data["proprio"][idx] - pol.mean) / pol.std
            motion, _ = pol.model(data["rgb"][idx], prop, pol.model.encode_text(ids))
            vy.append((motion * pol.caps)[:, 1])
    return torch.cat(vy).numpy()


def summarise(values, t, lo, hi):
    m = (t >= lo) & (t <= hi)
    v = values[m]
    return {"n": int(m.sum()), "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)), "p75": float(np.percentile(v, 75))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=REPO / "data/v0.2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    episodes = [e for e in load_episodes(args.data) if e["split"] == args.split]
    data = load_rows(args.data, episodes)
    aug, check = counterfactual_relabel(args.data, episodes, data, TaskConfig())
    n = len(data["instr"])
    cf_instr = aug["instr"][n:]
    exp_own = data["action"][:, 1].numpy()
    exp_cf = aug["action"][n:, 1].numpy()
    t = step_index(data["episode_id"])

    res = {"data": str(args.data), "split": args.split, "rows": n,
           "episodes": len(episodes), "relabel_check": check,
           "share_of_rows": {name: float(((t >= lo) & (t <= hi)).mean())
                             for name, lo, hi in BUCKETS},
           "expert": {"swap_gap_mps": {name: summarise(np.abs(exp_own - exp_cf), t, lo, hi)
                                       for name, lo, hi in BUCKETS}},
           "policies": {}}
    first = t == 0
    for run in args.runs:
        pol = LearnedPolicy(pathlib.Path(run))
        own = policy_vy(pol, data, data["instr"])
        oth = policy_vy(pol, data, cf_instr)
        dir_ok = np.sign(own - oth)[first] == np.sign(exp_own - exp_cf)[first]
        res["policies"][pathlib.Path(run).name] = {
            "no_language": pol.no_language,
            "swap_gap_mps": {name: summarise(np.abs(own - oth), t, lo, hi)
                             for name, lo, hi in BUCKETS},
            "lateral_err_mps": {name: summarise(np.abs(own - exp_own), t, lo, hi)
                                for name, lo, hi in BUCKETS},
            "first_frame_direction_correct": f"{int(dir_ok.sum())}/{int(first.sum())}",
            "first_frame": {"policy_gap": np.abs(own - oth)[first].tolist(),
                            "expert_gap": np.abs(exp_own - exp_cf)[first].tolist()},
        }
        print(run, {k: round(v["median"], 4) for k, v in
                    res["policies"][pathlib.Path(run).name]["swap_gap_mps"].items()},
              "dir", res["policies"][pathlib.Path(run).name]["first_frame_direction_correct"])
    print("expert", {k: round(v["median"], 4) for k, v in res["expert"]["swap_gap_mps"].items()})
    print("share of rows", res["share_of_rows"], "relabel check", check)
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
