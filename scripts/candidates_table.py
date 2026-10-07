"""Summarise the single-seed Season 2 screen (reports/eval_candidates_s0_val.json).

Per policy, on the 10 val pairs: which target each episode went for (closest approach), split
into pairs flown correctly (both sentences to their own target), inverted (both to the other
one) and collapsed (both to the same target); plus success, outcomes, the "nearer"/"left"
choice rates, parameters and training time.

    python scripts/candidates_table.py --eval reports/eval_candidates_s0_val.json \
        --out reports/candidates_s0_val_summary.json
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import pyarrow.parquet as pq

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA = REPO / "data" / "v0.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=pathlib.Path, default=REPO / "reports/eval_candidates_s0_val.json")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "reports/candidates_s0_val_summary.json")
    args = ap.parse_args()
    ev = json.loads(args.eval.read_text())
    eps = {e["episode_id"]: e for e in pq.read_table(DATA / "episodes.parquet").to_pylist()}
    lays = {}

    def lay(i):
        if i not in lays:
            lays[i] = json.loads((DATA / "layouts" / f"{i}.json").read_text())
        return lays[i]

    out = {}
    for name, pol in ev["policies"].items():
        by_pair = collections.defaultdict(dict)
        near = left = 0
        for t in pol["trials"]:
            e = eps[t["episode_id"]]
            L = lay(e["layout_id"])
            p = np.asarray(t["path"])[:, :2]
            hp = np.asarray(L["hover_points"])[:, :2]
            chosen = int(np.argmin([np.min(np.linalg.norm(p - h, axis=1)) for h in hp]))
            start = np.asarray(L["start_xyz"][:2])
            near += chosen == int(np.argmin([np.linalg.norm(h - start) for h in hp]))
            left += chosen == int(np.argmax(hp[:, 1]))
            by_pair[t["pair_id"]][t["goal"]] = (chosen, t["outcome"])
        correct = inverted = collapsed = pairs_ok = 0
        for g in by_pair.values():
            c0, c1 = g[0][0], g[1][0]
            if c0 == 0 and c1 == 1:
                correct += 1
            elif c0 == 1 and c1 == 0:
                inverted += 1
            else:
                collapsed += 1
            pairs_ok += g[0][1] == "success" and g[1][1] == "success"
        cfg = pol.get("policy_config") or {}
        out[name] = {
            "success": sum(t["outcome"] == "success" for t in pol["trials"]),
            "pairs_success": pairs_ok,
            "pairs_correct": correct, "pairs_inverted": inverted, "pairs_collapsed": collapsed,
            "chose_nearer": int(near), "chose_left": int(left),
            "outcomes": dict(collections.Counter(t["outcome"] for t in pol["trials"])),
            "parameters": cfg.get("parameters"),
            "train_s": (cfg.get("train") or {}).get("train_s"),
        }
    args.out.write_text(json.dumps(out, indent=1))
    hdr = f"{'policy':44s} succ pairOK  correct inverted collapsed  nearer left   params  train_s"
    print(hdr)
    for k, v in out.items():
        print(f"{k[:44]:44s} {v['success']:>4} {v['pairs_success']:>6}  {v['pairs_correct']:>7} "
              f"{v['pairs_inverted']:>8} {v['pairs_collapsed']:>9}  {v['chose_nearer']:>6} "
              f"{v['chose_left']:>4}  {str(v['parameters']):>7}  {str(round(v['train_s'] or 0))}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
