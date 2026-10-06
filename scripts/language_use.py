"""Success and language use per policy from a `dronevla.evaluate` report, averaged over seeds.

    .venv/bin/python scripts/language_use.py reports/eval_bc_scale_val.json

Which target an episode went for is scored by closest approach over the whole path, so
episodes that end out of bounds are still read correctly. "different targets" counts pairs where
the two instructions sent the drone to different targets; a policy that ignores the words scores
0 there and exactly half on "chose the instructed target". Runs named `<config>_s<seed>` are
grouped by `<config>` (taken from the policy name).
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dronevla.dataset import load_episodes  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("report", type=pathlib.Path)
ap.add_argument("--data", type=pathlib.Path, default=None,
                help="dataset holding the evaluated split's layouts (default: from the report)")
args = ap.parse_args()
rep = json.loads(args.report.read_text())
root = args.data or REPO_ROOT / f"data/v{rep['dataset'].rsplit('.', 1)[0]}"
hover = {}
for e in load_episodes(root):
    if e["split"] == rep["split"]:
        lay = json.loads((root / "layouts" / f"{e['layout_id']}.json").read_text())
        hover[e["pair_id"]] = np.asarray(lay["hover_points"])[:, :2]

print(f"{rep['split']} split of {root.name}, {rep['scenarios']} episodes")
print(f"{'policy':30s} {'success':>8s} {'pairs ok':>9s} {'diff targets':>12s} {'instructed':>11s}  outcomes")
groups = {}
for name, v in rep["policies"].items():
    s = v["summary"]
    chosen, by_pair = 0, {}
    for t in v["trials"]:
        p = np.asarray(t["path"])[:, :2]
        d = np.linalg.norm(p[:, None, :] - hover[t["pair_id"]][None], axis=-1).min(0)
        pick = int(np.argmin(d))
        chosen += pick == int(t["goal"])
        by_pair.setdefault(t["pair_id"], []).append(pick)
    diff = sum(len(set(x)) == 2 for x in by_pair.values())
    oc = {k: n for k, n in s["outcomes"].items() if n}
    print(f"{name:30s} {s['success']:3d}/{s['episodes']:<4d} {s['pair_success']:4d}/{s['pairs']:<4d} "
          f"{diff:6d}/{len(by_pair):<5d} {chosen:5d}/{len(v['trials']):<5d}  {oc}")
    if not v["oracle"]:
        key = re.sub(r"_s\d+$", "", name.split(" (")[0])
        groups.setdefault(key, []).append((s["success"], s["pair_success"], diff, chosen))
print()
for k, r in groups.items():
    r = np.array(r)
    print(f"{k}: {len(r)} seeds -- success {r[:, 0].mean():.1f} (per seed {r[:, 0].tolist()}), "
          f"pairs ok {r[:, 1].mean():.1f}, different targets {r[:, 2].mean():.1f} "
          f"(per seed {r[:, 2].tolist()}), chose instructed {r[:, 3].mean():.1f}")
