"""World-model stage-1 diagnostics beyond logged-action rollouts. Val only, never test.

    .venv/bin/python scripts/wm_diagnostics.py --run runs/wm_v0 --out reports/wm_v0_diagnostics.json

1. k=0 offset error (perception): split by whether the colour's hover point is inside the
   camera's horizontal field of view, and for the instructed target binned by its true distance
   (the near bins decide whether a closed-loop planner can finish the task).
2. Drift: the WM rollout vs command integration started from the WM's *own* k=0 estimate, so
   the learned dynamics are compared with an integrator on equal footing. Also on random-walk
   flights alone, where the scene cannot predict the motion.
3. Action probe: unroll k steps under a constant command and compare the predicted offset change
   with the zero-command rollout. Moving at a (body frame) should shift every offset by -a*DT*k,
   so slope 1.0 = moves exactly as commanded, 0 = ignores the action, <0 = wrong sign.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from dronevla.world_model import (COLOUR_ORDER, DT, OFFSET_SCALE, REPO_ROOT,  # noqa: E402
                                  batch_tensors, episode_list, load_model, load_sources, windows)

HFOV_HALF = math.atan(math.tan(math.radians(47 / 2)) * 128 / 96)       # ~30.1 deg
DIST_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, math.inf))


def med(x):
    return float(np.median(x)) if len(x) else None


def binned(err, dist):
    out = {}
    for lo, hi in DIST_BINS:
        sel = (dist >= lo) & (dist < hi)
        out[f"{lo}-{hi} m"] = {"n": int(sel.sum()), "median_error_m": med(err[sel]),
                               "frac_error_below_0.4m": float((err[sel] < 0.4).mean()) if sel.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, default=REPO_ROOT / "runs/wm_v0")
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "reports/wm_v0_diagnostics.json")
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()
    model, norm, cfg = load_model(args.run)
    H = args.horizon
    out = {"run": args.run.name, "hfov_half_deg": round(math.degrees(HFOV_HALF), 2)}

    demo_root = REPO_ROOT / "data/v0.1"
    goal_colour = {}
    for e in episode_list(demo_root, "val"):
        lay = json.loads((demo_root / "layouts" / f"{e['layout_id']}.json").read_text())
        goal_colour[e["episode_id"]] = COLOUR_ORDER.index(lay["targets"][e["goal_index"]]["color"])
    sets = {"val expert demos": load_sources([(demo_root, "val")]),
            "val exploration": load_sources([(REPO_ROOT / "data/explore_v0.1/val", None)])}
    sets["val random_walk only"] = [e for e in sets["val exploration"] if "random_walk" in e["episode_id"]]

    for name, eps in sets.items():
        win = windows(eps, H, stride=2)
        e_in, e_out, goal_err, goal_dist = [], [], [], []
        ks = (1, 5, H)
        drift_wm, drift_cmd = {k: [] for k in ks}, {k: [] for k in ks}
        with torch.no_grad():
            for s in range(0, len(win), 64):
                idx = win[s:s + 64]
                rgb, prop, act, off, mask = batch_tensors(eps, idx, H)
                B = len(idx)
                z = model.encode(rgb[:, 0], norm.prop(prop[:, 0]))
                p0 = (model.head_off(z) * OFFSET_SCALE).reshape(B, -1, 2)
                t0 = off[:, 0].reshape(B, -1, 2)
                e0 = torch.linalg.norm(p0 - t0, dim=-1)
                inview = (t0[..., 0] > 0) & (torch.atan2(t0[..., 1], t0[..., 0]).abs() < HFOV_HALF)
                m0 = mask[:, 0] > 0.5
                e_in += e0[m0 & inview].tolist()
                e_out += e0[m0 & ~inview].tolist()
                if name == "val expert demos":
                    g = torch.tensor([goal_colour[eps[ei]["episode_id"]] for ei, _ in idx])
                    goal_err += e0[torch.arange(B), g].tolist()
                    goal_dist += torch.linalg.norm(t0[torch.arange(B), g], dim=-1).tolist()
                cum = torch.zeros(B, 2)
                for k in range(1, H + 1):
                    z = model.step(z, norm.act(act[:, k - 1]))
                    cum = cum + act[:, k - 1] * DT
                    if k in ks:
                        tk = off[:, k].reshape(B, -1, 2)
                        mk = mask[:, k] > 0.5
                        pw = (model.head_off(z) * OFFSET_SCALE).reshape(B, -1, 2)
                        drift_wm[k] += torch.linalg.norm(pw - tk, dim=-1)[mk].tolist()
                        drift_cmd[k] += torch.linalg.norm(p0 - cum[:, None] - tk, dim=-1)[mk].tolist()
        r = {"windows": len(win),
             "k0_offset_error_m": {"in_view": med(e_in), "out_of_view": med(e_out),
                                   "n_in_view": len(e_in), "n_out_of_view": len(e_out)},
             "offset_error_m_from_wm_k0": {k: {"wm_rollout": med(drift_wm[k]),
                                               "cmd_int_from_wm_k0": med(drift_cmd[k])} for k in ks}}
        if goal_err:
            r["instructed_target_k0_error_all_m"] = med(goal_err)
            r["instructed_target_k0_error_by_true_distance"] = binned(np.array(goal_err), np.array(goal_dist))
        out[name] = r

    eps = sets["val expert demos"] + sets["val exploration"]
    win = windows(eps, 1, stride=4)
    probe = {}
    with torch.no_grad():
        rgb = torch.stack([torch.from_numpy(eps[ei]["rgb"][t]) for ei, t in win])
        prop = torch.stack([torch.from_numpy(eps[ei]["prop"][t]) for ei, t in win])
        msk = torch.stack([torch.from_numpy(eps[ei]["mask"][t]) for ei, t in win]) > 0.5
        z0 = model.encode(rgb, norm.prop(prop))
        B = len(win)

        def roll(a, k):
            z = z0
            for _ in range(k):
                z = model.step(z, norm.act(a.expand(B, 2)))
            return (model.head_off(z) * OFFSET_SCALE).reshape(B, -1, 2)

        for k in (1, 3, 5):
            base = roll(torch.zeros(2), k)
            for axis, lab in ((0, "x"), (1, "y")):
                for sgn, sname in ((1, "+"), (-1, "-")):
                    a = torch.zeros(2)
                    a[axis] = 0.5 * sgn
                    d = roll(a, k) - base
                    expect = float(-a[axis] * DT * k)
                    probe[f"k{k} {sname}{lab}"] = {
                        "slope": float((d[..., axis][msk] / expect).median()),
                        "expected_shift_m": abs(expect),
                        "cross_axis_abs_m": float(d[..., 1 - axis][msk].abs().median())}
    out["action_probe"] = {"states": B, "note": "slope 1.0 = offset moves exactly as commanded",
                           "results": probe}
    print(json.dumps(out, indent=1))
    args.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
