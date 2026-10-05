"""Closed-loop evaluation on a dataset split's scenarios (roadmap v3 §7.1-7.2, Phase 1 item 6).

    python -m dronevla.evaluate --split test --policy expert --policy runs/bc_text \
        --policy runs/bc_nolang --out reports/eval_v0.1_test.json

Every trial is reset snapshot -> fixed instruction -> current RGB/state -> policy ->
adapter/PID -> physics -> new RGB, until the env ends the episode. Each policy runs on the
same scenarios (the split's pairs: same layout, seed and instruction family), so results
compare like with like.

**Learned policies get the observation only.** `LearnedPolicy.act(obs)` has no access to
`info`, so it cannot read the goal, coordinates or anything else privileged -- the
interface enforces §7.1's "evaluator만 privileged state를 읽는다". The scripted expert
needs `info["privileged"]` by construction; it is labelled `oracle` and is a reference,
not a competitor (§7.2).
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

from dronevla.action_adapter import ActionLimits
from dronevla.dataset import load_episodes
from dronevla.env import OUTCOMES, DroneTargetPairsEnv
from dronevla.expert import StraightLineExpert
from dronevla.model import TinyBC, Vocab, pad_batch, upgrade_state_dict
from dronevla.task import Layout, Target, TaskConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- policies
class OraclePolicy:
    oracle = True

    def __init__(self, cfg):
        self.name = "expert (oracle)"
        self.expert = StraightLineExpert(cfg)

    def reset(self, instruction):
        pass

    def act_privileged(self, obs, info):
        return self.expert.act(info)


class LearnedPolicy:
    oracle = False

    def __init__(self, run_dir: pathlib.Path):
        run_dir = pathlib.Path(run_dir)
        self.config = json.loads((run_dir / "config.json").read_text())
        prep = self.config["prep"]
        self.no_language = prep["no_language"]
        self.name = f"{run_dir.name} ({'no language' if self.no_language else 'RGB+text'} BC)"
        self.vocab = Vocab([])
        self.vocab.itos = prep["vocab"]
        self.vocab.stoi = {t: i for i, t in enumerate(self.vocab.itos)}
        kwargs = self.config.get("model_kwargs", {})
        self.model = TinyBC(len(self.vocab.itos), **kwargs)
        self.model.load_state_dict(upgrade_state_dict(
            torch.load(run_dir / "model.pt", weights_only=True)))
        if kwargs.get("film"):
            self.name = self.name.replace(" BC)", " BC, FiLM)")
        self.model.eval()
        self.mean = torch.tensor(prep["proprio_mean"])
        self.std = torch.tensor(prep["proprio_std"])
        self.caps = torch.tensor(prep["caps"])
        self.threshold = self.config["stop_threshold"]
        self.unknown_token_rate = 0.0

    def reset(self, instruction):
        """Encode the instruction once per episode (§3.1: cached at task start)."""
        ids = [[1]] if self.no_language else [self.vocab.encode(instruction)]
        self.unknown_token_rate = 0.0 if self.no_language else self.vocab.unknown_rate(instruction)
        with torch.no_grad():
            self.text_vec = self.model.encode_text(pad_batch(ids))

    def act(self, obs) -> np.ndarray:
        rgb = torch.from_numpy(np.ascontiguousarray(obs["rgb"])).unsqueeze(0)
        prop = (torch.from_numpy(obs["proprio"]).unsqueeze(0) - self.mean) / self.std
        with torch.no_grad():
            motion, logit = self.model(rgb, prop, self.text_vec)
        m = (motion[0] * self.caps).numpy()
        # positive exactly when the logit clears the threshold chosen on validation
        return np.array([*m, float(logit[0]) - self.threshold], dtype=np.float32)


def load_policy(spec, cfg):
    return OraclePolicy(cfg) if spec == "expert" else LearnedPolicy(pathlib.Path(spec))


# -------------------------------------------------------------------------- scenarios
def layout_from_dict(d) -> Layout:
    return Layout(start_xyz=tuple(d["start_xyz"]), start_yaw=d["start_yaw"],
                  targets=tuple(Target(**t) for t in d["targets"]),
                  hover_points=tuple(tuple(h) for h in d["hover_points"]),
                  sample_seed=d["sample_seed"], rejected_before=d["rejected_before"])


def scenarios(root, split, only_pair=None):
    eps = [e for e in load_episodes(root) if e["split"] == split]
    out = []
    for e in sorted(eps, key=lambda e: e["episode_id"]):
        if only_pair is not None and e["pair_id"] != only_pair:
            continue
        lay = layout_from_dict(json.loads((root / "layouts" / f"{e['layout_id']}.json").read_text()))
        out.append({"episode_id": e["episode_id"], "pair_id": e["pair_id"],
                    "seed": e["snapshot_seed"], "goal": e["goal_index"],
                    "family": e["instruction_family"], "layout": lay})
    return out


# ------------------------------------------------------------------------- one trial
def run_trial(env, policy, sc):
    obs, info = env.reset(seed=sc["seed"], options={"layout": sc["layout"], "goal_index": sc["goal"],
                                                    "instruction_family": sc["family"]})
    policy.reset(obs["instruction"])
    latencies, capped, path = [], 0, [info["privileged"]["true_pos"]]
    while True:
        t0 = time.perf_counter()
        if policy.oracle:
            action = policy.act_privileged(obs, info)
        else:
            action = policy.act(obs)                   # obs only
        latencies.append((time.perf_counter() - t0) * 1e3)
        obs, _, term, trunc, info = env.step(action)
        path.append(info["privileged"]["true_pos"])
        if info["action"]["flags"].get("horizontal_capped"):
            capped += 1
        if term or trunc:
            break
    p = info["privileged"]
    return {"episode_id": sc["episode_id"], "pair_id": sc["pair_id"], "goal": sc["goal"],
            "instruction": obs["instruction"], "outcome": info["outcome"],
            "steps": len(latencies), "episode_t": info["episode_t"],
            "final_dist_to_goal": p["dist_to_goal"], "final_dist_to_other": p["dist_to_other"],
            "reached_goal": p["reached_goal"], "reached_other": p["reached_other"],
            "contact_with": p["contact_with"], "capped_actions": capped,
            "latency_ms": latencies, "path": path,
            "unknown_token_rate": getattr(policy, "unknown_token_rate", 0.0)}


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    # clamp: at k = 0 the lower bound is 0 up to rounding, which printed as "-0.0"
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def summarise(trials):
    n = len(trials)
    succ = sum(t["outcome"] == "success" for t in trials)
    pairs = {}
    for t in trials:
        pairs.setdefault(t["pair_id"], []).append(t["outcome"] == "success")
    pair_ok = sum(all(v) and len(v) == 2 for v in pairs.values())
    reach = sum(t["reached_goal"] for t in trials)
    lat = np.concatenate([t["latency_ms"] for t in trials])
    acts = sum(t["steps"] for t in trials)
    ne = np.array([t["final_dist_to_goal"] for t in trials])
    return {
        "episodes": n, "success": succ, "SR": succ / n, "SR_wilson95": wilson(succ, n),
        "pairs": len(pairs), "pair_success": pair_ok, "pair_SR": pair_ok / len(pairs),
        "pair_SR_wilson95": wilson(pair_ok, len(pairs)),
        "outcomes": {o: sum(t["outcome"] == o for t in trials) for o in OUTCOMES},
        "wrong_target_rate": sum(t["outcome"] == "wrong_target" for t in trials) / n,
        "OSR": reach / n, "stop_gap": (reach - succ) / n,
        "NE_m": {"median": float(np.median(ne)), "mean": float(ne.mean())},
        "policy_latency_ms": {"p50": float(np.percentile(lat, 50)),
                              "p95": float(np.percentile(lat, 95)),
                              "p99": float(np.percentile(lat, 99))},
        "action_clip_rate": sum(t["capped_actions"] for t in trials) / max(acts, 1),
    }


def figure(results, split, out_png, max_pairs=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from dronevla.task import COLORS

    names = list(results)
    pair_ids = list(dict.fromkeys(t["pair_id"] for t in results[names[0]]["trials"]))[:max_pairs]
    fig, axes = plt.subplots(len(pair_ids), len(names), figsize=(3.6 * len(names), 3.0 * len(pair_ids)),
                             squeeze=False)
    for c, name in enumerate(names):
        res = results[name]
        for r, pid in enumerate(pair_ids):
            ax = axes[r][c]
            for t in [t for t in res["trials"] if t["pair_id"] == pid]:
                lay = res["layouts"][pid]
                for i, tg in enumerate(lay["targets"]):
                    col = COLORS[tg["color"]][:3]
                    x, y = tg["xy"]
                    rr = tg["radius"]
                    ax.add_patch(Rectangle((x - rr, y - rr), 2 * rr, 2 * rr, color=col) if tg["shape"] == "box"
                                 else Circle((x, y), rr, color=col))
                    hx, hy, _ = lay["hover_points"][i]
                    ax.add_patch(Circle((hx, hy), 0.4, fill=False, ls="--", color=col, lw=0.8))
                goal_col = COLORS[lay["targets"][t["goal"]]["color"]][:3]
                p = np.array(t["path"])
                ax.plot(p[:, 0], p[:, 1], "-" if t["goal"] == 0 else "--", color=goal_col, lw=1.6,
                        label=f"{lay['targets'][t['goal']]['color']}: {t['outcome']}")
            ax.set_xlim(-3.5, 3.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
            ax.tick_params(labelsize=6); ax.legend(fontsize=6, loc="lower left")
            if r == 0:
                ax.set_title(name, fontsize=8)
    fig.suptitle(f"Closed loop on {split} pairs: line colour = the instructed target", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.evaluate")
    ap.add_argument("--data", type=pathlib.Path, default=REPO_ROOT / "data/v0.1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--policy", action="append", required=True,
                    help='"expert" or a run directory from dronevla.train; repeatable')
    ap.add_argument("--pair", default=None, help="evaluate one pair_id only")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    manifest = json.loads((args.data / "manifest.json").read_text())
    # evaluate under exactly the config the dataset was recorded with
    raw = manifest["config"]
    cfg = TaskConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in raw.items() if k != "limits"},
                     limits=ActionLimits(**raw["limits"]))
    assert cfg.config_hash() == manifest["config_hash"], "config did not round-trip"
    scs = scenarios(args.data, args.split, args.pair)
    env = DroneTargetPairsEnv(cfg)
    results = {}
    try:
        for spec in args.policy:
            pol = load_policy(spec, cfg)
            t0 = time.perf_counter()
            trials = [run_trial(env, pol, sc) for sc in scs]
            summary = summarise(trials)
            results[pol.name] = {"spec": spec, "oracle": pol.oracle, "summary": summary,
                                 "trials": trials,
                                 "layouts": {sc["pair_id"]: sc["layout"].to_dict() for sc in scs},
                                 "wall_s": time.perf_counter() - t0,
                                 "policy_config": getattr(pol, "config", None)}
            s = summary
            print(f"{pol.name:34s} SR {s['success']:2d}/{s['episodes']} {s['SR_wilson95']}  "
                  f"pairs {s['pair_success']}/{s['pairs']}  wrong-target {s['outcomes']['wrong_target']}  "
                  f"timeout {s['outcomes']['timeout']}  collision {s['outcomes']['collision']}  "
                  f"OSR {s['OSR']:.2f}  NE med {s['NE_m']['median']:.2f} m  "
                  f"latency p50 {s['policy_latency_ms']['p50']:.2f} ms", flush=True)
    finally:
        env.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"split": args.split, "dataset": manifest["dataset_version"],
              "dataset_manifest_rgb_tree": manifest["rgb"]["tree_sha256"],
              "scenarios": len(scs),
              "policies": {k: {kk: vv for kk, vv in v.items() if kk not in ("layouts",)}
                           for k, v in results.items()}}
    for v in report["policies"].values():
        for t in v["trials"]:
            t["latency_ms"] = [round(x, 3) for x in t["latency_ms"]]
            t["path"] = [[round(c, 3) for c in p] for p in t["path"]]
    args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    figure(results, args.split, args.out.with_suffix(".png"))
    print(f"wrote {args.out} and {args.out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
