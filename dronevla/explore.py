"""Exploration data for the world model: diverse flights, not demonstrations.

    python -m dronevla.explore --split train --episodes-per-layout 8 --out data/explore_v0.1/train
    python -m dronevla.explore --split val   --episodes-per-layout 2 --out data/explore_v0.1/val

A dynamics model needs (state, action, next state) over a wide spread of actions and states,
including the ones a policy reaches by mistake -- overshooting a goal, drifting toward a wall.
Expert demonstrations only cover straight lines into the goal, so this module flies
deliberately varied behaviours over the *same layouts* as dataset v0.1 (by split), keeping
the world-model vs behaviour-cloning comparison about the method, not about seeing more
scenes. Language is not involved: the dynamics do not depend on the words.

Behaviours, chosen per episode with a fixed seed:
* noisy_expert -- toward a random one of the two goals, a random speed scale, plus slowly
  varying (Gauss-Markov) velocity noise
* random_walk -- slowly varying random velocity inside the 0.5 m/s disk, steered back when
  near the room's edge
* overshoot -- toward a random goal, but aiming 1.2 m past it and never stopping, so the data
  contains the near-target and past-target states a failing policy ends up in
No behaviour ever asks to Stop. Episodes end at `--timeout-s` or on collision / out of bounds.

The output uses the dataset step schema (`dronevla.dataset.STEP_SCHEMA`), so the same
loaders read it; the episode table records the behaviour instead of an instruction outcome.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import time

import numpy as np
import pyarrow as pa

from dronevla.action_adapter import ActionLimits, world_to_body_velocity
from dronevla.dataset import STEP_SCHEMA, sha256_bytes, write_table
from dronevla.env import DroneTargetPairsEnv
from dronevla.evaluate import layout_from_dict
from dronevla.dataset import load_episodes
from dronevla.record import png_bytes
from dronevla.sim import GaussMarkov
from dronevla.task import TaskConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BEHAVIOURS = ("noisy_expert", "random_walk", "overshoot")

EXPLORE_EPISODE_SCHEMA = pa.schema([
    ("episode_id", pa.string()), ("split", pa.string()), ("layout_id", pa.string()),
    ("snapshot_seed", pa.int64()), ("behaviour", pa.string()), ("aim_goal_index", pa.int8()),
    ("outcome", pa.string()), ("n_steps", pa.int32()), ("episode_t", pa.float64()),
])


class Behaviour:
    """Acts from privileged state (it is a data collector, not a policy under test)."""

    def __init__(self, kind, layout, rng, cfg):
        self.kind, self.layout, self.cfg = kind, layout, cfg
        self.goal = int(rng.integers(2))
        self.speed = float(rng.uniform(0.3, 1.0)) * cfg.limits.horizontal_mps
        sigma = 0.15 if kind != "random_walk" else 0.35
        self.noise = GaussMarkov((sigma, sigma, 0.0), 1.0 if kind != "random_walk" else 1.5,
                                 1.0 / cfg.policy_hz, rng.integers(2 ** 31))
        h = np.asarray(layout.hover_points[self.goal][:2])
        if kind == "overshoot":                     # aim beyond the hover point, toward the target
            t = np.asarray(layout.targets[self.goal].xy)
            d = (t - h) / np.linalg.norm(t - h)
            h = h + d * 1.2
        self.aim = h

    def act(self, info):
        p = info["privileged"]
        pos = np.asarray(p["true_pos"][:2])
        n = self.noise.step()[:2]
        if self.kind == "random_walk":
            v = n.copy()
            lo, hi = np.asarray(self.cfg.room_min[:2]) + 1.0, np.asarray(self.cfg.room_max[:2]) - 1.0
            v += np.clip(lo - pos, 0, None) * 0.5 - np.clip(pos - hi, 0, None) * 0.5
        else:
            d = self.aim - pos
            dist = float(np.linalg.norm(d))
            v = (d / max(dist, 1e-6)) * min(self.speed, dist if self.kind == "noisy_expert" else self.speed) + n
        cap = self.cfg.limits.horizontal_mps * (1.0 - 1e-5)
        s = float(np.linalg.norm(v))
        if s > cap:
            v *= cap / s
        vb = world_to_body_velocity([v[0], v[1], 0.0], p["true_yaw"])
        return np.array([vb[0], vb[1], 0.0, 0.0, -1.0], dtype=np.float32)


def fly(env, beh, root, episode_id, seed, layout):
    obs, info = env.reset(seed=seed, options={"layout": layout, "goal_index": beh.goal})
    rows, t, done = [], 0, False
    (root / "rgb" / episode_id).mkdir(parents=True, exist_ok=True)
    while True:
        data = png_bytes(obs["rgb"])
        rel = f"rgb/{episode_id}/{t:05d}.png"
        (root / rel).write_bytes(data)
        priv = info["privileged"]
        row = {"episode_id": episode_id, "t": t, "sim_t": info["sim_t"],
               "episode_t": info["episode_t"], "wall_t_monotonic": time.monotonic(),
               "rgb_path": rel, "rgb_sha256": sha256_bytes(data),
               "proprio": obs["proprio"].tolist(), "action_mask": not done,
               "action_raw": None, "action_applied": None, "stop_positive": None,
               "stop_trigger": False, "flag_horizontal_capped": None, "flag_planar_zeroed": None,
               "terminal": done, "priv_true_pos": priv["true_pos"],
               "priv_dist_to_goal": priv["dist_to_goal"], "priv_dist_to_other": priv["dist_to_other"]}
        if done:
            rows.append(row)
            return rows, info
        action = beh.act(info)
        obs, _, term, trunc, info = env.step(action)
        a = info["action"]
        row.update(action_raw=[float(x) for x in a["raw"]], action_applied=a["applied"],
                   stop_positive=a["stop_positive"],
                   flag_horizontal_capped=a["flags"].get("horizontal_capped"),
                   flag_planar_zeroed=a["flags"].get("planar_zeroed"))
        rows.append(row)
        done = term or trunc
        t += 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.explore")
    ap.add_argument("--data", type=pathlib.Path, default=REPO_ROOT / "data/v0.1",
                    help="the demonstration dataset whose layouts to reuse")
    ap.add_argument("--split", default="train")
    ap.add_argument("--episodes-per-layout", type=int, default=8)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    manifest = json.loads((args.data / "manifest.json").read_text())
    raw = manifest["config"]
    cfg = TaskConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()
                        if k != "limits"}, limits=ActionLimits(**raw["limits"]))
    cfg = dataclasses.replace(cfg, timeout_s=args.timeout_s)
    layouts = {}
    for e in load_episodes(args.data):
        if e["split"] == args.split:
            layouts[e["layout_id"]] = (e["snapshot_seed"], layout_from_dict(json.loads(
                (args.data / "layouts" / f"{e['layout_id']}.json").read_text())))

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    rng = np.random.default_rng([args.seed, 0xE7])
    env = DroneTargetPairsEnv(cfg)
    episodes, n_rows, t0 = [], 0, time.perf_counter()
    try:
        for li, (lid, (seed, layout)) in enumerate(sorted(layouts.items())):
            for k in range(args.episodes_per_layout):
                kind = BEHAVIOURS[k % len(BEHAVIOURS)]
                beh = Behaviour(kind, layout, rng, cfg)
                eid = f"{args.split}-x{li:03d}-{k:02d}-{kind}"
                rows, info = fly(env, beh, args.out, eid, seed, layout)
                write_table(rows, STEP_SCHEMA, args.out / "steps" / f"{eid}.parquet")
                n_rows += len(rows)
                episodes.append({"episode_id": eid, "split": args.split, "layout_id": lid,
                                 "snapshot_seed": seed, "behaviour": kind,
                                 "aim_goal_index": beh.goal, "outcome": info["outcome"],
                                 "n_steps": len(rows), "episode_t": info["episode_t"]})
    finally:
        env.close()
    write_table(episodes, EXPLORE_EPISODE_SCHEMA, args.out / "episodes.parquet")
    for lid, (_, layout) in layouts.items():
        (args.out / "layouts").mkdir(exist_ok=True)
        (args.out / "layouts" / f"{lid}.json").write_text(json.dumps(layout.to_dict(), indent=1) + "\n")
    outcomes = {}
    for e in episodes:
        outcomes[(e["behaviour"], e["outcome"])] = outcomes.get((e["behaviour"], e["outcome"]), 0) + 1
    summary = {"split": args.split, "episodes": len(episodes), "rows": n_rows,
               "layouts": len(layouts), "wall_s": round(time.perf_counter() - t0, 1),
               "outcomes": {f"{b}/{o}": n for (b, o), n in sorted(outcomes.items())},
               "source_dataset_manifest_rgb_tree": manifest["rgb"]["tree_sha256"],
               "timeout_s": args.timeout_s, "seed": args.seed}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
