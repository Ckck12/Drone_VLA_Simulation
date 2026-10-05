"""Record a `DroneTargetPairs` dataset with the oracle expert.

    python -m dronevla.record --out data/v0.1                      # 20 / 10 / 10 pairs
    python -m dronevla.record --out /tmp/tiny --pairs 1 1 1        # a tiny one for tests

Roadmap v3 Phase 1 item 3 and §4.4-4.5. What it does, in order, for each split:

1. Walk that split's own seed range (train 1000+, val 2000+, test 3000+), so splits never
   share a seed, and none overlap the seeds used by the tests (0-9) or the held-out expert
   check (100-119).
2. Sample a layout from the seed; reject it if either target is too small in the first frame
   or the goal would be too small from its own hover point (`visibility_report`).
3. Fly the counterfactual pair: the same seed and layout with goal 0, then goal 1. Keep the
   pair only if the expert succeeds on both; otherwise delete both episodes and log why
   (§4.2 item 2: failed expert episodes are excluded, but recorded).
4. Write steps, images, episodes, layouts, splits, the datasheet and the manifest; then run
   `dronevla.dataset.validate` on what was written and exit non-zero if any check fails.

Every layout is its own family in v0.1 (nothing is mirrored or translated), so splitting by
layout is splitting by family. The instruction family alternates by seed so both phrasings
appear in every split; both members of a pair always share one.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import shutil
import sys
import time

import numpy as np
from PIL import Image

from dronevla import __version__
from dronevla.dataset import (EPISODE_SCHEMA, SCHEMA_VERSION, STEP_SCHEMA, failed,
                              rgb_tree_sha256, sha256_bytes, sha256_file, validate, write_table)
from dronevla.env import DroneTargetPairsEnv
from dronevla.expert import StraightLineExpert
from dronevla.profile_env import git_info
from dronevla.task import INSTRUCTION_FAMILIES, TaskConfig, sample_layout

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPLIT_SEED_BASE = {"train": 1000, "val": 2000, "test": 3000}
DATASET_VERSION = "0.1.0"


def png_bytes(rgb) -> bytes:
    import io
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def record_episode(env, expert, root, episode_id, seed, layout, goal, family):
    """Fly one episode with the expert and return (episode_row, step_rows)."""
    obs, info = env.reset(seed=seed, options={"layout": layout, "goal_index": goal,
                                              "instruction_family": family})
    rows, t, done = [], 0, False
    rgb_dir = root / "rgb" / episode_id
    rgb_dir.mkdir(parents=True, exist_ok=True)
    while True:
        wall = time.monotonic()                     # §4.3: the recorder stamps wall time
        data = png_bytes(obs["rgb"])
        rel = f"rgb/{episode_id}/{t:05d}.png"
        (root / rel).write_bytes(data)
        priv = info["privileged"]
        row = {
            "episode_id": episode_id, "t": t, "sim_t": info["sim_t"],
            "episode_t": info["episode_t"], "wall_t_monotonic": wall,
            "rgb_path": rel, "rgb_sha256": sha256_bytes(data),
            "proprio": obs["proprio"].tolist(), "action_mask": not done,
            "action_raw": None, "action_applied": None, "stop_positive": None,
            "stop_trigger": False, "flag_horizontal_capped": None, "flag_planar_zeroed": None,
            "terminal": done, "priv_true_pos": priv["true_pos"],
            "priv_dist_to_goal": priv["dist_to_goal"], "priv_dist_to_other": priv["dist_to_other"],
        }
        if done:
            rows.append(row)
            break
        action = expert.act(info)
        obs, _, term, trunc, info = env.step(action)
        a = info["action"]
        row.update(action_raw=[float(v) for v in a["raw"]], action_applied=a["applied"],
                   stop_positive=a["stop_positive"],
                   stop_trigger=bool((term or trunc) and info["stop_count"] >= env.cfg.stop_consecutive),
                   flag_horizontal_capped=a["flags"].get("horizontal_capped"),
                   flag_planar_zeroed=a["flags"].get("planar_zeroed"))
        rows.append(row)
        done = term or trunc
        t += 1

    goal_t, other_t = layout.targets[goal], layout.targets[1 - goal]
    # yaw is fixed at 0, so +y is the drone's left
    side = "left" if goal_t.xy[1] > layout.start_xyz[1] else "right"
    episode = {
        "episode_id": episode_id, "pair_id": info["pair_id"], "layout_id": layout.layout_id,
        "family_id": layout.layout_id, "split": None, "snapshot_seed": seed, "goal_index": goal,
        "goal_color": goal_t.color, "goal_shape": goal_t.shape,
        "other_color": other_t.color, "other_shape": other_t.shape, "goal_side": side,
        "instruction": obs["instruction"], "instruction_family": family,
        "outcome": info["outcome"], "n_steps": len(rows), "n_actions": len(rows) - 1,
        "episode_t": info["episode_t"], "config_hash": info["config_hash"],
        "first_rgb_sha256": rows[0]["rgb_sha256"],
    }
    return episode, rows


def visibility_ok(env, info, cfg):
    vis = env.visibility_report()
    first = info["privileged"]["first_frame_target_px"]
    floor = cfg.min_first_frame_target_px
    hover_floor = int(0.2 * 128 * 96)
    if min(first) < floor or min(vis["start_px"]) < floor:
        return False, f"a target covers fewer than {floor} px at the start ({first})", vis
    if min(vis["own_target_px_at_hover"]) < hover_floor:
        return False, f"goal covers fewer than {hover_floor} px from its hover point", vis
    return True, "", vis


def generate(root: pathlib.Path, pairs_per_split: dict, cfg: TaskConfig):
    env = DroneTargetPairsEnv(cfg)
    expert = StraightLineExpert(cfg)
    episodes, steps_by_episode, layouts, splits, rejected = [], {}, {}, {}, []
    t_start = time.perf_counter()
    try:
        for split, n_pairs in pairs_per_split.items():
            seed = SPLIT_SEED_BASE[split]
            splits[split] = {"pair_ids": [], "layout_ids": [], "seeds": []}
            while len(splits[split]["pair_ids"]) < n_pairs:
                seed += 1
                layout = sample_layout(seed, cfg)
                family = seed % len(INSTRUCTION_FAMILIES)
                k = len(splits[split]["pair_ids"])
                ids = [f"{split}-{k:03d}-g{g}" for g in (0, 1)]

                _, info = env.reset(seed=seed, options={"layout": layout, "goal_index": 0})
                ok, why, vis = visibility_ok(env, info, cfg)
                if not ok:
                    rejected.append({"split": split, "seed": seed, "layout_id": layout.layout_id,
                                     "stage": "layout", "reason": why, "visibility": vis})
                    continue

                pair = [record_episode(env, expert, root, ids[g], seed, layout, g, family)
                        for g in (0, 1)]
                outcomes = [ep["outcome"] for ep, _ in pair]
                if outcomes != ["success", "success"]:
                    for eid in ids:
                        shutil.rmtree(root / "rgb" / eid, ignore_errors=True)
                    rejected.append({"split": split, "seed": seed, "layout_id": layout.layout_id,
                                     "stage": "expert", "reason": f"expert outcomes {outcomes}",
                                     "outcomes": outcomes})
                    continue
                for ep, rows in pair:
                    ep["split"] = split
                    episodes.append(ep)
                    steps_by_episode[ep["episode_id"]] = rows
                layouts[layout.layout_id] = layout.to_dict()
                splits[split]["pair_ids"].append(pair[0][0]["pair_id"])
                splits[split]["layout_ids"].append(layout.layout_id)
                splits[split]["seeds"].append(seed)
                print(f"  {split:5s} pair {k + 1:3d}/{n_pairs}  seed {seed}  "
                      f"{pair[0][0]['instruction']!r} / {pair[1][0]['goal_color']} "
                      f"{pair[1][0]['goal_shape']}", flush=True)
    finally:
        env.close()
    return episodes, steps_by_episode, layouts, splits, rejected, time.perf_counter() - t_start


def contingency(episodes, a, b):
    table = collections.Counter((str(e[a]), str(e[b])) for e in episodes)
    rows = sorted({k[0] for k in table})
    cols = sorted({k[1] for k in table})
    lines = [f"| {a} \\ {b} | " + " | ".join(cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for r in rows:
        lines.append(f"| {r} | " + " | ".join(str(table[(r, c)]) for c in cols) + " |")
    return "\n".join(lines)


def write_datasheet(root, episodes, steps_by_episode, splits, rejected, cfg, gen_s, command):
    n_steps = np.array([e["n_steps"] for e in episodes])
    ep_t = np.array([e["episode_t"] for e in episodes])
    rgb_bytes = sum((root / r["rgb_path"]).stat().st_size
                    for rows in steps_by_episode.values() for r in rows)
    n_img = sum(len(rows) for rows in steps_by_episode.values())
    by_stage = collections.Counter(r["stage"] for r in rejected)
    split_lines = "\n".join(
        f"| {s} | {len(v['pair_ids'])} | {2 * len(v['pair_ids'])} | "
        f"{min(v['seeds'])}-{max(v['seeds'])} |" for s, v in splits.items() if v["seeds"])
    text = f"""# Datasheet — DroneTargetPairs v{DATASET_VERSION}

Generated {time.strftime('%Y-%m-%d')} by `{command}`. Every number below is computed from the
recorded files.

## Purpose
Thin-slice dataset for roadmap v3 Phase 1: train and evaluate a tiny RGB + text policy that
must fly to *the target the instruction names* and stop. Built as counterfactual pairs: the
same start snapshot flown twice, once per target. **Not for**: claims about general language
understanding, real-world flight, or anything beyond two coloured pillars in an empty room.

## Contents
| split | pairs | episodes | seeds |
|---|---:|---:|---|
{split_lines}

- {len(episodes)} episodes, {int(n_steps.sum())} observations ({n_img} PNG images, {rgb_bytes / 1e6:.2f} MB,
  {rgb_bytes / max(n_img, 1):.0f} bytes/image on average)
- observations per episode: median {int(np.median(n_steps))}, range {n_steps.min()}-{n_steps.max()}
- episode length: median {np.median(ep_t):.1f} sim-s, range {ep_t.min():.1f}-{ep_t.max():.1f}
- generation took {gen_s:.0f} wall-s on the development laptop (CPU only)

## Scene
8 x 8 x 3 m room, no obstacles. The drone starts at x = {cfg.start_x} m (lateral jitter
+-{cfg.start_y_jitter} m), altitude {cfg.altitude_m} m, yaw fixed at 0; planar motion only.
Two {cfg.target_height_m} m pillars of radius {cfg.target_radius_m} m, box or cylinder, in
two different colours of red/blue/green/yellow, within +-{cfg.max_bearing_deg:.0f} deg of the
start heading.

## Expert and what it knew (privileged inputs)
`StraightLineExpert`: true position and velocity from the simulator, plus the goal hover
point ({cfg.standoff_m} m in front of the goal target). It flies through the same action
adapter, caps, PID and physics as a policy, and its actions stay inside the caps, so
`action_raw` equals the applied action. It is an oracle, not a perception model.

## Observations vs ground truth
Policy inputs: `rgb` (PNG, 128x96x3), `proprio` (11 floats: body velocity, roll, pitch,
sin/cos yaw, body angular velocity, altitude -- true sim state), the instruction. Everything
in `priv_*` columns and in `layouts/` (hover points, target coordinates) is ground truth for
analysis and evaluation only.

## Instructions
Two human-written families, no LLM: {"; ".join(f'`{f}`' for f in INSTRUCTION_FAMILIES)}.
Both appear in every split; a pair always shares one family. Counts:

{contingency(episodes, "split", "instruction_family")}

## Balance (§4.5 contingency tables)
Goal colour x goal side (left/right of the start heading):

{contingency(episodes, "goal_color", "goal_side")}

Goal shape x split:

{contingency(episodes, "goal_shape", "split")}

These are reported, not enforced: v0.1 samples layouts at random.

## Splits and leakage
Split by layout; in v0.1 each layout is its own family (no mirrored or translated copies),
so no layout, family or seed appears in two splits. Both episodes of a pair are always in the
same split. `validate()` checks all of this.

## Rejections
{len(rejected)} candidates were generated and discarded: {dict(by_stage) or 'none'}. Each is in
`rejected.jsonl` with its reason.

## Labels and alignment
Row t = observation t + the action applied after it; row t+1 is 0.2 sim-s later. The last
row is terminal with no action (`action_mask` False). The row with `stop_trigger` True holds
the action that completed the Stop: it was not flown -- the env switched to a 1 s pose hold,
so the next row is 1 s later. Rates: physics {cfg.pyb_hz} Hz, control {cfg.ctrl_hz} Hz,
policy {cfg.policy_hz} Hz. Frames: actions body FLU (x forward, y left, z up), world ENU.
Both simulation time and monotonic wall time are stored.

## Outcomes
Only episodes the expert completed successfully are kept (success = within
{cfg.success_radius_m} m of the goal hover point, speed <= {cfg.stop_speed_mps} m/s for
{cfg.hold_s} s after Stop, no collision).

## Personal information
None. Synthetic scenes only; no people, no real images, no operator data.

## Licences
Code: MIT. Dataset: MIT, same as the code. Simulator assets used to render: gym-pybullet-drones
CF2X model (MIT), PyBullet `plane.urdf` (zlib). The generator does not use any third-party
dataset or pretrained weights.

## Known limitations and bias
- yaw is fixed and motion is planar, so the camera always faces +x
- the two targets always differ in colour; shape alone never decides the goal in v0.1
- targets are tall pillars chosen to stay visible from the hover point; the sky and floor
  are untextured
- position-estimate noise is simulated (sigma 0.03 m horizontal); wind is off
- 40 pairs is a thin slice: enough to wire training and evaluation, not to measure
  generalisation

## Regenerate
`{command}` from the repository at the commit in `manifest.json`. Same seeds give the same
layouts and, on the same machine, byte-identical images.

## Version
v{DATASET_VERSION} — first version.
"""
    (root / "datasheet.md").write_text(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.record")
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "data/v0.1")
    ap.add_argument("--pairs", type=int, nargs=3, default=[20, 10, 10],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args(argv)

    root = args.out
    if root.exists() and any(root.iterdir()):
        if not args.force:
            print(f"{root} is not empty; pass --force to overwrite", file=sys.stderr)
            return 2
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = TaskConfig()
    pairs = dict(zip(("train", "val", "test"), args.pairs))
    command = "python -m dronevla.record --out <dir> --pairs " + " ".join(map(str, args.pairs))
    print(f"recording {sum(pairs.values())} pairs into {root}")

    episodes, steps, layouts, splits, rejected, gen_s = generate(root, pairs, cfg)

    write_table(episodes, EPISODE_SCHEMA, root / "episodes.parquet")
    for eid, rows in steps.items():
        write_table(rows, STEP_SCHEMA, root / "steps" / f"{eid}.parquet")
    for lid, lay in layouts.items():
        (root / "layouts").mkdir(exist_ok=True)
        (root / "layouts" / f"{lid}.json").write_text(json.dumps(lay, indent=2) + "\n")
    (root / "splits.json").write_text(json.dumps(splits, indent=2) + "\n")
    with (root / "rejected.jsonl").open("w") as fh:
        for r in rejected:
            fh.write(json.dumps(r) + "\n")
    write_datasheet(root, episodes, steps, splits, rejected, cfg, gen_s, command)

    from dronevla.camera import FrontCamera
    files = {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.parts[len(root.parts)] != "rgb"}
    images = [(r["rgb_path"], r["rgb_sha256"]) for rows in steps.values() for r in rows]
    manifest = {
        "schema": SCHEMA_VERSION, "dataset_version": DATASET_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command,
        "code": {"dronevla_version": __version__, **git_info(REPO_ROOT)},
        "simulator": git_info(REPO_ROOT / "third_party/gym-pybullet-drones"),
        "config": cfg.as_dict(), "config_hash": cfg.config_hash(),
        "camera": FrontCamera(mount=cfg.camera_mount, tilt_deg=cfg.camera_tilt_deg,
                              fov_deg=cfg.camera_vfov_deg, near=0.01).describe(),
        "controller": "DSLPIDControl via dronevla.sim.TiltLimitedDSLPIDControl, "
                      f"max tilt {cfg.max_tilt_deg} deg",
        "expert": "dronevla.expert.StraightLineExpert(gain=1.0, stop_radius_m=0.12, "
                  "stop_speed_mps=0.08)",
        "counts": {"pairs_per_split": pairs, "episodes": len(episodes),
                   "observations": sum(len(r) for r in steps.values()),
                   "rejected": len(rejected)},
        "files": files,
        "rgb": {"count": len(images), "tree_sha256": rgb_tree_sha256(images)},
        "license": {"code": "MIT", "dataset": "MIT",
                    "assets": "gym-pybullet-drones CF2X (MIT), PyBullet plane.urdf (zlib)"},
        "generation_wall_s": round(gen_s, 1),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    report = validate(root)
    (root / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    bad = failed(report)
    print(f"\n{len(episodes)} episodes, {manifest['counts']['observations']} observations, "
          f"{len(rejected)} rejected candidates, {gen_s:.0f} s")
    for name, v in report.items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {name}")
    if bad:
        print(f"VALIDATION FAILED: {', '.join(bad)}", file=sys.stderr)
        return 1
    print(f"wrote {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
