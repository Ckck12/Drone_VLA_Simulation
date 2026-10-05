"""Dataset schema (roadmap v3 §4.5, project schema v0.x), loader, and validator.

Layout on disk:

    <root>/
      manifest.json        code/simulator SHA, config + hash, camera, counts, SHA256 of every
                           non-image file, a tree hash over every image
      datasheet.md         §4.5 datasheet, filled from the recorded data
      splits.json          split -> pair ids and layout ids
      episodes.parquet     one row per episode
      steps/<episode>.parquet   one row per observation (the last one is terminal)
      rgb/<episode>/<t>.png     the 128x96 RGB observation of each row
      layouts/<layout_id>.json  the geometry, including the evaluator-only hover points
      rejected.jsonl       every candidate that was generated and thrown away, with why
      validation.json      the result of `validate()` on the finished dataset

Alignment (§4.3): row t holds obs_t and the action applied *after* it; row t+1 is
0.2 sim-seconds later. Two exceptions, both explicit columns: the last row is terminal and
has no action (`action_mask` False), and the row whose action completed the Stop has
`stop_trigger` True -- that action switched the env to pose hold instead of being flown, so
the gap to the terminal row is the 1 s hold, not 0.2 s.

Columns prefixed `priv_` are simulator ground truth kept for analysis and the evaluator.
They are not policy inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION = "dronevla.dataset/0.1"
STOP_OUTCOMES = ("success", "stop_not_settled", "wrong_target", "stop_elsewhere")

EPISODE_SCHEMA = pa.schema([
    ("episode_id", pa.string()), ("pair_id", pa.string()), ("layout_id", pa.string()),
    ("family_id", pa.string()), ("split", pa.string()), ("snapshot_seed", pa.int64()),
    ("goal_index", pa.int8()), ("goal_color", pa.string()), ("goal_shape", pa.string()),
    ("other_color", pa.string()), ("other_shape", pa.string()), ("goal_side", pa.string()),
    ("instruction", pa.string()), ("instruction_family", pa.int8()), ("outcome", pa.string()),
    ("n_steps", pa.int32()), ("n_actions", pa.int32()), ("episode_t", pa.float64()),
    ("config_hash", pa.string()), ("first_rgb_sha256", pa.string()),
])

STEP_SCHEMA = pa.schema([
    ("episode_id", pa.string()), ("t", pa.int32()),
    ("sim_t", pa.float64()), ("episode_t", pa.float64()), ("wall_t_monotonic", pa.float64()),
    ("rgb_path", pa.string()), ("rgb_sha256", pa.string()),
    ("proprio", pa.list_(pa.float32(), 11)),
    ("action_mask", pa.bool_()),
    # null on the terminal row. Variable-length on purpose: a null in a fixed-size list
    # column round-trips through parquet as a 0-length entry that pyarrow then refuses to
    # read back. Lengths 5 and 4 are enforced by validate() instead.
    ("action_raw", pa.list_(pa.float32())),
    ("action_applied", pa.list_(pa.float32())),
    ("stop_positive", pa.bool_()), ("stop_trigger", pa.bool_()),
    ("flag_horizontal_capped", pa.bool_()), ("flag_planar_zeroed", pa.bool_()),
    ("terminal", pa.bool_()),
    ("priv_true_pos", pa.list_(pa.float64(), 3)),
    ("priv_dist_to_goal", pa.float64()), ("priv_dist_to_other", pa.float64()),
])


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(pathlib.Path(path).read_bytes())


def rgb_tree_sha256(pairs) -> str:
    """Hash of sorted (relative path, sha256) pairs: one fingerprint for every image."""
    h = hashlib.sha256()
    for rel, digest in sorted(pairs):
        h.update(f"{rel}\t{digest}\n".encode())
    return h.hexdigest()


def write_table(rows, schema, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def load_episodes(root) -> list[dict]:
    return pq.read_table(pathlib.Path(root) / "episodes.parquet").to_pylist()


def load_steps(root, episode_id) -> list[dict]:
    return pq.read_table(pathlib.Path(root) / "steps" / f"{episode_id}.parquet").to_pylist()


# ------------------------------------------------------------------------ validator
def validate(root) -> dict:
    """Re-read a finished dataset and check it against the schema and §4.3-4.5 rules.

    Nothing is taken from the generator's memory: every check reads files back.
    Returns {check_name: {"pass": bool, ...details}}.
    """
    root = pathlib.Path(root)
    out = {}

    def check(name, ok, **detail):
        out[name] = {"pass": bool(ok), **detail}
        return ok

    required = ["manifest.json", "episodes.parquet", "splits.json", "datasheet.md",
                "rejected.jsonl"]
    missing = [f for f in required if not (root / f).exists()]
    if not check("required_files", not missing, missing=missing):
        return out

    manifest = json.loads((root / "manifest.json").read_text())
    cfg = manifest["config"]
    episodes = load_episodes(root)
    splits = json.loads((root / "splits.json").read_text())

    # manifest integrity
    bad = [rel for rel, digest in manifest["files"].items()
           if not (root / rel).exists() or sha256_file(root / rel) != digest]
    check("manifest_file_hashes", not bad, mismatched=bad[:10], files=len(manifest["files"]))
    check("schema_version", manifest.get("schema") == SCHEMA_VERSION,
          found=manifest.get("schema"))

    # counts
    planned = manifest["counts"]["pairs_per_split"]
    got = {s: sum(e["split"] == s for e in episodes) // 2 for s in planned}
    check("pairs_per_split", got == planned, planned=planned, found=got)
    check("episodes_are_two_per_pair", len(episodes) == 2 * sum(planned.values()),
          episodes=len(episodes))
    check("all_episodes_succeeded", all(e["outcome"] == "success" for e in episodes),
          outcomes=sorted({e["outcome"] for e in episodes}))

    # pairs (§4.2 item 5)
    by_pair = {}
    for e in episodes:
        by_pair.setdefault(e["pair_id"], []).append(e)
    pair_problems = []
    first_proprio = {}
    for pid, eps in by_pair.items():
        eps = sorted(eps, key=lambda e: e["goal_index"])
        if [e["goal_index"] for e in eps] != [0, 1]:
            pair_problems.append((pid, "needs exactly goal 0 and goal 1"))
            continue
        a, b = eps
        for key in ("split", "layout_id", "snapshot_seed", "first_rgb_sha256", "instruction_family"):
            if a[key] != b[key]:
                pair_problems.append((pid, f"{key} differs"))
        if a["instruction"] == b["instruction"]:
            pair_problems.append((pid, "same instruction for both goals"))
        pa0 = load_steps(root, a["episode_id"])[0]["proprio"]
        pb0 = load_steps(root, b["episode_id"])[0]["proprio"]
        first_proprio[pid] = pa0 == pb0
        if pa0 != pb0:
            pair_problems.append((pid, "first proprio differs"))
    check("pairs_complete_and_identical_at_start", not pair_problems,
          problems=pair_problems[:10], pairs=len(by_pair))

    # split leakage (§4.5): no layout or family in more than one split
    seen = {}
    leaks = []
    for e in episodes:
        for key in ("layout_id", "family_id"):
            prev = seen.setdefault((key, e[key]), e["split"])
            if prev != e["split"]:
                leaks.append((key, e[key], prev, e["split"]))
    check("splits_disjoint_by_layout_and_family", not leaks, leaks=leaks[:10])
    listed = {pid for s in splits.values() for pid in s["pair_ids"]}
    check("splits_json_matches_episodes", listed == set(by_pair), listed=len(listed),
          episodes_pairs=len(by_pair))

    # instructions name the goal
    wrong_instr = [e["episode_id"] for e in episodes
                   if f"{e['goal_color']} {e['goal_shape']}" not in e["instruction"]]
    check("instruction_names_the_goal", not wrong_instr, wrong=wrong_instr[:10])

    # steps, alignment, actions, images
    dt = 1.0 / cfg["policy_hz"]
    lim = cfg["limits"]
    step_problems, image_pairs, image_problems, n_rows = [], [], [], 0
    for e in episodes:
        rows = load_steps(root, e["episode_id"])
        n_rows += len(rows)
        eid = e["episode_id"]
        if len(rows) < 2:
            step_problems.append((eid, "fewer than 2 rows: no action was ever recorded"))
            continue
        if [r["t"] for r in rows] != list(range(len(rows))):
            step_problems.append((eid, "t is not 0..n-1"))
        if len(rows) != e["n_steps"]:
            step_problems.append((eid, "n_steps mismatch"))
        sim = np.array([r["sim_t"] for r in rows])
        gaps = np.diff(sim)
        if np.any(gaps <= 0):
            step_problems.append((eid, "sim_t not strictly increasing"))
        last_gap = cfg["hold_s"] if rows[-2]["stop_trigger"] else dt
        if not (np.allclose(gaps[:-1], dt, atol=1e-9) and math.isclose(gaps[-1], last_gap, abs_tol=1e-9)):
            step_problems.append((eid, "sim_t spacing is not 0.2 s (or the hold before terminal)"))
        if np.any(np.diff([r["wall_t_monotonic"] for r in rows]) < 0):
            step_problems.append((eid, "wall clock went backwards"))
        if [r["action_mask"] for r in rows] != [True] * (len(rows) - 1) + [False]:
            step_problems.append((eid, "action_mask must be True except on the terminal row"))
        if not rows[-1]["terminal"] or any(r["terminal"] for r in rows[:-1]):
            step_problems.append((eid, "terminal flag must be on the last row only"))
        if sum(bool(r["stop_trigger"]) for r in rows) > 1:
            step_problems.append((eid, "more than one stop_trigger"))
        for r in rows[:-1]:
            raw, app = r["action_raw"], r["action_applied"]
            if raw is None or app is None or len(raw) != 5 or len(app) != 4 \
                    or not all(map(math.isfinite, raw + app)):
                step_problems.append((eid, f"bad action at t={r['t']}"))
                break
            if math.hypot(app[0], app[1]) > lim["horizontal_mps"] + 1e-5 \
                    or (cfg["planar"] and (app[2] != 0 or app[3] != 0)):
                step_problems.append((eid, f"applied action breaks the contract at t={r['t']}"))
                break
        if rows[-1]["action_raw"] is not None:
            step_problems.append((eid, "terminal row carries an action"))
        for r in rows:
            if len(r["proprio"]) != 11 or not all(map(math.isfinite, r["proprio"])):
                step_problems.append((eid, f"bad proprio at t={r['t']}"))
                break
            p = root / r["rgb_path"]
            if not p.exists():
                image_problems.append((r["rgb_path"], "missing"))
                continue
            digest = sha256_file(p)
            if digest != r["rgb_sha256"]:
                image_problems.append((r["rgb_path"], "sha256 mismatch"))
            image_pairs.append((r["rgb_path"], digest))
        if rows[0]["rgb_sha256"] != e["first_rgb_sha256"]:
            step_problems.append((eid, "first_rgb_sha256 does not match row 0"))
    check("steps_alignment_and_actions", not step_problems, problems=step_problems[:10],
          rows=n_rows)
    check("images_present_and_hashes_match", not image_problems,
          problems=image_problems[:10], images=len(image_pairs))
    check("rgb_tree_hash_matches_manifest",
          rgb_tree_sha256(image_pairs) == manifest["rgb"]["tree_sha256"])

    # decode a sample of images: shape and dtype
    from PIL import Image
    sample = image_pairs[:: max(1, len(image_pairs) // 20)]
    shapes = {np.asarray(Image.open(root / rel)).shape for rel, _ in sample}
    check("images_decode_to_96x128x3", shapes == {(96, 128, 3)}, found=sorted(map(str, shapes)))
    return out


def failed(report: dict) -> list[str]:
    return [k for k, v in report.items() if not v["pass"]]
