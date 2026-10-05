"""Video of one counterfactual pair flown in closed loop, both instructions side by side.

    python scripts/rollout_video.py --policy runs/bc_text                # val pair 0
    python scripts/rollout_video.py --policy expert --pair 3
    python scripts/rollout_video.py --policy runs/bc_cf --split val --pair 2

Writes an animated GIF to results/videos/ (no ffmpeg needed). One frame per policy step at
5 frames per second, so it plays in real time. Layout:

    [ drone camera, instruction A ]   [ drone camera, instruction B ]   <- what the policy sees
    [            top-down map: targets, goal regions, both paths so far             ]

Learned policies get the observation only, exactly as in dronevla/evaluate.py. The default
split is val: test is kept for final reporting (docs/phase1_policy.md), so test videos are
for the record, not for choosing what to try next.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dronevla.action_adapter import ActionLimits                    # noqa: E402
from dronevla.env import DroneTargetPairsEnv                        # noqa: E402
from dronevla.evaluate import load_policy, scenarios                # noqa: E402
from dronevla.task import COLORS, TaskConfig                        # noqa: E402

CAM_SCALE = 3                        # 128x96 -> 384x288
PX_PER_M = 70
MAP_X, MAP_Y = (-3.6, 3.6), (-2.6, 2.6)
PAD = 8


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                # Pillow without FreeType sizing
        return ImageFont.load_default()


def rgb255(color):
    return tuple(int(c * 255) for c in COLORS[color][:3])


def fly(env, policy, sc):
    """Run one episode and keep what the video needs, per step."""
    obs, info = env.reset(seed=sc["seed"], options={"layout": sc["layout"], "goal_index": sc["goal"],
                                                    "instruction_family": sc["family"]})
    policy.reset(obs["instruction"])
    steps = [{"rgb": obs["rgb"], "pos": info["privileged"]["true_pos"], "action": None,
              "stop_count": 0, "t": info["episode_t"]}]
    while True:
        action = policy.act_privileged(obs, info) if policy.oracle else policy.act(obs)
        obs, _, term, trunc, info = env.step(action)
        steps[-1]["action"] = [float(a) for a in action]
        steps.append({"rgb": obs["rgb"], "pos": info["privileged"]["true_pos"], "action": None,
                      "stop_count": info["stop_count"], "t": info["episode_t"]})
        if term or trunc:
            return obs["instruction"], info["outcome"], steps


def to_map(x, y, origin):
    ox, oy = origin
    return (ox + (x - MAP_X[0]) * PX_PER_M, oy + (MAP_Y[1] - y) * PX_PER_M)


def draw_map(d, origin, lay, runs, k):
    w = (MAP_X[1] - MAP_X[0]) * PX_PER_M
    h = (MAP_Y[1] - MAP_Y[0]) * PX_PER_M
    ox, oy = origin
    d.rectangle([ox, oy, ox + w, oy + h], fill=(250, 250, 250), outline=(180, 180, 180))
    for gx in range(int(MAP_X[0]), int(MAP_X[1]) + 1):
        x0, _ = to_map(gx, 0, origin)
        d.line([x0, oy, x0, oy + h], fill=(232, 232, 232))
    for gy in range(int(MAP_Y[0]), int(MAP_Y[1]) + 1):
        _, y0 = to_map(0, gy, origin)
        d.line([ox, y0, ox + w, y0], fill=(232, 232, 232))
    for i, t in enumerate(lay.targets):
        cx, cy = to_map(t.xy[0], t.xy[1], origin)
        r = t.radius * PX_PER_M
        col = rgb255(t.color)
        (d.rectangle if t.shape == "box" else d.ellipse)([cx - r, cy - r, cx + r, cy + r], fill=col)
        hx, hy = to_map(lay.hover_points[i][0], lay.hover_points[i][1], origin)
        rr = 0.4 * PX_PER_M
        d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], outline=col, width=2)
    sx, sy = to_map(lay.start_xyz[0], lay.start_xyz[1], origin)
    d.polygon([(sx, sy - 7), (sx - 6, sy + 5), (sx + 6, sy + 5)], fill=(0, 0, 0))
    for run in runs:
        col = rgb255(lay.targets[run["goal"]].color)
        upto = run["steps"][: min(k, len(run["steps"]) - 1) + 1]
        pts = [to_map(s["pos"][0], s["pos"][1], origin) for s in upto]
        if len(pts) > 1:
            d.line(pts, fill=col, width=3)
        px, py = pts[-1]
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col, outline=(0, 0, 0))


def compose(runs, lay, k, title):
    cam_w, cam_h = 128 * CAM_SCALE, 96 * CAM_SCALE
    head = 46
    map_w = int((MAP_X[1] - MAP_X[0]) * PX_PER_M)
    map_h = int((MAP_Y[1] - MAP_Y[0]) * PX_PER_M)
    width = max(2 * cam_w + 3 * PAD, map_w + 2 * PAD)
    height = 30 + head + cam_h + PAD + map_h + 2 * PAD
    img = Image.new("RGB", (width, height), (28, 28, 28))
    d = ImageDraw.Draw(img)
    d.text((PAD, 8), title, fill=(235, 235, 235), font=font(15))
    for c, run in enumerate(runs):
        s = run["steps"][min(k, len(run["steps"]) - 1)]
        x0 = PAD + c * (cam_w + PAD)
        y0 = 30 + head
        col = rgb255(lay.targets[run["goal"]].color)
        # colour swatch + white text: dark target colours (blue) are unreadable on dark grey
        d.rectangle([x0, 34, x0 + 12, 46], fill=col, outline=(255, 255, 255))
        d.text((x0 + 18, 32), run["instruction"], fill=(240, 240, 240), font=font(14))
        done = k >= len(run["steps"]) - 1
        if done:
            status = f"t={s['t']:.1f}s  -> {run['outcome'].upper()}"
        elif s["action"] is not None:
            a = s["action"]
            stop = "  STOP" if a[4] > 0 else ""
            status = f"t={s['t']:.1f}s  v=({a[0]:+.2f}, {a[1]:+.2f}) m/s{stop}"
        else:
            status = f"t={s['t']:.1f}s"
        d.text((x0, 52), status, fill=(200, 200, 200), font=font(13))
        img.paste(Image.fromarray(s["rgb"]).resize((cam_w, cam_h), Image.NEAREST), (x0, y0))
        if done:
            ok = run["outcome"] == "success"
            d.rectangle([x0, y0, x0 + cam_w - 1, y0 + cam_h - 1],
                        outline=(40, 200, 80) if ok else (220, 50, 50), width=5)
    draw_map(d, ((width - map_w) // 2, 30 + head + cam_h + PAD), lay, runs, k)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="expert", help='"expert" or a run directory')
    ap.add_argument("--data", type=pathlib.Path, default=REPO_ROOT / "data/v0.1")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pair", type=int, default=0, help="index of the pair within the split")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    raw = manifest["config"]
    cfg = TaskConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()
                        if k != "limits"}, limits=ActionLimits(**raw["limits"]))
    scs = scenarios(args.data, args.split)
    pair_ids = list(dict.fromkeys(s["pair_id"] for s in scs))
    if not 0 <= args.pair < len(pair_ids):
        sys.exit(f"--pair must be 0..{len(pair_ids) - 1} for split {args.split}")
    pair = [s for s in scs if s["pair_id"] == pair_ids[args.pair]]

    policy = load_policy(args.policy, cfg)
    env = DroneTargetPairsEnv(cfg)
    runs = []
    try:
        for sc in sorted(pair, key=lambda s: s["goal"]):
            instr, outcome, steps = fly(env, policy, sc)
            runs.append({"goal": sc["goal"], "instruction": instr, "outcome": outcome, "steps": steps})
            print(f"  {instr!r:52s} -> {outcome} ({len(steps) - 1} steps)")
    finally:
        env.close()

    lay = pair[0]["layout"]
    title = f"{policy.name}  |  {args.split} pair {args.pair}  |  same start, two instructions"
    n = max(len(r["steps"]) for r in runs)
    frames = [compose(runs, lay, k, title) for k in range(n)]
    frames += [frames[-1]] * 10                          # hold the outcome for 2 s
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    name = "expert" if args.policy == "expert" else pathlib.Path(args.policy).name
    out = args.out or (REPO_ROOT / "results/videos" / f"{name}_{args.split}_pair{args.pair}.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=200, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
