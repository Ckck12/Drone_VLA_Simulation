"""Look at one recorded episode: every frame, labelled with what the dataset says about it.

Roadmap v3 Phase 1 item 3: "metadata/image/action alignment를 3개 episode에서 수동
확인한다". The validator checks counts, hashes and spacing; only a person can check that
the image in row t is the scene the drone was in, and that the action in row t is what moved
it to row t+1. This puts all of that on one picture.

    python scripts/inspect_episode.py                      # first train episode of data/v0.1
    python scripts/inspect_episode.py --episode val-003-g1
    python scripts/inspect_episode.py --dataset /tmp/dvla_tiny --episode test-000-g0

Each tile: the 128x96 observation (2x), an arrow for the commanded horizontal velocity in
the body frame (up = forward, left = left), and below it t, sim time, the command, the
Stop flag and the ground-truth distance to the goal. Read it as: the arrow on tile t should
explain the change from tile t to tile t+1.
"""
import argparse
import json
import pathlib
import sys

from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dronevla.dataset import load_episodes, load_steps  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", type=pathlib.Path, default=REPO_ROOT / "data/v0.1")
ap.add_argument("--episode", default=None, help="episode id; default: the first train episode")
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--out", type=pathlib.Path, default=None)
args = ap.parse_args()

episodes = {e["episode_id"]: e for e in load_episodes(args.dataset)}
eid = args.episode or sorted(k for k in episodes if k.startswith("train"))[0]
if eid not in episodes:
    sys.exit(f"no episode {eid!r}; e.g. {sorted(episodes)[:4]}")
ep = episodes[eid]
rows = load_steps(args.dataset, eid)
layout = json.loads((args.dataset / "layouts" / f"{ep['layout_id']}.json").read_text())

S, LABEL, PAD = 2, 30, 4
tw, th = 128 * S, 96 * S
n = len(rows)
cols = args.cols
grid_rows = (n + cols - 1) // cols
header = 46
sheet = Image.new("RGB", (cols * (tw + PAD) + PAD, header + grid_rows * (th + LABEL + PAD)),
                  (24, 24, 24))
d = ImageDraw.Draw(sheet)
d.text((PAD, 4), f"{eid}   {ep['instruction']}   -> {ep['outcome']}   "
                 f"(goal: {ep['goal_color']} {ep['goal_shape']}, on the {ep['goal_side']})",
       fill=(235, 235, 235))
d.text((PAD, 20), f"pair {ep['pair_id']}   seed {ep['snapshot_seed']}   {n} rows   "
                  f"layout targets: " + ", ".join(
                      f"{t['color']} {t['shape']} @({t['xy'][0]:.2f},{t['xy'][1]:.2f})"
                      for t in layout["targets"]), fill=(170, 170, 170))

for i, r in enumerate(rows):
    x = PAD + (i % cols) * (tw + PAD)
    y = header + (i // cols) * (th + LABEL + PAD)
    sheet.paste(Image.open(args.dataset / r["rgb_path"]).resize((tw, th), Image.NEAREST), (x, y))
    if r["action_mask"]:
        vx, vy = r["action_applied"][0], r["action_applied"][1]
        cx, cy = x + tw - 26, y + 26                     # arrow origin, top-right corner
        scale = 40                                       # px per m/s
        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], outline=(255, 255, 255))
        d.line([cx, cy, cx - vy * scale, cy - vx * scale], fill=(255, 0, 255), width=3)
        stop = "STOP" if r["stop_positive"] else ""
        trig = " (trigger)" if r["stop_trigger"] else ""
        txt = f"t{r['t']} {r['sim_t']:.1f}s v=({vx:+.2f},{vy:+.2f}) {stop}{trig}"
    else:
        txt = f"t{r['t']} {r['sim_t']:.1f}s TERMINAL"
    d.text((x + 2, y + th + 2), txt, fill=(220, 220, 220))
    d.text((x + 2, y + th + 15), f"dist to goal {r['priv_dist_to_goal']:.2f} m",
           fill=(150, 200, 150))

out = args.out or (REPO_ROOT / "results/inspect" / f"{eid}.png")
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(f"{eid}: {n} rows, outcome {ep['outcome']} -> {out}")
