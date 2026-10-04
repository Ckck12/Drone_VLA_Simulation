"""Build a contact sheet from a profile_env recording so the frames can be looked at.

Roadmap v3 Phase 0 item 5 rule: "image 저장 후 직접 확인한다. pixel 변화·frame checksum
만으로 scene가 맞는지 확인했다고 끝내지 않는다." The automated checks in profile_env
confirm shape, dtype, uniqueness and non-blankness; they cannot confirm the scene is right.

    python scripts/contact_sheet.py [--frames-dir ...] [--out ...]
"""
import argparse
import pathlib

from PIL import Image, ImageDraw

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--frames-dir", type=pathlib.Path,
                default=REPO_ROOT / "results/env_profile/frames")
ap.add_argument("--out", type=pathlib.Path,
                default=REPO_ROOT / "results/phase0/env_profile_contact_sheet.png")
ap.add_argument("--cols", type=int, default=5)
ap.add_argument("--rows", type=int, default=4)
ap.add_argument("--scale", type=int, default=2)
args = ap.parse_args()

files = sorted(args.frames_dir.glob("frame_*.png"))
if not files:
    raise SystemExit(f"no frames in {args.frames_dir}")

n = args.cols * args.rows
picks = [files[round(i * (len(files) - 1) / (n - 1))] for i in range(n)]

w, h = Image.open(picks[0]).size
cw, ch = w * args.scale, h * args.scale
pad, label_h = 4, 14
sheet = Image.new("RGB",
                  (args.cols * (cw + pad) + pad,
                   args.rows * (ch + pad + label_h) + pad),
                  (24, 24, 24))
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(picks):
    col, row = i % args.cols, i // args.cols
    x = pad + col * (cw + pad)
    y = pad + row * (ch + pad + label_h)
    sheet.paste(Image.open(path).resize((cw, ch), Image.NEAREST), (x, y))
    draw.text((x + 2, y + ch + 1), path.stem.replace("frame_", "#"), fill=(200, 200, 200))

args.out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.out)
print(f"{len(files)} frames in {args.frames_dir}, sampled {n}")
print(f"each frame {w}x{h}, shown at {args.scale}x -> {args.out} ({sheet.size[0]}x{sheet.size[1]})")
