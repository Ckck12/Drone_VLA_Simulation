"""Where does an attention policy look? First frame of each val pair, both sentences.

The two sentences of a pair share one first frame, so any difference between the two maps
is caused by the words alone. Weights are averaged over heads and upsampled from the 6x8
patch grid to the 96x128 image.

    python scripts/attention_maps.py --run runs/bc_attn_v0.2_s0 --pairs 4 \
        --out reports/figures/attention_maps_s0.png
"""
import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dronevla.dataset import load_steps  # noqa: E402
from dronevla.evaluate import LearnedPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", type=pathlib.Path, default=REPO / "data/v0.2")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    pol = LearnedPolicy(pathlib.Path(args.run))
    assert getattr(pol.model, "attn", False), "not an attention run"
    eps = sorted([e for e in pq.read_table(args.data / "episodes.parquet").to_pylist()
                  if e["split"] == args.split], key=lambda e: e["episode_id"])
    pairs = [eps[2 * i: 2 * i + 2] for i in range(args.pairs)]
    fig, axes = plt.subplots(args.pairs, 3, figsize=(10, 2.6 * args.pairs))
    for r, pair in enumerate(pairs):
        st = load_steps(args.data, pair[0]["episode_id"])[0]
        rgb = np.asarray(Image.open(args.data / st["rgb_path"]))
        maps = []
        for e in pair:
            obs = {"rgb": rgb, "proprio": np.asarray(st["proprio"], dtype=np.float32)}
            pol.reset(e["instruction"])
            pol.act(obs)
            w = pol.model.last_attn[0].mean(0).reshape(6, 8).numpy()
            maps.append(w)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title("first frame (shared)", fontsize=9)
        for c, (e, w) in enumerate(zip(pair, maps), start=1):
            big = np.kron(w / w.max(), np.ones((16, 16)))
            axes[r, c].imshow(rgb)
            axes[r, c].imshow(big, cmap="magma", alpha=0.55, vmin=0, vmax=1)
            axes[r, c].set_title(f"\"{e['instruction']}\"", fontsize=8)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{pathlib.Path(args.run).name}: attention over the 6x8 patches, mean of heads",
                 x=0.02, ha="left", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
