"""Figures for the single-seed Season 2 screen.

    python scripts/candidates_table.py
    python scripts/figures/candidates_s0.py       # -> reports/figures/candidates_s0/
"""
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports"
DATA = REPO / "data" / "v0.2"
OUT = REPORTS / "figures" / "candidates_s0"
OUT.mkdir(parents=True, exist_ok=True)
TARGET = {"red": "#e34948", "blue": "#2a78d6", "green": "#008300", "yellow": "#eda100"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "xtick.color": MUTED, "ytick.color": MUTED, "savefig.dpi": 200})

LABEL = {
    "expert (oracle)": "scripted expert (oracle)",
    "plan_oracle (true goal offset, oracle)": "planner, true target position (oracle)",
    "bc_v0.2_s0 (RGB+text BC)": "BC baseline",
    "bc_cf_v0.2_s0 (RGB+text BC)": "BC + counterfactual",
    "bc_cf_film_v0.2_s0 (RGB+text BC, FiLM)": "BC + counterfactual + FiLM",
    "bc_aux_v0.2_s0 (RGB+text BC)": "BC + target-position head",
    "bc_attn_v0.2_s0 (RGB+text BC, attention)": "BC + cross-attention",
    "bc_attn_cf_v0.2_s0 (RGB+text BC, attention)": "BC + cross-attention + counterfactual",
    "plan_integrator_wm_v1_s0 (WM planner)": "world model + planner",
    "plan_wm_wm_v1_s0 (WM planner)": "world model + planner, learned dynamics",
}


def fig_pairs():
    s = json.loads((REPORTS / "candidates_s0_val_summary.json").read_text())
    names = list(LABEL)[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for y, n in enumerate(names):
        v = s[n]
        x = 0
        for k, c in [("pairs_correct", "#1baf7a"), ("pairs_inverted", "#eb6834"),
                     ("pairs_collapsed", "#c3c2b7")]:
            w = v[k]
            if w:
                ax.barh(y, w - 0.06, left=x, height=0.6, color=c)
                ax.text(x + w / 2, y, str(w), ha="center", va="center", fontsize=8.5,
                        color="white" if c != "#c3c2b7" else INK)
            x += w
        ax.text(10.3, y, f"{v['success']}/20", va="center", fontsize=9, color=INK)
        ax.text(11.6, y, f"{v['pairs_success']}/10", va="center", fontsize=9, color=INK)
    ax.text(10.3, len(names) - 0.3, "success", fontsize=8.5, color=INK2)
    ax.text(11.6, len(names) - 0.3, "pair success", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(names)), [LABEL[n] for n in names], fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 13); ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_xlabel("validation pairs (10)")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in ("#1baf7a", "#eb6834", "#c3c2b7")]
    ax.legend(handles, ["both sentences → their own target", "both → the other target (inverted)",
                        "both → the same target (word-blind)"],
              frameon=False, fontsize=8.5, ncol=3, loc="upper center", bbox_to_anchor=(0.4, -0.12))
    ax.set_title("Which target did each sentence lead to? Single training seed, 10 val pairs",
                 loc="left", fontsize=10.5, color=INK)
    fig.savefig(OUT / "01_pairs.png", bbox_inches="tight")
    print("wrote", OUT / "01_pairs.png")


def draw_scene(ax, lay):
    for t, hp in zip(lay["targets"], lay["hover_points"]):
        c = TARGET[t["color"]]
        if t["shape"] == "box":
            r = t["radius"]
            ax.add_patch(Rectangle((t["xy"][0] - r, t["xy"][1] - r), 2 * r, 2 * r, color=c))
        else:
            ax.add_patch(Circle(t["xy"], t["radius"], color=c))
        ax.add_patch(Circle(hp[:2], 0.4, fill=False, ec=c, lw=1.1))
    ax.plot(*lay["start_xyz"][:2], marker=">", ms=9, color=INK)


def fig_paths(pairs=(0, 1, 2)):
    ev = json.loads((REPORTS / "eval_candidates_s0_val.json").read_text())
    eps = sorted([e for e in pq.read_table(DATA / "episodes.parquet").to_pylist()
                  if e["split"] == "val"], key=lambda e: e["episode_id"])
    show = ["expert (oracle)", "bc_v0.2_s0 (RGB+text BC)",
            "bc_attn_cf_v0.2_s0 (RGB+text BC, attention)", "plan_integrator_wm_v1_s0 (WM planner)"]
    fig, axes = plt.subplots(len(pairs), len(show), figsize=(12, 2.7 * len(pairs)))
    for r, pi in enumerate(pairs):
        pair = eps[2 * pi: 2 * pi + 2]
        lay = json.loads((DATA / "layouts" / f"{pair[0]['layout_id']}.json").read_text())
        for c, name in enumerate(show):
            ax = axes[r, c]
            draw_scene(ax, lay)
            trials = {t["episode_id"]: t for t in ev["policies"][name]["trials"]}
            for e in pair:
                p = np.asarray(trials[e["episode_id"]]["path"])
                ax.plot(p[:, 0], p[:, 1], color=TARGET[e["goal_color"]], lw=1.8,
                        ls="-" if e["goal_index"] == 0 else (0, (4, 2)))
            if r == 0:
                ax.set_title(LABEL[name], fontsize=9, color=INK)
            ax.set_xlim(-3.5, 3.6); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Val pairs 0-2, top view. Line colour = the target the sentence named.",
                 x=0.02, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "02_paths.png", bbox_inches="tight")
    print("wrote", OUT / "02_paths.png")


if __name__ == "__main__":
    fig_pairs()
    fig_paths()
