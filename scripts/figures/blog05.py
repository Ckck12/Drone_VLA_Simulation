"""Figures for blog part 5 (borrowing from VLAs). Specs: docs/FIG_SPEC_blog05.md.

    python scripts/figures/blog05.py            # -> reports/figures/blog05/
"""
import collections
import textwrap
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from figstyle import (BLUE, DGRAY, GRAY, GREEN, LBLUE, LGREEN, LORANGE, ORANGE, PANEL_EC,  # noqa: E402
                      PANEL_FC, VERM, Canvas, FS, FS_S, FS_T, plt)
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

R = REPO / "reports"
DATA = REPO / "data" / "v0.2"
OUT = R / "figures" / "blog05"
OUT.mkdir(parents=True, exist_ok=True)
EPISODES = pq.read_table(DATA / "episodes.parquet").to_pylist()


def first_frame(split, pair_index, goal=0):
    eps = sorted([e for e in EPISODES if e["split"] == split], key=lambda e: e["episode_id"])
    e = eps[2 * pair_index + goal]
    st = pq.read_table(DATA / "steps" / f"{e['episode_id']}.parquet").to_pylist()[0]
    return e, st


# ======================================================================= F5-1 method
def mini_hist(c, x, y, w, h, peak, color, n=24, width=2.2):
    """Schematic bar histogram: probability over bins, peaked at fraction `peak`."""
    xs = np.linspace(0, 1, n)
    p = np.exp(-((xs - peak) ** 2) / (2 * 0.05 ** 2)) + 0.03
    p = p / p.max()
    for i, v in enumerate(p):
        c.rect(x + i * w / n, y, w / n * 0.8, h * v, fc=color, ec="none", lw=0, z=5)
    c.line([x - 1, x + w], [y, y], color=GRAY, lw=0.6, z=5)


def fig_method():
    W, H = 980, 500
    c = Canvas(W, H)
    c.txt(12, H - 14, "Same backbone, three ways to output an action", FS_T + 1, ha="left", weight="bold")

    # ---------------- inputs
    ep, st = first_frame("val", 0)
    lines = textwrap.wrap(f'"{ep["instruction"]}"', 19)
    im = np.asarray(Image.open(DATA / st["rgb_path"]))
    c.ax.imshow(im, extent=(14, 144, 312, 312 + 130 * 96 / 128), zorder=3, interpolation="nearest")
    c.rect(14, 312, 130, 130 * 96 / 128, fc="none", ec=BLUE, lw=0.9, z=4)
    c.txt(79, 300, "camera RGB  96×128×3", FS_S, BLUE)
    c.rbox(14, 205, 130, 54, fc=LBLUE, ec=BLUE, r=4)
    for i, ln in enumerate(lines):
        c.txt(79, 232 + 7 * (len(lines) - 1) - 14 * i, ln, FS_S, DGRAY)
    c.txt(79, 195, "instruction", FS_S, BLUE)
    c.rbox(14, 110, 130, 54, fc=LBLUE, ec=BLUE, r=4)
    c.txt(79, 145, "velocity · attitude", FS_S, DGRAY)
    c.txt(79, 130, "rates · altitude", FS_S, DGRAY)
    c.txt(79, 100, "state (11 numbers)", FS_S, BLUE)

    # ---------------- shared backbone
    c.rbox(166, 80, 352, 385, fc=PANEL_FC, ec=PANEL_EC, r=6)
    c.txt(342, 450, "shared encoder — same layers as Part 4", FS, DGRAY, weight="bold")
    # CNN as shrinking feature maps
    xs, hs = [186, 214, 238, 258], [92, 74, 56, 40]
    for i, (xx, hh) in enumerate(zip(xs, hs)):
        c.rect(xx, 372 - hh / 2, 16 - 2 * i, hh, fc="#DADFE6", ec=GRAY, lw=0.6, z=4)
    c.txt(228, 314, "4 conv layers (stride 2)", FS_S, DGRAY)
    c.txt(228, 300, "96×128 → 6×8×64", FS_S, DGRAY)
    c.arrow((146, 372), (182, 372))
    c.rbox(298, 356, 62, 32, fc="white", ec=GRAY, r=3)
    c.txt(329, 372, "FC", FS_S, DGRAY)
    c.arrow((276, 372), (296, 372))
    c.tokens(380, 372, 5, "#C9D2DC", GRAY, s=8, step=10)
    c.txt(404, 352, "image 128", FS_S, DGRAY)
    c.txt(404, 339, "(b, c: 104)", FS_S, ORANGE)
    c.arrow((362, 372), (378, 372))
    # words
    c.arrow((146, 232), (182, 232))
    c.tokens(186, 232, 7, "#C9D2DC", GRAY, s=8, step=11)
    c.txt(222, 252, "word embeddings", FS_S, DGRAY)
    c.txt(222, 212, "17-word vocab", FS_S, DGRAY)
    c.arrow((262, 232), (296, 232))
    c.rbox(298, 216, 62, 32, fc="white", ec=GRAY, r=3)
    c.txt(329, 232, "mean", FS_S, DGRAY)
    c.arrow((362, 232), (378, 232))
    c.tokens(380, 232, 3, "#C9D2DC", GRAY, s=8, step=10)
    c.txt(392, 212, "text 32", FS_S, DGRAY)
    # state
    c.arrow((146, 137), (284, 137))
    c.mlp(288, 130, 44, 16, fc="#DADFE6", ec=GRAY, label="MLP")
    c.arrow((334, 137), (378, 137))
    c.tokens(380, 137, 3, "#C9D2DC", GRAY, s=8, step=10)
    c.txt(392, 117, "state 32", FS_S, DGRAY)
    # concat
    c.rbox(466, 110, 30, 290, fc="#C9D2DC", ec=GRAY, r=3)
    c.txt(481, 255, "concat  192   (b, c: 168)", FS_S, DGRAY).set_rotation(90)
    for yy in (372, 232, 137):
        c.arrow((432, yy), (464, yy))

    # ---------------- three heads
    rows = [(372, "(a)  Part 4: regression", "#F2F2F2", GRAY, "3/20 · pair 0/10"),
            (248, "(b)  action tokens", LORANGE, ORANGE, "8/20 · pair 1/10"),
            (124, "(c)  8-step token chunk", LGREEN, GREEN, "8/20 · pair 3/10")]
    for yc, title, fc, ec, res in rows:
        c.rbox(540, yc - 58, 428, 112, fc=fc, ec=ec, r=6, lw=1.0 if ec != GRAY else 0.8)
        c.txt(552, yc + 40, title, FS, DGRAY, ha="left", weight="bold")
        c.pill(962, yc + 40, res, ec)
        c.polyarrow([(498, 255), (520, 255), (520, yc), (548, yc)])
    # (a) regression
    y = 372
    c.mlp(554, y - 8, 46, 16, fc="#E2E2E2", ec=GRAY, label="MLP")
    c.arrow((602, y), (628, y))
    c.rbox(630, y - 16, 96, 32, fc="white", ec=GRAY, r=3)
    c.txt(678, y + 3, "tanh × limit", FS_S, DGRAY)
    c.txt(678, y - 9, "→ vx, vy", FS_S, DGRAY)
    c.arrow((728, y), (750, y))
    c.txt(756, y + 4, "one number per axis", FS_S, DGRAY, ha="left")
    c.txt(756, y - 10, "e.g. vy = +0.04 m/s", FS_S, DGRAY, ha="left")
    c.txt(552, y - 34, "train: Huber loss against the expert's speed", FS_S, GRAY, ha="left", style="italic")
    # (b) tokens
    y = 248
    c.mlp(554, y - 8, 46, 16, fc="#F6D49A", ec=ORANGE, label="MLP")
    c.arrow((602, y), (622, y))
    mini_hist(c, 626, y - 4, 70, 22, 0.72, ORANGE)
    c.txt(661, y - 14, "vx: 256 bins", FS_S, DGRAY)
    mini_hist(c, 704, y - 4, 70, 22, 0.30, ORANGE)
    c.txt(739, y - 14, "vy: 256 bins", FS_S, DGRAY)
    c.arrow((778, y + 6), (800, y + 6))
    c.txt(806, y + 12, "pick the most", FS_S, DGRAY, ha="left")
    c.txt(806, y - 1, "likely bin → its", FS_S, DGRAY, ha="left")
    c.txt(806, y - 14, "centre speed", FS_S, DGRAY, ha="left")
    c.txt(552, y - 34, "train: cross-entropy against the expert's bin", FS_S, ORANGE, ha="left",
          style="italic")
    c.txt(552, y - 46, "bins span the 1st–99th percentile of the training speeds", FS_S, ORANGE,
          ha="left", style="italic")
    # (c) chunk
    y = 124
    c.mlp(554, y - 8, 46, 16, fc="#BFE3D3", ec=GREEN, label="trunk")
    c.arrow((602, y), (616, y))
    c.txt(642, y + 20, "+ step k", FS_S, DGRAY)
    c.tokens(620, y + 8, 8, "#BFE3D3", GREEN, s=5, step=6)
    c.txt(642, y - 12, "k = 1…8", FS_S, DGRAY)
    c.arrow((668, y), (686, y))
    c.mlp(688, y - 8, 52, 16, fc="#BFE3D3", ec=GREEN, label="decoder")
    c.arrow((742, y), (758, y))
    for k in range(4):
        mini_hist(c, 762 + k * 4, y - 8 + k * 6, 46, 10, 0.3 + 0.03 * k, GREEN, n=16)
    c.txt(826, y + 14, "bins + Stop for", FS_S, DGRAY, ha="left")
    c.txt(826, y + 1, "the next 8 steps", FS_S, DGRAY, ha="left")
    c.txt(826, y - 12, "(1.6 s)", FS_S, DGRAY, ha="left")
    c.txt(552, y - 34, "train: cross-entropy over all 8 future steps", FS_S, GREEN, ha="left",
          style="italic")
    c.txt(552, y - 46, "fly: execute step 1 only, ask again 0.2 s later", FS_S, GREEN, ha="left",
          style="italic")

    # legend / note
    c.rect(170, 52, 14, 10, fc="#C9D2DC", ec=GRAY, lw=0.5, z=5)
    c.txt(190, 57, "unchanged from Part 4", FS_S, DGRAY, ha="left")
    c.rect(340, 52, 14, 10, fc=LORANGE, ec=ORANGE, lw=0.8, z=5)
    c.txt(360, 57, "changed in this post", FS_S, DGRAY, ha="left")
    c.txt(540, 58, "Every head also outputs a Stop logit (binary cross-entropy).", FS_S, GRAY,
          ha="left")
    c.txt(540, 43, "Results: validation, single training seed. Bar shapes are illustrative.",
          FS_S, GRAY, ha="left")
    c.txt(540, 28, "Parameters: (a) 480,901  (b) 469,609  (c) 470,633 — no pretrained parts.",
          FS_S, GRAY, ha="left")
    c.fig.savefig(OUT / "01_method.png", dpi=220)
    c.fig.savefig(OUT / "01_method.svg")
    plt.close(c.fig)
    print("wrote", OUT / "01_method.png")


# ======================================================================= data helpers
import torch  # noqa: E402

from dronevla.evaluate import LearnedPolicy  # noqa: E402
from dronevla.model import pad_batch  # noqa: E402
from dronevla.train import load_rows  # noqa: E402

EVALS = ["eval_candidates_s0_val.json", "eval_tokens_s0_val.json", "eval_tokens2_s0_val.json",
         "eval_attn_tok_s0_val.json", "eval_explore_s0_val.json"]
SUMMARY = {r["label"]: r for r in json.loads((R / "season2_summary.json").read_text())}
OBJ = {"red": "#C8352B", "blue": "#2E5FB8", "green": "#2E9A4A", "yellow": "#D9B310"}
# (summary label, short name, eval policy key prefix)
MODELS = {
    "base": ("BC baseline (regression)", "Part 4 regression", "bc_v0.2_s0 ("),
    "tok": ("256-bin action tokens", "256-bin tokens", "bc_tok256_v0.2_s0 ("),
    "chunk": ("256-bin + chunk 8 (execute 1)", "tokens + chunk 8", "bc_tokchunk8_v0.2_s0 (RGB+text BC, 256-bin tokens, chunk 8, exec 1)"),
    "chunk8": ("256-bin + chunk 8 (execute 8)", "chunk 8, run all 8", None),
    "explore": ("chunk 8 + exploration states", "chunk 8 + exploration", None),
    "wm": ("world model + planner (integrator)", "world model + planner*", None),
    "expert": ("scripted expert (oracle)", "scripted expert*", None),
}


def split_pairs(split):
    eps = sorted([e for e in EPISODES if e["split"] == split], key=lambda e: e["episode_id"])
    pairs = [eps[2 * i: 2 * i + 2] for i in range(len(eps) // 2)]
    assert all(a["pair_id"] == b["pair_id"] for a, b in pairs)
    return pairs


def layout(lid):
    return json.loads((DATA / "layouts" / f"{lid}.json").read_text())


def trials(prefix):
    for f in EVALS:
        for name, p in json.loads((R / f).read_text())["policies"].items():
            if name.startswith(prefix):
                return {t["episode_id"]: t for t in p["trials"]}
    raise KeyError(prefix)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def seg_label(ax, x, y, n, col, hatch):
    kw = dict(ha="center", va="center", fontsize=FS_S)
    if hatch:
        kw["bbox"] = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
        ax.text(x, y, str(n), color="black", **kw)
    else:
        ax.text(x, y, str(n), color="white" if col in (VERM, DGRAY, GREEN) else "black", **kw)


def style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=FS_S)


# ======================================================================= F5-2 midpoint vs pick
def first_frames(split):
    """First recorded frame of every episode of a split, in pair order (A, B, A, B, ...)."""
    eps = [e for p in split_pairs(split) for e in p]
    rows = [pq.read_table(DATA / "steps" / f"{e['episode_id']}.parquet").to_pylist()[0] for e in eps]
    assert all(r["action_mask"] for r in rows)
    return eps, {"rgb": torch.from_numpy(np.stack([np.asarray(Image.open(DATA / r["rgb_path"])) for r in rows])),
                 "proprio": torch.tensor([r["proprio"] for r in rows], dtype=torch.float32),
                 "action": torch.tensor([r["action_applied"] for r in rows], dtype=torch.float32)}


def fig_midpoint(n_show=4):
    eps, data = first_frames("train")
    reg = LearnedPolicy(REPO / "runs" / "bc_v0.2_s0")
    tok = LearnedPolicy(REPO / "runs" / "bc_tok256_v0.2_s0")

    def run(pol):
        ids = pad_batch([pol.vocab.encode(e["instruction"]) for e in eps])
        prop = (data["proprio"] - pol.mean) / pol.std
        with torch.no_grad():
            out = pol.model(data["rgb"], prop, pol.model.encode_text(ids), return_logits=True)
        return (out[0][:, 1] * pol.caps[1]).numpy(), out[2]

    reg_vy, _ = run(reg)
    tok_vy, logits = run(tok)
    probs = torch.softmax(logits[:, 1], -1).numpy()                     # vy bins
    centres = (tok.model.bin_centres()[1] * tok.caps[1]).numpy()
    expert = data["action"][:, 1].numpy()
    partner = expert.reshape(-1, 2)[:, ::-1].reshape(-1)

    # aggregate over all training pairs: which expert answer is each output closer to?
    def closer(v):
        own, oth = np.abs(v - expert), np.abs(v - partner)
        return int((own < oth).sum()), int((oth < own).sum())
    mid = np.abs(reg_vy - (expert + partner) / 2)
    stats = {"frames": len(eps), "regression_own_vs_partner": closer(reg_vy),
             "tokens_own_vs_partner": closer(tok_vy),
             "regression_mean_err": float(np.abs(reg_vy - expert).mean()),
             "regression_mean_dist_to_midpoint": float(mid.mean()),
             "tokens_mean_err": float(np.abs(tok_vy - expert).mean()),
             "tokens_median_err": float(np.median(np.abs(tok_vy - expert)))}
    (OUT / "02_midpoint_vs_pick.json").write_text(json.dumps(stats, indent=1))

    fig, axes = plt.subplots(2, n_show, figsize=(10.5, 4.8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.7]})
    lim = 0.22
    for j in range(n_show):
        a, b = 2 * j, 2 * j + 1
        cols = [OBJ[eps[i]["goal_color"]] for i in (a, b)]
        ax = axes[0, j]
        for k, (i, mk, dy) in enumerate([(a, "o", 0.15), (b, "s", -0.15)]):
            ax.scatter(expert[i], 1 + dy, s=55, marker=mk, color=cols[k], zorder=3)
            ax.scatter(reg_vy[i], dy, s=55, marker=mk, facecolor="white", edgecolor=cols[k],
                       lw=1.6, zorder=3)
        ax.axvline(expert[[a, b]].mean(), color=GRAY, ls=":", lw=1)
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1], ["regression\n(Part 4)", "expert"] if j == 0 else ["", ""])
        ax.set_title(f"training pair {j + 1}", fontsize=FS)
        style_ax(ax)
        ax = axes[1, j]
        for k, (i, sign) in enumerate([(a, 1), (b, -1)]):
            nz = probs[i] > 0.005
            ax.vlines(centres[nz], 0, sign * probs[i][nz], color=cols[k], lw=2.2)
            ax.scatter(expert[i], sign * 1.08, marker="v" if sign > 0 else "^", color=cols[k],
                       s=30, zorder=3)
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([-1, 0, 1], ["1", "0", "1"])
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("sideways speed vy (m/s)", fontsize=FS_S)
        if j == 0:
            ax.set_ylabel("token head, probability\nsentence B ↓    sentence A ↑", fontsize=FS_S)
        style_ax(ax)
    h = [plt.Line2D([], [], color=DGRAY, marker="o", ls="none", label="sentence A"),
         plt.Line2D([], [], color=DGRAY, marker="s", ls="none", label="sentence B"),
         plt.Line2D([], [], color=DGRAY, marker="o", mfc="white", ls="none", label="open = model output"),
         plt.Line2D([], [], color=GRAY, ls=":", label="midpoint of the two expert answers"),
         plt.Line2D([], [], color=DGRAY, marker="v", ls="none", label="▼▲ expert answer")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=FS_S,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Same first frame, two sentences: regression gives one in-between answer;\n"
                 "tokens put their probability on one expert answer (not always the right one)",
                 x=0.01, ha="left", fontsize=FS_T, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "02_midpoint_vs_pick.png", dpi=200)
    plt.close(fig)
    print("wrote 02", stats)


# ======================================================================= F5-3 attention
def fig_attention(pair_ids=(0, 1)):
    runs = [("bc_attn_v0.2_s0", "attention + regression"),
            ("bc_attn_tok256_v0.2_s0", "attention + tokens")]
    pols = [LearnedPolicy(REPO / "runs" / r) for r, _ in runs]
    pairs = split_pairs("val")
    fig, axes = plt.subplots(len(pair_ids), 5, figsize=(11, 1.95 * len(pair_ids) + 0.5))
    for r, pi in enumerate(pair_ids):
        pair = pairs[pi]
        _, st = first_frame("val", pi)
        rgb = np.asarray(Image.open(DATA / st["rgb_path"]))
        obs = {"rgb": rgb, "proprio": np.asarray(st["proprio"], dtype=np.float32)}
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(f"val pair {pi + 1}: first frame", fontsize=FS_S)
        for m, pol in enumerate(pols):
            for g, e in enumerate(pair):
                pol.reset(e["instruction"])
                pol.act(obs)
                w = pol.model.last_attn[0].mean(0).reshape(6, 8).numpy()
                ax = axes[r, 1 + 2 * m + g]
                ax.imshow(rgb)
                ax.imshow(np.kron(w / w.max(), np.ones((16, 16))), cmap="magma", alpha=0.55,
                          vmin=0, vmax=1)
                ax.set_title("\n".join(textwrap.wrap(f'"{e["instruction"]}"', 26)), fontsize=7)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    for m, (_, name) in enumerate(runs):
        x = (axes[0, 1 + 2 * m].get_position().x0 + axes[0, 2 + 2 * m].get_position().x1) / 2
        fig.text(x, 0.985, name, ha="center", va="top", fontsize=FS, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for m, (_, name) in enumerate(runs):
        p0, p1 = axes[0, 1 + 2 * m].get_position(), axes[0, 2 + 2 * m].get_position()
        fig.texts[m].set_x((p0.x0 + p1.x1) / 2)
    fig.savefig(OUT / "03_attention.png", dpi=200)
    plt.close(fig)
    print("wrote 03")


# ======================================================================= F5-4 scoreboard
def fig_scoreboard():
    keys = ["expert", "wm", "base", "tok", "chunk", "chunk8", "explore"]
    rows = [SUMMARY[MODELS[k][0]] for k in keys][::-1]
    names = [MODELS[k][1] for k in keys][::-1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True,
                                     gridspec_kw={"width_ratios": [2.2, 1.1, 1.6]})
    segs = [("pairs_correct", GREEN, "", "each sentence → its own target"),
            ("pairs_inverted", VERM, "////", "swapped"),
            ("pairs_same_target", "#C9C9C9", "", "both → the same target")]
    for y, r in enumerate(rows):
        x = 0
        for k, col, hatch, _ in segs:
            if r[k]:
                a1.barh(y, r[k], left=x, height=0.6, color=col, hatch=hatch, edgecolor="white", lw=0.5)
                seg_label(a1, x + r[k] / 2, y, r[k], col, hatch)
            x += r[k]
        a2.barh(y, r["success"], height=0.6, color=BLUE)
        a2.text(r["success"] + 0.4, y, f"{r['success']}", va="center", fontsize=FS_S)
        lo, hi = wilson(r["pair_success"], 10)
        a3.plot([10 * lo, 10 * hi], [y, y], color=GRAY, lw=1.4)
        a3.scatter(r["pair_success"], y, color=ORANGE if r["group"] == "token BC" else DGRAY,
                   zorder=3, s=36)
        a3.text(10.9, y, f"{r['pair_success']}/10", va="center", fontsize=FS_S)
    a1.set_yticks(range(len(rows)), names, fontsize=FS_S)
    a1.axhline(4.5, color=PANEL_EC, lw=1)
    for a in (a2, a3):
        a.axhline(4.5, color=PANEL_EC, lw=1)
    a1.set_xlim(0, 10); a1.set_title("which target each sentence led to\n(10 val pairs)", fontsize=FS)
    a2.set_xlim(0, 22); a2.set_title("success /20", fontsize=FS)
    a3.set_xlim(0, 12.5); a3.set_xticks([0, 5, 10])
    a3.set_title("pair success /10\n(both sentences succeed; 95% Wilson)", fontsize=FS)
    for a in (a1, a2, a3):
        style_ax(a); a.tick_params(axis="y", length=0)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=hh, ec="white") for _, c, hh, _ in segs]
    a1.legend(h, [s[3] for s in segs], ncol=3, frameon=False, fontsize=FS_S, loc="upper left",
              bbox_to_anchor=(-0.55, -0.1), handlelength=1.2, columnspacing=1)
    fig.suptitle("End-to-end models in this post, with references (*)", x=0.01, ha="left",
                 fontsize=FS_T, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "04_scoreboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 04")


# ======================================================================= F5-5 paths
def fig_paths(pair_ids=(0, 1, 2)):
    keys = ["base", "tok", "chunk"]
    tr = {k: trials(MODELS[k][2]) for k in keys}
    pairs = split_pairs("val")
    fig, axes = plt.subplots(len(pair_ids), 3, figsize=(10.5, 2.75 * len(pair_ids) + 0.6))
    for r, pi in enumerate(pair_ids):
        pair = pairs[pi]
        L = layout(pair[0]["layout_id"])
        pts = [L["start_xyz"][:2]] + [t["xy"] for t in L["targets"]]
        pts += [p_[:2] for k in keys for e in pair for p_ in tr[k][e["episode_id"]]["path"]]
        pts = np.asarray(pts)
        lo, hi = pts.min(0) - 0.55, pts.max(0) + 0.55
        for c_, k in enumerate(keys):
            ax = axes[r, c_]
            for t in L["targets"]:
                x0, y0 = t["xy"]
                if t["shape"] == "box":
                    ax.add_patch(plt.Rectangle((x0 - 0.2, y0 - 0.2), 0.4, 0.4, color=OBJ[t["color"]], zorder=3))
                else:
                    ax.add_patch(plt.Circle(t["xy"], t["radius"], color=OBJ[t["color"]], zorder=3))
            for g, e in enumerate(pair):
                hp = L["hover_points"][e["goal_index"]]
                col = OBJ[e["goal_color"]]
                ax.add_patch(plt.Circle(hp[:2], 0.4, fill=False, ec=col, ls=":", lw=1))
                t = tr[k][e["episode_id"]]
                path = np.asarray(t["path"])
                ok = t["outcome"] == "success"
                ax.plot(path[:, 0], path[:, 1], color=col, lw=1.8, ls="-" if g == 0 else "--",
                        label=f'"{e["goal_color"]} {e["goal_shape"]}" → {t["outcome"].replace("_", " ")}')
                ax.scatter(*path[-1, :2], color=col, marker="*" if ok else "x", s=90 if ok else 45,
                           zorder=5, edgecolor="black" if ok else None, linewidth=0.4 if ok else 1.5)
            ax.scatter(*L["start_xyz"][:2], color="black", marker="^", s=40, zorder=4)
            ax.set_aspect("equal")
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
            ax.set_xticks([]); ax.set_yticks([])
            ax.legend(fontsize=7, loc="upper left", frameon=False, handlelength=1.8,
                      bbox_to_anchor=(0.0, 0.0), borderaxespad=0.2)
            if r == 0:
                ax.set_title(MODELS[k][1], fontsize=FS, weight="bold")
            if c_ == 0:
                ax.set_ylabel(f"val pair {pi + 1}", fontsize=FS)
    fig.suptitle("Top view of the first three validation pairs (not selected).  ▲ start   ★ success   "
                 "× failure   dotted circle = 0.4 m goal region", x=0.01, ha="left", fontsize=FS)
    fig.tight_layout(h_pad=2.6)
    fig.savefig(OUT / "05_paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 05")


# ======================================================================= F5-6 reaction vs stop
def fig_reaction():
    sens = {}
    for f in ["instruction_sensitivity_v0.2_train.json", "instruction_sensitivity_tok256_train.json",
              "instruction_sensitivity_tokens2_train.json", "instruction_sensitivity_explore_train.json"]:
        sens.update(json.loads((R / f).read_text())["policies"])
    expert_gap = None
    keys = [("base", "bc_v0.2_s0"), ("tok", "bc_tok256_v0.2_s0"), ("chunk", "bc_tokchunk8_v0.2_s0"),
            ("explore", "bc_tokchunk8_explore_v0.2_s0")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1, 1.5]})
    for x, (k, run) in enumerate(keys):
        g = np.asarray(sens[run]["first_frame"]["policy_gap"])
        expert_gap = np.asarray(sens[run]["first_frame"]["expert_gap"])
        jit = np.random.default_rng(0).uniform(-0.22, 0.22, len(g))
        a1.scatter(x + jit, g, s=5, alpha=0.45, lw=0, color=ORANGE if k != "base" else GRAY)
        a1.plot([x - 0.3, x + 0.3], [np.median(g)] * 2, color="black", lw=2)
        a1.text(x, 1.02, f"{int((g > 0.02).sum())}/200 react", transform=a1.get_xaxis_transform(),
                ha="center", fontsize=6.8, color=DGRAY)
    a1.axhline(np.median(expert_gap), color=BLUE, ls="--", lw=1.2)
    a1.text(-0.45, np.median(expert_gap) + 0.01, "expert", color=BLUE, fontsize=FS_S, ha="left")
    a1.set_xticks(range(len(keys)), ["\n".join(textwrap.wrap(MODELS[k][1], 12)) for k, _ in keys],
                  fontsize=FS_S)
    a1.set_ylabel("change in vy when only the\nsentence is swapped (m/s)", fontsize=FS_S)
    a1.set_xlim(-0.5, len(keys) - 0.5)
    a1.set_title("first frame of 200 training episodes\n(dot = one episode, bar = median)",
                 fontsize=FS, pad=16)
    style_ax(a1)

    cats = [("success", GREEN, ""), ("wrong_target", VERM, "////"), ("stop_elsewhere", "#CC79A7", ""),
            ("stop_not_settled", "#F0E442", ""), ("out_of_bounds", "#56B4E9", "xxx"),
            ("collision", DGRAY, ""), ("timeout", "#C9C9C9", "")]
    for y, (k, _) in enumerate(keys[::-1]):
        oc = SUMMARY[MODELS[k][0]]["outcomes"]
        x = 0
        for c, col, hatch in cats:
            n = oc.get(c, 0)
            if n:
                a2.barh(y, n, left=x, height=0.6, color=col, hatch=hatch, edgecolor="white", lw=0.5)
                seg_label(a2, x + n / 2, y, n, col, hatch)
                x += n
    a2.set_yticks(range(len(keys)), [MODELS[k][1] for k, _ in keys[::-1]], fontsize=FS_S)
    a2.set_xlim(0, 20)
    a2.set_title("what happened in the 20 validation flights", fontsize=FS)
    style_ax(a2); a2.tick_params(axis="y", length=0)
    h = [plt.Rectangle((0, 0), 1, 1, color=c, hatch=hh, ec="white") for _, c, hh in cats]
    a2.legend(h, [c.replace("_", " ") for c, _, _ in cats], ncol=4, frameon=False, fontsize=6.8,
              loc="upper center", bbox_to_anchor=(0.45, -0.08), handlelength=1.2)
    fig.suptitle("Exploration states make the words count — and the drone stops stopping",
                 x=0.01, ha="left", fontsize=FS_T, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_reaction_vs_stop.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 06; expert median gap %.3f" % np.median(expert_gap))


if __name__ == "__main__":
    which = sys.argv[1:] or ["method", "midpoint", "attention", "scoreboard", "paths", "reaction"]
    for w in which:
        globals()[f"fig_{w}"]()
