"""Figures for blog part 6 (world model + planner).

    python scripts/figures/blog06.py            # -> reports/figures/blog06/
"""
import json
import pathlib
import sys
import textwrap

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import blog05 as b5  # noqa: E402
from figstyle import (BLUE, DGRAY, GRAY, GREEN, LBLUE, LGREEN, LORANGE, ORANGE, PANEL_EC,  # noqa: E402
                      PANEL_FC, VERM, Canvas, FS, FS_S, FS_T, plt)
from PIL import Image  # noqa: E402

R, DATA = b5.R, b5.DATA
OUT = R / "figures" / "blog06"
OUT.mkdir(parents=True, exist_ok=True)
OBJ = b5.OBJ

MODELS = {
    "expert": ("scripted expert (oracle)", "scripted expert*", None),
    "oracle": ("planner + true target position (oracle)", "planner, true target position*", None),
    "int": ("world model + planner (integrator)", "world model + planner (integrator)",
            "plan_integrator_wm_v1_s0 ("),
    "wmdyn": ("world model + planner (learned dynamics)", "world model + planner (learned dynamics)",
              "plan_wm_wm_v1_s0 ("),
    "base": ("BC baseline (regression)", "BC, Part 4", "bc_v0.2_s0 ("),
    "aux": ("+ target-position aux head", "BC + the same position labels", "bc_aux_v0.2_s0 ("),
    "chunk": ("256-bin + chunk 8 (execute 1)", "BC tokens + chunk 8 (Part 5)", None),
}


# ======================================================================= F6-1 system
def fig_system():
    W, H = 1180, 500
    c = Canvas(W, H)
    c.txt(12, H - 14, "Perceive every target, let the words pick one, then plan to it",
          FS_T + 1, ha="left", weight="bold")

    # inputs
    _, st = b5.first_frame("val", 0)
    ep = sorted([e for e in b5.EPISODES if e["split"] == "val"], key=lambda e: e["episode_id"])[0]
    im = np.asarray(Image.open(DATA / st["rgb_path"]))
    c.ax.imshow(im, extent=(14, 134, 340, 430), zorder=3, interpolation="nearest")
    c.rect(14, 340, 120, 90, fc="none", ec=BLUE, lw=0.9, z=4)
    c.txt(74, 328, "camera RGB", FS_S, BLUE)
    c.rbox(14, 270, 120, 40, fc=LBLUE, ec=BLUE, r=4)
    c.txt(74, 290, "state (11)", FS_S, DGRAY)
    c.rbox(14, 148, 120, 56, fc=LBLUE, ec=BLUE, r=4)
    for i, ln in enumerate(textwrap.wrap(f'"{ep["instruction"]}"', 19)):
        c.txt(74, 188 - 13 * i, ln, FS_S, DGRAY)
    c.txt(74, 136, "instruction", FS_S, BLUE)

    # 1 perceive
    c.rbox(156, 262, 404, 200, fc=LBLUE, ec=BLUE, r=6)
    c.badge(172, 446, 1, fc=BLUE)
    c.txt(184, 446, "perceive: world-model encoder (learned, sees no words)", FS, DGRAY, ha="left",
          weight="bold")
    c.arrow((136, 385), (172, 385))
    c.polyarrow([(136, 290), (160, 290), (160, 362), (172, 362)])
    for i, (xx, hh) in enumerate(zip([176, 198, 216, 230], [76, 60, 46, 34])):
        c.rect(xx, 375 - hh / 2, 14 - 2 * i, hh, fc="#CFE0F0", ec=BLUE, lw=0.6, z=4)
    c.arrow((242, 375), (258, 375))
    c.tokens(262, 375, 6, "#CFE0F0", BLUE, s=7, step=9)
    c.txt(284, 357, "latent z (64)", FS_S, DGRAY)
    c.arrow((314, 375), (330, 375))
    c.mlp(332, 367, 44, 16, fc="#CFE0F0", ec=BLUE, label="head")
    c.arrow((378, 375), (396, 375))
    for i, col in enumerate(["red", "blue", "green", "yellow"]):
        y = 420 - 22 * i
        c.rect(400, y - 7, 14, 14, fc=OBJ[col], ec="none", z=5)
        c.txt(420, y, f"{col}: (dx, dy)", FS_S, DGRAY, ha="left")
    c.txt(400, 316, "offset from the drone to", FS_S, DGRAY, ha="left")
    c.txt(400, 304, "each colour's hover point", FS_S, DGRAY, ha="left")
    c.txt(172, 286, "head trained against true offsets (simulator, training only, never an input)",
          FS_S, BLUE, ha="left", style="italic")
    c.txt(172, 273, "it also learns z' = LN(z + f(z, a)): what happens after a command", FS_S, BLUE,
          ha="left", style="italic")

    # 2 select
    c.rbox(156, 112, 404, 116, fc=LBLUE, ec=BLUE, r=6)
    c.badge(172, 212, 2, fc=BLUE)
    c.txt(184, 212, "select: goal selector (learned, 340 parameters)", FS, DGRAY, ha="left",
          weight="bold")
    c.arrow((136, 176), (176, 176))
    c.tokens(180, 176, 6, "#CFE0F0", BLUE, s=7, step=9)
    c.txt(206, 158, "word embeddings, mean", FS_S, DGRAY)
    c.arrow((232, 176), (262, 176))
    c.mlp(264, 168, 48, 16, fc="#CFE0F0", ec=BLUE, label="linear")
    c.arrow((314, 176), (330, 176))
    for i, (col, p) in enumerate(zip(["red", "blue", "green", "yellow"], [0.02, 0.94, 0.03, 0.01])):
        c.rect(336 + 18 * i, 168, 12, 30 * p + 1, fc=OBJ[col], ec="none", z=5)
    c.txt(370, 158, "which colour?", FS_S, DGRAY)
    c.txt(172, 124, "the only place the words enter: they choose, they do not steer", FS_S, BLUE,
          ha="left", style="italic")

    # mux: pick slot
    c.rbox(578, 352, 70, 48, fc="white", ec=DGRAY, r=4)
    c.txt(613, 382, "pick the", FS_S, DGRAY)
    c.txt(613, 369, "named slot", FS_S, DGRAY)
    c.polyarrow([(520, 398), (566, 398), (566, 376), (576, 376)])
    c.polyarrow([(410, 176), (613, 176), (613, 350)])
    c.txt(618, 190, "colour", FS_S, DGRAY, ha="left")

    # 3 filter
    c.rbox(668, 334, 182, 84, fc="#F2F2F2", ec=GRAY, r=4)
    c.badge(682, 404, 3, fc=GRAY)
    c.txt(694, 404, "filter (rule)", FS_S, DGRAY, ha="left", weight="bold")
    c.txt(680, 384, "pred = g − last command·Δt", FS_S, DGRAY, ha="left")
    c.txt(680, 370, "g = pred + 0.3·(new − pred)", FS_S, DGRAY, ha="left")
    c.txt(680, 350, "→ goal offset g (2 numbers)", FS_S, DGRAY, ha="left")
    c.arrow((650, 376), (666, 376))

    # 4 plan
    c.rbox(870, 262, 300, 200, fc=PANEL_FC, ec=PANEL_EC, r=6)
    c.badge(886, 446, 4, fc=DGRAY)
    c.txt(898, 446, "plan: cross-entropy method (rule)", FS, DGRAY, ha="left", weight="bold")
    c.txt(882, 426, "try 64 command sequences, 8 steps (1.6 s) each;", FS_S, DGRAY, ha="left")
    c.txt(882, 413, "keep the best 8, resample, 4 rounds; fly step 1", FS_S, DGRAY, ha="left")
    c.txt(882, 398, "cost: |g + predicted move|², jerkiness, end speed", FS_S, DGRAY, ha="left")
    c.arrow((852, 376), (868, 376))
    c.rbox(882, 330, 276, 56, fc=LGREEN, ec=GREEN, r=4, lw=1.0)
    c.txt(890, 372, "(A) integrator: move = −Σ command·Δt", FS_S, DGRAY, ha="left", weight="bold")
    c.txt(890, 358, "assumes the drone does what it is told", FS_S, DGRAY, ha="left")
    c.pill(1152, 341, "15/20 · pair 6/10", GREEN)
    c.rbox(882, 270, 276, 54, fc=LORANGE, ec=ORANGE, r=4, lw=1.0)
    c.txt(890, 310, "(B) learned dynamics: roll z forward", FS_S, DGRAY, ha="left", weight="bold")
    c.txt(890, 296, "with the world model, read the head", FS_S, DGRAY, ha="left")
    c.pill(1152, 281, "0/20 · pair 0/10", ORANGE)
    c.polyarrow([(480, 262), (480, 246), (862, 246), (862, 296), (880, 296)], color=ORANGE, ls="--")

    # 5 stop
    c.rbox(870, 112, 300, 116, fc="#F2F2F2", ec=GRAY, r=6)
    c.badge(886, 212, 5, fc=GRAY)
    c.txt(898, 212, "stop (rule, same as the expert's)", FS, DGRAY, ha="left", weight="bold")
    c.txt(882, 192, "if |g| < 0.12 m and speed < 0.08 m/s:", FS_S, DGRAY, ha="left")
    c.txt(894, 178, "command 0, Stop logit +1", FS_S, DGRAY, ha="left")
    c.txt(882, 164, "else: fly the planned step 1, Stop −1", FS_S, DGRAY, ha="left")
    c.txt(882, 130, "→ (vx, vy, 0, 0, Stop) every 0.2 s", FS_S, DGRAY, ha="left", weight="bold")
    c.arrow((1020, 262), (1020, 230))

    # legend
    c.rect(160, 76, 14, 10, fc=LBLUE, ec=BLUE, lw=0.8, z=5)
    c.txt(180, 81, "learned", FS_S, DGRAY, ha="left")
    c.rect(250, 76, 14, 10, fc="#F2F2F2", ec=GRAY, lw=0.8, z=5)
    c.txt(270, 81, "hand-written rule", FS_S, DGRAY, ha="left")
    c.line([390, 412], [81, 81], color=ORANGE, lw=0.9, ls="--")
    c.txt(418, 81, "used only in variant (B)", FS_S, DGRAY, ha="left")
    c.txt(160, 56, "Results: validation, 20 flights (10 pairs), one training seed. The world model "
          "(574,515 parameters) was trained on demonstrations plus exploration", FS_S, GRAY, ha="left")
    c.txt(160, 42, "flights: 3x the states the BC policies saw, plus dense position labels. "
          "The goal selector is 100% correct on train and val sentences.", FS_S, GRAY, ha="left")
    c.fig.savefig(OUT / "01_system.png", dpi=220)
    c.fig.savefig(OUT / "01_system.svg")
    plt.close(c.fig)
    print("wrote 01")


# ======================================================================= F6-2 perception + dynamics
def fig_wm():
    d1 = json.loads((R / "wm_v1_diagnostics.json").read_text())
    d0 = json.loads((R / "wm_v0_diagnostics.json").read_text())
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    bins = list(d1["val expert demos"]["instructed_target_k0_error_by_true_distance"])
    x = np.arange(len(bins))
    for off, (d, name, col) in zip([-0.2, 0.2], [(d0, "20 training scenes", "#BBBBBB"),
                                                 (d1, "100 training scenes", BLUE)]):
        v = d["val expert demos"]["instructed_target_k0_error_by_true_distance"]
        med = [v[b]["median_error_m"] for b in bins]
        a1.bar(x + off, med, width=0.38, color=col, label=name)
        for xi, b in zip(x, bins):
            if d is d1:
                a1.text(xi + off, v[b]["median_error_m"] + 0.012, f"{100 * v[b]['frac_error_below_0.4m']:.0f}%",
                        ha="center", fontsize=6.8, color=BLUE)
    a1.axhline(0.4, color=VERM, ls="--", lw=1)
    a1.text(len(bins) - 0.5, 0.41, "goal radius 0.4 m", color=VERM, fontsize=FS_S, ha="right", va="bottom")
    a1.set_xticks(x, [b.replace("-inf", "+").replace("0.0-", "0-") for b in bins], fontsize=FS_S)
    a1.set_xlabel("true distance to the named target", fontsize=FS_S)
    a1.set_ylabel("error in the estimated\ngoal position (m, median)", fontsize=FS_S)
    a1.set_ylim(0, 0.6)
    a1.legend(frameon=False, fontsize=FS_S, loc="upper left")
    a1.set_title("A. Perception on unseen layouts\n(% = share of frames under 0.4 m)", fontsize=FS)

    ks = [1, 5, 10]
    for key, ls, lab in [("val expert demos", "-", "expert flights"), ("val random_walk only", "--", "random-walk flights")]:
        e = d1[key]["offset_error_m_from_wm_k0"]
        a2.plot(np.array(ks) * 0.2, [e[str(k)]["wm_rollout"] for k in ks], ls=ls, marker="o", color=ORANGE,
                label=f"learned dynamics, {lab}")
        a2.plot(np.array(ks) * 0.2, [e[str(k)]["cmd_int_from_wm_k0"] for k in ks], ls=ls, marker="s",
                color=GREEN, label=f"integrator, {lab}")
    a2.set_xlabel("seconds ahead", fontsize=FS_S)
    a2.set_ylabel("goal position error (m, median)", fontsize=FS_S)
    a2.set_ylim(0.25, 0.6)
    a2.legend(frameon=False, fontsize=6.5, loc="upper left")
    a2.set_title("B. Predicting ahead: both start\nfrom the same perceived position", fontsize=FS)

    pr = d1["action_probe"]["results"]
    dirs = ["+x", "-x", "+y", "-y"]
    names = ["forward", "backward", "left", "right"]
    sl = [pr[f"k5 {d_}"]["slope"] for d_ in dirs]
    a3.bar(range(4), sl, color=ORANGE, width=0.6)
    for i, s in enumerate(sl):
        a3.text(i, s - 0.08, f"{s:.2f}", ha="center", va="top", fontsize=FS_S, color="white")
    a3.axhline(1, color=DGRAY, ls="--", lw=1)
    a3.text(3.45, 1.04, "exact", fontsize=FS_S, ha="right", color=DGRAY)
    a3.set_xticks(range(4), names, fontsize=FS_S)
    a3.set_ylabel("predicted move / commanded move", fontsize=FS_S)
    a3.set_ylim(0, 2.3)
    a3.set_title("C. Hold one command for 1 s:\ndoes the learned model follow it?", fontsize=FS)
    for a in (a1, a2, a3):
        b5.style_ax(a)
    fig.suptitle("The world model is good at where, biased at what-happens-next", x=0.01, ha="left",
                 fontsize=FS_T, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "02_perception_dynamics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 02")


# ======================================================================= F6-3 scoreboard
def fig_scoreboard():
    keys = ["expert", "oracle", "int", "wmdyn", "base", "aux", "chunk"]
    rows = [b5.SUMMARY[MODELS[k][0]] for k in keys][::-1]
    names = [MODELS[k][1] for k in keys][::-1]
    gcol = [ORANGE if k in ("int", "wmdyn") else DGRAY for k in keys][::-1]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.7), sharey=True,
                                     gridspec_kw={"width_ratios": [2.2, 1.1, 1.6]})
    segs = [("pairs_correct", GREEN, "", "each sentence → its own target"),
            ("pairs_inverted", VERM, "////", "swapped"),
            ("pairs_same_target", "#C9C9C9", "", "both → the same target")]
    for y, r in enumerate(rows):
        x = 0
        for k, col, hatch, _ in segs:
            if r[k]:
                a1.barh(y, r[k], left=x, height=0.6, color=col, hatch=hatch, edgecolor="white", lw=0.5)
                b5.seg_label(a1, x + r[k] / 2, y, r[k], col, hatch)
            x += r[k]
        a2.barh(y, r["success"], height=0.6, color=BLUE)
        a2.text(r["success"] + 0.4, y, f"{r['success']}", va="center", fontsize=FS_S)
        lo, hi = b5.wilson(r["pair_success"], 10)
        a3.plot([10 * lo, 10 * hi], [y, y], color=GRAY, lw=1.4)
        a3.scatter(r["pair_success"], y, color=gcol[y], zorder=3, s=36)
        a3.text(10.9, y, f"{r['pair_success']}/10", va="center", fontsize=FS_S)
    a1.set_yticks(range(len(rows)), names, fontsize=FS_S)
    for a in (a1, a2, a3):
        a.axhline(4.5, color=PANEL_EC, lw=1)
        a.axhline(2.5, color=PANEL_EC, lw=1)
    a1.set_xlim(0, 10)
    a1.set_title("which target each sentence led to\n(10 val pairs, closest approach)", fontsize=FS)
    a2.set_xlim(0, 22); a2.set_title("success /20", fontsize=FS)
    a3.set_xlim(0, 12.5); a3.set_xticks([0, 5, 10])
    a3.set_title("pair success /10\n(both succeed; 95% Wilson)", fontsize=FS)
    for a in (a1, a2, a3):
        b5.style_ax(a); a.tick_params(axis="y", length=0)
    h = [plt.Rectangle((0, 0), 1, 1, color=c_, hatch=hh, ec="white") for _, c_, hh, _ in segs]
    a1.legend(h, [s[3] for s in segs], ncol=3, frameon=False, fontsize=FS_S, loc="upper left",
              bbox_to_anchor=(-0.75, -0.1), handlelength=1.2, columnspacing=1)
    fig.suptitle("Modular vs end-to-end, validation, one training seed (* = privileged reference)",
                 x=0.01, ha="left", fontsize=FS_T, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_scoreboard.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 03")


# ======================================================================= F6-4 paths
def fig_paths(pair_ids=(0, 1, 2)):
    keys = ["int", "wmdyn"]
    tr = {k: b5.trials(MODELS[k][2]) for k in keys}
    pairs = b5.split_pairs("val")
    fig, axes = plt.subplots(len(pair_ids), 2, figsize=(7.6, 2.75 * len(pair_ids) + 0.6))
    for r, pi in enumerate(pair_ids):
        pair = pairs[pi]
        L = b5.layout(pair[0]["layout_id"])
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
                ax.set_title(["(A) integrator", "(B) learned dynamics"][c_], fontsize=FS, weight="bold")
            if c_ == 0:
                ax.set_ylabel(f"val pair {pi}", fontsize=FS)
    fig.suptitle("World model + planner, validation pairs 0-2 (not selected)\n▲ start   ★ success   "
                 "× failure   dotted circle = 0.4 m goal region", x=0.01, ha="left", fontsize=FS)
    fig.tight_layout(h_pad=2.6)
    fig.savefig(OUT / "04_paths.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote 04")


if __name__ == "__main__":
    which = sys.argv[1:] or ["system", "wm", "scoreboard", "paths"]
    for w in which:
        globals()[f"fig_{w}"]()
