"""Figures for blog parts 2-4 (task and evaluation, dataset v0.2, the first policy).

Numbers come from the committed reports, the dataset files and the runs; a few values that
were measured in earlier write-ups are quoted with the doc they come from.

    python scripts/figures/blog02_04.py            # -> reports/figures/blog02, blog03, blog04
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
REPORTS = REPO / "reports"
DATA = REPO / "data" / "v0.2"
OUT = {k: REPORTS / "figures" / k for k in ("blog02", "blog03", "blog04")}
for d in OUT.values():
    d.mkdir(parents=True, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
TARGET = {"red": "#e34948", "blue": "#2a78d6", "green": "#008300", "yellow": "#eda100"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
EXPERT, POLICY = "#52514e", "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "savefig.dpi": 200,
})


def save(fig, part, name):
    p = OUT[part] / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def box(ax, x, y, w, h, text, fc, tc=INK, size=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=tc,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))


EPISODES = pq.read_table(DATA / "episodes.parquet").to_pylist()
EP = {e["episode_id"]: e for e in EPISODES}


def layout(layout_id):
    return json.loads((DATA / "layouts" / f"{layout_id}.json").read_text())


def steps(eid):
    return pq.read_table(DATA / "steps" / f"{eid}.parquet").to_pylist()


# ============================================================== part 2
def fig_interface_card():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    ins = [("camera image", "RGB uint8  (96, 128, 3)\ngimbal, tilt -20°, 47° vertical FOV"),
           ("proprioception", "float32  (11,)\nbody velocity 3 · roll, pitch 2\nsin/cos yaw 2 · body rates 3 · altitude 1"),
           ("instruction", "one sentence per episode\n\"Go to the red box and stop.\"")]
    for i, (t, d) in enumerate(ins):
        y = 3.2 - i * 1.35
        box(ax, 0.1, y, 3.3, 1.05, f"{t}\n{d}", "#cde2fb", size=9)
    box(ax, 4.2, 1.45, 2.2, 1.5, "policy\n5 Hz\n(one decision\nevery 0.2 s)", SERIES[0], "white",
        size=10, bold=True)
    for i in range(3):
        arrow(ax, 3.45, 3.72 - i * 1.35, 4.15, 2.2)
    box(ax, 7.2, 2.25, 3.6, 1.6,
        "action  float32  (5,)\nvx, vy, vz  [m/s]   yaw_rate  [rad/s]\nstop_logit\n"
        "body frame: x forward, y left, z up", "#ffe2b3", size=9)
    box(ax, 7.2, 0.35, 3.6, 1.55,
        "action adapter (the contract)\nreject NaN · |v_xy| <= 0.5 m/s\n"
        "|vz| <= 0.3 m/s · |yaw| <= 0.5 rad/s\nStop = 3 positives in a row", "#e1e0d9", size=9)
    arrow(ax, 6.45, 2.4, 7.15, 3.0)
    arrow(ax, 9.0, 2.2, 9.0, 1.95)
    ax.text(9.0, 0.12, "→ PID at 60 Hz → physics at 240 Hz", ha="center", fontsize=9, color=INK2)
    ax.set_title("The interface every policy in this series must satisfy", loc="left",
                 fontsize=11, color=INK)
    save(fig, "blog02", "01_interface_card")


def fig_camera_mounts():
    from dronevla.camera import FrontCamera
    from dronevla.env import DroneTargetPairsEnv
    from dronevla.task import TaskConfig

    env = DroneTargetPairsEnv(TaskConfig())
    env.reset(seed=2001, options={"goal_index": 0})
    for _ in range(2):                         # accelerate: the airframe pitches forward
        env.step(np.array([0.5, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))
    st = env._state
    pitch = math.degrees(st[8])
    cams = [("body-fixed, level\n(Phase 0 camera, 60° vertical FOV)",
             FrontCamera(mount="legacy", fov_deg=60.0)),
            ("body-fixed, tilted -20°\n(47° vertical FOV)",
             FrontCamera(mount="rigid", tilt_deg=-20.0, fov_deg=47.0, near=0.01)),
            ("gimbal, tilted -20°\n(47° vertical FOV, used from now on)",
             FrontCamera(mount="gimbal", tilt_deg=-20.0, fov_deg=47.0, near=0.01))]
    imgs = [c.render(st[0:3], st[3:7], client=env._client) for _, c in cams]
    env.close()
    # sky-share standard deviation over the same route, from docs/realism_mapping.md
    sky_std = [2.54, 3.06, 0.10]
    fig = plt.figure(figsize=(11, 3.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05], wspace=0.15)
    for i, ((label, _), im) in enumerate(zip(cams, imgs)):
        ax = fig.add_subplot(gs[i])
        ax.imshow(im, interpolation="nearest")
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c3c2b7")
    ax = fig.add_subplot(gs[3])
    ys = [2, 1, 0]
    for y, v, c in zip(ys, sky_std, [MUTED, MUTED, SERIES[0]]):
        ax.barh(y, v, height=0.5, color=c)
        ax.text(v + 0.08, y, f"{v:.2f} pp", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, ["level", "tilted", "gimbal"], fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 3.9)
    ax.set_xlabel("frame shake: std of sky share")
    ax.set_title("same route, lower = steadier", fontsize=9, color=INK, loc="left")
    fig.suptitle(f"Same instant, drone pitched {abs(pitch):.0f}° while accelerating: "
                 "three ways to mount the camera", x=0.08, ha="left", fontsize=11, color=INK)
    save(fig, "blog02", "02_camera_mounts")


def fig_noise_expert():
    # docs/realism_mapping.md, "Noise model and measurement": expert over 20 episodes
    rows = [("1 s, unsmoothed", 12, 0.152), ("1 s", 18, 0.138), ("2 s", 19, 0.102),
            ("3 s  (chosen)", 20, 0.087), ("no noise", 20, 0.059)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.1))
    ys = np.arange(len(rows))[::-1]
    for y, (n, s, v) in zip(ys, rows):
        c = SERIES[0] if "chosen" in n else MUTED
        axes[0].barh(y, s, height=0.5, color=c)
        axes[0].text(s + 0.3, y, f"{s}/20", va="center", fontsize=9, color=INK)
        axes[1].barh(y, v, height=0.5, color=c)
        axes[1].text(v + 0.003, y, f"{v:.3f}", va="center", fontsize=9, color=INK)
    axes[0].set_yticks(ys, [r[0] for r in rows], fontsize=9.5, color=INK)
    axes[1].set_yticks([])
    axes[0].set_xlim(0, 23); axes[0].set_xlabel("expert episodes that succeed")
    axes[1].axvline(0.1, color=SERIES[1], lw=1.2)
    axes[1].text(0.1, len(rows) - 0.45, " settle limit 0.1 m/s", color=INK2, fontsize=9)
    axes[1].set_xlim(0, 0.19); axes[1].set_xlabel("max speed while holding after Stop (m/s)")
    for a in axes:
        a.tick_params(axis="y", length=0)
    axes[0].set_title("Position-noise correlation time vs the scripted expert", loc="left",
                      fontsize=11, color=INK)
    save(fig, "blog02", "03_noise_vs_expert")


def draw_scene(ax, lay, goal=None):
    ax.add_patch(Rectangle((-4, -4), 8, 8, fill=False, ec="#c3c2b7", lw=1))
    for i, (t, hp) in enumerate(zip(lay["targets"], lay["hover_points"])):
        c = TARGET[t["color"]]
        if t["shape"] == "box":
            r = t["radius"]
            ax.add_patch(Rectangle((t["xy"][0] - r, t["xy"][1] - r), 2 * r, 2 * r, color=c))
        else:
            ax.add_patch(Circle(t["xy"], t["radius"], color=c))
        ax.add_patch(Circle(hp[:2], 0.4, fill=False, ec=c, lw=1.2, ls="-"))
    sx, sy, _ = lay["start_xyz"]
    ax.plot(sx, sy, marker=">", ms=10, color=INK)


def fig_task_topdown():
    pair = [e for e in EPISODES if e["split"] == "val"][:2]
    lay = layout(pair[0]["layout_id"])
    ev = json.loads((REPORTS / "eval_v0.1_val.json").read_text())   # val is identical in v0.2
    trials = {t["episode_id"]: t for t in ev["policies"]["expert (oracle)"]["trials"]}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    draw_scene(ax, lay)
    for e in pair:
        p = np.asarray(trials[e["episode_id"]]["path"])
        c = TARGET[e["goal_color"]]
        ax.plot(p[:, 0], p[:, 1], color=c, lw=2)
        ax.text(p[-1, 0] - 0.1, p[-1, 1] + (0.55 if e["goal_index"] == 0 else -0.65),
                f"\"{e['instruction']}\"", fontsize=8.5, color=INK, ha="center")
    ax.text(lay["start_xyz"][0], lay["start_xyz"][1] - 0.45, "start", ha="center",
            fontsize=9, color=INK2)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-2.6, 2.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("One counterfactual pair, top view: same start, two sentences, two flights.\n"
                 "Circles: 0.4 m success region around each hover point.", loc="left",
                 fontsize=10.5, color=INK)
    save(fig, "blog02", "04_task_topdown")


# ============================================================== part 3
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
    steps_ = [("seed", "#e1e0d9"), ("sample layout,\nreject bad ones", "#cde2fb"),
              ("one start\nsnapshot", "#cde2fb"), ("expert flies\nboth targets", "#9ec5f4"),
              ("recorder\nPNG + parquet", "#9ec5f4"), ("validator\n14 checks", SERIES[0]),
              ("manifest\n+ datasheet", SERIES[0])]
    w, gap = 1.42, 0.14
    for i, (t, c) in enumerate(steps_):
        x = 0.05 + i * (w + gap)
        box(ax, x, 0.95, w, 1.0, t, c, "white" if c == SERIES[0] else INK, size=8.5)
        if i:
            arrow(ax, x - gap + 0.02, 1.45, x - 0.02, 1.45)
    ax.text(0.05, 0.45, "Rejection rules: hover points >= 1.5 m apart · success regions clear of "
            "targets · straight path to one goal clears the other by 0.35 m · both targets "
            "visible in the first frame", fontsize=8.8, color=INK2)
    ax.set_title("How one counterfactual pair is made", loc="left", fontsize=11, color=INK)
    save(fig, "blog03", "01_pipeline")


def fig_pair_strip(pair_index=0):
    pair = sorted([e for e in EPISODES if e["split"] == "train"],
                  key=lambda e: e["episode_id"])[2 * pair_index: 2 * pair_index + 2]
    cols = 5
    fig, axes = plt.subplots(2, cols, figsize=(11, 3.7))
    for r, e in enumerate(pair):
        st = [s for s in steps(e["episode_id"])]
        idx = np.linspace(0, len(st) - 1, cols).round().astype(int)
        for c, k in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(Image.open(DATA / st[k]["rgb_path"]), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(TARGET[e["goal_color"]]); sp.set_linewidth(2)
            ax.set_title(f"t = {st[k]['episode_t']:.1f} s", fontsize=8.5, color=INK2)
            if c == 0:
                ax.set_ylabel(f"\"{e['instruction']}\"", fontsize=8.5, color=INK, rotation=0,
                              ha="right", va="center", labelpad=8)
    fig.suptitle("The two episodes of one pair: identical first frame, then the words decide",
                 x=0.12, ha="left", fontsize=11, color=INK)
    save(fig, "blog03", "02_pair_strip")


def fig_dataset_stats():
    n_obs, stop_rows, act_rows, capped = [], 0, 0, 0
    for e in EPISODES:
        st = steps(e["episode_id"])
        n_obs.append(len(st))
        for s in st:
            if s["action_mask"]:
                act_rows += 1
                stop_rows += bool(s["stop_positive"])
                v = s["action_applied"]
                capped += math.hypot(v[0], v[1]) >= 0.499
    side = {}
    for e in EPISODES:
        side[(e["goal_color"], e["goal_side"])] = side.get((e["goal_color"], e["goal_side"]), 0) + 1
    fig = plt.figure(figsize=(11, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    ax.hist(n_obs, bins=range(min(n_obs), max(n_obs) + 2), color=SERIES[0], rwidth=0.85)
    ax.set_xlabel("observations per episode (5 Hz)")
    ax.set_title(f"{len(EPISODES)} episodes, median {int(np.median(n_obs))} frames",
                 loc="left", fontsize=10, color=INK)
    ax = fig.add_subplot(gs[1])
    vals = [100 * stop_rows / act_rows, 100 * capped / act_rows]
    for y, (n, v) in enumerate(zip(["Stop label = 1", "speed at the 0.5 m/s cap"], vals)):
        ax.barh(1 - y, v, height=0.5, color=[SERIES[1], SERIES[2]][y])
        ax.text(v + 1.5, 1 - y, f"{v:.1f}%", va="center", fontsize=9.5, color=INK)
    ax.set_yticks([1, 0], ["Stop = 1", "at speed cap"], fontsize=9.5, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100); ax.set_xlabel(f"share of {act_rows:,} action rows")
    ax.set_title("labels are lopsided", loc="left", fontsize=10, color=INK)
    ax = fig.add_subplot(gs[2])
    colors = ["blue", "green", "red", "yellow"]
    m = np.array([[side.get((c, s), 0) for s in ("left", "right")] for c in colors])
    ax.imshow(m, cmap="Blues", vmin=0)
    for i in range(4):
        for j in range(2):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9.5,
                    color="white" if m[i, j] > m.max() * 0.6 else INK)
    ax.set_xticks([0, 1], ["left", "right"]); ax.set_yticks(range(4), colors)
    ax.set_title("goal colour x side (all splits)", loc="left", fontsize=10, color=INK)
    save(fig, "blog03", "03_dataset_stats")
    return {"episodes": len(EPISODES), "act_rows": act_rows, "stop_pct": vals[0],
            "capped_pct": vals[1], "median_obs": float(np.median(n_obs))}


def fig_splits():
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.6); ax.axis("off")
    for x0, n, lab, c in [(0, 100, "train: 100 pairs\nseeds 1001-1100", SERIES[0]),
                          (100, 10, "val: 10\n2001-10", SERIES[2]),
                          (110, 10, "test: 10\n3001-10", SERIES[1])]:
        ax.add_patch(Rectangle((x0 + 0.3, 0.55), n - 0.6, 0.85, color=c))
        ax.text(x0 + n / 2, 0.97, lab, ha="center", va="center", color="white", fontsize=8.5,
                fontweight="bold")
    ax.text(0, 0.15, "Split by layout: no layout or seed appears in two splits, and both "
            "episodes of a pair stay together. Test is held back: no result in this series "
            "uses it yet.", fontsize=9, color=INK2)
    save(fig, "blog03", "04_splits")


# ============================================================== part 4
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.set_xlim(0, 11); ax.set_ylim(0, 4.0); ax.axis("off")
    img = ["RGB\n96x128x3", "conv 5x5 /2\n48x64x16", "conv 3x3 /2\n24x32x32",
           "conv 3x3 /2\n12x16x64", "conv 3x3 /2\n6x8x64", "FC\n128"]
    for i, t in enumerate(img):
        box(ax, 0.1 + i * 1.12, 2.9, 1.0, 0.85, t, "#cde2fb" if i else "#e1e0d9", size=8.5)
        if i:
            arrow(ax, 0.1 + i * 1.12 - 0.12, 3.32, 0.1 + i * 1.12 - 0.01, 3.32)
    box(ax, 0.1, 1.55, 2.1, 0.85, "\"Go to the red box\nand stop.\"", "#e1e0d9", size=8.5)
    box(ax, 2.45, 1.55, 2.0, 0.85, "17-word vocab\nembed 32 each", "#ffe2b3", size=8.5)
    box(ax, 4.7, 1.55, 1.7, 0.85, "mean over\nwords → 32", "#ffe2b3", size=8.5)
    arrow(ax, 2.22, 1.97, 2.43, 1.97); arrow(ax, 4.47, 1.97, 4.68, 1.97)
    box(ax, 0.1, 0.25, 2.1, 0.85, "proprio\n11 floats", "#e1e0d9", size=8.5)
    box(ax, 2.45, 0.25, 2.0, 0.85, "MLP → 32", "#d6f0e5", size=8.5)
    arrow(ax, 2.22, 0.67, 2.43, 0.67)
    box(ax, 7.1, 1.4, 1.35, 1.15, "concat\n128+32+32\n= 192", "#e1e0d9", size=8.5)
    arrow(ax, 6.8, 3.3, 7.3, 2.58); arrow(ax, 6.42, 1.97, 7.08, 1.97); arrow(ax, 4.47, 0.67, 7.3, 1.38)
    box(ax, 8.75, 1.4, 1.0, 1.15, "MLP\n128 → 5", SERIES[0], "white", size=8.5, bold=True)
    arrow(ax, 8.47, 1.97, 8.73, 1.97)
    box(ax, 9.95, 2.05, 1.0, 0.9, "tanh x caps\nvx vy vz yaw", "#ffe2b3", size=8)
    box(ax, 9.95, 0.95, 1.0, 0.9, "Stop\nlogit", "#ffe2b3", size=8.5)
    arrow(ax, 9.77, 2.1, 9.93, 2.45); arrow(ax, 9.77, 1.8, 9.93, 1.4)
    ax.set_title("TinyBC: 480,901 parameters, no pretrained parts, trained from scratch on CPU",
                 loc="left", fontsize=11, color=INK)
    save(fig, "blog04", "01_architecture")


def fig_closed_loop_topdown(n_pairs=3):
    ev_bc = json.loads((REPORTS / "eval_bc_scale_val.json").read_text())
    bc = {t["episode_id"]: t for t in ev_bc["policies"]["bc_v0.2_s0 (RGB+text BC)"]["trials"]}
    ev_ex = json.loads((REPORTS / "eval_v0.1_val.json").read_text())
    ex = {t["episode_id"]: t for t in ev_ex["policies"]["expert (oracle)"]["trials"]}
    val = sorted([e for e in EPISODES if e["split"] == "val"], key=lambda e: e["episode_id"])
    fig, axes = plt.subplots(2, n_pairs, figsize=(11, 5.2), sharex=True, sharey=True)
    for c in range(n_pairs):
        pair = val[2 * c: 2 * c + 2]
        lay = layout(pair[0]["layout_id"])
        for r, (src, name) in enumerate([(ex, "scripted expert"), (bc, "TinyBC, seed 0")]):
            ax = axes[r, c]
            draw_scene(ax, lay)
            for e in pair:
                t = src[e["episode_id"]]
                p = np.asarray(t["path"])
                ax.plot(p[:, 0], p[:, 1], color=TARGET[e["goal_color"]], lw=2,
                        ls="-" if e["goal_index"] == 0 else (0, (4, 2)))
            outs = " / ".join(src[e["episode_id"]]["outcome"].replace("_", " ") for e in pair)
            ax.set_title(f"{name}\n{outs}", fontsize=8.5, color=INK)
            ax.set_xlim(-3.5, 3.6); ax.set_ylim(-2.4, 2.4); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Val pairs 0-2, top view. Line colour = the target the sentence named. "
                 "The expert's two lines split; the policy's mostly don't.",
                 x=0.06, ha="left", fontsize=10.5, color=INK)
    save(fig, "blog04", "02_closed_loop_topdown")


def fig_language_use():
    """Per seed: how often it flew to the target the sentence named (closest approach),
    and pairs where the two sentences led to different targets. From scripts/language_use.py."""
    import subprocess
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "language_use.py"),
                          str(REPORTS / "eval_bc_scale_val.json")],
                         capture_output=True, text=True, check=True).stdout
    vals = {}
    for line in out.splitlines():
        if line.startswith("bc_v0.2_s"):
            parts = line.split()
            seed = parts[0]
            nums = [p for p in parts if "/" in p]
            vals[seed] = {"success": nums[0], "pairs_ok": nums[1], "diff": nums[2],
                          "instructed": nums[3]}
    seeds = sorted(vals)
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    specs = [("diff", 10, "pairs: two sentences → two targets", 10, "expert 10/10"),
             ("instructed", 20, "flew to the named target", 10, "any word-blind policy: 10/20"),
             ("success", 20, "episodes that succeed", 20, "expert 20/20")]
    for ax, (key, tot, title, ref, reflab) in zip(axes, specs):
        v = [int(vals[s][key].split("/")[0]) for s in seeds]
        ax.bar(range(3), v, width=0.5, color=POLICY)
        ax.axhline(ref, color=EXPERT, lw=1.2, ls="-")
        ax.text(2.35, ref + tot * (0.09 if key == "instructed" else 0.02), reflab,
                ha="right", fontsize=8.5, color=INK2)
        for i, k in enumerate(v):
            ax.text(i, k + tot * 0.02, f"{k}/{tot}", ha="center", fontsize=9.5, color=INK)
        ax.set_ylim(0, tot * 1.12)
        ax.set_xticks(range(3), ["seed 0", "seed 1", "seed 2"])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.suptitle("TinyBC on dataset v0.2, three training seeds, closed loop on the 20 val episodes",
                 x=0.06, ha="left", fontsize=10.5, color=INK, y=1.04)
    save(fig, "blog04", "03_language_use")
    return vals


def fig_swap_gap():
    r = json.loads((REPORTS / "instruction_sensitivity_v0.2_train.json").read_text())
    buckets = list(r["expert"]["swap_gap_mps"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"wspace": 0.3})
    x = np.arange(len(buckets))
    w = 0.2
    axes[0].bar(x - 1.5 * w, [r["expert"]["swap_gap_mps"][b]["median"] for b in buckets], w,
                color=EXPERT, label="scripted expert")
    for i, (name, p) in enumerate(sorted(r["policies"].items())):
        axes[0].bar(x + (i - 0.5) * w, [p["swap_gap_mps"][b]["median"] for b in buckets], w,
                    color=POLICY, alpha=[1.0, 0.7, 0.45][i],
                    label=f"TinyBC seed {name[-1]}")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, buckets)
    axes[0].set_ylabel("m/s (log scale)")
    axes[0].set_title("A. swap only the sentence:\nchange in the lateral command",
                      loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center",
                   bbox_to_anchor=(1.1, -0.15))
    for i, (name, p) in enumerate(sorted(r["policies"].items())):
        axes[1].plot(x, [p["lateral_err_mps"][b]["median"] for b in buckets], marker="o",
                     color=POLICY, alpha=[1.0, 0.7, 0.45][i], lw=2,
                     label=f"seed {name[-1]}")
    share = r["share_of_rows"]
    for i, b in enumerate(buckets):
        axes[1].text(i, -0.012, f"{100 * share[b]:.0f}% of rows", ha="center", fontsize=8.5,
                     color=INK2)
    axes[1].set_xticks(x, buckets)
    axes[1].set_ylim(-0.02, 0.12)
    axes[1].set_ylabel("|policy − expert| lateral, m/s")
    axes[1].set_title("B. training-set error vs the expert,\nby step in the episode",
                      loc="left", fontsize=10, color=INK)
    save(fig, "blog04", "04_swap_gap")


def fig_action_heads():
    fig, ax = plt.subplots(figsize=(11, 3.3))
    ax.set_xlim(0, 11); ax.set_ylim(0, 3.3); ax.axis("off")
    cols = [("Discrete action tokens\n(RT-2, OpenVLA)",
             "each action dimension is cut\ninto 256 bins; the language\nmodel predicts bin ids\nas tokens, one per dimension"),
            ("Continuous action chunk\n(SmolVLA)",
             "an action expert trained with\nflow matching outputs a chunk\nof future continuous actions;\nno action tokens"),
            ("One continuous step\n(this project, so far)",
             "a regression head outputs\n(5,): 4 velocities + Stop logit,\nevery 0.2 s; no tokens,\nno chunk")]
    for i, (t, d) in enumerate(cols):
        x = 0.1 + i * 3.65
        box(ax, x, 2.15, 3.45, 0.95, t, [SERIES[1], SERIES[2], SERIES[0]][i], "white", 9.5, True)
        box(ax, x, 0.25, 3.45, 1.7, d, "#f0efec", INK, 9)
    ax.set_title("Three ways a policy can emit actions", loc="left", fontsize=11, color=INK)
    save(fig, "blog04", "05_action_heads")


if __name__ == "__main__":
    fig_interface_card()
    fig_camera_mounts()
    fig_noise_expert()
    fig_task_topdown()
    fig_pipeline()
    fig_pair_strip()
    print("dataset stats", fig_dataset_stats())
    fig_splits()
    fig_architecture()
    fig_closed_loop_topdown()
    print("language use", fig_language_use())
    fig_swap_gap()
    fig_action_heads()
