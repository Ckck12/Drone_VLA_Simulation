"""Figures for blog post 1 (Phase 0: the environment).

Every number is read from the committed reports, never retyped, so a figure cannot
drift from the measurement it shows.

    python scripts/figures/blog01_environment.py            # -> reports/figures/blog01/
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon  # noqa: E402
from PIL import Image  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
REPORTS = REPO / "reports"
OUT = REPORTS / "figures" / "blog01"
OUT.mkdir(parents=True, exist_ok=True)

# Palette: validated categorical order (blue, orange, aqua, yellow, magenta, green) + ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": 200,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


A = json.loads((REPORTS / "env.json").read_text())
S = json.loads((REPORTS / "env_shadow_seg.json").read_text())


# ---------------------------------------------------------------- 0. the stack
def fig_stack():
    layers = [  # (label, detail, color) bottom -> top
        ("Windows 11 laptop", "i5-1340P, 16 threads · Intel Iris Xe · no NVIDIA GPU", "#e1e0d9"),
        ("WSL2 · Ubuntu 24.04", "code lives in ~/dronevla on Linux ext4, not in OneDrive", "#cde2fb"),
        ("Python 3.12 venv", "torch 2.14.1+cpu · pybullet 3.2.7 compiled from source", "#9ec5f4"),
        ("gym-pybullet-drones", "pinned by commit SHA in one lock file · PID controller", "#6da7ec"),
        ("PyBullet DIRECT\n+ TinyRenderer", "headless, CPU rasteriser → 128 x 96 RGB frames", "#2a78d6"),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for i, (name, detail, c) in enumerate(layers):
        y = i * 0.9
        ax.add_patch(FancyBboxPatch((0, y), 10, 0.72, boxstyle="round,pad=0,rounding_size=0.12",
                                    color=c, lw=0))
        dark = i >= 3
        ax.text(0.3, y + 0.36, name, va="center", fontsize=11, fontweight="bold",
                color="white" if dark else INK)
        ax.text(3.7, y + 0.36, detail, va="center", fontsize=9.5,
                color="white" if dark else INK2)
    ax.set_xlim(-0.1, 10.1)
    ax.set_ylim(-0.1, len(layers) * 0.9)
    ax.axis("off")
    ax.set_title("The whole stack, bottom to top", loc="left", fontsize=11, color=INK)
    save(fig, "00_stack")


# ---------------------------------------------------------------- 1. clock rates
def fig_rates():
    frame = 0.2
    rows = [("physics  240 Hz", 240, SERIES[0], True),
            ("controller  60 Hz", 60, SERIES[2], True),
            ("controller  48 Hz\n(examples' default)", 48, SERIES[1], False),
            ("camera + policy  5 Hz", 5, INK, True)]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, (label, hz, color, ok) in enumerate(rows):
        y = len(rows) - 1 - i
        ticks = np.arange(0, frame + 1e-9, 1 / hz)
        h = 0.32 if hz >= 60 else 0.42
        ax.vlines(ticks, y - h, y + h, color=color, lw=1.2 if hz == 240 else 2.4)
        ax.text(-0.004, y, label, ha="right", va="center", color=INK, fontsize=10.5)
        if hz == 48:
            last = ticks[-1]
            ax.annotate(f"next update at {last + 1 / hz:.4f} s\nmisses the frame edge",
                        xy=(frame, y), xytext=(frame + 0.012, y), color=INK2, fontsize=9.5,
                        va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.axvspan(frame - 0.0006, frame + 0.0006, color=MUTED, alpha=0.25)
    ax.text(frame, len(rows) - 0.35, "next recorded frame", ha="center", color=INK2, fontsize=9.5)
    ax.text(0.1, -1.0, "per recorded frame:  48 physics steps  ·  12 controller updates  ·  1 image",
            ha="center", color=INK, fontsize=10.5)
    ax.set_xlim(-0.002, 0.27)
    ax.set_ylim(-1.3, len(rows) - 0.1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks([0, 0.05, 0.1, 0.15, 0.2])
    ax.set_xlabel("simulated time (s)")
    ax.set_title("A 0.2 s frame holds exactly 12 updates at 60 Hz, but 9.6 at 48 Hz",
                 loc="left", fontsize=11, color=INK)
    save(fig, "01_clock_rates")


# ---------------------------------------------------------------- 2. frame budget
def fig_budget():
    order = ["control", "physics", "render", "encode", "write", "reset"]
    names = {"control": "PID controller (12 calls)", "physics": "physics (48 steps)",
             "render": "camera render", "encode": "PNG encode", "write": "file write",
             "reset": "episode reset (amortised)"}
    frames_per_ep = A["config"].get("frames_per_episode", 100) if "config" in A else 100

    def per_frame(rep):
        b = rep["timing"]["buckets"]
        vals = {k: b[k]["p50_ms"] for k in order if k != "reset"}
        vals["reset"] = b["reset"]["p50_ms"] / frames_per_ep
        return vals

    bars = [("this project's camera", per_frame(A)),
            ("library defaults\n(shadow + segmentation)", per_frame(S))]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    gap = 0.12  # ms of surface between segments
    for yi, (label, vals) in enumerate(bars):
        y = 1 - yi
        x = 0.0
        for k, color in zip(order, SERIES):
            w = vals[k]
            ax.barh(y, max(w - gap, 0.05), left=x, height=0.46, color=color)
            if w > 2.6:
                ax.text(x + w / 2, y, f"{w:.1f}", ha="center", va="center", color="white",
                        fontsize=9.5, fontweight="bold")
            x += w
        ax.text(x + 0.4, y, f"{x:.1f} ms / frame", va="center", color=INK, fontsize=10)
        ax.text(-0.6, y, label, ha="right", va="center", color=INK, fontsize=10)
    ctl, phy = A["timing"]["buckets"]["control"]["p50_ms"], A["timing"]["buckets"]["physics"]["p50_ms"]
    fps = A["throughput"]["recorded_frames_per_wall_second"]
    ax.set_title(f"Where one recorded frame's wall time goes (p50).  Controller = {ctl / phy:.1f}x "
                 f"the physics it drives;  whole pipeline {fps:.1f} frames/s", loc="left",
                 fontsize=11, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SERIES]
    ax.legend(handles, [names[k] for k in order], ncol=3, frameon=False, fontsize=9.5,
              loc="upper center", bbox_to_anchor=(0.45, -0.28))
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("milliseconds per recorded frame")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "02_frame_budget")


# ---------------------------------------------------------------- 3. what is in a frame
def fig_frame_contents():
    import pybullet as p
    from dronevla.camera import FrontCamera
    from dronevla.profile_env import build_targets, episode_plan, waypoint
    from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics

    # identical setup to scripts/inspect_frame_contents.py
    rng = np.random.default_rng([0, 0])
    plan = episode_plan(rng, 20.0)
    start_pos, start_yaw = waypoint(plan, 0.0)
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=start_pos.reshape(1, 3),
                     initial_rpys=np.array([[0.0, 0.0, start_yaw]]),
                     physics=Physics.PYB, pyb_freq=240, ctrl_freq=60,
                     gui=False, obstacles=False, user_debug_gui=False)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset(seed=0)
    scene = build_targets(env.CLIENT, rng)
    action = np.zeros((1, 4))
    for _ in range(60):
        obs, _, _, _, _ = env.step(action)
        action[0], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=obs[0],
            target_pos=start_pos, target_rpy=np.array([0.0, 0.0, start_yaw]))
    state = env._getDroneStateVector(0)
    cam = FrontCamera(segmentation=True)
    w, h = cam.width, cam.height
    _, _, rgba, _, seg = p.getCameraImage(
        width=w, height=h, viewMatrix=cam.view_matrix(state[0:3], state[3:7]),
        projectionMatrix=cam._projection, shadow=0,
        flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
    rgb = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    seg = np.asarray(seg).reshape(h, w)
    body = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
    classes = [(-1, "sky (nothing hit)", "#cfd8e3"),
               (int(env.PLANE_ID), "ground plane", "#9aa7b5"),
               (scene["red_cube_id"], "red cube (target)", "#e34948"),
               (scene["blue_sphere_id"], "blue sphere (target)", "#2a78d6"),
               (int(env.DRONE_IDS[0]), "drone's own airframe", "#3b3a37")]
    env.close()

    shares = [(name, 100 * (body == bid).sum() / body.size, c) for bid, name, c in classes]
    seg_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for bid, _, c in classes:
        seg_rgb[body == bid] = [int(c[i:i + 2], 16) for i in (1, 3, 5)]

    fig = plt.figure(figsize=(11, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.12)
    for i, (img, title) in enumerate([(rgb, "what the policy sees (128 x 96)"),
                                      (seg_rgb, "same frame, labelled per pixel")]):
        ax = fig.add_subplot(gs[i])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color("#c3c2b7")
    ax = fig.add_subplot(gs[2])
    ys = np.arange(len(shares))[::-1]
    for y, (name, s, c) in zip(ys, shares):
        ax.barh(y, s, height=0.55, color=c)
        ax.text(s + 1, y, f"{s:.1f}%  {name}", va="center", color=INK, fontsize=9.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50])
    ax.set_xlabel("share of pixels")
    ax.set_title("the two targets are under 8% of the image", fontsize=10, color=INK, loc="left")
    ax.tick_params(axis="y", length=0)
    save(fig, "03_frame_contents")


# ---------------------------------------------------------------- 4. clipped targets
def fig_clipping():
    half_fov = 30.0
    dist, z_target = 1.2, 0.15
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
    for ax, z_drone, ok in [(axes[0], 1.0, False), (axes[1], 0.5, True)]:
        dep = math.degrees(math.atan((z_drone - z_target) / dist))
        L = 1.6
        lo = math.radians(-half_fov)
        cone = Polygon([[0, z_drone], [L, z_drone + L * math.tan(math.radians(half_fov))],
                        [L, z_drone + L * math.tan(lo)]], closed=True,
                       color=SERIES[0], alpha=0.10, lw=0)
        ax.add_patch(cone)
        ax.plot([0, L], [z_drone, z_drone + L * math.tan(lo)], color=SERIES[0], lw=1.2)
        ax.plot([0, dist], [z_drone, z_target], color=SERIES[1] if not ok else SERIES[2],
                lw=2, ls="-")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.add_patch(FancyBboxPatch((dist - 0.08, 0), 0.16, 0.3, boxstyle="round,pad=0,rounding_size=0.02",
                                    color="#e34948"))
        ax.plot(0, z_drone, "o", ms=9, color=INK, mec="white", mew=2)
        verdict = "target below the image edge" if not ok else "target inside the view"
        ax.set_title(f"flying at {z_drone:.1f} m: look-down {dep:.0f}°  vs  half-FOV {half_fov:.0f}°\n"
                     f"→ {verdict}", fontsize=10, color=INK, loc="left")
        ax.set_xlim(-0.1, 1.7)
        ax.set_ylim(-0.05, 1.45)
        ax.set_aspect("equal")
        ax.set_xlabel("distance ahead (m)")
    axes[0].set_ylabel("height (m)")
    save(fig, "04_clipping_geometry")


# ---------------------------------------------------------------- 5. bytes per frame
def fig_bytes():
    bp = A["bytes_per_frame"]
    rows = [("PNG (lossless, written)", bp["png_on_disk"]["mean"]),
            ("JPEG quality 75", bp["jpeg_in_memory_subsample"]["75"]["mean"]),
            ("JPEG quality 90", bp["jpeg_in_memory_subsample"]["90"]["mean"])]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axvspan(6000, 15000, color=SERIES[3], alpha=0.18, lw=0)
    ax.text(10500, 2.62, "roadmap's planning assumption\n(JPEG, 6–15 KB)", ha="center",
            va="center", color=INK2, fontsize=9.5)
    for y, (name, v) in zip([2, 1, 0], rows):
        ax.barh(y, v, height=0.46, color=SERIES[0])
        ax.text(v + 200, y, f"{v / 1000:.1f} KB", va="center", color=INK, fontsize=10)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows], color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 16000)
    ax.set_ylim(-0.5, 3.1)
    ax.set_xticks([0, 5000, 10000, 15000], ["0", "5 KB", "10 KB", "15 KB"])
    ax.set_title(f"Bytes per 128 x 96 frame (raw RGB = {bp['raw_rgb_uint8'] / 1000:.1f} KB): "
                 "lossless PNG is the smallest", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    save(fig, "05_bytes_per_frame")


# ---------------------------------------------------------------- 6. onboard GIF
def gif_onboard(episode=0, n=100, scale=3):
    frames_dir = REPO / "results" / "env_profile" / "frames"
    first = episode * n
    imgs = []
    for i in range(first, first + n):
        im = Image.open(frames_dir / f"frame_{i:06d}.png").convert("RGB")
        imgs.append(im.resize((im.width * scale, im.height * scale), Image.NEAREST))
    out = OUT / "06_onboard_camera.gif"
    # 5 Hz recording shown at 2x speed (100 ms per frame)
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=100, loop=0,
                 optimize=True)
    print("wrote", out, f"{out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    fig_stack()
    fig_rates()
    fig_budget()
    fig_frame_contents()
    fig_clipping()
    fig_bytes()
    gif_onboard()
