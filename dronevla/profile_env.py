"""Profile the CPU execution path that produces drone camera frames.

    python -m dronevla.profile_env --frames 1000 --renderer tiny --out reports/env.json

Roadmap v3 Phase 0 item 6 and its exit gate. What this does and why, in one place:

* Records `--frames` real RGB frames to disk at the contract's resolution and rate
  (§4.3: 128x96x3 uint8, no alpha, physics 240 Hz, control 60 Hz, record 5 Hz, so exactly
  48 physics steps per recorded frame).
* Splits wall time into reset / control / physics / render / encode / write, and follows
  the §7.4 protocol: warm-up excluded, percentiles rather than means only, per-episode
  spread, timer overhead measured, host conditions recorded.
* Measures bytes/frame in each candidate storage format, because §4.4's GB estimate rests
  on an assumed 6-15 KB/frame that this run is meant to replace.
* Verifies the output by reading it back from disk -- frame count, decoded shape/dtype,
  NaN, sim-time monotonicity and spacing, and that the frames are not all identical.

Episodes are 100 frames = 20 sim-seconds by default, matching §4.4's episode assumption,
so `reset` is sampled once per episode instead of once per run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pybullet as p
from PIL import Image

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from dronevla import __version__
from dronevla.camera import FrontCamera

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- measurement
def stats(samples) -> dict:
    """Summary used for every timing bucket. Seconds in, milliseconds out."""
    a = np.asarray(samples, dtype=float)
    if a.size == 0:
        return {"n": 0, "total_s": 0.0}
    return {
        "n": int(a.size),
        "total_s": float(a.sum()),
        "mean_ms": float(a.mean() * 1e3),
        "p50_ms": float(np.percentile(a, 50) * 1e3),
        "p95_ms": float(np.percentile(a, 95) * 1e3),
        "p99_ms": float(np.percentile(a, 99) * 1e3),
        "min_ms": float(a.min() * 1e3),
        "max_ms": float(a.max() * 1e3),
    }


def proc_status_kb(field: str):
    """One `/proc/self/status` field in kB, or None if the kernel does not expose it."""
    try:
        for line in pathlib.Path("/proc/self/status").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def timer_overhead_ns(n: int = 200000) -> float:
    """Cost of one `perf_counter()` call, so sub-millisecond buckets can be judged."""
    t0 = time.perf_counter()
    for _ in range(n):
        time.perf_counter()
    return (time.perf_counter() - t0) / n * 1e9


def git_info(path: pathlib.Path) -> dict:
    def run(*args):
        try:
            out = subprocess.run(["git", "-C", str(path), *args],
                                 capture_output=True, text=True, timeout=15)
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    dirty = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD"),
        "dirty": None if dirty is None else bool(dirty),
        "remote": run("config", "--get", "remote.origin.url"),
    }


# ------------------------------------------------------------------------------- scene
def build_targets(client: int, rng: np.random.Generator) -> dict:
    """The two coloured targets from scripts/render_two_objects.py, with placement jitter.

    `reset()` calls `p.resetSimulation()`, which destroys these, so they are rebuilt every
    episode and that cost sits inside the `reset` bucket.
    """
    j = rng.uniform(-0.25, 0.25, size=4)
    red_xyz = [float(j[0]), float(0.5 + j[1]), 0.15]
    blue_xyz = [float(j[2]), float(-0.5 + j[3]), 0.15]
    red = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15] * 3, physicsClientId=client),
        p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15] * 3, rgbaColor=[1, 0, 0, 1],
                            physicsClientId=client),
        red_xyz, physicsClientId=client)
    blue = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_SPHERE, radius=0.15, physicsClientId=client),
        p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[0, 0, 1, 1],
                            physicsClientId=client),
        blue_xyz, physicsClientId=client)
    return {"red_cube_id": int(red), "red_xyz": red_xyz,
            "blue_sphere_id": int(blue), "blue_xyz": blue_xyz}


def episode_plan(rng: np.random.Generator, period: float) -> dict:
    """A slow circuit around the targets, facing them, so consecutive frames differ.

    `period` is the lap time, independent of the episode length: a 100-frame episode at
    5 Hz with a 20 s lap flies exactly one lap, and a short episode flies part of one.
    Tangential speed is kept under the §4.3 horizontal cap of 0.5 m/s.

    Altitude is chosen against the camera, not picked for looks. The camera looks along
    body +x with a 60 deg vertical FOV, so the bottom of the frame is 30 deg below the
    horizon. Targets sit at z = 0.15 m about `radius` away, so flying at 1.0 m would put
    them atan(0.85 / 1.2) = 35 deg down -- below the frame. A first recording at 0.9-1.1 m
    did exactly that: the top 40% of every frame was empty sky and the targets were
    clipped by the bottom edge. 0.40-0.60 m keeps them inside the lower half.
    """
    radius = float(rng.uniform(1.0, 1.4))
    altitude = float(rng.uniform(0.40, 0.60))
    phase0 = float(rng.uniform(0.0, 2 * np.pi))
    speed = 2 * np.pi * radius / period
    if speed > 0.5:
        raise SystemExit(
            f"planned tangential speed {speed:.3f} m/s exceeds the 0.5 m/s contract cap; "
            "lengthen --lap-period-s or shrink the radius")
    return {"radius_m": radius, "altitude_m": altitude, "phase0_rad": phase0,
            "lap_period_s": period, "tangential_speed_ms": speed}


def waypoint(plan: dict, t: float):
    """Position on the circuit at sim time `t`, and the yaw that points at the origin."""
    ang = plan["phase0_rad"] + 2 * np.pi * t / plan["lap_period_s"]
    pos = np.array([plan["radius_m"] * np.cos(ang),
                    plan["radius_m"] * np.sin(ang),
                    plan["altitude_m"]])
    yaw = float(np.arctan2(-pos[1], -pos[0]))   # face the origin, where the targets are
    return pos, yaw


# -------------------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m dronevla.profile_env",
        description="Profile the CPU physics+render+encode path (roadmap v3 Phase 0 item 6).")
    ap.add_argument("--frames", type=int, default=1000,
                    help="recorded frames in total (default: 1000, the Phase 0 exit gate)")
    ap.add_argument("--frames-per-episode", type=int, default=100,
                    help="frames before a reset (default: 100 = 20 sim-seconds at 5 Hz)")
    ap.add_argument("--warmup-frames", type=int, default=50,
                    help="frames run and discarded before timing starts (default: 50)")
    ap.add_argument("--renderer", choices=["tiny"], default="tiny",
                    help="only 'tiny' is offered; dronevla/camera.py says why")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--pyb-hz", type=int, default=240)
    ap.add_argument("--ctrl-hz", type=int, default=60)
    ap.add_argument("--record-hz", type=float, default=5.0)
    ap.add_argument("--lap-period-s", type=float, default=20.0,
                    help="lap time of the circuit the drone flies (default: 20 s, which is "
                         "one lap per 100-frame episode and §4.4's episode length)")
    ap.add_argument("--png-compress-level", type=int, default=6)
    ap.add_argument("--jpeg-qualities", type=int, nargs="*", default=[90, 75],
                    help="measured in memory on a subsample, never written to disk")
    ap.add_argument("--jpeg-every", type=int, default=10,
                    help="measure JPEG sizes on every Nth frame (default: 10)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shadow", action="store_true", help="enable shadows (off by default)")
    ap.add_argument("--segmentation", action="store_true",
                    help="also compute the segmentation mask (off by default)")
    ap.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "reports/env.json")
    ap.add_argument("--frames-dir", type=pathlib.Path,
                    default=REPO_ROOT / "results/env_profile/frames")
    ap.add_argument("--meta-out", type=pathlib.Path,
                    default=REPO_ROOT / "results/env_profile/frames.jsonl")
    ap.add_argument("--host-state", type=pathlib.Path,
                    default=REPO_ROOT / "reports/host_state.json",
                    help="JSON from scripts/host_state.ps1; skipped if absent")
    ap.add_argument("--host-state-max-age-s", type=float, default=900.0,
                    help="ignore the host_state file if it is older than this, so a "
                         "committed snapshot is never reported as this run's conditions")
    ap.add_argument("--label", default="", help="free-text tag stored in the report")
    return ap.parse_args(argv)


def resolve_rates(args):
    """Fail loudly on rate combinations that cannot hold the contract exactly."""
    steps_per_frame = args.pyb_hz / args.record_hz
    if steps_per_frame != int(steps_per_frame):
        raise SystemExit(
            f"--pyb-hz {args.pyb_hz} / --record-hz {args.record_hz} = {steps_per_frame} "
            "physics steps per recorded frame, which is not a whole number")
    steps_per_ctrl = args.pyb_hz / args.ctrl_hz
    if steps_per_ctrl != int(steps_per_ctrl):
        raise SystemExit("--pyb-hz must be a whole multiple of --ctrl-hz")
    ctrl_per_frame = steps_per_frame / steps_per_ctrl
    if ctrl_per_frame != int(ctrl_per_frame):
        raise SystemExit(
            f"{steps_per_frame:.0f} physics steps per frame is not a whole number of "
            f"{steps_per_ctrl:.0f}-step control periods; pick rates that divide")
    return int(steps_per_frame), int(ctrl_per_frame)


# -------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    t_process_start = time.perf_counter()
    steps_per_frame, ctrl_per_frame = resolve_rates(args)
    record_dt = 1.0 / args.record_hz

    rss_after_imports_kb = proc_status_kb("VmRSS")
    overhead_ns = timer_overhead_ns()

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.frames_dir.glob("frame_*.png"):
        stale.unlink()
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    warmup_dir = args.frames_dir.parent / "warmup"

    cam = FrontCamera(width=args.width, height=args.height, renderer=args.renderer,
                      shadow=args.shadow, segmentation=args.segmentation)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 1.0]]), initial_rpys=np.zeros((1, 3)),
        physics=Physics.PYB, pyb_freq=args.pyb_hz, ctrl_freq=args.ctrl_hz,
        gui=False, record=False, obstacles=False, user_debug_gui=False)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

    buckets = {k: [] for k in ("reset", "control", "physics", "render", "encode", "write")}
    jpeg_bytes = {q: [] for q in args.jpeg_qualities}
    jpeg_times = {q: [] for q in args.jpeg_qualities}
    episodes_meta = []
    state_box = {"rss_after_first_reset_kb": None}

    def run_episode(index, n_frames, out_dir, record, first_global):
        """One reset plus `n_frames` frames. Returns the metadata rows."""
        rng = np.random.default_rng([args.seed, index & 0xFFFFFFFF])
        plan = episode_plan(rng, args.lap_period_s)
        start_pos, start_yaw = waypoint(plan, 0.0)

        t0 = time.perf_counter()
        env.INIT_XYZS = start_pos.reshape(1, 3)
        env.INIT_RPYS = np.array([[0.0, 0.0, start_yaw]])
        obs, _ = env.reset(seed=args.seed + abs(index))
        scene = build_targets(env.CLIENT, rng)
        ctrl.reset()
        t_reset = time.perf_counter() - t0
        if record:
            buckets["reset"].append(t_reset)
        if state_box["rss_after_first_reset_kb"] is None:
            state_box["rss_after_first_reset_kb"] = proc_status_kb("VmRSS")

        action = np.zeros((1, 4))
        rows = []
        for i in range(n_frames):
            sim_t = env.step_counter / env.PYB_FREQ

            t0 = time.perf_counter()
            target_pos, target_yaw = waypoint(plan, sim_t)
            target_rpy = np.array([0.0, 0.0, target_yaw])
            t_control = time.perf_counter() - t0

            t_physics = 0.0
            for _ in range(ctrl_per_frame):
                t0 = time.perf_counter()
                obs, _, _, _, _ = env.step(action)
                t_physics += time.perf_counter() - t0
                t0 = time.perf_counter()
                action[0], _, _ = ctrl.computeControlFromState(
                    control_timestep=env.CTRL_TIMESTEP, state=obs[0],
                    target_pos=target_pos, target_rpy=target_rpy)
                t_control += time.perf_counter() - t0

            state = env._getDroneStateVector(0)
            t0 = time.perf_counter()
            rgb = cam.render(state[0:3], state[3:7], client=env.CLIENT)
            t_render = time.perf_counter() - t0

            t0 = time.perf_counter()
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="PNG",
                                      compress_level=args.png_compress_level)
            png = buf.getvalue()
            t_encode = time.perf_counter() - t0

            gi = first_global + i
            path = out_dir / f"frame_{gi:06d}.png"
            t0 = time.perf_counter()
            path.write_bytes(png)
            t_write = time.perf_counter() - t0

            if not record:
                continue

            buckets["control"].append(t_control)
            buckets["physics"].append(t_physics)
            buckets["render"].append(t_render)
            buckets["encode"].append(t_encode)
            buckets["write"].append(t_write)
            if args.jpeg_every and gi % args.jpeg_every == 0:
                for q in args.jpeg_qualities:
                    jb = io.BytesIO()
                    tq = time.perf_counter()
                    Image.fromarray(rgb).save(jb, format="JPEG", quality=q)
                    jpeg_times[q].append(time.perf_counter() - tq)
                    jpeg_bytes[q].append(jb.getbuffer().nbytes)
            rows.append({
                "episode": index, "frame_in_episode": i, "global_frame": gi,
                "sim_t": float(sim_t), "wall_t_monotonic": float(time.monotonic()),
                "pos": [float(v) for v in state[0:3]],
                "quat": [float(v) for v in state[3:7]],
                "rpy": [float(v) for v in state[7:10]],
                "vel": [float(v) for v in state[10:13]],
                "ang_v": [float(v) for v in state[13:16]],
                "png_bytes": len(png),
                "png_sha256_12": hashlib.sha256(png).hexdigest()[:12],
                "control_s": t_control, "physics_s": t_physics, "render_s": t_render,
                "encode_s": t_encode, "write_s": t_write, "file": path.name,
            })

        if record:
            per_frame = [r["control_s"] + r["physics_s"] + r["render_s"]
                         + r["encode_s"] + r["write_s"] for r in rows]
            episodes_meta.append({
                "episode": index, "frames": n_frames, "reset_s": t_reset,
                "plan": plan, "scene": scene,
                "render_p50_ms": float(np.percentile([r["render_s"] for r in rows], 50) * 1e3),
                "frame_total_p50_ms": float(np.percentile(per_frame, 50) * 1e3),
            })
        return rows

    # --- warm-up: run and discard, so imports, caches and first-touch are not timed ----
    if args.warmup_frames > 0:
        warmup_dir.mkdir(parents=True, exist_ok=True)
        run_episode(-1, args.warmup_frames, warmup_dir, False, 0)
        shutil.rmtree(warmup_dir, ignore_errors=True)

    # --- timed run --------------------------------------------------------------------
    pipeline_t0 = time.perf_counter()
    all_rows, remaining, ep = [], args.frames, 0
    while remaining > 0:
        n = min(args.frames_per_episode, remaining)
        all_rows += run_episode(ep, n, args.frames_dir, True, args.frames - remaining)
        remaining -= n
        ep += 1
    pipeline_wall_s = time.perf_counter() - pipeline_t0

    with args.meta_out.open("w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row) + "\n")

    rss_end_kb = proc_status_kb("VmRSS")
    hwm_kb = proc_status_kb("VmHWM")
    ru_maxrss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    env.close()

    # --- verification: read the artefacts back; nothing here trusts the loop -----------
    checks = verify(args, record_dt, steps_per_frame)

    bucket_stats = {k: stats(v) for k, v in buckets.items()}
    recorded_bucket_s = sum(s["total_s"] for s in bucket_stats.values())
    png_sizes = np.array([r["png_bytes"] for r in all_rows], dtype=float)
    sim_seconds = args.frames * record_dt

    report = {
        "schema": "dronevla.profile_env/1",
        "label": args.label,
        "captured_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roadmap": "v3 Phase 0 item 6; exit gate '1000 RGB frames, no NaN, no time reversal'",
        "config": {
            "frames": args.frames,
            "frames_per_episode": args.frames_per_episode,
            "episodes": ep,
            "warmup_frames": args.warmup_frames,
            "pyb_hz": args.pyb_hz,
            "ctrl_hz": args.ctrl_hz,
            "record_hz": args.record_hz,
            "lap_period_s": args.lap_period_s,
            "physics_steps_per_recorded_frame": steps_per_frame,
            "control_updates_per_recorded_frame": ctrl_per_frame,
            "physics_model": "Physics.PYB",
            "drone_model": "CF2X",
            "controller": "DSLPIDControl",
            "connection_mode": "DIRECT",
            "png_compress_level": args.png_compress_level,
            "seed": args.seed,
        },
        "camera": cam.describe(),
        "timing": {
            "buckets": bucket_stats,
            "per_episode": episodes_meta,
            "pipeline_wall_s": pipeline_wall_s,
            "recorded_buckets_total_s": recorded_bucket_s,
            "unaccounted_s": pipeline_wall_s - recorded_bucket_s,
            "process_wall_s": None,
            "timer_overhead_ns": overhead_ns,
            "note": ("unaccounted_s is pipeline time outside the six buckets: the JPEG "
                     "size subsample, metadata construction and loop overhead. Throughput "
                     "is computed from pipeline_wall_s, so it includes that overhead."),
        },
        "throughput": {
            "recorded_frames_per_wall_second": args.frames / pipeline_wall_s,
            "sim_seconds_recorded": sim_seconds,
            "real_time_factor": sim_seconds / pipeline_wall_s,
            "roadmap_4_4_planning_assumption": "5-20 recorded frames/wall-second",
        },
        "bytes_per_frame": {
            "raw_rgb_uint8": args.width * args.height * 3,
            "png_on_disk": {
                "mean": float(png_sizes.mean()),
                "p50": float(np.percentile(png_sizes, 50)),
                "min": float(png_sizes.min()), "max": float(png_sizes.max()),
                "compress_level": args.png_compress_level,
            },
            "jpeg_in_memory_subsample": {
                str(q): {
                    "n": len(jpeg_bytes[q]),
                    "mean": float(np.mean(jpeg_bytes[q])) if jpeg_bytes[q] else None,
                    "p50": float(np.percentile(jpeg_bytes[q], 50)) if jpeg_bytes[q] else None,
                    "encode_p50_ms": (float(np.percentile(jpeg_times[q], 50) * 1e3)
                                      if jpeg_times[q] else None),
                    "subsampling": "PIL default for this quality",
                } for q in args.jpeg_qualities
            },
            "note": ("PNG is what was written to disk, so the recorded frames stay "
                     "losslessly re-encodable. JPEG is the format assumed by roadmap v3 "
                     "§4.4 and is measured in memory on a subsample; its encode time is "
                     "excluded from the buckets."),
        },
        "memory_kb": {
            "vmrss_after_imports": rss_after_imports_kb,
            "vmrss_after_first_reset": state_box["rss_after_first_reset_kb"],
            "vmrss_at_end": rss_end_kb,
            "vmhwm_peak": hwm_kb,
            "ru_maxrss": ru_maxrss_kb,
            "note": ("ru_maxrss is a lifetime peak that includes interpreter and import "
                     "cost. Frames are streamed to disk, never accumulated in memory."),
        },
        "provenance": {
            "dronevla_version": __version__,
            "repo": git_info(REPO_ROOT),
            "gym_pybullet_drones": git_info(REPO_ROOT / "third_party/gym-pybullet-drones"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "pybullet_api_version": p.getAPIVersion(),
            "pybullet_numpy_enabled": int(p.isNumpyEnabled()),
            "torch_imported": "torch" in sys.modules,
        },
        "host": host_block(args),
        "checks": checks,
        "artifacts": {"frames_dir": str(args.frames_dir),
                      "frames_metadata": str(args.meta_out)},
    }
    report["timing"]["process_wall_s"] = time.perf_counter() - t_process_start

    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    print_summary(report)
    print(f"\nwrote {args.out}")
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("pass") is False]
    if failed:
        print(f"FAILED CHECKS: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def host_block(args) -> dict:
    block = {
        "uname": " ".join(platform.uname()),
        "cpu_count_affinity": len(os.sched_getaffinity(0)),
        "cpu_count_total": os.cpu_count(),
        "loadavg_1_5_15": list(os.getloadavg()),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS")},
        "renderer_env": {k: os.environ.get(k) for k in
                         ("GALLIUM_DRIVER", "LIBGL_ALWAYS_SOFTWARE", "DISPLAY",
                          "WAYLAND_DISPLAY")},
    }
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:")):
                key, val = line.split(":")
                block[f"meminfo_{key.lower()}_kb"] = int(val.split()[0])
    except OSError:
        pass
    try:
        usage = shutil.disk_usage(REPO_ROOT)
        block["disk_free_gb"] = round(usage.free / 1e9, 2)
        block["disk_total_gb"] = round(usage.total / 1e9, 2)
    except OSError:
        pass
    block["windows"] = windows_block(args.host_state, args.host_state_max_age_s)
    return block


def windows_block(path, max_age_s: float) -> dict:
    """Merge the Windows-side snapshot only if it actually describes *this* run.

    The file is committed, so a stale one would otherwise be reported as the conditions of
    every later run -- including a CI run on a GitHub Ubuntu runner, which has no Windows
    host at all.
    """
    if not path or not path.exists():
        return {"collected": False,
                "reason": "no host_state file; run scripts/host_state.ps1 on Windows"}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {"collected": False, "reason": f"could not read {path}: {exc}"}

    age = host_state_age_s(data.get("captured_local"))
    if age is None:
        return {"collected": False, "reason": "captured_local missing or unparseable",
                "captured_local": data.get("captured_local")}
    if age > max_age_s:
        return {"collected": False, "reason": "stale", "stale": True,
                "age_s": round(age, 1), "max_age_s": max_age_s,
                "captured_local": data.get("captured_local"),
                "note": "re-run scripts/host_state.ps1 just before the profile to record "
                        "the conditions of that run"}
    data["collected"] = True
    data["age_s_at_run"] = round(age, 1)
    return data


def host_state_age_s(stamp):
    """Seconds between `stamp` and now. PowerShell writes 7 fractional digits, which
    `datetime.fromisoformat` rejects, so the fraction is trimmed to microseconds."""
    if not isinstance(stamp, str):
        return None
    trimmed = re.sub(r"\.(\d{6})\d+", r".\1", stamp)
    try:
        then = datetime.fromisoformat(trimmed)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.astimezone()
    return (datetime.now(timezone.utc) - then).total_seconds()


def verify(args, record_dt: float, steps_per_frame: int) -> dict:
    """Read the written artefacts back and check them."""
    out = {}

    files = sorted(args.frames_dir.glob("frame_*.png"))
    out["frame_files_on_disk"] = {"pass": len(files) == args.frames,
                                  "found": len(files), "expected": args.frames}

    rows = [json.loads(ln) for ln in args.meta_out.read_text().splitlines() if ln]
    out["metadata_rows"] = {"pass": len(rows) == args.frames,
                            "found": len(rows), "expected": args.frames}
    if not rows:
        return out

    numeric = []
    for r in rows:
        numeric += [r[k] for k in ("sim_t", "wall_t_monotonic", "render_s", "encode_s",
                                   "physics_s", "control_s", "write_s")]
        for k in ("pos", "quat", "rpy", "vel", "ang_v"):
            numeric += r[k]
    arr = np.asarray(numeric, dtype=float)
    n_bad = int((~np.isfinite(arr)).sum())
    out["no_nan_or_inf"] = {"pass": n_bad == 0, "non_finite_values": n_bad,
                            "values_checked": int(arr.size)}

    wall = np.array([r["wall_t_monotonic"] for r in rows])
    out["wall_clock_monotonic"] = {"pass": bool(np.all(np.diff(wall) >= 0)),
                                   "backward_steps": int((np.diff(wall) < 0).sum())}

    sim_ok, interval_ok, worst = True, True, 0.0
    for epi in sorted({r["episode"] for r in rows}):
        st = np.array([r["sim_t"] for r in rows if r["episode"] == epi])
        if st.size < 2:
            continue
        d = np.diff(st)
        sim_ok = sim_ok and bool(np.all(d > 0))
        worst = max(worst, float(np.abs(d - record_dt).max()))
        interval_ok = interval_ok and bool(np.allclose(d, record_dt, atol=1e-9))
    out["sim_time_strictly_increasing_per_episode"] = {"pass": sim_ok}
    out["sim_time_interval_equals_record_period"] = {
        "pass": interval_ok, "expected_s": record_dt,
        "worst_abs_deviation_s": worst, "physics_steps_per_frame": steps_per_frame}

    # decode a subsample off disk: shape, dtype, and that the scene actually moves
    idx = sorted(set(np.linspace(0, len(files) - 1, min(40, len(files))).astype(int)))
    shapes, digests, stds = set(), set(), []
    for i in idx:
        a = np.asarray(Image.open(files[i]))
        shapes.add((a.shape, str(a.dtype)))
        digests.add(hashlib.sha256(a.tobytes()).hexdigest())
        stds.append(float(a.std()))
    want = ((args.height, args.width, 3), "uint8")
    out["decoded_shape_and_dtype"] = {
        "pass": shapes == {want}, "expected": [list(want[0]), want[1]],
        "found": [[list(s[0]), s[1]] for s in sorted(shapes, key=str)],
        "frames_decoded": len(idx)}
    out["frames_are_not_identical"] = {"pass": len(digests) == len(idx),
                                       "unique_images": len(digests), "sampled": len(idx)}
    out["frames_are_not_blank"] = {"pass": bool(min(stds) > 1.0),
                                   "min_pixel_std": min(stds), "max_pixel_std": max(stds)}
    return out


def print_summary(rep: dict) -> None:
    c, t = rep["config"], rep["timing"]
    print(f"\n--- dronevla.profile_env --- {c['frames']} frames in {c['episodes']} episodes, "
          f"{rep['camera']['width']}x{rep['camera']['height']} "
          f"{rep['camera']['renderer']} renderer")
    print(f"rates: physics {c['pyb_hz']} Hz / control {c['ctrl_hz']} Hz / record "
          f"{c['record_hz']} Hz -> {c['physics_steps_per_recorded_frame']} physics steps "
          f"and {c['control_updates_per_recorded_frame']} control updates per frame")

    print(f"\n{'bucket':<10}{'n':>6}{'total s':>10}{'p50 ms':>9}{'p95 ms':>9}"
          f"{'p99 ms':>9}{'% wall':>9}")
    for name, s in t["buckets"].items():
        if not s.get("n"):
            continue
        print(f"{name:<10}{s['n']:>6}{s['total_s']:>10.3f}{s['p50_ms']:>9.3f}"
              f"{s['p95_ms']:>9.3f}{s['p99_ms']:>9.3f}"
              f"{100 * s['total_s'] / t['pipeline_wall_s']:>8.1f}%")
    print(f"{'(other)':<10}{'':>6}{t['unaccounted_s']:>10.3f}{'':>27}"
          f"{100 * t['unaccounted_s'] / t['pipeline_wall_s']:>8.1f}%")

    th = rep["throughput"]
    print(f"\npipeline wall    : {t['pipeline_wall_s']:.2f} s   "
          f"(whole process {t['process_wall_s']:.2f} s, "
          f"perf_counter overhead {t['timer_overhead_ns']:.0f} ns/call)")
    print(f"throughput       : {th['recorded_frames_per_wall_second']:.2f} recorded "
          f"frames/wall-second   (§4.4 assumed {th['roadmap_4_4_planning_assumption']})")
    print(f"real-time factor : {th['real_time_factor']:.2f}x   "
          f"({th['sim_seconds_recorded']:.0f} sim-s in {t['pipeline_wall_s']:.1f} wall-s)")

    eps = t["per_episode"]
    if len(eps) > 1:
        tot = [e["frame_total_p50_ms"] for e in eps]
        rst = [e["reset_s"] * 1e3 for e in eps]
        print(f"per-episode      : frame p50 {min(tot):.2f}..{max(tot):.2f} ms, "
              f"reset {min(rst):.1f}..{max(rst):.1f} ms  (n={len(eps)})")

    b = rep["bytes_per_frame"]
    line = (f"\nbytes/frame      : raw {b['raw_rgb_uint8']}  "
            f"png {b['png_on_disk']['mean']:.0f} (p50 {b['png_on_disk']['p50']:.0f})")
    for q, v in b["jpeg_in_memory_subsample"].items():
        if v["mean"] is not None:
            line += f"  jpeg-q{q} {v['mean']:.0f}"
    print(line)

    m = rep["memory_kb"]
    print(f"RSS kB           : imports {m['vmrss_after_imports']}  "
          f"after first reset {m['vmrss_after_first_reset']}  end {m['vmrss_at_end']}  "
          f"VmHWM {m['vmhwm_peak']}  ru_maxrss {m['ru_maxrss']}")

    print("\nchecks:")
    for name, v in rep["checks"].items():
        if not isinstance(v, dict):
            continue
        mark = "PASS" if v.get("pass") else "FAIL"
        extra = {k: x for k, x in v.items() if k != "pass"}
        print(f"  [{mark}] {name}  {extra}")


if __name__ == "__main__":
    raise SystemExit(main())
