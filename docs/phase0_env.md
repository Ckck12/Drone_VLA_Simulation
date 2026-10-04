# Phase 0 — environment record

- Recorded: 2026-10-04T21:06:58+09:00
- Status: **lock candidate**, not a verified lock (roadmap v3 line 891).

## Pinned source

- gym-pybullet-drones 2.2.0 @ `7ebad1ecabd28a7000add2d05f888aa2e837c2cc` (editable install from `third_party/`)
- Upstream: https://github.com/learnsyslab/gym-pybullet-drones

## Key decisions

- torch installed from https://download.pytorch.org/whl/cpu as `2.14.1+cpu`.
  Default PyPI torch on Linux pulls cuda-toolkit 13.0.3 + 6 more NVIDIA deps, useless on Iris Xe.
- pybullet 3.2.7 has no cp312 wheel; built from source (gcc 13.3.0). isNumpyEnabled() = 1.
- OpenGL renderer defaults to llvmpipe (software). GALLIUM_DRIVER=d3d12 enables
  D3D12 (Intel Iris Xe) but measured slower on glxgears (37 vs 425 FPS), so it is NOT set by default.
  Proceed with PyBullet DIRECT + ER_TINY_RENDERER per roadmap v3 line 310.

## Machine

```
$ uname -a
Linux <host> 6.18.40.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Fri Jul 31 22:12:15 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux

$ lsb_release -a
Distributor ID:	Ubuntu
Description:	Ubuntu 24.04.5 LTS
Release:	24.04
Codename:	noble

$ free -h
               total        used        free      shared  buff/cache   available
Mem:           7.5Gi       593Mi       6.1Gi       4.8Mi       1.0Gi       7.0Gi
Swap:          2.0Gi          0B       2.0Gi

$ df -h / /mnt/c
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G  3.9G  952G   1% /
C:\             456G  351G  106G  77% /mnt/c

$ lscpu
CPU(s):                                  16
Model name:                              13th Gen Intel(R) Core(TM) i5-1340P
Thread(s) per core:                      2
Core(s) per socket:                      8
Virtualization:                          VT-x
Virtualization type:                     full
```

## Verification gates passed

```
torch.__version__          = 2.14.1+cpu
torch.cuda.is_available()  = False
pybullet.isNumpyEnabled()  = 1
nvidia/cuda pip packages   = 0
pip check                  = No broken requirements found.
venv size                  = 1.5G
```

Full package list: `env-lock-candidate.txt`

## Smoke test (headless, DIRECT + ER_TINY_RENDERER)

Single cold run, NOT a throughput benchmark.

```
physics 240 steps : 33.9 ms
getCameraImage    : 13.0 ms   (128x96, ER_TINY_RENDERER)
rgb returned      : ndarray (96, 128, 4) uint8   <- numpy path confirmed end to end
HoverAviary       : obs (1, 72), action (1, 4), reward 1.3789, reset/step OK
```

Roadmap v3 targets 128x96 recording at 5 Hz (200 ms/frame budget). One cold frame took 13 ms,
so the budget looks comfortable, but a sustained throughput measurement has not been run.

---

## Roadmap v3 line 302 — executed 2026-10-04

Scripts live in `scripts/`, outputs in `results/phase0/`. All headless (`DIRECT`, matplotlib `Agg`).

### 1. Two-object render, visually inspected

`scripts/render_two_objects.py` -> `two_objects_inspect_640x480.png`, `two_objects_policy_128x96.png`

Red cube at world y = +0.5, blue sphere at world y = -0.5, camera at (2.0, 0, 0.7) looking at the
origin with up = +z.

- The images were opened and looked at, not just checksummed: checkerboard ground plane, correct
  perspective and shading, both objects the right colour and shape.
- **Axis convention confirmed empirically: world +y maps to image RIGHT** for this camera pose
  (verified from the segmentation mask, not assumed). Red (y=+0.5) renders right, blue (y=-0.5) left.
- At the roadmap's recording size of 128x96 the two objects remain **visually distinguishable** by
  colour and by left/right position. That is a rendering fact only — it says nothing yet about
  whether a policy can learn topic A's counterfactual pairs at this resolution.

### 2. Single-drone hover — `scripts/hover_pid.py`

Adapted from `examples/pid.py`, which flies a *circular* trajectory with 3 drones. Same controller
(`DSLPIDControl`) and env (`CtrlAviary`), constant target instead. Start z = 0.10 m, target z = 1.00 m.

```
rise time (10-90%)      : 0.750 s
overshoot               : +0.64 %
settling time (+-2%)    : 1.354 s
steady-state error (z)  : +5.0 mm   (mean of last 2 s)
max xy drift (last 2 s) : 0.0 mm
```

Well-damped, with a +5 mm standing offset in z.

**xy drift is exactly 0.000 mm.** With `Physics.PYB` and a perfectly symmetric start, nothing
perturbs xy. `utils/enums.py` does offer `PYB_GND`, `PYB_DRAG`, `PYB_DW` and `PYB_GND_DRAG_DW`, but
those aero models are deterministic too. So randomization for the data flywheel has to be added
explicitly — no physics mode supplies it.

### 3. Velocity step response — `scripts/velocity_step.py`

Adapted from `examples/pid_velocity.py`, which switches 4 drones between several targets at once.
One drone, one clean step: 1 s hold, 3 s at vx = 0.20 m/s, 2 s back to 0.

```
SPEED_LIMIT            : 0.250 m/s   (= 0.03 * MAX_SPEED_KMH / 3.6 for the CF2X)
rise time (10-90%)     : 0.208 s
overshoot              : +19.5 %
mean vx over hold      : 0.1999 m/s  (target 0.200)
vx after step off      : -0.008 m/s
altitude hold          : 1.000 .. 1.004 m
```

**Finding — the velocity loop never settles.** Computed from `velocity_step.csv` over the
1.71-3.98 s window (the hold minus its 0.7 s transient, n = 110):

```
std                            : 0.0145 m/s   (7.2% of target)
peak-to-peak                   : 0.0520 m/s   (26.0% of target)
min / max                      : 0.1727 / 0.2247 m/s
inside the +-2% band           : 10.4% of the 3 s hold
last excursion outside the band: 3.98 s  (the hold ends at 4.00 s)
```

The mean tracks the target almost exactly (+0.0 %), which is why a mean-only "steady-state error"
reads as near-perfect and hides this entirely. The ripple does not decay: the signal is still
leaving the +-2 % band 20 ms before the step ends. Step-off undershoots to -0.06 m/s before damping.
Altitude is held well throughout (4 mm rise over 6 s).

*Ripple frequency is not reported.* The dominant FFT bin is 0.44 Hz, which is exactly the lowest
resolvable frequency for a 2.27 s window, so this window cannot characterise it. A longer hold would
be needed to state a frequency.

If the policy drives `VelocityAviary`'s action interface, it sits on top of this tracker, so a
26 % peak-to-peak velocity ripple would belong to the controller rather than the policy, and
closed-loop success metrics must not read it as policy jitter. The roadmap has not fixed the action
interface yet, so this is conditional.

**Bug caught while writing this.** The first attempt commanded 0.5 m/s, which needs
`fraction = 2.0` — outside the action space bound of [0, 1]. `VelocityAviary._preprocessAction`
uses `np.abs(action[3])` with no clipping, so it ran and produced a plausible-looking 0.5 m/s plot.
The script now asserts `0 <= fraction <= 1`. Treat the action-space bounds as unenforced here.

### Watching a run in a GUI window

WSLg is available (`DISPLAY=:0`, `wayland-0`, `/mnt/wslg`, `explorer.exe` bridge), so PyBullet's GUI
opens as a normal Windows window. Run it from an interactive terminal, and redirect the logs so the
pinned clone stays clean — `pid.py` defaults `--output_folder results` relative to the working
directory and always calls `logger.save_as_csv`:

```
python third_party/gym-pybullet-drones/gym_pybullet_drones/examples/pid.py \
    --gui True --num_drones 1 --duration_sec 12 \
    --output_folder ~/dronevla/results/gui_run
```

A matplotlib window also opens at the end because `--plot` defaults to True. Note the `--duration_sec`
help text says "default: 5" but `DEFAULT_DURATION_SEC` is 12, so pass it explicitly.

### Still not done

- Sustained camera throughput measurement (only a single cold frame at 13.0 ms so far).
- Ripple frequency over a longer hold.

---

## Roadmap v3 step 4 — renderer raw logs (added 2026-10-04)

The conclusions above were recorded before the raw output was kept. The logs now exist:

- `docs/phase0/glxinfo-default.log` — environment as it comes, no override
- `docs/phase0/glxinfo-d3d12.log` — same command under `GALLIUM_DRIVER=d3d12`
- `docs/phase0/capture_glxinfo.sh` — regenerates both, with the environment variables,
  `DISPLAY`/`WAYLAND_DISPLAY`, mesa package versions and capture time in each header

Both were captured from a **non-interactive** `wsl.exe -d Ubuntu-24.04 -- bash <script>` shell,
and WSLg still supplied `DISPLAY=:0` / `WAYLAND_DISPLAY=wayland-0` there.

```
default : llvmpipe (LLVM 20.1.2, 256 bits)        Accelerated: no
d3d12   : D3D12 (Intel(R) Iris(R) Xe Graphics)    Accelerated: yes
both    : Mesa 25.2.8-0ubuntu0.24.04.4, mesa-utils 9.0.0-2
```

**New finding — forcing d3d12 lowers the GL version ceiling.** `diff` of the two logs:

| | default (llvmpipe) | `GALLIUM_DRIVER=d3d12` |
|---|---|---|
| Max core profile | 4.5 | **4.1** |
| Max compat profile | 4.5 | **4.1** |
| Max GLES[23] | 3.2 | **3.0** |
| GLSL | 4.50 | 4.10 |
| Reported video memory | 7722 MB (system RAM, `Accelerated: no`) | 8099 MB |

So the renderer choice is not a pure speed trade: the accelerated path also offers a **lower**
GL feature level. That matters for Phase 5 (Gazebo) rather than now, because PyBullet's
TinyRenderer is CPU software rasterisation and uses no GL context at all.

The `llvmpipe` branch of the roadmap's renderer table is therefore the one in force, with the
d3d12 option recorded but not enabled. The earlier `glxgears` numbers (37 vs 425 FPS) compare a
trivial scene and are **not** evidence about PyBullet or Gazebo throughput; they only showed that
enabling d3d12 did not make the default case faster.

## Repo layout decisions (2026-10-04)

`git init -b main` in `~/dronevla` (Linux home, ASCII path, outside OneDrive). What is tracked:

| Path | Tracked? | Why |
|---|---|---|
| `docs/`, `scripts/`, `robotics_project/` | yes | specs, and the code behind every number in this file |
| `results/phase0/` (244 KB) | yes | small, and it is the evidence for the plots cited here |
| `env-lock-candidate.txt` | yes | the dependency lock |
| `.venv/` (1.5 GB) | no | rebuildable from the lock |
| `third_party/` (136 MB) | no | pinned clone, see below |
| `results/gui_run/` (876 KB) | no | GUI sanity-check logs, not a deliverable |
| `results/**/frames/`, `*.npy` | no | bulk recordings |

**`third_party/gym-pybullet-drones` is ignored, not a submodule.** Verified at init time:
`git rev-parse HEAD` = `7ebad1ecabd28a7000add2d05f888aa2e837c2cc`, remote
`learnsyslab/gym-pybullet-drones`, working tree clean — i.e. exactly the SHA already pinned in
`env-lock-candidate.txt`. A submodule would add a *second* record of that SHA (the gitlink) which
can silently disagree with the lock file. Since the editable install also needs a per-machine venv
step with a non-default torch index, a clean checkout can never be "clone and run" regardless, so
one authoritative SHA in the lock plus a documented setup step is the smaller surface. Trade-off
accepted: `git clone` alone does not fetch the simulator.

**Caveat on `robotics_project/`.** These Markdown files are *copies* of the authoritative
originals in the Windows OneDrive workspace. Tracking them makes divergence visible in `git diff`
instead of invisible, but edits made on the OneDrive side will not appear here automatically.
