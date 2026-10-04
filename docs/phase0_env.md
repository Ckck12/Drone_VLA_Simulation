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

*(Superseded later the same day — see "Still open" at the end of this file.)*

- ~~Sustained camera throughput measurement (only a single cold frame at 13.0 ms so far).~~
  Done: 1000 frames, 34.48 recorded frames/wall-second. The single cold frame at 13.0 ms
  turned out to be representative — the sustained render p50 is 12.84 ms.
- Ripple frequency over a longer hold. Still open.

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
n---n
## Roadmap v3 Phase 0 item 6 — the 1000-frame profile (2026-10-04)

```
python -m dronevla.profile_env --frames 1000 --renderer tiny --out reports/env.json
```

Code: `dronevla/profile_env.py`, `dronevla/camera.py`. Reports: `reports/env.json` (run A),
`reports/env_runB.json` (repeat), `reports/env_shadow_seg.json` (comparison).
`scripts/summarize_env_profile.py` prints every number quoted below straight from those
files, so the doc cannot drift from the JSON.

### What was measured, and under what rules

Protocol from §7.4: 50 warm-up frames discarded before timing; percentiles not just means;
the whole run repeated in a second process for spread; `perf_counter` overhead measured
(145-147 ns/call, so the sub-millisecond buckets are still ~2000x the timer cost); host
conditions recorded. Rates from §4.3: physics 240 Hz, control 60 Hz, record 5 Hz, which is
**48 physics steps and 12 control updates per recorded frame, exactly**.

The earlier hover and velocity-step scripts used a 48 Hz controller. 48 Hz cannot hold this
contract — 240/5 = 48 physics steps per frame is not a whole number of 240/48 = 5-step
control periods. 60 Hz divides both. The controller rate changed for that reason, not by
preference.

Ten episodes of 100 frames = 20 sim-seconds each, matching §4.4's episode-length
assumption, so `reset` is sampled ten times rather than once.

### Where the wall time goes

p50 per recorded frame, run A and the repeat run B:

| bucket | A p50 | B p50 | share of wall (A) | run-to-run spread |
|---|---:|---:|---:|---:|
| reset (per episode, incl. rebuilding the scene) | 48.31 ms | 47.57 ms | 1.7 % | 1.6 % |
| control — 12x `DSLPIDControl` | 9.50 ms | 9.54 ms | 33.9 % | 0.4 % |
| physics — 48 PyBullet steps | 3.53 ms | 3.46 ms | 12.5 % | 1.9 % |
| render — one 128x96 TinyRenderer frame | 12.84 ms | 12.77 ms | 44.6 % | 0.6 % |
| encode — PNG, compress level 6 | 1.49 ms | 1.48 ms | 5.2 % | 0.4 % |
| write — one file to ext4 | 0.37 ms | 0.27 ms | 1.3 % | 36.2 % |

```
run A : 34.48 recorded frames/wall-second, real-time factor 6.90x, pipeline 29.00 s
run B : 35.05 recorded frames/wall-second, real-time factor 7.01x, pipeline 28.53 s
run-to-run throughput spread : 1.65 %
per-episode frame p50 (n=10)  : 27.38 - 28.82 ms;  reset 47.2 - 52.5 ms
```

**Finding — the PID controller costs 2.7x the physics it drives.** 12 calls to
`computeControlFromState` take 9.50 ms while the 48 physics steps they schedule take
3.53 ms. Control is the second-largest bucket in the whole pipeline. This is a pure-Python
NumPy controller called 60 times per sim-second; the figure is a fact about this
implementation, not about PID control. It matters because §4.3 keeps attitude
stabilisation on this controller while the policy runs at 5 Hz, so the controller is a
fixed per-frame cost under every policy, learned or scripted.

**Finding — `write` is the only bucket that is not repeatable.** 36 % spread between runs
against under 2 % everywhere else, which is what filesystem buffering looks like. It is
1.3 % of the pipeline, so it changes nothing here, but a bytes/frame or storage claim
should not be built on a single write measurement.

### Why this camera and not `BaseAviary._getDroneImages`

`_getDroneImages` hardcodes 64x48, never passes `renderer=`, and defaults to shadows on
plus `ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`. Running this profile with those two
options turned on (`--shadow --segmentation`, `reports/env_shadow_seg.json`):

```
render p50   12.84 ms  ->  26.52 ms     (2.07x)
throughput   34.48     ->  23.31 recorded frames/wall-second
png/frame    1421 B    ->  2436 B
```

So the library's defaults would cost **twice the render time** for a segmentation mask that
§4.3 explicitly excludes from policy input and a shadow nothing needs yet. It also renders
at `aspect=1.0` into a 4:3 image, which stretches the pixels. `dronevla/camera.py` keeps
the view transform identical and lists every deviation in `camera.describe()`, which lands
in the report.

### bytes/frame — §4.4's compression assumption is inverted

| format | bytes/frame | vs raw | vs PNG |
|---|---:|---:|---:|
| raw uint8 RGB 128x96x3 | 36 864 | 1.0x | |
| **PNG, compress level 6 (what was written)** | **1 421** (p50 1 411, range 1 022-1 847) | 25.9x smaller | 1.00x |
| JPEG q90 (in memory) | 3 922 | 9.4x smaller | 2.76x larger |
| JPEG q75 (in memory) | 2 651 | 13.9x smaller | 1.87x larger |

§4.4 assumes "JPEG 평균 6-15 KB/frame". Measured, **JPEG is 2.7x *larger* than lossless
PNG**, and PNG is 4-10x smaller than the low end of that assumption. At 128x96 the image is
flat colour over large areas, which PNG's filtering plus DEFLATE handles far better than a
DCT, and JPEG's fixed overhead is not amortised over so few pixels. Lossy was the wrong
default here, both for size and for fidelity.

Recomputing §4.4 with the measured PNG size and run A's throughput, keeping the roadmap's
own 1.5x overhead factor:

| stage | frames | §4.4 raw-GB figure | measured PNG | §4.4 time estimate | measured |
|---|---:|---:|---:|---:|---:|
| v0.1 thin slice | 8 000 | 0.295 GB | **0.011 GB** | 10-40 min | **5.8 min** |
| v0.2 total | 200 000 | 7.373 GB | **0.284 GB** | 4.2-16.7 h | **2.42 h** |

**These replace planning assumptions, not the measurement.** Two caveats, both material:
the scene is a ground plane plus two untextured primitives, and richer scenes compress
worse and render slower; and the 1.5x factor still covers planning, rejection and QA, which
were not measured. The disk budget in §4.4 (20 GB for dataset and staging) is far larger
than it needs to be at these sizes.

### What is actually in a frame

`scripts/inspect_frame_contents.py` renders one pose with the segmentation mask on and
counts pixels per body, because guessing from a thumbnail is not evidence:

```
pose xyz=[1.213, 0.319, 0.454]  yaw=-165.3 deg,  128x96 = 12288 px

  50.00 %  background / sky (nothing hit)
  40.82 %  ground plane
   4.87 %  red cube target
   3.10 %  blue sphere target
   1.21 %  the drone's own airframe
```

Three things follow. **Half of every frame is empty sky**, because the camera looks
horizontally so the horizon sits exactly on the image centre line. **The two targets --
the entire content the policy has to tell apart -- occupy under 8 % of the pixels.** And
the drone sees **its own airframe** in 1.2 % of the frame, a fixed artifact in the bottom
of every observation. None of this is a Phase 0 blocker, but all three are inputs to the
§4.1 scene generator and to any later decision about camera pitch or FOV.

**Trouble-shooting note — this was caught by looking, not by a check.** The first 1000-frame
recording passed every automated check: 1000 files, right shape and dtype, no NaN, no
repeated images, non-blank. The contact sheet
(`results/phase0/env_profile_contact_sheet.png`, built by `scripts/contact_sheet.py`)
showed the targets clipped by the bottom edge in most frames. Cause: the drone flew at
0.9-1.1 m while the targets sit at z = 0.15 m about 1.2 m away, so the depression angle to
a target was `atan(0.85/1.2) = 35 deg` against a half-FOV of 30 deg. Flying at 0.40-0.60 m
fixed it and the run was repeated. The checks cannot catch this: a frame of sky and ground
is non-blank and unique.

### Memory

```
VmRSS after imports        68.3 MB
VmRSS after the first reset 108.2 MB
VmRSS at the end           113.2 MB
VmHWM (peak)               113.2 MB      ru_maxrss 112.7 MB
WSL MemTotal 7.5 GiB, MemAvailable 6.3 GiB at the time of the run
```

Frames are streamed to disk, never accumulated, so RSS is flat after the first reset. Peak
is 1.5 % of what WSL has — the §4.4 worry about approaching the WSL memory ceiling does not
apply at this stage. `torch` is never imported by the profiler, which the report records.

### Conditions, and what they mean for these numbers

AC power, battery at 100 %, Windows power scheme `SAMSUNG MODE`, 16 logical CPUs visible to
WSL, load average 0.73. **The desktop was busy**: the editor and browser were running, the
host reported 23 % instant CPU load and only 432 MB free RAM. Every figure above is
therefore a **lower bound on throughput and an upper bound on per-frame time**. That is the
conservative direction for planning — the measured 34.5 frames/wall-second already beats
§4.4's 5-20 assumption by 1.7-6.9x with a loaded machine. A quiet-machine repeat would
refine it, not rescue it. Full host record in `reports/host_state.json`, collected by
`scripts/host_state.ps1` (WSL cannot see AC state or the host power scheme).

CPU temperature is not exposed by this hardware through WMI, so thermal behaviour over a
long generation run is unrecorded. §7.4's thread sweep (1/2/4/8) is not part of Phase 0.

### Phase 0 exit criteria

| §5 Phase 0 exit criterion | status |
|---|---|
| 1000 real RGB frames recorded | met — `frame_files_on_disk` 1000/1000, decoded off disk |
| no NaN, no state-time reversal | met — 23 000 values checked finite; sim time strictly increasing per episode, interval 0.2 s to 2.8e-15 s; wall clock monotonic |
| renderer name and peak RSS/FPS recorded with no blanks | met — TinyRenderer, VmHWM 113.2 MB, 34.48 frames/wall-second |
| runnable headless with one command | met — the command at the top of this section, `DIRECT` mode, no display |
| clone outside OneDrive | met — `~/dronevla` on the WSL ext4 home |

All nine automated checks pass in both runs. The checks are run against the artefacts read
back off disk, not against the loop that wrote them.

### Minimal CI

`.github/workflows/ci.yml` runs `scripts/ci_smoke.sh` on push: the same camera, rates,
encoder and verification as the full profile, at 10 frames. It completes locally in 0.72 s
and the gate re-reads the report rather than trusting the exit code. The workflow skips
torch entirely, since the profiler never imports it.

**It has never executed.** There is no git remote yet, so the YAML is unverified beyond
being written against the documented actions. Treat the local `bash scripts/ci_smoke.sh`
run as the only evidence so far.

### Still open

- Sustained throughput is now measured, but only on a two-primitive scene and only on a
  busy desktop.
- Velocity-loop ripple frequency over a longer hold (from the earlier step-response work).
- Thread sweep and a combined-load benchmark (§7.4 item 3) — Phase 4 work, not Phase 0.
