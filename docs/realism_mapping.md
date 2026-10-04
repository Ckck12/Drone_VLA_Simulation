# Realism mapping — what this simulation takes from a real drone, and what it does not

Recorded 2026-10-04, extended 2026-10-05. Reference spec: **DJI Mavic 4 Pro**, from the DJI
Korean spec sheet the user supplied. Simulated airframe: **Bitcraze Crazyflie 2.x**
(`DroneModel.CF2X` in gym-pybullet-drones), whose mass, inertia, thrust and drag
coefficients come from the real product.

The goal is not to clone the Mavic. It is to make the simulation as close to a real drone
as this laptop (CPU only, PyBullet TinyRenderer) allows, and to write down where it stops
and why. Every number below was measured with `scripts/playground.py`, with air drag on,
unless it is marked as a spec value.

## The scale problem comes first

| | Mavic 4 Pro (spec) | Crazyflie 2.x (simulated) |
|---|---|---|
| take-off mass | ~1063 g | 27 g (`cf2x.urdf`) |
| use | outdoor, km range | indoor, a 2 m arena here |
| max horizontal speed | 15-25 m/s | roadmap §4.3 cap 0.5 m/s |

A ~40x mass difference means absolute numbers such as 25 m/s or +-0.3 m hover error do not
transfer. Each spec item therefore falls into one of three bins.

## ① Scale-independent — taken directly

| Spec | Before | Now | Status |
|---|---|---|---|
| Main camera FOV 72°, 4/3" sensor (4:3) | 60° vertical = 88° diagonal | `CAMERA_VFOV_DEG = 47` -> 72° diagonal | **done**. Assumes DJI's 72° is diagonal, DJI's usual convention -- not confirmed from the sheet |
| 3-axis gimbal, tilt -90° ... +70° | camera pitched with the airframe | `CAMERA_MOUNT = "gimbal"`, `CAMERA_TILT_DEG` | **done**, ideal stabilisation (spec residual is +-0.003° in normal mode) |
| camera at the front of the body | lens above the centre, own airframe in frame | lens at body (+0.035, 0, -0.015) m | **done** |
| max pitch 35° | no tilt limit in `DSLPIDControl` | `MAX_TILT_DEG = 35` (`dronevla.sim.TiltLimitedDSLPIDControl`) | **done**, limits the *commanded* tilt -- see below |
| video 30/60 fps, policy slower | records 5 Hz | unchanged | consistent already |

### Measured: camera mount

Same route, same seed, no noise, no objects. "Sky" is the share of pure-background pixels in
each frame, which only changes if the camera's orientation changes -- so its spread measures
how much the image shakes.

| Mount | sky mean | sky range | sky std | own airframe in frame |
|---|---:|---:|---:|---:|
| legacy (Phase 0), 60° | 49.5 % | 36.5-61.5 % | 2.54 % | 1.21 % |
| rigid, tilt -20°, 47° | 7.8 % | 0.0-24.1 % | 3.06 % | 0.00 % |
| **gimbal, tilt -20°, 47°** | **8.3 %** | **7.3-8.3 %** | **0.10 %** | **0.00 %** |

(own-airframe share from a segmentation render while hovering)

- By sky std the gimbal image is 25x steadier than legacy and 31x steadier than rigid.
- Tilting 20° down cuts the sky from half the frame to about 8 %, so the pixels go to the
  ground and the targets -- the "50 % sky" finding from Phase 0.
- Moving the lens to the front removes the drone's own airframe from the frame.
- With the default objects in the scene the gimbal's sky range widens to 4.0-8.3 %: the
  0.6 m green cylinder rises above a 0.5 m flight altitude and covers sky. Confirmed by
  removing the objects. The remaining one-row (1 %) variation without objects is *probably*
  the finite ground plane's edge shifting with altitude and heading; not verified.

### Measured: tilt cap

The default route at 1.5 m/s, three times the contract cap, to force aggressive corners. No
noise, no wind.

| | max actual tilt | lean the position loop asked for | cap engaged | tracking mean / max | hover max afterwards |
|---|---:|---:|---:|---:|---:|
| cap off | 65.3° | 88.3° | -- | 0.508 / 1.094 m | 0.102 m |
| **cap 35°** | **39.7°** | 91.1° | 13.5 % of steps | 0.591 / 0.961 m | 0.065 m |

**The cap limits the commanded tilt, not the achieved tilt.** The attitude loop overshoots
its command by about 5°, so the airframe still reached 39.7°. That is the honest behaviour
of capping a setpoint; a hard guarantee would need the attitude loop itself changed. At the
default 0.3 m/s the cap never engages (the controller asks for at most ~21°).

## ② Present in reality, size must be scaled

| Spec | Before | Now | Status |
|---|---|---|---|
| hover accuracy +-0.1 m vertical / +-0.3 m horizontal (vision) | controller given the exact true position | `POSITION_NOISE_XY_M = 0.03`, `POSITION_NOISE_Z_M = 0.01` | **done** |
| wind resistance 12 m/s | no wind, no drag | `WIND_SPEED_MPS`, `WIND_TOWARD_DEG`, `WIND_GUST_MPS`; drag always on | **done**, with a model limit -- see below |
| obstacle sensing (front LiDAR 0.5-25 m, downward IR 0.3-8 m) | none | -- | later; feasible with ray casting |

### Noise model and measurement

A first-order Gauss-Markov process per axis, 1 s correlation time, so the error drifts
slowly the way GNSS or visual-odometry error does, instead of buzzing at 60 Hz. The
controller sees `truth + error`; the camera renders from the truth, because it is physically
on the drone. The 3:1 horizontal-to-vertical ratio follows the Mavic spec. **The absolute
size, 0.03 m, is a choice, not a derivation** -- roughly a third of the Crazyflie's ~9 cm
span, picked to be visible in a 2 m arena.

Gimbal camera, default route and objects, same seed. Hover = true position vs the final
waypoint over the last 3.6 s, after a 1 s settle.

| | tracking mean / max | hover horizontal max (p95) | hover vertical max (p95) |
|---|---:|---:|---:|
| no noise | 0.018 / 0.092 m | 0.006 m (0.006) | 0.004 m (0.004) |
| **noise 0.03 / 0.01 m** | 0.053 / 0.146 m | 0.045 m (0.045) | 0.018 m (0.017) |

The drone drives what it *believes* its position is onto the target, so the true position
misses by roughly the estimate error -- the controller's belief was off by 0.045 m on average.
That is how a real drone ends up with a hover tolerance at all.

Caveat: 3.6 s of hover with a 1 s correlation time is only a few independent samples. The
hover figures are indicative; a 30 s+ hover (raise `DURATION_S`) would pin them down.

### Wind model and measurement

gym-pybullet-drones already has a drag model (`BaseAviary._drag`, from the Crazyflie system
identification in Förster 2015): a force proportional to rotor speed times velocity. It uses
velocity relative to the *ground*. `dronevla.sim.WindyCtrlAviary` changes exactly one thing,
velocity relative to the *air* (`v_drone - v_wind`), applied every physics substep (240 Hz).
Checked: with zero wind, its trajectory is bit-identical to the library's own `PYB_DRAG`
physics; and with the cap off, `TiltLimitedDSLPIDControl` is bit-identical to
`DSLPIDControl`. Gusts use the same Gauss-Markov process as the noise, 2 s correlation.

Hovering at a single point for 20 s, no noise:

| wind | max lean | hover horizontal max (p95) over 17 s |
|---|---:|---:|
| 1 m/s steady | 2.3° | 0.013 m (0.012) |
| 2 m/s steady | 3.5° | 0.023 m (0.019) |
| 4 m/s steady | 5.8° | 0.040 m (0.039) |
| 8 m/s steady | 10.7° | 0.082 m (0.075) |
| 2 m/s + 1 m/s gusts (5.39 m/s peak) | 7.2° | 0.038 m (0.033) |

**Model limit -- strong wind is optimistic here.** The drag is linear in airspeed and was
identified for a Crazyflie flying slowly. Body drag on a real airframe grows roughly with
the square of airspeed, and that term is absent. So the model says the Crazyflie holds
position in 8 m/s of wind with only 10.7° of lean. That is very unlikely for a 27 g indoor
drone. Treat results above about 2 m/s as a lower bound on difficulty. For the same reason
the Mavic's "12 m/s wind resistance" is not compared against anything here.

## ③ Out of reach on this laptop -- recorded, not attempted

| Spec | Why not |
|---|---|
| 1 kg airframe dynamics | gym-pybullet-drones ships an 830 g `racer.urdf`, but `DSLPIDControl` refuses any model except CF2X/CF2P. Needs a different controller (`MRACControl` mentions the racer; untested) and retuning |
| 100 MP sensor, Hasselblad colour, photoreal rendering | TinyRenderer is a CPU rasteriser; photoreal rendering is Isaac Sim territory and needs an RTX GPU. The policy input is 128x96 anyway |
| flight time, battery, O4+ transmission, GNSS constellations, storage | not relevant to a short-range language-conditioned flight task |

## What this changes, and what it does not

- Camera mount, tilt and FOV are camera extrinsics/intrinsics under the roadmap §4.3
  contract. The contract says to record them, which `FrontCamera.describe()` does, so changing
  them is allowed -- but they should be fixed **before** dataset v0.1 is generated.
- The same goes for drag, wind, noise and the tilt cap: they change what the expert's
  demonstrations look like, so they belong in the dataset manifest and should be fixed first.
- `dronevla.profile_env` still uses the `legacy` mount and `Physics.PYB` (no drag), so every
  Phase 0 number stands. Checked: the 10-frame CI smoke produces byte-identical frames before
  and after the camera change.
- Not modelled: quadratic body drag, velocity-estimate noise, image sensor noise, motion
  blur, gimbal lag.
