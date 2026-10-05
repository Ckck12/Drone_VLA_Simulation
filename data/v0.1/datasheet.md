# Datasheet — DroneTargetPairs v0.1.0

Generated 2026-10-05 by `python -m dronevla.record --out <dir> --pairs 20 10 10`. Every number below is computed from the
recorded files.

## Purpose
Thin-slice dataset for roadmap v3 Phase 1: train and evaluate a tiny RGB + text policy that
must fly to *the target the instruction names* and stop. Built as counterfactual pairs: the
same start snapshot flown twice, once per target. **Not for**: claims about general language
understanding, real-world flight, or anything beyond two coloured pillars in an empty room.

## Contents
| split | pairs | episodes | seeds |
|---|---:|---:|---|
| train | 20 | 40 | 1001-1020 |
| val | 10 | 20 | 2001-2010 |
| test | 10 | 20 | 3001-3010 |

- 80 episodes, 4101 observations (4101 PNG images, 6.44 MB,
  1570 bytes/image on average)
- observations per episode: median 51, range 40-65
- episode length: median 10.8 sim-s, range 8.6-13.6
- generation took 188 wall-s on the development laptop (CPU only)

## Scene
8 x 8 x 3 m room, no obstacles. The drone starts at x = -3.0 m (lateral jitter
+-0.5 m), altitude 1.0 m, yaw fixed at 0; planar motion only.
Two 1.2 m pillars of radius 0.2 m, box or cylinder, in
two different colours of red/blue/green/yellow, within +-22 deg of the
start heading.

## Expert and what it knew (privileged inputs)
`StraightLineExpert`: true position and velocity from the simulator, plus the goal hover
point (0.75 m in front of the goal target). It flies through the same action
adapter, caps, PID and physics as a policy, and its actions stay inside the caps, so
`action_raw` equals the applied action. It is an oracle, not a perception model.

## Observations vs ground truth
Policy inputs: `rgb` (PNG, 128x96x3), `proprio` (11 floats: body velocity, roll, pitch,
sin/cos yaw, body angular velocity, altitude -- true sim state), the instruction. Everything
in `priv_*` columns and in `layouts/` (hover points, target coordinates) is ground truth for
analysis and evaluation only.

## Instructions
Two human-written families, no LLM: `Go to the {color} {shape} and stop.`; `Approach the {color} {shape}, then hold position.`.
Both appear in every split; a pair always shares one family. Counts:

| split \ instruction_family | 0 | 1 |
|---|---:|---:|
| test | 10 | 10 |
| train | 20 | 20 |
| val | 10 | 10 |

## Balance (§4.5 contingency tables)
Goal colour x goal side (left/right of the start heading):

| goal_color \ goal_side | left | right |
|---|---:|---:|
| blue | 7 | 10 |
| green | 12 | 13 |
| red | 10 | 13 |
| yellow | 9 | 6 |

Goal shape x split:

| goal_shape \ split | test | train | val |
|---|---:|---:|---:|
| box | 11 | 24 | 8 |
| cylinder | 9 | 16 | 12 |

These are reported, not enforced: v0.1 samples layouts at random.

## Splits and leakage
Split by layout; in v0.1 each layout is its own family (no mirrored or translated copies),
so no layout, family or seed appears in two splits. Both episodes of a pair are always in the
same split. `validate()` checks all of this.

## Rejections
0 candidates were generated and discarded: none. Each is in
`rejected.jsonl` with its reason.

## Labels and alignment
Row t = observation t + the action applied after it; row t+1 is 0.2 sim-s later. The last
row is terminal with no action (`action_mask` False). The row with `stop_trigger` True holds
the action that completed the Stop: it was not flown -- the env switched to a 1 s pose hold,
so the next row is 1 s later. Rates: physics 240 Hz, control 60 Hz,
policy 5 Hz. Frames: actions body FLU (x forward, y left, z up), world ENU.
Both simulation time and monotonic wall time are stored.

## Outcomes
Only episodes the expert completed successfully are kept (success = within
0.4 m of the goal hover point, speed <= 0.1 m/s for
1.0 s after Stop, no collision).

## Personal information
None. Synthetic scenes only; no people, no real images, no operator data.

## Licences
Code: MIT. Dataset: MIT, same as the code. Simulator assets used to render: gym-pybullet-drones
CF2X model (MIT), PyBullet `plane.urdf` (zlib). The generator does not use any third-party
dataset or pretrained weights.

## Known limitations and bias
- yaw is fixed and motion is planar, so the camera always faces +x
- the two targets always differ in colour; shape alone never decides the goal in v0.1
- targets are tall pillars chosen to stay visible from the hover point; the sky and floor
  are untextured
- position-estimate noise is simulated (sigma 0.03 m horizontal); wind is off
- 40 pairs is a thin slice: enough to wire training and evaluation, not to measure
  generalisation

## Regenerate
`python -m dronevla.record --out <dir> --pairs 20 10 10` from the repository at the commit in `manifest.json`. Same seeds give the same
layouts and, on the same machine, byte-identical images.

## Version
v0.1.0 — first version.
