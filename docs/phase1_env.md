# Phase 1 — the `DroneTargetPairs` environment, expert and evaluator

Recorded 2026-10-05. Roadmap v3 Phase 1 items 1-2: a Gymnasium environment under the §4.3
contract, a scripted oracle expert (§4.2), and an evaluator checked by deliberately
producing every outcome. Dataset recording (item 3) is the next step and is not here.

**Order note.** The roadmap opens Phase 1 with the C++ warm-up and C1. This was built
first instead, at the user's go-ahead; the C++ work follows. Everything C1 will own lives
in one module, `dronevla/action_adapter.py`, so that C1 replaces it rather than running
beside it (§3.2 forbids a parallel production Python copy).

## What exists

| File | What it is |
|---|---|
| `dronevla/task.py` | `TaskConfig` (every episode-defining number), layout sampling and rejection, `layout_id`, instructions |
| `dronevla/env.py` | `DroneTargetPairsEnv`, the Gymnasium env |
| `dronevla/action_adapter.py` | caps, NaN rejection, body<->world velocity. **The C1 stand-in** |
| `dronevla/expert.py` | `StraightLineExpert`, the oracle expert |
| `dronevla/sim.py` | wind, tilt cap, and the `GaussMarkov` noise process (see `realism_mapping.md`) |
| `tests/` | 51 tests: the action contract, the env API, pair determinism, one fixture per outcome |
| `scripts/try_env.py` | knob script: a layout seed or hand-placed targets (`CUSTOM_TARGETS`, `START_Y`, re-checked by the same rejection rules), a goal or the whole pair, expert / manual / wrong. GUI mode follows the drone; **only `--headless` has been run so far** |

```
python -m pytest tests -q            # 51 passed in 36 s
python scripts/try_env.py            # GUI; add --headless for a fast run
```

## The contract, and where each part is implemented

| §4.3 item | Implementation |
|---|---|
| Vision: front RGB 128x96x3 uint8, no alpha, intrinsics recorded | gimbal camera, tilt -20°, 47° vertical FOV; `FrontCamera.describe()` |
| Proprioception, 11 float32, marked as sim state | body-FLU velocity, roll, pitch, sin/cos yaw, body angular velocity, altitude. **True sim state.** PyBullet reports angular velocity in the world frame; it is rotated into the body frame |
| Language: fixed per episode | `obs["instruction"]`; two paraphrase families |
| Action `(vx, vy, vz, yaw_rate)` + Stop logit, body FLU, caps 0.5 / 0.3 / 0.5 | `action_adapter.adapt`; horizontal cap on the vector norm, fixed order. Phase 1 is planar, so vz and yaw rate are forced to zero and the zeroing is flagged |
| Rates 240 / 60 / 5 Hz, 48 physics steps per policy step | one `step()` = 12 control updates = 48 physics steps |
| Stop: 3 consecutive positives, then pose hold, then 1 sim-second check | as specified; the hold freezes the setpoint at the estimated position |
| Success: within 0.4 m, speed <= 0.1 m/s for 1 s after Stop, no collision; timeout 30 s | as specified, judged on the **true** state, 3-D distance to the hover point |
| Store simulation time and monotonic wall time | `info["sim_t"]`, `info["episode_t"]`; wall time is stamped by the recorder (see below) |

Outcomes, one per finished episode: `success`, `stop_not_settled`, `wrong_target`,
`stop_elsewhere`, `collision`, `out_of_bounds`, `timeout` (a truncation, not a termination),
`invalid_action`. Reach for both targets is tracked separately (`reached_goal`,
`reached_other`) so §7.1's OSR and stop-gap can be computed.

## Decisions, with the evidence behind them

**Scene.** §4.1's 8 x 8 x 3 m room; altitude 1.0 m and yaw fixed at 0 (§4.1, Phase 1). The
drone starts at x = -3 m with up to +-0.5 m of lateral jitter; two targets sit at
x in [1, 3], y in [-1.6, 1.6], within +-22° of the start heading (the camera's half-FOV is
~30°). v0.1 requires the two targets to differ in colour, a simplification noted for §4.5's
held-out colour x shape design.

**Targets are 1.2 m pillars, 0.2 m radius.** A target near the floor would drop out of the
frame exactly where the policy has to decide to Stop. Measured with segmentation over 20
seeds:

```
at the start         each target 183 - 556 px   (1.5 - 4.5 % of the frame)
at the hover point   own target 4405 - 5315 px  (36 - 43 % of the frame)
```

**Hover point** (§4.1: "target 앞 0.75 m"): 0.75 m outside the target's footprint, on the
line toward the start. Defined this way so it is deterministic, and so the straight expert
path never passes its own target. Layouts are rejected if the hover points are closer than
1.5 m, if a 0.4 m success region overlaps a target, if the straight path to one goal passes
within 0.35 m of the other target, or if a target is outside the bearing limit.

**Velocity tracking.** The commanded velocity is integrated into a position setpoint that
the PID chases, with the velocity as feed-forward. A zero command leaves the setpoint where
it is, which is the pose hold. The setpoint may not lead the estimated position by more than
0.3 m, so it cannot run away if the drone is held back.

**Noise model changed because of the expert.** With a 1 s correlation time the oracle expert
failed 2-8 of 20 Stops on estimator drift alone. A 3 s correlation time plus 0.3 s smoothing
gives 20/20. Full table in `realism_mapping.md`.

**Pair determinism.** Layout, estimate noise and gusts all derive from the episode seed,
never from the goal index. Tested: the two members of a pair produce byte-identical first
RGB and proprio for 5 seeds (§4.2 item 5).

**No wall-clock time in `info`.** Gymnasium's `check_env` requires `info` to be identical
for the same seed and actions, and wall time never is. §4.3 still wants it stored, so the
recorder stamps `time.monotonic()` when it receives each observation.

**`check_env` passes with 5 warnings, all deliberate:** the action space is in physical
units (m/s, rad/s), not normalised to [-1, 1], because that is the contract; proprio is
unbounded because the contract gives no bounds; and two float64->float32 precision notes
come from gym-pybullet-drones' own spaces.

**The expert's privileged inputs** (for the datasheet): the true pose and velocity, and the
goal hover point. It acts through the same adapter, caps, PID and physics as a policy, and
its commands stay inside the caps, so its emitted action equals the applied action.

## Measured

| seeds | role | expert success | pairs | max speed in Stop hold | median episode |
|---|---|---:|---:|---:|---:|
| 0-9, both goals | **selection set** -- used to choose the noise model | 20/20 | 10/10 | 0.087 m/s | 10.1 s |
| 100-119, both goals | **held out** -- not looked at while choosing | **40/40** | **20/20** | 0.083 m/s (p95 0.078) | 11.1 s |

The selection-set result is not evidence on its own, since the noise model was picked to make
it pass; the held-out set is. The margin to the 0.1 m/s settle limit is about 0.02 m/s, so a
noisier configuration would start producing rejected expert episodes -- which the recorder
must log, not hide (§4.2 item 2).

The episode length matters for §4.4, which assumed 100 frames (20 s) per episode. At ~11 s
the frame count is about half the §4.4 estimate. That is a projection from one expert on
Phase 1 layouts, not a dataset measurement.

## Next

1. Dataset recorder v0.1 (§4.4-4.5): 80 episodes as 40 pairs, splits by layout, a manifest,
   expert failures logged to a rejected manifest. Carry over from this step:
   - enforce `visibility_report()` at generation time; balance colours and left/right placement
   - stamp `time.monotonic()` per observation (the env deliberately does not)
   - the policy step that completes the Stop is not applied as velocity -- the env switches
     to pose hold instead. Mark it in the step record so the teacher label is not misread
   - the expert and the test fixtures read `info["privileged"]`. A learned policy must get
     `obs` only (§7.1: only the evaluator reads privileged state); the eval harness has to
     enforce that interface
2. C++ warm-up, then C1 replacing `action_adapter.py`.
