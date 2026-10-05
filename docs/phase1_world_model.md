# Phase 1 — World model, stage 1: does it predict?

The BC policy ([phase1_policy.md](phase1_policy.md)) learned an image shortcut and ignored the
instruction. The world-model track asks a different question first: can a model learn *what
happens next* from (image, proprio, action), well enough that a planner could later use it?
Stage 1 measures only that. Language and planning come in stages 2 and 3.

## Model and data
`dronevla/world_model.py`, 574,515 parameters, CPU only.

- **encoder** (RGB 128x96 + 11 proprio) → latent z (64)
- **dynamics** z' = LayerNorm(z + f(z, a)), a = commanded (vx, vy) / 0.5 m/s
- **heads**: proprio (11), and the body-frame offset from the drone to each colour's hover point
  (4 colours x 2, masked where the colour is absent). Offsets come from logged true positions
  and are training targets only, never inputs.
- **loss**: 4-step unroll of latent (stop-grad encoder target) + proprio + offsets, discounted 0.9
- **training**: 12 epochs fixed in advance, last epoch reported, seed 0, Adam 1e-3, batch 32

Data = expert demonstrations + exploration flights (`dronevla/explore.py`: noisy expert,
random walk, overshoot past the goal; never Stop) over the *same* train layouts. Val is the
same in both runs; test was not touched.

| run | data | train layouts | train rows | train time |
|---|---|---:|---:|---:|
| wm_v0 | v0.1 + explore_v0.1 (8 flights/layout) | 20 | 10,101 | 709 s |
| wm_v1 | v0.2 + explore_v0.2 (4 flights/layout) | 100 | 30,448 | 2,164 s |

Dataset v0.2 continues the train seed range (1001-1100). Its val and test splits, and its first
20 train pairs, were checked byte-identical to v0.1: 4,101 images and actions compared. The
exploration val flights are identical too.

## Results (val, medians)
Reproduce: `python -m dronevla.world_model eval --run runs/wm_vN` and
`python scripts/wm_diagnostics.py --run runs/wm_vN --out reports/wm_vN_diagnostics.json`.
Numbers: `reports/wm_v{0,1}_{prediction,diagnostics}.json`.

### Perception: where are the targets, right now (k = 0)
| offset error, m | wm_v0 | wm_v1 |
|---|---:|---:|
| all colours, val demos | 0.556 | **0.301** |
| colour in view (±30° horizontal) | 0.465 | **0.250** |
| colour out of view | 1.141 | 0.775 |
| instructed target, all distances | 0.410 | **0.224** |
| instructed target, < 0.5 m from it (n = 31) | 0.324 (55% < 0.4 m) | **0.169 (84% < 0.4 m)** |
| instructed target, 0.5–1 m (n = 48) | 0.317 (69%) | **0.145 (92%)** |

wm_v0 had a train/val loss gap of 0.59 vs 1.33 (val best at epoch 6). It had memorised 20
scenes. With 100 scenes the gap is 0.61 vs 0.95, and every perception number roughly halves.
**The bottleneck was the number of distinct scenes, not the model.** The BC policy was trained
on the same 20 scenes.

### Dynamics: how the error grows over 2 s (10 steps, val demos)
| offset error, m | k=0 | k=5 (1 s) | k=10 (2 s) |
|---|---:|---:|---:|
| wm_v0 rollout | 0.556 | 0.586 | 0.632 |
| wm_v1 rollout | 0.301 | 0.323 | 0.395 |
| wm_v1 k=0 estimate + command integration | 0.301 | — | 0.298 |
| *privileged*: true k=0 + command integration | 0 | 0.023 | 0.033 |

Velocity error stays at 0.03–0.04 m/s. **The learned dynamics are worse than integrating the
commanded velocity**: 0.395 vs 0.298 at 2 s on demos, and 0.570 vs 0.341 on random-walk
flights, where the scene cannot predict the motion. In this simulator the PID tracks commands
closely, so an integrator is near-perfect dynamics. A learned model has to earn its place
through perception and planning, not by beating it.

### Does the prediction follow the action? (constant-command probe, 519 val states)
Slope of the predicted offset change against the commanded displacement (1.0 = exact).
Second value: median leak into the other axis.

| | +x | −x | +y | −y |
|---|---|---|---|---|
| wm_v1, 3 steps | 0.90 / 0.08 m | 1.38 / 0.22 m | 1.25 / 0.26 m | 0.33 / 0.17 m |
| wm_v1, 5 steps | 0.91 / 0.09 m | 1.97 / 0.23 m | 0.95 / 0.28 m | 0.58 / 0.28 m |

The signs are right from 3 steps on, but the response is biased: backward is over-predicted and
right is under-predicted. Slopes near 0 at 1 step are partly real, because the PID needs time
to accelerate. The commands seen in training are mostly forward, toward the targets, and this
probe exposes that. A sampling planner (CEM/MPPI) would exploit these biases.

## What this means for stages 2–3
- **Use** the world model's perception (its target offsets) as the planner's state estimate.
- **Do not** trust the learned dynamics for planning yet. Use a command integrator for the
  rollout, and report a pure-WM rollout planner as a separate variant.
- The BC baseline must be retrained on v0.2 (100 train layouts) before any comparison.
  Otherwise the comparison measures data, not method.
- Not yet done: multiple seeds (the stage-3 closed-loop comparison will use ≥ 3), test split.
