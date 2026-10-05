# Phase 1 — the first learned policy, and why it ignores language

Recorded 2026-10-05. Roadmap v3 Phase 1 items 4-6: a tiny RGB + text behaviour-cloning
policy trained on CPU from dataset v0.1, compared in closed loop with a no-language
baseline and the scripted expert.

**Headline: the language policy does not use the instruction.** On both held-out splits,
and even on the training pairs, the two instructions of a pair almost always send it to the
same target. Its success rate is higher than the no-language baseline's, but that gap is
inside the confidence intervals and is not evidence of grounding. The cause is diagnosed
below and points at a specific next experiment.

```
python -m dronevla.train --out runs/bc_text                     # 187 s
python -m dronevla.train --out runs/bc_nolang --no-language     # 196 s
python -m dronevla.evaluate --split test --policy expert --policy runs/bc_text \
    --policy runs/bc_nolang --out reports/eval_v0.1_test.json
```

## Model and training

`dronevla/model.py`, as §3.1 specifies: a 4-layer CNN keeping a 6x8 spatial map, a
train-vocabulary embedding mean-pooled over tokens, an MLP on the 11 proprio values,
concatenated into an MLP that outputs 4 motion values (tanh, scaled by the §4.3 caps) and a
Stop logit. **480,901 parameters** (budget 2M). No privileged input.

`dronevla/train.py`: Huber loss on cap-scaled motion plus BCE on Stop with
`pos_weight = 13.9` (135 Stop-positive rows of 2015, train only); proprio mean/std from train;
Adam 1e-3, batch 16, 40 epochs, best epoch by validation loss; Stop threshold chosen on
validation. The no-language baseline is the same network trained separately with every
instruction replaced by one constant token (§7.2).

**Pilot (item 5):** 100 minibatches, 24.5 ms per step (p95 29.8), 3.2 s per epoch, 572 MB
peak RSS, 8 torch threads. The projected 2.1 min for 40 epochs became 187-196 s in practice,
because every epoch also runs a validation pass. The roadmap's 2-hour limit is far away.

## Debugging on one memorised pair first (item 4)

Trained and validated on a single train pair for 200 epochs: losses near zero, horizontal
RMSE 0.008 m/s, Stop separable (F1 1.0). In closed loop on that same pair it still failed one
of two episodes -- **which exposed a bug**. The Stop threshold was the lowest logit that
maximised F1, i.e. exactly the smallest positive logit (8.53). At its goal the policy produced
7.24, 9.71, 8.77, missed the three-in-a-row, drifted past the goal into states it had never
seen, and flew into the target. Thresholds are now midpoints between consecutive logits,
widest gap on ties (here 1.33); a test pins it. After the fix the collision is gone, and the
remaining failure is `stop_not_settled`: the policy signals Stop while still moving, where the
expert only stops below 0.08 m/s -- with 6 Stop examples it never learned that timing. That is
a policy-quality issue, not a pipeline bug, and was left alone.

## Closed loop on held-out layouts (item 6)

20 episodes = 10 counterfactual pairs per split, identical scenarios for every policy.
Wilson 95% intervals in brackets.

| split | policy | success | pair success | wrong target | out of bounds | other failures | reached goal (OSR) |
|---|---|---:|---:|---:|---:|---:|---:|
| val | expert (oracle) | 20/20 [0.84, 1] | 10/10 | 0 | 0 | 0 | 1.00 |
| val | RGB+text BC | 7/20 [0.18, 0.57] | **0/10** [0, 0.28] | 5 | 8 | 0 | 0.35 |
| val | no-language BC | 3/20 [0.05, 0.36] | 0/10 | 4 | 10 | 3 | 0.20 |
| test | expert (oracle) | 20/20 [0.84, 1] | 10/10 | 0 | 0 | 0 | 1.00 |
| test | RGB+text BC | 8/20 [0.22, 0.61] | **0/10** [0, 0.28] | 5 | 6 | 1 | 0.40 |
| test | no-language BC | 4/20 [0.08, 0.42] | 0/10 | 5 | 4 | 7 | 0.40 |

Policy latency (one forward pass, CPU, batch 1): p50 2.8 ms, p95 3.5-4.0 ms. Figures:
`reports/eval_v0.1_val.png`, `reports/eval_v0.1_test.png` -- the expert's two paths per pair
split apart; the text policy's two paths lie on top of each other.

**Pair success is 0/10 for both learned policies.** With a 40% episode success rate that
means at most one instruction of any pair succeeded: the behaviour of a policy that picks one
target per scene, whatever it is told.

The other big failure is **flying out of the room**: offline the Stop head reaches F1 0.85 on
validation rows, but in closed loop it often does not fire, and the drone overshoots its goal
and keeps going.

## Diagnosis: the image is a shortcut, so language is never needed

**1. It is not a generalisation gap.** Closed loop on the 20 *training* pairs: 10/40 success,
0/20 pairs; the two instructions led to different targets in **2 of 20** training pairs.

| | pairs where the two instructions led to different targets |
|---|---:|
| expert | val 10/10, test 10/10 |
| RGB+text BC | val 2/10, test 1/10, train 2/20 |
| no-language BC | 0 everywhere (by construction) |

**2. Open loop, the instruction barely moves the output.** On each training pair's identical
first frame, swapping only the instruction:

```
expert lateral speed differs by     median 0.191 m/s  (0.068-0.284)
text BC lateral speed differs by    median 0.008 m/s  (0.001-0.017)   -- 24x smaller
both lateral directions correct:    3 of 20 training pairs
later in the episode (t >= 10):     swapping the instruction changes vy by 0.005 m/s
```

**3. Why.** Within an episode, the image soon tells you where the drone is heading: once it
has turned toward one target, every later frame implies that target. Only the first few steps
genuinely need the instruction -- the first five steps are 200 of the 2015 training rows, about
10%. Behaviour cloning minimises the loss on all rows equally, so the cheapest solution is to
read the goal off the image and leave the text pathway nearly unused. In closed loop the first
decision is then made without the instruction, and every following frame confirms whichever
way the drone happened to lean.

**4. Was it simply undertrained?** The kept model is epoch 17 of 40, chosen by validation
loss, so it might have stopped before learning to use the words. Tested by training 150
epochs and keeping the last one (`--select last`, `runs/bc_text_long`, 738 s). Lateral-command
error on the *training* rows, by step:

| step | epoch 17 (kept) | epoch 150 |
|---|---:|---:|
| **t = 0** -- identical frame for both instructions; only the words decide | 0.090 m/s | **0.084 m/s** |
| t = 1-4 | 0.014 | 0.006 |
| t = 5-9 | 0.020 | 0.004 |
| t >= 10 | 0.014 | 0.005 |

Longer training memorised every row the image can explain down to ~5 mm/s, and left t = 0
where it was: about half the expert's 0.191 m/s instruction gap, which is what predicting the
*average* of the two instructions' labels gives. Those 40 rows (2% of the data) are
essentially all of the remaining training loss (estimated 0.0026 against a measured 0.002).
Swapping the instruction on the first frame moved the output 0.014 m/s (was 0.008); in closed
loop the two instructions still led to different targets in only 2 of 20 training pairs; and
the extra epochs overfit -- validation loss rose from 0.22 to 2.17 and test success fell to
5/20. So it is not a matter of training longer. The rows that need language are too few for
the loss to care about once the image explains the rest.

This is the failure the project's counterfactual pairs were meant to expose, and it shows up
in the first experiment. It also means the higher text-policy success rate (8/20 vs 4/20 on
test) should not be read as a language effect: the intervals overlap and the pair metric is
identical.

## Smaller observations

- **65-70% of the learned policies' actions are capped.** The motion head is a per-axis tanh
  scaled by the caps, so forward 0.5 plus any lateral component exceeds the 0.5 m/s norm and
  the adapter scales it down. Harmless, but it inflates §7.1's clipping metric for a reason
  that is the head's parametrisation, not the policy's intent.
- The expert's Stop-label flicker (`docs/phase1_dataset.md`) and the 6.7% Stop rate both
  surface here as a Stop head that is fine offline and unreliable in closed loop.

## What to try next

Ordered by how directly each one attacks the diagnosed cause:

1. **Counterfactual relabelling (recommended).** For every recorded state, add a second
   training sample with the *other* instruction and the oracle expert's action toward the
   *other* goal, computed from the logged true pose. Then the same image comes with two
   different labels depending on the words, at every step and not just the first five, so the
   shortcut stops working. It needs no new simulation, uses the expert the dataset already
   relies on, and is a direct application of the project's counterfactual-pair idea to
   training rather than only to evaluation.
2. **Stronger text conditioning:** FiLM, i.e. the instruction scaling and shifting CNN channels,
   instead of concatenating a 32-number summary at the end.
3. **Up-weighting the early steps** where the decision is made.

Each would be judged by the same closed-loop pair metric on the same test pairs.

## Phase 1 exit criteria (§5) -- status

| criterion | status |
|---|---|
| three baselines run on the fixed test rollouts, results and failure reasons saved | **met** -- `reports/eval_v0.1_{val,test}.json` |
| instruction and RGB pass through the API | met -- obs-only `LearnedPolicy.act(obs)`, tested |
| invalid timestamp / action shape rejected | met -- env `invalid_action`, dataset validator |
| evaluator fixtures for success / wrong target / collision / timeout pass | met -- `tests/test_env.py` |
| high success rate | **not an exit condition** (§5 says so explicitly) |
| localhost service in the loop | not done |
| paired rollout video | not done (the trajectory figures stand in for now) |
| C1 C++ core used by the Python loop | not done |
