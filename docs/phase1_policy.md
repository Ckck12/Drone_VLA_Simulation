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

**Videos** (`scripts/rollout_video.py`, animated GIF, real time; val pair 0, since test is
reserved for final reporting): `reports/videos/expert_val_pair0.gif` and
`reports/videos/bc_text_val_pair0.gif`. In the second, both instructions command the same
first action, v = (+0.50, -0.13) m/s -- rightward, toward the green cylinder -- including
"Approach the blue box", whose box is on the left; the blue episode then leaves the room.

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
*average* of the two instructions' labels gives. Measured exactly on the training rows with the
training loss definitions: those 40 rows are 2% of the data but **36% of the remaining
training loss** (0.0017 in total), with a per-row motion loss 32x that of the other rows. (A
first version of this paragraph estimated them at "essentially all" of the remaining loss; the
estimate had forgotten that the motion loss is averaged over four action dimensions.)

The same measurement on the kept epoch-17 model shows something else: its training loss of
0.100 is **94% Stop BCE** and only 6% motion, and the t = 0 rows are 0.9% of it. At the point
where validation loss selected the model, the objective was almost entirely about *when to
stop*; *where to go* -- the only part language could change -- barely registered.
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

## Experiments on the shortcut (2026-10-05) -- and what was wrong with how they were run

### Experiment 1: counterfactual relabelling (data only)

`--counterfactual`: every training state also gets the *other* instruction and the oracle
expert's action toward the *other* goal, recomputed from the logged true position and the
velocity rebuilt from the observation. Model, loss, Stop pos_weight (13.9, from the original
rows) and selection rule unchanged. Check first: recomputing toward the episode's *own* goal
reproduced every recorded action (max error 7e-9 m/s, Stop agreement 100%), so the
reconstruction is exact. 1.5% of the counterfactual labels (30 of 2015) point along a straight
line that passes within 0.35 m of a target -- small, not zero.

It **did make the policy use the words**: first-frame lateral gap when only the instruction
changes rose from 0.008 to 0.073 m/s on train pairs (expert 0.191); in closed loop on val the
two instructions led to different targets in 5 of 10 pairs (closest-approach scoring), against
2 of 10 for the baseline. It **broke the flying**: val success 7/20 -> 0/20, mostly out of
bounds and collisions. Mechanism, measured on val rows: within 0.3 m of the goal the expert
flies at a median 0.105 m/s, the baseline 0.053, the relabelled policy **0.210** -- the same
image now carries "stop here" and "full speed to the other one", and a policy that separates
them only partly by the words outputs something in between.

### Experiment 2: + FiLM (architecture), inconclusive

`--film`: the instruction scales and shifts every conv layer's channels (zero-initialised, so
it starts as exactly the plain network -- tested; 492,517 parameters). Near-goal speed came
back to 0.113 m/s (expert 0.105), and val pairs with different targets were 6 of 10. Val
success stayed 0/20, with 8 stops in the wrong place. **But validation loss selected epoch 5**,
so this model was barely trained, and the comparison with Experiment 1 (epoch 15) and the
baseline (epoch 17) says little.

### Problems with the protocol, stated plainly

1. **Test was used for decisions.** Every iteration was scored on the 10 test pairs and the
   next step followed from it. With 10 pairs that is a real leak. From here, test is frozen;
   decisions use val, and if 10 val pairs are too noisy, a separate dev split from a fresh seed
   range (4000+). The test numbers in this file stay as recorded, with that caveat.
2. **Single seeds.** A 480k-parameter model on 20 training pairs; seed variance is probably as
   large as the 2-vs-5-vs-6 differences above. Each configuration needs at least three seeds,
   reported as mean and range.
3. **The selection rule favours early epochs.** Validation loss is dominated by Stop BCE, so it
   picks "before the Stop head overfits", not "best policy" -- epoch 5, 15 and 17 above. One
   rule for every configuration: a fixed epoch count, or validation motion loss only, with the
   Stop threshold tuned on val in closed loop under the three-in-a-row rule.
4. **"Which target did it choose" was scored by final position**, which misreads episodes that
   end out of bounds. It is now scored by closest approach over the whole path.

The fixed comparison is baseline vs relabelling vs relabelling + FiLM, three seeds each, one
selection rule, decided on val -- roughly 1-1.5 hours of CPU. Whether to run it is open.

This is roadmap **Phase 3** work (grounding and generalisation). Phase 1's exit criteria were
met before these experiments.

## Phase 1 exit criteria (§5) -- status

| criterion | status |
|---|---|
| three baselines run on the fixed test rollouts, results and failure reasons saved | **met** -- `reports/eval_v0.1_{val,test}.json` |
| instruction and RGB pass through the API | met -- obs-only `LearnedPolicy.act(obs)`, tested |
| invalid timestamp / action shape rejected | met -- env `invalid_action`, dataset validator |
| evaluator fixtures for success / wrong target / collision / timeout pass | met -- `tests/test_env.py` |
| high success rate | **not an exit condition** (§5 says so explicitly) |
| localhost service in the loop | not done |
| paired rollout video | **met** -- `reports/videos/*_val_pair0.gif`, any pair via `scripts/rollout_video.py` |
| C1 C++ core used by the Python loop | not done |
