# Season 2 screen — can the drone be made to listen?

Recorded 2026-10-07. Dataset v0.2 (100 training pairs), validation split only (10 pairs = 20
episodes); test untouched. **One training seed per model** (seed 0, user decision), 20 epochs,
last epoch kept. Every learned BC variant is within 80–100% of the baseline's 480,901 parameters,
except the two regression variants trained before that rule existed (FiLM 102%, aux head 103%).

```
python scripts/season2_summary.py        # every number below, from the eval JSONs
```
Table: `reports/season2_summary.md` (+ `.json`). Figure: `reports/figures/season2/all_models_pairs.png`.

## Metrics
- **success** (/20): stop within 0.4 m of the named target's hover point, Stop three times in a
  row, then hold under 0.1 m/s for 1 s, no collision, within 30 s.
- **pair success** (/10): both sentences of the same layout succeed. A policy that ignores the
  words scores 0 here, however well it flies. This is the series' main metric.
- **correct / inverted / same** (/10 pairs): which target each sentence's flight came closest to.
  This is a behaviour proxy, not proof of grounding. A planner with an explicit selector still
  scores "same" when it fails to fly.
- **first-frame rows changed** (/200, training set): on each training episode's first frame,
  swapping only the sentence changes the lateral command by more than 0.02 m/s
  (`scripts/instruction_sensitivity.py`).

## Results

| model | params | correct / inverted / same | success /20 | pair success /10 | reached goal /20 | final dist (median) | first-frame rows changed /200 |
|---|---:|---|---:|---:|---:|---:|---:|
| scripted expert (oracle) | — | 10 / 0 / 0 | 20 | 10 | 20 | 0.07 m | — |
| planner + true target position (oracle) | — | 10 / 0 / 0 | 20 | 10 | 20 | 0.05 m | — |
| BC baseline (regression) | 480,901 | 0 / 0 / 10 | 3 | 0 | 6 | 1.38 m | 0 |
| + counterfactual relabelling | 480,901 | 2 / 3 / 5 | 0 | 0 | 3 | 2.93 m | — |
| + counterfactual + FiLM | 492,517 | 4 / 3 / 3 | 0 | 0 | 6 | 3.30 m | — |
| + target-position aux head | 493,773 | 0 / 0 / 10 | 8 | 0 | 10 | 0.72 m | — |
| + cross-attention | 420,997 | 0 / 0 / 10 | 0 | 0 | 3 | 2.33 m | — |
| + cross-attention + counterfactual | 420,997 | 5 / 3 / 2 | 0 | 0 | 6 | 2.67 m | — |
| 256-bin action tokens | 469,609 | 3 / 0 / 7 | 8 | 1 | 12 | 0.22 m | 74 |
| 64-bin action tokens | 471,289 | 1 / 1 / 8 | 7 | 1 | 10 | 0.99 m | 62 |
| 1024-bin action tokens | 475,609 | 2 / 0 / 8 | 4 | 0 | 12 | 0.85 m | 66 |
| 256-bin + paired goal loss | 480,555 | 2 / 0 / 8 | 7 | 0 | 11 | 0.42 m | 82 |
| 256-bin + cross-attention | 477,289 | 1 / 0 / 9 | 9 | 0 | 11 | 0.33 m | 102 |
| **256-bin + chunk 8 (execute 1)** | 470,633 | 4 / 1 / 5 | 8 | **3** | 13 | 0.29 m | 114 |
| 256-bin + chunk 8 (execute 8) | 470,633 | 1 / 0 / 9 | 9 | 1 | 11 | 0.29 m | — |
| chunk 8 + exploration states | 470,633 | 5 / 2 / 3 | 6 | 0 | 13 | 2.11 m | 174 |
| chunk 8 + exploration + aux head | 468,909 | 5 / 2 / 3 | 2 | 0 | 10 | 1.60 m | 160 |
| **world model + planner (integrator)** | 574,515 + selector | 10 / 0 / 0 | **15** | **6** | 20 | 0.21 m | — |
| world model + planner (learned dynamics) | same | 3 / 0 / 7 | 0 | 0 | 8 | 1.63 m | — |

The baseline is evaluated in several files. BC evaluation is deterministic, and the summary
script asserts that the repeats give identical outcomes.

## What the screen says

1. **Action tokens fix the flying.**
   - Discretising (vx, vy) into 256 bins with cross-entropy (OpenVLA-style) changes only the head
     and the loss.
   - Success rises 3 → 8 and the median final distance falls 1.38 → 0.22 m.
   - Attention goes from 0 → 9 successes with the same change.
   - Mechanism (measured on the baseline): at the first frame the regression head predicts the
     **midpoint** of the two sentences' labels; its training error is 0.10 m/s, half the expert's
     0.20 m/s gap. Over the 200 training first frames it is closer to its own sentence's answer
     in exactly 100 and to the partner's in 100, sitting on average 0.03 m/s from the midpoint.
   - The token head picks one of them: median t=0 training error 0.0005 m/s (mean 0.066). It
     lands closer to the right answer in 135/200 frames and to the partner's in 65/200, so it
     commits, though not always to the right answer.
     Source: `reports/figures/blog05/02_midpoint_vs_pick.json`.
   - So the regression policy hovers between pillars ("stop elsewhere" 8/20); the token policy
     commits.
2. **Tokens alone don't make it choose by the sentence.** 7–9 of 10 pairs still go to one target.
   Among the token variants, 256 bins was best. The bin-count differences are within single-seed
   noise.
3. **An 8-step action chunk, executed one step at a time, is the best end-to-end model:** pair
   success 3/10. Executing all 8 steps open-loop flies as well (9/20) but listens less (1 correct
   pair).
4. **Covering the world model's states helps the choosing, not the stopping.**
   - Adding the exploration states (19,896, relabelled by the oracle expert) brings the open-loop
     first-frame reaction to the expert's level: median 0.196 vs 0.200 m/s, right direction in
     180/200 rows.
   - In closed loop it gives 5 correct pairs and 0 wrong-target stops.
   - But 8/20 flights overshoot the right pillar and leave the room.
   - Candidate causes, not separated:
     - Stop supervision diluted: 6 Stop positives among the exploration rows; pos_weight was
       recomputed to 43.4.
     - The lateral bin range widened from ±0.35 to ±0.85 of the speed cap (about ±0.17 to ±0.42 m/s), so each bin is 2.4x coarser.
     - Overshoot recovery labels near the goal.
5. **Counterfactual relabelling makes the words matter but not reliably** (correct ≈ inverted),
   and flight collapses. Part of that collapse is a confound: the relabelled rows add zero Stop
   positives while pos_weight stayed at the original 13.9.
6. **The modular system is the only one that reliably listens** (10/10 correct, pair success 6/10).
   Its advantage is not from one thing:
   - explicit colour selection
   - 3x the states, through exploration
   - dense position supervision
   - a filter and a rule-based Stop
   
   Its remaining 5 failures happen after reaching the goal region (it reaches it 20/20). With the
   true target position the same controller succeeds 20/20. Planning with the *learned* dynamics
   fails (0/20, 13 timeouts): CEM exploits the model's biased predictions, and the model was
   trained for 4 steps but asked for 8.

## What this does not show

- **One training seed per model and n = 10 pairs.** Single-pair differences are noise. Pairs with
  a different target and correct direction (2/5, 4/7, 5/8 for the counterfactual variants) are
  not distinguishable from chance. Wilson 95% for 0/10 is [0, 0.28] and for 6/10 is [0.31, 0.83].
- **Validation was used for screening and for the Stop thresholds.** These are not
  generalisation estimates. The original test seeds were looked at in an earlier round, so final
  numbers need a fresh test split.
- **First-frame sensitivity is measured on training frames.** Near-zero t=0 training errors in the
  token models may partly be memorisation.
- **The language task is easy.** The two pillars always differ in colour, so one colour word
  decides the target.
- **The planner's CEM RNG differs between the two episodes of a pair.** BC evaluation is
  deterministic.
- **Simulation only.**

## Code added for this screen

- `dronevla/model.py`
  - `attn` (language-query cross-attention over 6x8 patches)
  - `action_bins` (OpenVLA-style tokens: 1st–99th percentile bins, greedy decoding)
  - `chunk` (shared step decoder, H future steps of tokens + Stop)
  - `paired_goal`
  - `img_dim` / `head_hidden` (for parameter parity)
- `dronevla/train.py`
  - `--attn`, `--action-bins`, `--chunk`, `--paired-goal`, `--explore`, `--img-dim`, `--head-hidden`
  - for exploration rows, chunk steps beyond the current one are masked
- `dronevla/evaluate.py`: `run#exec=N` executes N chunk steps per query.
- Scripts:
  - `scripts/run_candidates_s0.sh`, `run_tokens_s0.sh`, `run_tokens2_s0.sh`,
    `run_attn_tok_s0.sh`, `run_explore_s0.sh`
  - `scripts/attention_maps.py`, `candidates_table.py`, `season2_summary.py`
- Tests: `tests/test_model_attn.py`, `tests/test_action_tokens.py` (parameter parity, token round
  trip, checkpoint round trip, chunk shapes).

## Independent review

Codex reviewed the first half of the screen (`codex_review.md` in the OneDrive review folder).
It confirmed:
- the Stop-balance confound in the counterfactual runs
- the planner RNG difference between pair members
- that "same target" is not equivalent to "word-blind"
- the world model's extra data and supervision

Those points are reflected in the wording above.
