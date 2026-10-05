# Phase 1 — dataset v0.1

Recorded 2026-10-05. Roadmap v3 Phase 1 item 3 and §4.4-4.5: the first recorded dataset,
40 counterfactual pairs flown by the oracle expert.

```
python -m dronevla.record --out data/v0.1          # 188 s on the development laptop
python scripts/inspect_episode.py --episode val-004-g1
```

Images and parquet files are not in git; `data/v0.1/manifest.json`, `datasheet.md`,
`splits.json` and `validation.json` are. The manifest pins the code commit (`ded033e`,
clean), the simulator commit, the full `TaskConfig` and its hash, the camera intrinsics and
extrinsics, a SHA256 for every non-image file, and one tree hash over all 4101 images
(`8d4a2499eae4...`). Re-running the command on the same machine reproduces the images byte
for byte (tested on a small dataset in `tests/test_dataset.py`).

## What is in it

| split | pairs | episodes | seeds |
|---|---:|---:|---|
| train | 20 | 40 | 1001-1020 |
| val | 10 | 20 | 2001-2010 |
| test | 10 | 20 | 3001-3010 |

- 4101 observations, median 51 per episode (range 40-65); median episode 10.8 sim-s
- 0 candidates rejected; the expert succeeded on all 80 episodes
- `validate()`: 14 of 14 checks pass, from files read back off disk
- the full datasheet, with the balance tables, is `data/v0.1/datasheet.md`

**Against the plan.** §4.4 assumed 100 frames per episode and, for v0.1, 10-40 minutes and
0.06-0.15 GB. Phase 0 projected 0.011 GB and 5.8 minutes from its own measurements.
Measured: **51 frames per episode, 188 s, 6.44 MB of images** (1570 bytes per PNG -- a little
above Phase 0's 1421 bytes, because the targets now fill much more of the frame) and 18 MB
on disk with the parquet files and filesystem overhead.

## Label properties the next step has to deal with

Computed from the recorded steps (`action_mask` rows only, 4021 of them):

- **Stop is rare: 6.7% of action rows** (median 6.4% per episode). The roadmap already plans
  a class weight for the Stop BCE, computed on train only.
- **Stop labels flicker near the goal: 16 times in 14 of 80 episodes**, the expert's Stop went
  positive and then negative again before the triggering run of three. The expert says
  "stop" only when it is within 0.12 m *and* slower than 0.08 m/s, and noise-driven drift
  crosses that speed line. The state barely changes between those rows, so a classifier sees
  nearly identical inputs with opposite labels. It is a property of the teacher, not a
  recording error; the 3-in-a-row rule still triggers correctly in every episode.
- **75.7% of commanded speeds are at the cap** (0.499995 m/s). The motion target is mostly
  "full speed in a direction", with a short slow-down at the end.

## Balance

Reported, not enforced -- v0.1 samples layouts at random:

| goal colour | left | right |
|---|---:|---:|
| blue | 7 | 10 |
| green | 12 | 13 |
| red | 10 | 13 |
| yellow | 9 | 6 |

Yellow is the least frequent goal (15 of 80). Within val, cylinders outnumber boxes 12 to 8
while train goes the other way (16 to 24). Neither is large, and with 20-episode splits a
balanced design would be the job of the generator, not of post-hoc filtering.

## Problems found by building it

Each of these was caught by reading the output back, and each is now covered by a test.

1. **A good layout was rejected (visibility check rendered from inside the drone).** The first
   full recording discarded seed 3006 because "a target covers 0 px at the start", while the
   real first frame showed it at 346 px. The check rendered from the nominal start position,
   but estimate noise leaves the hovering drone a few centimetres away, so the hypothetical
   camera sat inside the real drone's arm and propeller, which filled the view. The start is
   now rendered from the pose of the first observation. With the fix, seed 3006 is test pair
   5 and both its targets are clearly visible.
2. **37% of the expert's actions were flagged as capped.** It commanded exactly 0.5 m/s; float32
   turned that into 0.5000001 and the adapter capped it. Harmless numerically, but it would have
   reported a 37% action-clipping rate (§7.1) for a teacher that never exceeds the cap. The
   expert now stays at 0.5 x (1 - 1e-5): 0 capped rows.
3. **The manifest called clean code dirty**, because the recorder's own untracked output under
   `data/` counted. `git status` now ignores untracked files for that flag.
4. **A null in a fixed-size list column does not round-trip through parquet** with pyarrow
   25.0.1: written as a zero-length entry, then refused on read. The action columns are
   variable-length lists and `validate()` checks their lengths instead.

Fixes 2 and 3 changed the expert's trajectories very slightly, so the final recording has
4101 observations where the first had 4143; every check passed on all three recordings.

## Manual alignment check (§5 Phase 1 item 3)

`scripts/inspect_episode.py` draws every frame of an episode with the commanded velocity as an
arrow, the time, the Stop flag and the true distance to the goal. Checked by the assistant on
train-000-g0, val-004-g1 and test-005-g1 (the formerly rejected layout): in each, the arrow on
frame t explains the change to frame t+1 (forward plus left makes the left target grow toward
the centre and the right one slide out), the distance falls monotonically to ~0.05-0.1 m, the
Stop run and its trigger are where the expert stopped, and the terminal row is exactly 1 s
after the trigger. These were looked at on earlier recordings of the same seeds; the sheets in
`results/inspect/` are regenerated from the final data. **The roadmap assigns this check to
the user, and that is still open.**

## Next

Phase 1 item 4: a <= 2M-parameter CNN + text behaviour-cloning policy, trained on CPU from
this dataset, with the Stop class weight from train only and an obs-only policy interface
for closed-loop evaluation (§7.1).
