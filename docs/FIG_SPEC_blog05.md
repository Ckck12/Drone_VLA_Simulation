# Figure specs: blog part 5 (borrowing from VLAs)

These follow the paper-writing super prompt: PART 4.3 (method figures), 4.6 (spec fields),
4.7 (captions) and 4.8 (readability and honesty).
- Style: `scripts/figures/figstyle.py`, copied from the driving-VLA project's `ibdstyle.py`.
- Colour roles:
  - Okabe–Ito
  - **gray** = unchanged from Part 4
  - **orange** = changed in this post
  - **blue** = inputs
  - **green** = the chunk addition
- Encoding: line style and labels always carry the same distinction, so no meaning rests on
  colour alone.

All figures are produced by `scripts/figures/blog05.py`. Measured data come from the reports
named below. Panels without data are schematics and are labelled as such.

## F5-1 Method: same backbone, three heads (schematic)

| field | content |
|---|---|
| claim | Only the action head and its loss change between Part 4 and this post; the backbone, the inputs and the data are the same. |
| one-sentence message | A regression head outputs one number per axis. A token head outputs a probability for each of 256 speed bins. A chunk head does that for the next 8 steps at once. |
| reader question | What exactly changed, and what is trained against what? |
| data | None (schematic). The parameter counts come from `reports/season2_summary.json`. |
| panels | (left) inputs; (middle) shared backbone, gray; (right) three heads stacked: (a) regression, (b) 256-bin tokens, (c) 8-step token chunk, each with its loss and decoding rule |
| panel order | Information flows left to right. The heads are ordered as the post introduces them. |
| comparison unit | Same backbone and the same fused 168-d feature for every head. |
| colour / labels | Gray = unchanged; orange/green = the head that differs. The italic "train:" line in each head names its loss. The image FC is narrowed 128 → 104 in (b) and (c) for parameter parity; this is marked in orange. |
| uncertainty | Not applicable (schematic). |
| caption draft | **Same backbone, three ways to output an action.** The image, the sentence and the state are encoded exactly as in Part 4 (gray). (a) Part 4 regresses vx, vy with a tanh and a Huber loss. (b) Action tokens: each axis becomes 256 bins spanning the 1st–99th percentile of the training actions; the head is trained with cross-entropy and decoded by taking the most likely bin. (c) Chunk: one trunk plus a learned embedding per future step predicts the bins and Stop for the next 8 steps; only the first step is executed, and the model is queried again 0.2 s later. Parameter counts are within 80–100% of Part 4's. |
| possible misreading | That the model became a VLA. It has no pretrained vision-language backbone; only the *output format* is borrowed. The caption says so. |

## F5-2 Midpoint versus pick (measured)

| field | content |
|---|---|
| claim | On the shared first frame of a pair, the regression head predicts the average of the two sentences' answers, while the token head assigns its probability to one of them. |
| data | `runs/bc_v0.2_s0`, `runs/bc_tok256_v0.2_s0`, training pairs, first frame. Recomputed by `blog05.py`. |
| panels | One column per example pair. Top row: the regression output (one marker per sentence) against the two expert labels. Bottom row: the token probabilities over the vy bins for each sentence. |
| comparison unit | The same frame, the same state, two sentences. |
| axes | vy in m/s, with the same range in every panel. |
| possible misreading | That this holds for every frame, or that tokens pick the *right* answer. The panels show the first four training pairs in order (not selected). Aggregate over all 200 first frames, from `02_midpoint_vs_pick.json`: regression is closer to its own answer in 100/200 and to the partner's in 100/200; tokens 135 and 65, with a median error of 0.0005 m/s and a mean of 0.066. Training pair 0, sentence B (green) is a wrong pick: most of its mass sits on the red answer. Training pair 1, sentence A is split across three bins, though its argmax is right. Pair numbering is 0-based throughout, matching `--pair` and the GIF headers. Rendered as mirrored stems: sentence A up, sentence B down. |

## F5-3 Attention maps (measured)
Two frames: attention-only regression versus attention with tokens, the same first frames, both
sentences. Head-averaged weights, normalised per map; the caption says so. Attention maps are not
causal explanations (Jain & Wallace, 2019), so the text pairs them with the action-swap measurement.

## F5-4 Scoreboard (measured)
End-to-end models plus the planner as a reference. The data come from `reports/season2_summary.json`.
Single seed and n = 10 pairs; the caption gives Wilson intervals for 0/10 and 3/10.

## F5-5 Top-view paths and GIF (measured)
- Val pairs 0–2 (not selected) for the baseline, 256-bin tokens and chunk 8.
  - On these three pairs chunk 8 looks *worse* than tokens. The caption must say so, and point
    to the aggregate (3 vs 1 pair successes, n = 10).
  - The dotted goal circle sits in front of each pillar because the hover point is offset
    toward the start.
- GIFs (`pair7_*.gif`): val pair 7, chosen because chunk 8 succeeds on both sentences there; the
  caption states that selection rule.
  - On the same pair the baseline is stop elsewhere ×2 and tokens are stop not settled ×2
    (checked against the eval JSONs).
  - Render with `scripts/figures/render_blog05_videos.sh`.

## F5-6 Reaction versus stopping (measured)
- Left: first-frame swap reaction for baseline → tokens → chunk → chunk + exploration, with the
  expert's level drawn as a line.
- Right: outcome counts for the same models.
- Message: exploration fixes the choice but breaks the stop.
