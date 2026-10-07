| model | params | correct / inverted / same pairs | success /20 | pair success /10 | reached goal /20 | final dist (median) | first-frame rows changed by the sentence /200 |
|---|---:|---|---:|---:|---:|---:|---:|
| scripted expert (oracle) | — | 10 / 0 / 0 | 20 | 10 | 20 | 0.07 m | — |
| planner + true target position (oracle) | — | 10 / 0 / 0 | 20 | 10 | 20 | 0.05 m | — |
| BC baseline (regression) | 480901 | 0 / 0 / 10 | 3 | 0 | 6 | 1.38 m | 0 |
| + counterfactual relabelling | 480901 | 2 / 3 / 5 | 0 | 0 | 3 | 2.93 m | — |
| + counterfactual + FiLM | 492517 | 4 / 3 / 3 | 0 | 0 | 6 | 3.30 m | — |
| + target-position aux head | 493773 | 0 / 0 / 10 | 8 | 0 | 10 | 0.72 m | — |
| + cross-attention | 420997 | 0 / 0 / 10 | 0 | 0 | 3 | 2.33 m | — |
| + cross-attention + counterfactual | 420997 | 5 / 3 / 2 | 0 | 0 | 6 | 2.67 m | — |
| 256-bin action tokens | 469609 | 3 / 0 / 7 | 8 | 1 | 12 | 0.22 m | 74 |
| 64-bin action tokens | 471289 | 1 / 1 / 8 | 7 | 1 | 10 | 0.99 m | 62 |
| 1024-bin action tokens | 475609 | 2 / 0 / 8 | 4 | 0 | 12 | 0.85 m | 66 |
| 256-bin + paired goal loss | 480555 | 2 / 0 / 8 | 7 | 0 | 11 | 0.42 m | 82 |
| 256-bin + cross-attention | 477289 | 1 / 0 / 9 | 9 | 0 | 11 | 0.33 m | 102 |
| 256-bin + chunk 8 (execute 1) | 470633 | 4 / 1 / 5 | 8 | 3 | 13 | 0.29 m | 114 |
| 256-bin + chunk 8 (execute 8) | 470633 | 1 / 0 / 9 | 9 | 1 | 11 | 0.29 m | — |
| chunk 8 + exploration states | 470633 | 5 / 2 / 3 | 6 | 0 | 13 | 2.11 m | 174 |
| chunk 8 + exploration + aux head | 468909 | 5 / 2 / 3 | 2 | 0 | 10 | 1.60 m | 160 |
| world model + planner (integrator) | — | 10 / 0 / 0 | 15 | 6 | 20 | 0.21 m | — |
| world model + planner (learned dynamics) | — | 3 / 0 / 7 | 0 | 0 | 8 | 1.63 m | — |
