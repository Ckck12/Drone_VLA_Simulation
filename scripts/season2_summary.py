"""One table and one figure for every Season 2 screen (v0.2, val, single training seed).

Reads every eval JSON of the screen, de-duplicates policies evaluated more than once (BC
evaluation is deterministic, so repeats must agree -- checked), and joins the open-loop
instruction-swap measurements.

    python scripts/season2_summary.py      # -> reports/season2_summary.{json,md}, figures/season2/
"""
import collections
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parent.parent
R = REPO / "reports"
DATA = REPO / "data" / "v0.2"
EVALS = ["eval_candidates_s0_val.json", "eval_tokens_s0_val.json", "eval_tokens2_s0_val.json",
         "eval_attn_tok_s0_val.json", "eval_explore_s0_val.json"]
SENS = ["instruction_sensitivity_v0.2_train.json", "instruction_sensitivity_tok256_train.json",
        "instruction_sensitivity_tokens2_train.json", "instruction_sensitivity_attn_tok_train.json",
        "instruction_sensitivity_explore_train.json"]

# display order, label, group
ROWS = [
    ("expert (oracle)", "scripted expert (oracle)", "reference"),
    ("plan_oracle (true goal offset, oracle)", "planner + true target position (oracle)", "reference"),
    ("bc_v0.2_s0 (RGB+text BC)", "BC baseline (regression)", "regression BC"),
    ("bc_cf_v0.2_s0 (RGB+text BC)", "+ counterfactual relabelling", "regression BC"),
    ("bc_cf_film_v0.2_s0 (RGB+text BC, FiLM)", "+ counterfactual + FiLM", "regression BC"),
    ("bc_aux_v0.2_s0 (RGB+text BC)", "+ target-position aux head", "regression BC"),
    ("bc_attn_v0.2_s0 (RGB+text BC, attention)", "+ cross-attention", "regression BC"),
    ("bc_attn_cf_v0.2_s0 (RGB+text BC, attention)", "+ cross-attention + counterfactual", "regression BC"),
    ("bc_tok256_v0.2_s0 (RGB+text BC, 256-bin tokens)", "256-bin action tokens", "token BC"),
    ("bc_tok64_v0.2_s0 (RGB+text BC, 64-bin tokens)", "64-bin action tokens", "token BC"),
    ("bc_tok1024_v0.2_s0 (RGB+text BC, 1024-bin tokens)", "1024-bin action tokens", "token BC"),
    ("bc_tokpaired_v0.2_s0 (RGB+text BC, paired goal, 256-bin tokens)", "256-bin + paired goal loss", "token BC"),
    ("bc_attn_tok256_v0.2_s0 (RGB+text BC, attention)", "256-bin + cross-attention", "token BC"),
    ("bc_tokchunk8_v0.2_s0 (RGB+text BC, 256-bin tokens, chunk 8, exec 1)", "256-bin + chunk 8 (execute 1)", "token BC"),
    ("bc_tokchunk8_v0.2_s0 (RGB+text BC, 256-bin tokens, chunk 8, exec 8)", "256-bin + chunk 8 (execute 8)", "token BC"),
    ("bc_tokchunk8_explore_v0.2_s0 (RGB+text BC, 256-bin tokens, chunk 8, exec 1)", "chunk 8 + exploration states", "token BC"),
    ("bc_tokchunk8_explore_aux_v0.2_s0 (RGB+text BC, 256-bin tokens, chunk 8, exec 1)", "chunk 8 + exploration + aux head", "token BC"),
    ("plan_integrator_wm_v1_s0 (WM planner)", "world model + planner (integrator)", "modular"),
    ("plan_wm_wm_v1_s0 (WM planner)", "world model + planner (learned dynamics)", "modular"),
]


def main():
    eps = {e["episode_id"]: e for e in pq.read_table(DATA / "episodes.parquet").to_pylist()}
    lays = {}

    def lay(i):
        if i not in lays:
            lays[i] = json.loads((DATA / "layouts" / f"{i}.json").read_text())
        return lays[i]

    pols, seen = {}, {}
    for f in EVALS:
        for name, p in json.loads((R / f).read_text())["policies"].items():
            sig = [t["outcome"] for t in p["trials"]]
            if name in seen:
                assert seen[name] == sig, f"non-deterministic repeat: {name}"
                continue
            seen[name] = sig
            pols[name] = p
    sens = {}
    for f in SENS:
        for k, p in json.loads((R / f).read_text())["policies"].items():
            g = np.asarray(p["first_frame"]["policy_gap"])
            sens[k] = {"first_frame_changed_rows": int((g > 0.02).sum()),
                       "first_frame_mean_gap_mps": float(g.mean()),
                       "first_frame_direction_correct": p["first_frame_direction_correct"]}

    out = []
    for key, label, group in ROWS:
        p = pols[key]
        tr = p["trials"]
        by_pair = collections.defaultdict(dict)
        for t in tr:
            L = lay(eps[t["episode_id"]]["layout_id"])
            path = np.asarray(t["path"])[:, :2]
            hp = np.asarray(L["hover_points"])[:, :2]
            chosen = int(np.argmin([np.min(np.linalg.norm(path - h, axis=1)) for h in hp]))
            by_pair[t["pair_id"]][t["goal"]] = (chosen, t["outcome"])
        c = collections.Counter()
        for g in by_pair.values():
            c["correct" if (g[0][0], g[1][0]) == (0, 1) else
              "inverted" if (g[0][0], g[1][0]) == (1, 0) else "same"] += 1
            c["pair_success"] += g[0][1] == "success" and g[1][1] == "success"
        cfg = p.get("policy_config") or {}
        run = key.split(" (")[0]
        out.append({
            "label": label, "group": group, "policy": key,
            "parameters": cfg.get("parameters"),
            "success": sum(t["outcome"] == "success" for t in tr),
            "pair_success": c["pair_success"],
            "pairs_correct": c["correct"], "pairs_inverted": c["inverted"], "pairs_same_target": c["same"],
            "reached_goal": int(sum(t["reached_goal"] for t in tr)),
            "final_dist_median_m": float(np.median([t["final_dist_to_goal"] for t in tr])),
            "outcomes": dict(collections.Counter(t["outcome"] for t in tr)),
            **(sens.get(run, {}) if "exec 8" not in key else {}),
        })
    (R / "season2_summary.json").write_text(json.dumps(out, indent=1))

    lines = ["| model | params | correct / inverted / same pairs | success /20 | pair success /10 | "
             "reached goal /20 | final dist (median) | first-frame rows changed by the sentence /200 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for r in out:
        ch = r.get("first_frame_changed_rows")
        lines.append(f"| {r['label']} | {r['parameters'] or '—'} | {r['pairs_correct']} / "
                     f"{r['pairs_inverted']} / {r['pairs_same_target']} | {r['success']} | "
                     f"{r['pair_success']} | {r['reached_goal']} | {r['final_dist_median_m']:.2f} m | "
                     f"{'—' if ch is None else ch} |")
    (R / "season2_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # figure: pair breakdown + success / pair success, grouped
    od = REPO / "reports" / "figures" / "season2"
    od.mkdir(parents=True, exist_ok=True)
    rows = out[::-1]
    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    for y, r in enumerate(rows):
        x = 0
        for k, col in [("pairs_correct", "#1baf7a"), ("pairs_inverted", "#eb6834"),
                       ("pairs_same_target", "#c3c2b7")]:
            w = r[k]
            if w:
                ax.barh(y, w - 0.06, left=x, height=0.62, color=col)
                ax.text(x + w / 2, y, str(w), ha="center", va="center", fontsize=8,
                        color="white" if col != "#c3c2b7" else "#0b0b0b")
            x += w
        ax.text(10.4, y, f"{r['success']}/20", va="center", fontsize=8.5)
        ax.text(11.8, y, f"{r['pair_success']}/10", va="center", fontsize=8.5,
                fontweight="bold" if r["pair_success"] >= 3 else "normal")
    ax.text(10.4, len(rows) - 0.3, "success", fontsize=8.5, color="#52514e")
    ax.text(11.8, len(rows) - 0.3, "pair success", fontsize=8.5, color="#52514e")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=8.5)
    prev = None
    for y, r in enumerate(rows):
        if prev is not None and r["group"] != prev:
            ax.axhline(y - 0.5, color="#e1e0d9", lw=1)
        prev = r["group"]
    ax.set_xlim(0, 13.3)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("validation pairs (10): which target did each sentence lead to (closest approach)")
    from matplotlib.patches import Rectangle
    h = [Rectangle((0, 0), 1, 1, color=c) for c in ("#1baf7a", "#eb6834", "#c3c2b7")]
    ax.legend(h, ["each sentence → its own target", "inverted", "both → same target"],
              ncol=3, frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.4, -0.07))
    ax.set_title("Season 2 screen, dataset v0.2, single training seed", loc="left", fontsize=10.5)
    fig.savefig(od / "all_models_pairs.png", dpi=200, bbox_inches="tight")
    print("wrote", od / "all_models_pairs.png")


if __name__ == "__main__":
    main()
