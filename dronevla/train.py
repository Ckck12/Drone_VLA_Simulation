"""Train the tiny behaviour-cloning policy on a recorded dataset (roadmap v3 Phase 1 item 4-5).

    python -m dronevla.train --data data/v0.1 --out runs/bc_text
    python -m dronevla.train --data data/v0.1 --out runs/bc_nolang --no-language
    python -m dronevla.train --data data/v0.1 --out runs/overfit --overfit-pair 0
    python -m dronevla.train --data data/v0.1 --out runs/pilot --pilot

Rules kept from the roadmap:
* vocabulary, proprio mean/std and the Stop class weight come from **train only** (§4.3,
  Phase 1 item 4); the Stop threshold is chosen on **val only** (§4.3)
* motion: Huber loss on actions scaled by the §4.3 caps; Stop: BCE with pos_weight
* `--no-language` replaces every instruction with the same constant token and is trained
  separately with the same architecture and budget (§7.2 "No-language BC")
* `--overfit-pair` trains and validates on one train pair only, to debug the training path
  (Phase 1 item 4). Its result is never a generalisation number.
* `--pilot` times 100 minibatches and extrapolates the full run (Phase 1 item 5)
* `--aux-offsets` adds the world model's auxiliary target (body-frame offset to every colour's
  hover point, from logged true positions) as a second loss with weight 1, so BC gets the same
  privileged supervision as the world model in the BC-vs-planner comparison
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import resource
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

from dronevla.dataset import load_episodes, load_steps, sha256_file
from dronevla.model import TinyBC, Vocab, count_parameters, pad_batch
from dronevla.profile_env import git_info, proc_status_kb, stats

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PROPRIO_STD_FLOOR = 1e-2   # features that barely vary on train (e.g. sin yaw at fixed yaw)


def load_rows(root, episodes):
    """Every action row of the given episodes, images decoded into memory, plus the
    privileged body-frame offsets to every colour's hover point (for `--aux-offsets`)."""
    from dronevla.world_model import body_offsets
    rgb, proprio, action, stop, instr, eid, tpos = [], [], [], [], [], [], []
    offs, masks = [], []
    for e in episodes:
        lay = json.loads((root / "layouts" / f"{e['layout_id']}.json").read_text())
        n0 = len(tpos)
        for r in load_steps(root, e["episode_id"]):
            if not r["action_mask"]:
                continue
            rgb.append(np.asarray(Image.open(root / r["rgb_path"])))
            proprio.append(r["proprio"])
            action.append(r["action_applied"])
            stop.append(1.0 if r["stop_positive"] else 0.0)
            instr.append(e["instruction"])
            eid.append(e["episode_id"])
            tpos.append(r["priv_true_pos"])
        p = np.asarray(proprio[n0:], dtype=np.float64)
        o, m = body_offsets(lay, np.asarray(tpos[n0:]), np.arctan2(p[:, 5], p[:, 6]))
        offs.append(o.reshape(len(o), -1))
        masks.append(np.repeat(m, 2, axis=1))
    return {"rgb": torch.from_numpy(np.stack(rgb)),
            "proprio": torch.tensor(proprio, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "stop": torch.tensor(stop, dtype=torch.float32),
            "instr": instr, "episode_id": eid,
            "priv_true_pos": np.asarray(tpos, dtype=np.float64),
            "offsets": torch.from_numpy(np.concatenate(offs)).float(),
            "offset_mask": torch.from_numpy(np.concatenate(masks)).float()}


def expert_label(expert, true_pos, proprio, goal_hover_point):
    """What the oracle expert would command from a recorded state toward a given goal.

    The expert needs the true position (logged as `priv_true_pos`), the world-frame velocity
    and yaw. The velocity is rebuilt from the observation itself: body-frame velocity rotated
    by roll, pitch and yaw (PyBullet's convention, R = Rz(yaw) Ry(pitch) Rx(roll)).
    """
    from scipy.spatial.transform import Rotation
    p = np.asarray(proprio, dtype=np.float64)
    yaw = math.atan2(p[5], p[6])
    v_world = Rotation.from_euler("xyz", [p[3], p[4], yaw]).apply(p[0:3])
    a = expert.act({"privileged": {"true_pos": list(true_pos), "true_vel": list(v_world),
                                   "true_yaw": yaw, "goal_hover_point": list(goal_hover_point)}})
    return a[:4].astype(np.float32), float(a[4] > 0)


def counterfactual_relabel(root, episodes, data, cfg):
    """Counterfactual relabelling (docs/phase1_policy.md, "What to try next" 1).

    For every recorded state, add a sample with the *other* target's instruction and the
    action the oracle expert would take from that state toward the *other* goal. The same
    image then carries two different labels depending on the words, at every step rather than
    only at t = 0, so the image alone can no longer explain the label.

    Returns (augmented rows, consistency report). The report recomputes the label toward the
    episode's *own* goal and compares it with the recorded action: if that does not match, the
    state reconstruction is wrong and so would be every counterfactual label.
    """
    from dronevla.expert import StraightLineExpert
    from dronevla.task import Target, make_instruction
    expert = StraightLineExpert(cfg)
    lay = {e["episode_id"]: json.loads((root / "layouts" / f"{e['layout_id']}.json").read_text())
           for e in episodes}
    meta = {e["episode_id"]: e for e in episodes}
    cf_action, cf_stop, cf_instr = [], [], []
    own_err, own_stop_agree = [], []
    for i, eid in enumerate(data["episode_id"]):
        e, L = meta[eid], lay[eid]
        g = e["goal_index"]
        state = (data["priv_true_pos"][i], data["proprio"][i].numpy())
        a_own, s_own = expert_label(expert, *state, L["hover_points"][g])
        own_err.append(float(np.abs(a_own[:2] - data["action"][i, :2].numpy()).max()))
        own_stop_agree.append(s_own == float(data["stop"][i]))
        a_cf, s_cf = expert_label(expert, *state, L["hover_points"][1 - g])
        cf_action.append(a_cf)
        cf_stop.append(s_cf)
        other = Target(**L["targets"][1 - g])
        cf_instr.append(make_instruction(other, e["instruction_family"]))
    report = {"own_goal_max_abs_error_mps": {"median": float(np.median(own_err)),
                                             "p99": float(np.percentile(own_err, 99)),
                                             "max": float(np.max(own_err))},
              "own_goal_stop_agreement": float(np.mean(own_stop_agree)),
              "counterfactual_rows": len(cf_action),
              "counterfactual_stop_positives": int(sum(cf_stop))}
    aug = {"rgb": torch.cat([data["rgb"], data["rgb"]]),
           "proprio": torch.cat([data["proprio"], data["proprio"]]),
           "action": torch.cat([data["action"], torch.tensor(np.stack(cf_action))]),
           "stop": torch.cat([data["stop"], torch.tensor(cf_stop, dtype=torch.float32)]),
           "instr": data["instr"] + cf_instr,
           "episode_id": data["episode_id"] + [f"{x}#cf" for x in data["episode_id"]],
           "priv_true_pos": np.concatenate([data["priv_true_pos"], data["priv_true_pos"]])}
    return aug, report


class Prepared:
    """Normalisation and text encoding fixed from the training rows."""

    def __init__(self, train, caps, no_language):
        self.no_language = no_language
        self.vocab = Vocab.from_texts(train["instr"])
        self.mean = train["proprio"].mean(0)
        self.std = train["proprio"].std(0).clamp(min=PROPRIO_STD_FLOOR)
        self.caps = torch.tensor(caps, dtype=torch.float32)

    def text_ids(self, texts):
        if self.no_language:
            return pad_batch([[1] for _ in texts])         # one constant token for everyone
        return pad_batch([self.vocab.encode(t) for t in texts])

    def batch(self, data, idx):
        return (data["rgb"][idx], (data["proprio"][idx] - self.mean) / self.std,
                self.text_ids([data["instr"][i] for i in idx]),
                data["action"][idx] / self.caps, data["stop"][idx])

    def to_json(self):
        return {"vocab": self.vocab.itos, "proprio_mean": self.mean.tolist(),
                "proprio_std": self.std.tolist(), "caps": self.caps.tolist(),
                "no_language": self.no_language}


def evaluate_rows(model, prep, data, huber, bce, batch=256):
    model.eval()
    motion_l, stop_l, logits, preds = [], [], [], []
    with torch.no_grad():
        for s in range(0, len(data["stop"]), batch):
            idx = list(range(s, min(s + batch, len(data["stop"]))))
            rgb, prop, ids, act, stp = prep.batch(data, idx)
            m, logit = model(rgb, prop, model.encode_text(ids))
            motion_l.append(huber(m, act).item() * len(idx))
            stop_l.append(bce(logit, stp).item() * len(idx))
            logits.append(logit)
            preds.append(m)
    n = len(data["stop"])
    pred = torch.cat(preds) * prep.caps
    rmse = torch.sqrt(((pred[:, :2] - data["action"][:, :2]) ** 2).sum(1).mean()).item()
    return {"motion_huber": sum(motion_l) / n, "stop_bce": sum(stop_l) / n,
            "horizontal_rmse_mps": rmse, "logits": torch.cat(logits)}


def best_threshold(logits, labels):
    """Stop threshold maximising F1 on validation rows (§4.3: chosen on validation only).

    Candidates are the *midpoints* between consecutive distinct logits, and among equally
    good candidates the one in the widest gap wins. A first version used the logits
    themselves and kept the lowest F1-maximiser, which is exactly the smallest positive
    logit: the threshold then sat on the edge of the Stop class, and in closed loop a
    memorised policy at its goal produced logits of 7.2, 9.7, 8.8 against a threshold of
    8.5, missed the three-in-a-row, overshot and hit the target.
    """
    u = torch.unique(logits).sort().values
    if len(u) < 2:
        return float(u[0]) if len(u) else 0.0, 0.0, {}
    best = (None, -1.0, {}, -1.0)
    for lo, hi in zip(u[:-1].tolist(), u[1:].tolist()):
        th = (lo + hi) / 2
        pred = logits >= th
        tp = float((pred & (labels > 0.5)).sum()); fp = float((pred & (labels < 0.5)).sum())
        fn = float((~pred & (labels > 0.5)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1.0)
        gap = hi - lo
        if f1 > best[1] + 1e-12 or (abs(f1 - best[1]) <= 1e-12 and gap > best[3]):
            best = (th, f1, {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                             "margin": gap}, gap)
    return best[:3]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.train")
    ap.add_argument("--data", type=pathlib.Path, default=REPO_ROOT / "data/v0.1")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--no-language", action="store_true")
    ap.add_argument("--overfit-pair", type=int, default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)          # Phase 1 item 5 start value
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--film", action="store_true",
                    help="FiLM-condition the CNN on the instruction (see dronevla/model.py)")
    ap.add_argument("--counterfactual", action="store_true",
                    help="add a counterfactual-relabelled copy of every training state; the "
                         "Stop pos_weight and proprio statistics stay those of the original rows")
    ap.add_argument("--aux-offsets", action="store_true",
                    help="auxiliary loss on body-frame offsets to every colour's hover point")
    ap.add_argument("--select", choices=["best_val", "last"], default="best_val",
                    help="keep the epoch with the lowest validation loss (default) or the last "
                         "one -- 'last' tests whether early selection stopped training too soon")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    root = args.data
    manifest = json.loads((root / "manifest.json").read_text())
    cfg = manifest["config"]
    caps = [cfg["limits"]["horizontal_mps"], cfg["limits"]["horizontal_mps"],
            cfg["limits"]["vertical_mps"], cfg["limits"]["yaw_rate_radps"]]
    episodes = load_episodes(root)
    if args.overfit_pair is not None:
        pair = sorted({e["pair_id"] for e in episodes if e["split"] == "train"},
                      key=lambda p: min(e["episode_id"] for e in episodes if e["pair_id"] == p))
        chosen = pair[args.overfit_pair]
        train_eps = val_eps = [e for e in episodes if e["pair_id"] == chosen]
    else:
        train_eps = [e for e in episodes if e["split"] == "train"]
        val_eps = [e for e in episodes if e["split"] == "val"]

    t_load = time.perf_counter()
    train, val = load_rows(root, train_eps), load_rows(root, val_eps)
    load_s = time.perf_counter() - t_load
    prep = Prepared(train, caps, args.no_language)
    model_kwargs = {"film": args.film}
    if args.aux_offsets:
        assert not args.counterfactual, "--aux-offsets with --counterfactual is not implemented"
        model_kwargs["aux_offsets"] = True
    model = TinyBC(len(prep.vocab), **model_kwargs)
    n_params = count_parameters(model)
    pos = float(train["stop"].sum())
    pos_weight = (len(train["stop"]) - pos) / max(pos, 1.0)   # from the original rows only
    cf_report = None
    if args.counterfactual:
        from dronevla.action_adapter import ActionLimits
        from dronevla.task import TaskConfig
        task_cfg = TaskConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                                 for k, v in cfg.items() if k != "limits"},
                              limits=ActionLimits(**cfg["limits"]))
        train, cf_report = counterfactual_relabel(root, train_eps, train, task_cfg)
        unseen = sorted({w for t in train["instr"] for w in t.lower().replace(",", " ")
                         .replace(".", " ").split()} - set(prep.vocab.itos))
        assert not unseen, f"counterfactual instructions use words outside the vocabulary: {unseen}"
        print(f"counterfactual relabelling: {cf_report}")
    huber = nn.SmoothL1Loss(beta=0.1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = len(train["stop"])
    steps_per_epoch = (n + args.batch - 1) // args.batch
    print(f"train rows {n} ({len(train_eps)} episodes), val rows {len(val['stop'])} "
          f"({len(val_eps)} episodes), loaded in {load_s:.1f} s")
    print(f"model {n_params:,} parameters; vocab {len(prep.vocab)} "
          f"{'(not used: --no-language)' if args.no_language else ''}; Stop positives "
          f"{pos:.0f}/{n} -> pos_weight {pos_weight:.1f}; torch threads {torch.get_num_threads()}")
    assert n_params <= 2_000_000, "roadmap §3.1 budget is 2M parameters"

    def train_step(idx):
        model.train()
        rgb, prop, ids, act, stp = prep.batch(train, idx)
        if args.aux_offsets:
            m, logit, aux = model(rgb, prop, model.encode_text(ids), return_aux=True)
            om = train["offset_mask"][idx]
            target = train["offsets"][idx] / 3.0                  # world_model.OFFSET_SCALE
            aux_loss = (((aux - target) ** 2) * om).sum() / om.sum().clamp(min=1)
        else:
            m, logit = model(rgb, prop, model.encode_text(ids))
            aux_loss = 0.0
        loss = huber(m, act) + bce(logit, stp) + aux_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    if args.pilot:
        times = []
        order = rng.permutation(n)
        for k in range(100):
            idx = order[(k * args.batch) % n:][: args.batch].tolist() or order[: args.batch].tolist()
            t0 = time.perf_counter()
            train_step(idx)
            times.append(time.perf_counter() - t0)
        s = stats(times[5:])
        epoch_s = s["mean_ms"] / 1e3 * steps_per_epoch
        report = {"minibatches": 100, "excluded_warmup": 5, "batch": args.batch,
                  "step_ms": s, "steps_per_epoch": steps_per_epoch,
                  "projected_epoch_s": epoch_s, "projected_run_s": epoch_s * args.epochs,
                  "epochs": args.epochs, "vmhwm_mb": proc_status_kb("VmHWM") / 1024,
                  "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  "torch_threads": torch.get_num_threads(), "parameters": n_params,
                  "data_load_s": load_s}
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "pilot.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"pilot: step p50 {s['p50_ms']:.1f} ms (p95 {s['p95_ms']:.1f}), "
              f"{steps_per_epoch} steps/epoch -> {epoch_s:.1f} s/epoch, "
              f"{args.epochs} epochs ~ {epoch_s * args.epochs / 60:.1f} min; "
              f"peak RSS {report['vmhwm_mb']:.0f} MB")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    log, best = [], (float("inf"), None, -1)
    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        order = rng.permutation(n)
        losses = [train_step(order[s:s + args.batch].tolist()) for s in range(0, n, args.batch)]
        v = evaluate_rows(model, prep, val, huber, bce)
        val_loss = v["motion_huber"] + v["stop_bce"]
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_motion_huber": v["motion_huber"], "val_stop_bce": v["stop_bce"],
               "val_horizontal_rmse_mps": v["horizontal_rmse_mps"]}
        log.append(row)
        if val_loss < best[0] or (args.select == "last" and epoch == args.epochs - 1):
            best = (val_loss, {k: t.clone() for k, t in model.state_dict().items()}, epoch)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {row['train_loss']:.4f}  val {val_loss:.4f}  "
                  f"val horizontal RMSE {v['horizontal_rmse_mps']:.3f} m/s", flush=True)
    train_s = time.perf_counter() - t_train

    model.load_state_dict(best[1])
    v = evaluate_rows(model, prep, val, huber, bce)
    th, f1, pr = best_threshold(v["logits"], val["stop"])
    torch.save(model.state_dict(), args.out / "model.pt")
    with (args.out / "train_log.jsonl").open("w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    config = {
        "schema": "dronevla.policy/0.1", "model": "TinyBC", "parameters": n_params,
        "model_kwargs": model_kwargs,
        "prep": prep.to_json(), "stop_threshold": th,
        "stop_threshold_val": {"f1": f1, **pr},
        "best_epoch": best[2], "best_val_loss": best[0],
        "val_horizontal_rmse_mps": v["horizontal_rmse_mps"],
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr, "seed": args.seed,
                  "optimizer": "Adam", "loss": "SmoothL1(beta=0.1) + BCEWithLogits(pos_weight)",
                  "pos_weight": pos_weight, "train_rows": n, "val_rows": len(val["stop"]),
                  "train_episodes": [e["episode_id"] for e in train_eps],
                  "overfit_pair": args.overfit_pair, "train_s": train_s,
                  "counterfactual": cf_report, "select": args.select,
                  "torch_threads": torch.get_num_threads(), "torch": torch.__version__},
        "data": {"root": str(root), "dataset_version": manifest["dataset_version"],
                 "manifest_sha256": sha256_file(root / "manifest.json"),
                 "rgb_tree_sha256": manifest["rgb"]["tree_sha256"]},
        "code": git_info(REPO_ROOT),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"best epoch {best[2]} (val loss {best[0]:.4f}); val horizontal RMSE "
          f"{v['horizontal_rmse_mps']:.3f} m/s; Stop threshold {th:.3f} (val F1 {f1:.3f}, "
          f"precision {pr['precision']:.2f}, recall {pr['recall']:.2f}); trained {train_s:.0f} s")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
