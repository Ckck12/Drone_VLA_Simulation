"""Latent world model for DroneTargetPairs: stage 1 of the world-model track.

    python -m dronevla.world_model train --out runs/wm_v0
    python -m dronevla.world_model eval  --run runs/wm_v0 --out reports/wm_v0_prediction.json
    python -m dronevla.world_model train --data data/v0.2 --explore data/explore_v0.2 --out runs/wm_v1

What it is: an encoder from (RGB, proprio) to a 64-d latent z, a residual dynamics model
z_{t+1} = LN(z_t + f(z_t, a_t)) driven by the horizontal command a_t = (vx, vy), and two
read-out heads: the 11 proprio values, and the body-frame offset from the drone to each of the
four colours' hover points (masked where a colour is absent). Language is not an input:
dynamics do not depend on the words. A later stage adds a language-conditioned goal model and a
planner on top.

Training signal, unrolled H steps from each start state with the recorded actions:
    latent  || z_k(predicted) - stopgrad(encoder(obs_{t+k})) ||^2
    proprio || head(z_k) - proprio_{t+k} ||^2         for the encoded and the predicted z_k
    offsets || head(z_k) - offsets_{t+k} ||^2         same, masked by colour presence
discounted by 0.9^k. The offsets are computed from the logged true position and the layout --
privileged supervision used for training only, never as an input. (The BC policy did not get
it; the comparison stage gives BC the same auxiliary target so the comparison stays fair.)

Targets come from the *online* encoder with stop-gradient rather than an EMA copy: it halves
the image encodes per step on a CPU, and the supervised heads anchor the latent against
collapse.

Data: the demonstration dataset's train split plus exploration flights over the same train
layouts (`dronevla.explore`). Evaluation: val layouts only, never test.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
from PIL import Image
from torch import nn

from dronevla.dataset import load_steps
from dronevla.profile_env import git_info, stats
from dronevla.task import COLORS

import pyarrow.parquet as pq

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
COLOUR_ORDER = tuple(COLORS)          # red, blue, green, yellow
OFFSET_SCALE = 3.0                    # metres -> roughly unit scale
DT = 0.2


# ----------------------------------------------------------------------------- data
def episode_list(root: pathlib.Path, split: str | None):
    eps = pq.read_table(root / "episodes.parquet").to_pylist()
    return [e for e in eps if split is None or e["split"] == split]


def load_episode_arrays(root, e):
    """Per-row arrays for one episode, including colour-slot offsets from the layout."""
    lay = json.loads((root / "layouts" / f"{e['layout_id']}.json").read_text())
    rows = load_steps(root, e["episode_id"])
    n = len(rows)
    rgb = np.stack([np.asarray(Image.open(root / r["rgb_path"])) for r in rows])
    prop = np.array([r["proprio"] for r in rows], dtype=np.float32)
    act = np.zeros((n, 2), dtype=np.float32)
    flown = np.zeros(n, dtype=bool)
    for i, r in enumerate(rows):
        if r["action_mask"] and not r["stop_trigger"]:
            act[i] = r["action_applied"][:2]
            flown[i] = True
    pos = np.array([r["priv_true_pos"] for r in rows])
    off = np.zeros((n, len(COLOUR_ORDER), 2), dtype=np.float32)
    mask = np.zeros((n, len(COLOUR_ORDER)), dtype=np.float32)
    yaw = np.arctan2(prop[:, 5], prop[:, 6])
    for ti, t in enumerate(lay["targets"]):
        c = COLOUR_ORDER.index(t["color"])
        h = np.asarray(lay["hover_points"][ti][:2])
        d = h[None, :] - pos[:, :2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        off[:, c, 0] = cy * d[:, 0] + sy * d[:, 1]          # world -> body (yaw only)
        off[:, c, 1] = -sy * d[:, 0] + cy * d[:, 1]
        mask[:, c] = 1.0
    return {"rgb": rgb, "prop": prop, "act": act, "flown": flown,
            "off": off.reshape(n, -1), "mask": mask, "pos": pos, "episode_id": e["episode_id"]}


def load_sources(sources):
    """sources: list of (root, split-or-None). Returns per-episode arrays."""
    out = []
    for root, split in sources:
        root = pathlib.Path(root)
        for e in episode_list(root, split):
            ep = load_episode_arrays(root, e)
            ep["source"] = root.name
            out.append(ep)
    return out


def windows(episodes, horizon, stride=1):
    """(episode, start) pairs where the next `horizon` actions were all actually flown."""
    w = []
    for ei, ep in enumerate(episodes):
        n = len(ep["prop"])
        for t in range(0, n - horizon, stride):
            if ep["flown"][t:t + horizon].all():
                w.append((ei, t))
    return w


# ---------------------------------------------------------------------------- model
class WorldModel(nn.Module):
    CHANNELS = (16, 32, 64, 64)

    def __init__(self, z_dim: int = 64, proprio_dim: int = 11, action_dim: int = 2):
        super().__init__()
        c = self.CHANNELS
        self.cnn = nn.Sequential(
            nn.Conv2d(3, c[0], 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(c[0], c[1], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c[1], c[2], 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c[2], c[3], 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.img_fc = nn.Sequential(nn.Linear(64 * 6 * 8, 128), nn.ReLU())
        self.prop_fc = nn.Sequential(nn.Linear(proprio_dim, 32), nn.ReLU())
        self.to_z = nn.Sequential(nn.Linear(160, z_dim), nn.LayerNorm(z_dim))
        self.dyn = nn.Sequential(nn.Linear(z_dim + action_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, z_dim))
        self.dyn_norm = nn.LayerNorm(z_dim)
        self.head_prop = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(), nn.Linear(64, proprio_dim))
        self.head_off = nn.Sequential(nn.Linear(z_dim, 64), nn.ReLU(),
                                      nn.Linear(64, 2 * len(COLOUR_ORDER)))

    def encode(self, rgb_u8, prop_norm):
        x = rgb_u8.permute(0, 3, 1, 2).float().div(255.0).sub(0.5)
        return self.to_z(torch.cat([self.img_fc(self.cnn(x)), self.prop_fc(prop_norm)], dim=1))

    def step(self, z, a_scaled):
        return self.dyn_norm(z + self.dyn(torch.cat([z, a_scaled], dim=1)))


class Norm:
    def __init__(self, prop_mean, prop_std, cap=0.5):
        self.mean, self.std, self.cap = prop_mean, prop_std, cap

    def prop(self, p):
        return (p - self.mean) / self.std

    def act(self, a):
        return a / self.cap

    def to_json(self):
        return {"prop_mean": self.mean.tolist(), "prop_std": self.std.tolist(), "cap": self.cap}


def batch_tensors(episodes, idx, horizon):
    """Stack H+1 observations and H actions for each (episode, start)."""
    rgb, prop, act, off, mask = [], [], [], [], []
    for ei, t in idx:
        ep = episodes[ei]
        sl = slice(t, t + horizon + 1)
        rgb.append(ep["rgb"][sl]); prop.append(ep["prop"][sl]); off.append(ep["off"][sl])
        mask.append(ep["mask"][sl]); act.append(ep["act"][t:t + horizon])
    f = lambda xs, dt: torch.from_numpy(np.stack(xs)).to(dt)
    return (f(rgb, torch.uint8), f(prop, torch.float32), f(act, torch.float32),
            f(off, torch.float32), f(mask, torch.float32))


def losses(model, norm, batch, horizon, gamma=0.9):
    rgb, prop, act, off, mask = batch
    B = rgb.shape[0]
    flat_rgb = rgb.reshape(B * (horizon + 1), *rgb.shape[2:])
    flat_prop = norm.prop(prop.reshape(B * (horizon + 1), -1))
    z_enc = model.encode(flat_rgb, flat_prop).reshape(B, horizon + 1, -1)
    pnorm = norm.prop(prop)
    offn = off / OFFSET_SCALE
    m2 = mask.repeat_interleave(2, dim=-1)

    def head_losses(z, k):
        lp = ((model.head_prop(z) - pnorm[:, k]) ** 2).mean()
        lo = (((model.head_off(z) - offn[:, k]) ** 2) * m2[:, k]).sum() / m2[:, k].sum().clamp(min=1)
        return lp, lo

    lp0, lo0 = head_losses(z_enc[:, 0], 0)
    total = {"latent": 0.0, "prop": lp0, "off": lo0}
    z = z_enc[:, 0]
    for k in range(1, horizon + 1):
        z = model.step(z, norm.act(act[:, k - 1]))
        w = gamma ** k
        total["latent"] = total["latent"] + w * ((z - z_enc[:, k].detach()) ** 2).mean()
        lp, lo = head_losses(z, k)
        lpe, loe = head_losses(z_enc[:, k], k)             # encoder must also read state at t+k
        total["prop"] = total["prop"] + w * lp + lpe
        total["off"] = total["off"] + w * lo + loe
    total["sum"] = total["latent"] + total["prop"] + total["off"]
    return total


# ------------------------------------------------------------------------- training
def default_sources(split, data="data/v0.1", explore="data/explore_v0.1"):
    """Demonstrations of `split` plus the exploration flights over the same split's layouts."""
    return [(REPO_ROOT / data, split), (REPO_ROOT / explore / split, None)]


def val_sets(cfg):
    """The evaluation sets for a trained run: val demonstrations and val exploration."""
    d = cfg.get("data", {"demos": "data/v0.1", "explore": "data/explore_v0.1"})
    return (("val expert demos", [(REPO_ROOT / d["demos"], "val")]),
            ("val exploration", [(REPO_ROOT / d["explore"] / "val", None)]))


def train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    tr = load_sources(default_sources("train", args.data, args.explore))
    va = load_sources(default_sources("val", args.data, args.explore))
    print(f"loaded {sum(len(e['prop']) for e in tr)} train rows ({len(tr)} episodes), "
          f"{sum(len(e['prop']) for e in va)} val rows ({len(va)} episodes) "
          f"in {time.perf_counter() - t0:.0f} s")
    allp = np.concatenate([e["prop"] for e in tr])
    norm = Norm(torch.tensor(allp.mean(0)), torch.tensor(allp.std(0)).clamp(min=1e-2))
    model = WorldModel(z_dim=args.z_dim)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    wtr, wva = windows(tr, args.horizon), windows(va, args.horizon, stride=3)
    print(f"model {nparam:,} parameters; {len(wtr)} train windows, {len(wva)} val windows; "
          f"horizon {args.horizon}; torch threads {torch.get_num_threads()}")

    def val_loss():
        model.eval()
        acc, n = {}, 0
        with torch.no_grad():
            for s in range(0, len(wva), 64):
                b = batch_tensors(va, wva[s:s + 64], args.horizon)
                l = losses(model, norm, b, args.horizon)
                for k, v in l.items():
                    acc[k] = acc.get(k, 0.0) + float(v) * len(b[0])
                n += len(b[0])
        return {k: v / n for k, v in acc.items()}

    args.out.mkdir(parents=True, exist_ok=True)
    log, step_times = [], []
    t_train = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(wtr))
        ep_losses = []
        for s in range(0, len(order), args.batch):
            idx = [wtr[i] for i in order[s:s + args.batch]]
            ts = time.perf_counter()
            l = losses(model, norm, batch_tensors(tr, idx, args.horizon), args.horizon)
            opt.zero_grad()
            l["sum"].backward()
            opt.step()
            step_times.append(time.perf_counter() - ts)
            ep_losses.append(l["sum"].item())
        v = val_loss()
        row = {"epoch": epoch, "train": float(np.mean(ep_losses)), **{f"val_{k}": x for k, x in v.items()}}
        log.append(row)
        print(f"  epoch {epoch:2d}  train {row['train']:.4f}  val {v['sum']:.4f} "
              f"(latent {v['latent']:.4f}, prop {v['prop']:.4f}, offsets {v['off']:.4f})", flush=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    with (args.out / "train_log.jsonl").open("w") as fh:
        for row in log:
            fh.write(json.dumps(row) + "\n")
    cfg = {"schema": "dronevla.world_model/0.1", "parameters": nparam, "z_dim": args.z_dim,
           "horizon": args.horizon, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
           "seed": args.seed, "norm": norm.to_json(), "offset_scale": OFFSET_SCALE,
           "colour_order": list(COLOUR_ORDER), "train_s": time.perf_counter() - t_train,
           "step_ms": stats(step_times[5:]), "selection": "last epoch (fixed count)",
           "train_rows": int(sum(len(e["prop"]) for e in tr)),
           "train_episodes": len(tr), "data": {"demos": args.data, "explore": args.explore},
           "code": git_info(REPO_ROOT)}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"trained {cfg['train_s']:.0f} s; wrote {args.out}")


# ------------------------------------------------------------------------ evaluation
def load_model(run):
    run = pathlib.Path(run)
    cfg = json.loads((run / "config.json").read_text())
    m = WorldModel(z_dim=cfg["z_dim"])
    m.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    m.eval()
    n = cfg["norm"]
    return m, Norm(torch.tensor(n["prop_mean"]), torch.tensor(n["prop_std"]), n["cap"]), cfg


def evaluate(args):
    model, norm, cfg = load_model(args.run)
    H = args.horizon
    report = {"run": str(args.run), "horizon": H, "sets": {}}
    for name, sources in val_sets(cfg):
        eps = load_sources(sources)
        win = windows(eps, H, stride=2)
        err = {m: [[] for _ in range(H + 1)] for m in ("wm", "hold", "const_vel", "cmd_int")}
        verr = [[] for _ in range(H + 1)]
        with torch.no_grad():
            for s in range(0, len(win), 64):
                idx = win[s:s + 64]
                rgb, prop, act, off, mask = batch_tensors(eps, idx, H)
                z = model.encode(rgb[:, 0], norm.prop(prop[:, 0]))
                o0 = off[:, 0].reshape(len(idx), -1, 2)
                v0 = prop[:, 0, 0:2]                                  # body velocity at t
                cum = torch.zeros(len(idx), 2)
                for k in range(H + 1):
                    if k > 0:
                        z = model.step(z, norm.act(act[:, k - 1]))
                        cum = cum + act[:, k - 1] * DT
                    pred_off = (model.head_off(z) * OFFSET_SCALE).reshape(len(idx), -1, 2)
                    true_off = off[:, k].reshape(len(idx), -1, 2)
                    msk = mask[:, k] > 0.5
                    cands = {"wm": pred_off, "hold": o0,
                             "const_vel": o0 - (v0 * DT * k)[:, None, :],
                             "cmd_int": o0 - cum[:, None, :]}
                    for m, p in cands.items():
                        e = torch.linalg.norm(p - true_off, dim=-1)[msk]
                        err[m][k] += e.tolist()
                    pp = model.head_prop(z) * norm.std + norm.mean
                    verr[k] += torch.linalg.norm(pp[:, 0:2] - prop[:, k, 0:2], dim=-1).tolist()
        res = {"windows": len(win), "offset_error_m": {}, "velocity_error_mps": {}}
        for k in range(H + 1):
            res["offset_error_m"][k] = {m: float(np.median(err[m][k])) for m in err}
            res["velocity_error_mps"][k] = float(np.median(verr[k]))
        report["sets"][name] = res
        print(f"\n== {name}: {len(win)} windows of {H} steps (median errors)")
        print(f"   {'k':>3} {'t+':>5}  {'world model':>12} | {'hold':>6} {'const-vel':>9} "
              f"{'cmd-integr':>10}   (privileged baselines start from the TRUE state)  | WM velocity")
        for k in range(H + 1):
            r = res["offset_error_m"][k]
            print(f"   {k:3d} {k * DT:4.1f}s  {r['wm']:10.3f} m | {r['hold']:6.3f} {r['const_vel']:9.3f} "
                  f"{r['cmd_int']:10.3f}   {'':47s}| {res['velocity_error_mps'][k]:.3f} m/s")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.world_model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=pathlib.Path, required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--horizon", type=int, default=4)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--z-dim", type=int, default=64)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--data", default="data/v0.1", help="demonstration dataset, relative to the repo")
    t.add_argument("--explore", default="data/explore_v0.1", help="exploration root with train/ and val/")
    e = sub.add_parser("eval")
    e.add_argument("--run", type=pathlib.Path, required=True)
    e.add_argument("--horizon", type=int, default=10)
    e.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)
    (train if args.cmd == "train" else evaluate)(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
