"""World-model track, stages 2 and 3: pick the instructed target, then plan to it.

    python -m dronevla.planner train-selector --data data/v0.2 --out runs/goal_selector
    python -m dronevla.evaluate --data data/v0.2 --split val \
        --policy plan:runs/wm_v1_s0:integrator --policy plan:runs/wm_v1_s0:wm --out ...

Stage 2 -- `GoalSelector`: instruction -> which of the four colour slots of the world model's
offset head is the goal. A mean-pooled word embedding and one linear layer, trained on the
train split's instructions. In this task the two targets always differ in colour, so a colour
keyword alone decides the goal; the selector is learned so that no rule is hand-written, but it
is expected to be exact and is not a result in itself.

Stage 3 -- `PlannerPolicy`, an obs-only policy with the same interface as the BC policy:
1. perceive: encode (RGB, proprio) with the world model, read the goal slot's body-frame offset
2. filter: predict the offset forward by the last command (offset -= a * DT, valid because yaw
   is fixed in Phase 1, so body and world axes stay aligned) and blend in the new measurement
   with a fixed gain. Per-frame estimates are noisy (median 0.17 m near the goal on val); acting
   on them raw makes the commands jitter and the Stop rule never settles
3. plan: cross-entropy method over H-step sequences of (vx, vy) inside the speed cap, scored by
   the predicted offsets. The predicted *displacement* comes from either
   * `integrator`: -sum(a) * DT. Sampling MPC over an integrator is in effect a proportional
     controller with a speed profile; named as such, not as "planning with a learned model"
   * `wm`: the world model's latent rollout, head(z_k) - head(z_0) for the goal slot
   and is added to the filtered offset, so both variants share one state estimate
4. stop: request Stop (logit +1) while the filtered offset is inside `stop_radius_m` and the
   measured speed is below `stop_speed_mps`, with a zero command; the env's three-in-a-row rule
   does the rest, exactly as for the expert

`OraclePlannerPolicy` is the same planner fed the true goal offset (privileged, `oracle=True`):
it measures the planner and Stop logic on their own, without perception error.

Planner settings are fixed in `PlannerConfig` and were set with the oracle planner on *train*
scenes only, never by watching learned variants on val.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time

import numpy as np
import torch
from torch import nn

from dronevla.dataset import load_episodes
from dronevla.model import Vocab, pad_batch
from dronevla.profile_env import git_info
from dronevla.world_model import COLOUR_ORDER, DT, OFFSET_SCALE, load_model

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ stage 2: goal selector
class GoalSelector(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.fc = nn.Linear(dim, len(COLOUR_ORDER))

    def forward(self, ids):
        mask = (ids > 0).float().unsqueeze(-1)
        e = (self.embed(ids) * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.fc(e)


def _instructions(root, split):
    eps = [e for e in load_episodes(root) if e["split"] == split]
    return [e["instruction"] for e in eps], [COLOUR_ORDER.index(e["goal_color"]) for e in eps]


def train_selector(args):
    torch.manual_seed(args.seed)
    tr_x, tr_y = _instructions(args.data, "train")
    va_x, va_y = _instructions(args.data, "val")
    vocab = Vocab.from_texts(tr_x)
    model = GoalSelector(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    X = pad_batch([vocab.encode(t) for t in tr_x])
    Y = torch.tensor(tr_y)
    for _ in range(args.epochs):                       # full batch: 200 instructions
        loss = nn.functional.cross_entropy(model(X), Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = lambda xs, ys: float((model(pad_batch([vocab.encode(t) for t in xs])).argmax(1)
                                    == torch.tensor(ys)).float().mean())
        train_acc, val_acc = acc(tr_x, tr_y), acc(va_x, va_y)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    cfg = {"schema": "dronevla.goal_selector/0.1", "vocab": vocab.itos, "colour_order": list(COLOUR_ORDER),
           "train_instructions": len(tr_x), "val_instructions": len(va_x),
           "train_accuracy": train_acc, "val_accuracy": val_acc, "epochs": args.epochs,
           "seed": args.seed, "data": str(args.data), "code": git_info(REPO_ROOT)}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"goal selector: train accuracy {train_acc:.3f} ({len(tr_x)}), val accuracy {val_acc:.3f} "
          f"({len(va_x)}); vocab {len(vocab)}; wrote {args.out}")


def load_selector(run):
    run = pathlib.Path(run)
    cfg = json.loads((run / "config.json").read_text())
    vocab = Vocab([])
    vocab.itos = cfg["vocab"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
    m = GoalSelector(len(vocab.itos))
    m.load_state_dict(torch.load(run / "model.pt", weights_only=True))
    m.eval()
    return m, vocab


# ------------------------------------------------------------------------ stage 3: planner
@dataclasses.dataclass(frozen=True)
class PlannerConfig:
    horizon: int = 8                  # 1.6 s; the WM's response to a command needs > 1 step
    samples: int = 64
    iterations: int = 4
    elites: int = 8
    init_std_mps: float = 0.3
    filter_gain: float = 0.3          # weight of a new measurement
    stop_radius_m: float = 0.12       # expert value
    stop_speed_mps: float = 0.08      # expert value
    w_smooth: float = 0.5             # on squared command change, (m/s)^2
    w_end_speed: float = 1.0          # on the squared last command: arrive slowly
    cap_mps: float = 0.5 * (1.0 - 1e-5)


def cap_norm(a, cap):
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a * np.minimum(1.0, cap / np.maximum(n, 1e-9))


class _Planner:
    """CEM over (H, 2) command sequences; `displacement(A)` -> predicted offset change (N, H, 2)."""

    def __init__(self, cfg: PlannerConfig):
        self.cfg = cfg

    def reset(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((self.cfg.horizon, 2))
        self.a_prev = np.zeros(2)

    def plan(self, g0, displacement):
        c = self.cfg
        mean, std = self.mean.copy(), np.full((c.horizon, 2), c.init_std_mps)
        best = None
        for _ in range(c.iterations):
            A = cap_norm(mean + std * self.rng.standard_normal((c.samples, c.horizon, 2)), c.cap_mps)
            A[0] = cap_norm(mean, c.cap_mps)                      # always score the warm start
            off = g0[None, None, :] + displacement(A)
            cost = (off ** 2).sum(-1).mean(1)
            da = np.diff(np.concatenate([np.broadcast_to(self.a_prev, (c.samples, 1, 2)), A], 1), axis=1)
            cost += c.w_smooth * (da ** 2).sum(-1).mean(1) + c.w_end_speed * (A[:, -1] ** 2).sum(-1)
            idx = np.argsort(cost)[: c.elites]
            mean, std = A[idx].mean(0), A[idx].std(0) + 1e-3
            if best is None or cost[idx[0]] < best[0]:
                best = (cost[idx[0]], A[idx[0]].copy())
        seq = best[1]
        self.mean = np.concatenate([seq[1:], seq[-1:]], 0)          # warm start, shifted
        return seq[0]

    def finish(self, g, speed, a):
        """Action from the planned command `a`, the filtered goal offset `g` and the measured
        speed: a zero command with a positive Stop logit once inside the stop radius and slow."""
        c = self.cfg
        if np.linalg.norm(g) < c.stop_radius_m and speed < c.stop_speed_mps:
            a, logit = np.zeros(2), 1.0
        else:
            logit = -1.0
        self.a_prev = a
        return np.array([a[0], a[1], 0.0, 0.0, logit], dtype=np.float32)


def integrator_displacement(A):
    return -np.cumsum(A, axis=1) * DT


class PlannerPolicy:
    oracle = False

    def __init__(self, wm_run, dynamics="integrator", selector_run=REPO_ROOT / "runs/goal_selector",
                 cfg: PlannerConfig = PlannerConfig()):
        assert dynamics in ("integrator", "wm"), dynamics
        self.wm, self.norm, self.wm_cfg = load_model(wm_run)
        self.selector, self.vocab = load_selector(selector_run)
        self.dynamics, self.cfg = dynamics, cfg
        self.planner = _Planner(cfg)
        self.name = f"plan_{dynamics}_{pathlib.Path(wm_run).name} (WM planner)"
        self.config = {"wm_run": str(wm_run), "dynamics": dynamics, "selector": str(selector_run),
                       "planner": dataclasses.asdict(cfg)}
        self.unknown_token_rate = 0.0
        self._episode = 0

    def reset(self, instruction):
        with torch.no_grad():
            self.slot = int(self.selector(pad_batch([self.vocab.encode(instruction)])).argmax(1))
        self.unknown_token_rate = self.vocab.unknown_rate(instruction)
        self.g = None
        self.planner.reset(seed=self._episode)
        self._episode += 1

    def act(self, obs):
        rgb = torch.from_numpy(np.ascontiguousarray(obs["rgb"])).unsqueeze(0)
        prop = torch.from_numpy(obs["proprio"]).unsqueeze(0)
        with torch.no_grad():
            z0 = self.wm.encode(rgb, self.norm.prop(prop))
            off0 = (self.wm.head_off(z0) * OFFSET_SCALE).reshape(-1, 2)
        meas = off0[self.slot].numpy().astype(np.float64)
        if self.g is None:
            self.g = meas
        else:
            pred = self.g - self.planner.a_prev * DT
            self.g = pred + self.cfg.filter_gain * (meas - pred)

        if self.dynamics == "integrator":
            disp = integrator_displacement
        else:
            def disp(A):
                with torch.no_grad():
                    z = z0.expand(A.shape[0], -1)
                    out = []
                    for k in range(A.shape[1]):
                        z = self.wm.step(z, self.norm.act(torch.from_numpy(A[:, k]).float()))
                        out.append((self.wm.head_off(z) * OFFSET_SCALE).reshape(A.shape[0], -1, 2)[:, self.slot])
                    return (torch.stack(out, 1) - off0[self.slot]).numpy()
        a = self.planner.plan(self.g, disp)
        return self.planner.finish(self.g, float(np.linalg.norm(obs["proprio"][0:2])), a)


class OraclePlannerPolicy:
    """The same planner with the true goal offset (privileged) and integrator dynamics."""
    oracle = True

    def __init__(self, cfg: PlannerConfig = PlannerConfig()):
        self.cfg = cfg
        self.planner = _Planner(cfg)
        self.name = "plan_oracle (true goal offset, oracle)"
        self.config = {"planner": dataclasses.asdict(cfg)}
        self._episode = 0

    def reset(self, instruction):
        self.planner.reset(seed=self._episode)
        self._episode += 1

    def act_privileged(self, obs, info):
        p = info["privileged"]
        d = np.asarray(p["goal_hover_point"][:2]) - np.asarray(p["true_pos"][:2])
        cy, sy = np.cos(p["true_yaw"]), np.sin(p["true_yaw"])
        g = np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1]])
        a = self.planner.plan(g, integrator_displacement)
        return self.planner.finish(g, float(np.linalg.norm(obs["proprio"][0:2])), a)


def policy_from_spec(spec):
    """`plan-oracle`, or `plan:<world model run>:<integrator|wm>`."""
    if spec == "plan-oracle":
        return OraclePlannerPolicy()
    _, run, dyn = spec.split(":")
    return PlannerPolicy(REPO_ROOT / run if not pathlib.Path(run).is_absolute() else run, dyn)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dronevla.planner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("train-selector")
    s.add_argument("--data", type=pathlib.Path, default=REPO_ROOT / "data/v0.2")
    s.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "runs/goal_selector")
    s.add_argument("--epochs", type=int, default=300)
    s.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    train_selector(args)
    print(f"{time.perf_counter() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
