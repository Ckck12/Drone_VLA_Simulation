"""Try the Phase 1 task: same start, two instructions, and see where the drone ends up.

    python scripts/try_env.py              # GUI window, real time
    python scripts/try_env.py --headless   # no window, fast

Edit the block below and re-run. Each run writes results/try_env/<time>/:
    topview.png          the room from above: targets, goal regions, flown paths, outcomes
    first_frames.png     the first camera frame of each run (a pair must be identical)
    run<k>_sheet.png     camera frames over the run
and prints the instruction, the outcome and the distances for each run.
"""

# ============================== 여기를 바꾸세요 ==============================

LAYOUT_SEED = 0          # 장면 배치. 숫자를 바꾸면 물체의 색·모양·위치와 시작 위치가 바뀜

CUSTOM_TARGETS = None    # None이면 LAYOUT_SEED가 자동 배치. 직접 놓으려면 정확히 2개를 적으세요:
# CUSTOM_TARGETS = [
#     dict(color="red",  shape="box",      xy=(2.0,  1.0)),   # color: red blue green yellow
#     dict(color="blue", shape="cylinder", xy=(2.5, -1.0)),   # shape: box cylinder
# ]                                                          # 드론은 x=-3에서 +x 방향을 보고 시작
START_Y = None           # 시작 y 위치 [m]. None이면 LAYOUT_SEED가 정함

RUN = "pair"             # "pair" : 같은 시작 상태에서 두 지시를 각각 실행 (counterfactual pair)
                         # "goal0": 첫 번째 물체로 가라는 지시만 / "goal1": 두 번째 물체만

POLICY = "expert"        # "expert": 정답 위치를 아는 스크립트 비행 (데이터셋을 만들 선생님)
                         # "manual": 아래 MANUAL_COMMANDS 대로 직접 조종
                         # "wrong" : 일부러 다른 물체로 가는 전문가 (평가기가 오답을 잡는지 보기)
                         # "runs/bc_text" 같은 학습 결과 폴더: 학습된 정책 (관측만 받음)
                         #   val 장면을 보려면 LAYOUT_SEED를 2001~2010 중 하나로

INSTRUCTION_FAMILY = 0   # 0: "Go to the ... and stop." / 1: "Approach the ..., then hold position."

MANUAL_COMMANDS = [      # POLICY = "manual"일 때: (몇 초 동안, 앞 m/s, 왼쪽 m/s, 멈춤 요청?)
    (6.0, 0.5, 0.0, False),   # 6초 동안 앞으로 0.5 m/s
    (2.0, 0.0, 0.3, False),   # 2초 동안 왼쪽으로 0.3 m/s
    (1.0, 0.0, 0.0, True),    # 멈춤 요청 (3번 연속이면 정지 후 1초 동안 판정)
]                             # 목록이 끝나면 제자리 호버 -> 멈춤 요청이 없으면 결국 timeout

CONFIG_OVERRIDES = dict(  # 환경 설정 덮어쓰기 (dronevla/task.py의 TaskConfig 참고)
    # noise_xy_m=0.0,          # 위치 추정 노이즈 끄기
    # wind_mps=1.0,            # 바람
    # camera_tilt_deg=-35.0,   # 카메라 더 숙이기
    # success_radius_m=0.4,    # 성공 판정 반경
)

# ============================================================================

import argparse
import dataclasses
import pathlib
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))     # scripts/ is not the package root

from dronevla.env import DroneTargetPairsEnv              # noqa: E402
from dronevla.expert import StraightLineExpert            # noqa: E402
from dronevla.task import (COLORS, SHAPES, Layout, Target, TaskConfig,  # noqa: E402
                           check_geometry, hover_point, sample_layout)


def build_layout(cfg):
    """LAYOUT_SEED's layout, with CUSTOM_TARGETS / START_Y applied and re-checked."""
    base = sample_layout(LAYOUT_SEED, cfg)
    if CUSTOM_TARGETS is None and START_Y is None:
        return base
    if CUSTOM_TARGETS is None:
        targets = base.targets
    else:
        if len(CUSTOM_TARGETS) != 2:
            sys.exit("CUSTOM_TARGETS needs exactly 2 targets")
        targets = []
        for i, t in enumerate(CUSTOM_TARGETS):
            if t.get("color") not in COLORS or t.get("shape") not in SHAPES:
                sys.exit(f"CUSTOM_TARGETS[{i}]: color must be one of {list(COLORS)}, "
                         f"shape one of {list(SHAPES)}")
            targets.append(Target(color=t["color"], shape=t["shape"],
                                  xy=(float(t["xy"][0]), float(t["xy"][1])),
                                  radius=cfg.target_radius_m, height=cfg.target_height_m))
        targets = tuple(targets)
        if targets[0].description == targets[1].description:
            sys.exit("the two targets need different descriptions, or the instruction is ambiguous")
    start_y = base.start_xyz[1] if START_Y is None else float(START_Y)
    start = (cfg.start_x, start_y, cfg.altitude_m)
    hovers = tuple(hover_point(t, start[:2], cfg) for t in targets)
    reason = check_geometry(targets, start, hovers, cfg)
    if reason:
        print(f"layout rejected: {reason}")
        print("the rules (dronevla/task.py check_geometry): both targets within "
              f"+-{cfg.max_bearing_deg:.0f} deg of straight ahead from the start; goal regions "
              f"at least {cfg.min_hover_separation_m} m apart; a goal region must not touch "
              f"the other target; the straight path to one goal must pass the other target "
              f"with {cfg.path_clearance_m} m to spare; goals 0.5 m inside the room")
        sys.exit(1)
    return Layout(start_xyz=start, start_yaw=cfg.start_yaw, targets=targets,
                  hover_points=hovers, sample_seed=LAYOUT_SEED, rejected_before=0)


def manual_policy(cfg):
    actions = []
    for seconds, fwd, left, stop in MANUAL_COMMANDS:
        n = int(round(seconds * cfg.policy_hz))
        actions += [np.array([fwd, left, 0.0, 0.0, 1.0 if stop else -1.0], np.float32)] * n
    hover = np.array([0, 0, 0, 0, -1.0], np.float32)
    it = iter(actions)
    return lambda obs, info: next(it, hover)


def make_policy(cfg):
    if POLICY == "expert":
        e = StraightLineExpert(cfg)
        return lambda obs, info: e.act(info)
    if POLICY == "wrong":
        e = StraightLineExpert(cfg, goal_key="other_hover_point")
        return lambda obs, info: e.act(info)
    if POLICY == "manual":
        return manual_policy(cfg)
    run = pathlib.Path(POLICY)
    run = run if run.is_absolute() else REPO_ROOT / run
    if (run / "config.json").exists():
        from dronevla.evaluate import LearnedPolicy
        learned = LearnedPolicy(run)

        class Learned:                         # obs only, like dronevla/evaluate.py
            def reset(self, instruction):
                learned.reset(instruction)

            def __call__(self, obs, info):
                return learned.act(obs)
        return Learned()
    sys.exit(f'POLICY must be "expert", "manual", "wrong" or a run directory, got {POLICY!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    goals = {"pair": (0, 1), "goal0": (0,), "goal1": (1,)}.get(RUN)
    if goals is None:
        sys.exit('RUN must be "pair", "goal0" or "goal1"')

    cfg = dataclasses.replace(TaskConfig(), **CONFIG_OVERRIDES)
    layout = build_layout(cfg)
    out = REPO_ROOT / "results/try_env" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    env = DroneTargetPairsEnv(cfg, gui=not args.headless)

    print(f"layout seed {LAYOUT_SEED} (id {layout.layout_id}): "
          + "  vs  ".join(f"{t.description} at ({t.xy[0]:.2f}, {t.xy[1]:.2f})" for t in layout.targets))
    runs = []
    for k, goal in enumerate(goals):
        policy = make_policy(cfg)
        obs, info = env.reset(seed=LAYOUT_SEED, options={
            "layout": layout, "goal_index": goal, "instruction_family": INSTRUCTION_FAMILY})
        if hasattr(policy, "reset"):
            policy.reset(obs["instruction"])
        frames, path = [obs["rgb"]], [info["privileged"]["true_pos"]]
        if k == 0:
            vis = env.visibility_report()
            names = [t.description for t in layout.targets]
            print(f"visible : at the start {names[0]} {vis['start_px'][0]} px, "
                  f"{names[1]} {vis['start_px'][1]} px (of {128 * 96}; reject below "
                  f"{cfg.min_first_frame_target_px}); at its own goal each fills "
                  f"{vis['own_target_px_at_hover'][0]} / {vis['own_target_px_at_hover'][1]} px")
            if min(vis["start_px"]) < cfg.min_first_frame_target_px:
                print("WARNING : a target is (almost) out of the first frame -- "
                      "a policy could not know where it is")
        t0 = time.perf_counter()
        while True:
            obs, reward, term, trunc, info = env.step(policy(obs, info))
            frames.append(obs["rgb"])
            path.append(info["privileged"]["true_pos"])
            if term or trunc:
                break
        priv = info["privileged"]
        runs.append(dict(goal=goal, instruction=obs["instruction"], outcome=info["outcome"],
                         path=np.array(path), frames=frames, info=info))
        print(f"\nrun {k}: {obs['instruction']!r}")
        print(f"   outcome  : {info['outcome']}   ({info['episode_t']:.1f} sim-s, "
              f"{len(frames) - 1} policy steps, {time.perf_counter() - t0:.1f} wall-s)")
        print(f"   distance : to goal {priv['dist_to_goal']:.3f} m, to the other {priv['dist_to_other']:.3f} m "
              f"(success radius {cfg.success_radius_m} m)")
        print(f"   reached  : goal region {priv['reached_goal']}, other region {priv['reached_other']}"
              + (f", contact with {priv['contact_with']}" if priv["contact_with"] else ""))
        if priv["hold"]:
            h = priv["hold"]
            print(f"   stop hold: max distance to goal {h['max_dist_to_goal']:.3f} m, "
                  f"max speed {h['max_speed']:.3f} m/s (limit {cfg.stop_speed_mps})")
    env.close()

    if len(runs) == 2:
        same = (np.array_equal(runs[0]["frames"][0], runs[1]["frames"][0]))
        print(f"\npair: first camera frames identical = {same}; "
              f"pair success = {all(r['outcome'] == 'success' for r in runs)}")
    save_pictures(runs, layout, cfg, out)


def save_pictures(runs, layout, cfg, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle
    from PIL import Image

    fig, ax = plt.subplots(figsize=(8, 6.5))
    lo, hi = cfg.room_min, cfg.room_max
    ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fill=False, ls=":", color="0.5"))
    for i, t in enumerate(layout.targets):
        c = COLORS[t.color][:3]
        x, y = t.xy
        r = t.radius
        ax.add_patch(Rectangle((x - r, y - r), 2 * r, 2 * r, color=c) if t.shape == "box"
                     else Circle((x, y), r, color=c))
        hx, hy, _ = layout.hover_points[i]
        ax.add_patch(Circle((hx, hy), cfg.success_radius_m, fill=False, ls="--", color=c))
        ax.annotate(f"{t.description}\ngoal region", (hx, hy), fontsize=7, ha="center",
                    va="center", color="0.25")
    sx, sy, _ = layout.start_xyz
    ax.plot(sx, sy, "k^", ms=9, label="start")
    styles = ["-", "--"]
    for k, r in enumerate(runs):
        tgt = layout.targets[r["goal"]]
        ax.plot(r["path"][:, 0], r["path"][:, 1], styles[k % 2], lw=2,
                color=COLORS[tgt.color][:3], label=f"{r['instruction']}  ->  {r['outcome']}")
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.grid(alpha=0.25)
    ax.set_title(f"Layout seed {LAYOUT_SEED} — policy {POLICY!r}", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "topview.png", dpi=110); plt.close(fig)

    firsts = [Image.fromarray(r["frames"][0]).resize((256, 192), Image.NEAREST) for r in runs]
    strip = Image.new("RGB", (256 * len(firsts) + 4 * (len(firsts) - 1), 192), (24, 24, 24))
    for k, im in enumerate(firsts):
        strip.paste(im, (k * 260, 0))
    strip.save(out / "first_frames.png")

    for k, r in enumerate(runs):
        d = out / f"run{k}_frames"
        d.mkdir(exist_ok=True)
        for n, f in enumerate(r["frames"]):
            Image.fromarray(f).save(d / f"frame_{n:05d}.png")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/contact_sheet.py"),
                        "--frames-dir", str(d), "--out", str(out / f"run{k}_sheet.png")],
                       check=False, capture_output=True)
    print(f"\nsaved   : {out}/topview.png, first_frames.png, run*_sheet.png")


if __name__ == "__main__":
    main()
