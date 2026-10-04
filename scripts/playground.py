"""Playground — 변수를 바꿔가며 드론 비행을 직접 실험하는 스크립트.

    python scripts/playground.py              # GUI 창으로 비행을 보면서 실행
    python scripts/playground.py --headless   # 창 없이 빠르게, 결과만

아래 "여기를 바꾸세요" 블록의 값만 고치고 다시 실행하면 됩니다.
실행이 끝나면 results/playground/<시각>/ 에 다음이 남습니다:
    summary.png        위에서 본 계획 경로 vs 실제 경로, 시간별 속도·추종 오차
    camera_sheet.png   드론 앞 카메라(정책이 보게 될 128x96 화면) 샘플
    frames/            카메라 프레임 원본
그리고 터미널에 속도·추종 오차·물체와의 최소 거리·충돌 여부가 출력됩니다.
"""

# ============================== 여기를 바꾸세요 ==============================

SPEED_MPS = 0.3        # 수평 비행 속도 [m/s]. 로드맵 §4.3 계약 상한은 0.5
ALTITUDE_M = 0.5       # 비행 고도 [m]. 물체 높이보다 낮으면 부딪힐 수 있음

WAYPOINTS = [          # 차례로 지나갈 (x, y) 지점 [m]. 첫 지점에서 이륙
    (0.0, 0.0),
    (1.0, 0.0),
    (1.0, 1.0),
    (0.0, 1.0),
    (0.0, 0.0),
]
LOOP = False           # True: 마지막 지점 다음 처음으로 돌아가 반복 / False: 마지막에서 호버

YAW_MODE = "travel"    # "travel": 진행 방향을 바라봄 / "fixed": FIXED_YAW_DEG로 고정
FIXED_YAW_DEG = 0.0    # 0 = +x 방향, 90 = +y 방향

OBJECTS = [            # 바닥에 놓을 물체. 원하는 만큼 추가/삭제
    # shape: "box" | "sphere" | "cylinder"
    # color: "red" "blue" "green" "yellow" "white" "black" 또는 (r, g, b) 0~1
    # size : box는 한 변의 절반, sphere/cylinder는 반지름 [m]
    # height: cylinder 높이 [m] (cylinder만)
    dict(shape="box",      color="red",   xy=(1.5, 0.5),  size=0.15),
    dict(shape="sphere",   color="blue",  xy=(1.5, -0.5), size=0.15),
    dict(shape="cylinder", color="green", xy=(0.5, 1.5),  size=0.10, height=0.6),
]

DURATION_S = 20.0      # 시뮬레이션 길이 [sim-초]
CAMERA_HZ = 5          # 앞 카메라 저장 주기 [Hz]. 60의 약수(1 2 3 4 5 6 10 ...), 0이면 저장 안 함
GUI_FOLLOW = True      # GUI 시점이 드론을 따라감

# ============================================================================

import argparse
import pathlib
import subprocess
import sys
import time

import numpy as np
import pybullet as p

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))     # scripts/ is not the package root

from dronevla.camera import FrontCamera                                # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl    # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary             # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics        # noqa: E402
from gym_pybullet_drones.utils.utils import sync                       # noqa: E402

PYB_HZ, CTRL_HZ = 240, 60      # §4.3 contract rates; not knobs
TAKEOFF_S = 2.0                # hold over the first waypoint while climbing
START_Z = 0.1
COLORS = {"red": (1, 0, 0), "blue": (0, 0, 1), "green": (0, 0.7, 0),
          "yellow": (1, 0.85, 0), "white": (0.95, 0.95, 0.95), "black": (0.1, 0.1, 0.1)}


# ------------------------------------------------------------------ knob checks
def check_knobs():
    problems = []
    if SPEED_MPS <= 0:
        problems.append("SPEED_MPS must be > 0")
    if ALTITUDE_M <= 0:
        problems.append("ALTITUDE_M must be > 0")
    if len(WAYPOINTS) < 1:
        problems.append("WAYPOINTS needs at least one point")
    if YAW_MODE not in ("travel", "fixed"):
        problems.append('YAW_MODE must be "travel" or "fixed"')
    if CAMERA_HZ < 0 or (CAMERA_HZ and CTRL_HZ % CAMERA_HZ):
        problems.append(f"CAMERA_HZ must divide {CTRL_HZ} (or be 0)")
    for i, o in enumerate(OBJECTS):
        if o.get("shape") not in ("box", "sphere", "cylinder"):
            problems.append(f"OBJECTS[{i}]: unknown shape {o.get('shape')!r}")
        c = o.get("color")
        if not (c in COLORS or (isinstance(c, (tuple, list)) and len(c) == 3)):
            problems.append(f"OBJECTS[{i}]: unknown color {c!r}")
    if problems:
        sys.exit("knob error:\n  " + "\n  ".join(problems))
    if SPEED_MPS > 0.5:
        print(f"[note] SPEED_MPS {SPEED_MPS} > 0.5 m/s: allowed here, but above the "
              "roadmap §4.3 contract cap, so a policy would never be asked to fly this fast")


# ------------------------------------------------------------------- the route
class Route:
    """A point that slides along the waypoint polyline at SPEED_MPS ("carrot").

    The PID chases the carrot, and the carrot's velocity is passed as a feed-forward, so
    changing SPEED_MPS changes how fast the drone is *asked* to fly. Whether it keeps up
    is what the tracking-error numbers at the end tell you.
    """

    def __init__(self, waypoints, speed, loop):
        pts = [np.array(w, dtype=float) for w in waypoints]
        if loop and len(pts) > 1 and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts.append(pts[0].copy())
        self.pts, self.speed, self.loop = pts, speed, loop
        self.seg_len = [float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:])]
        self.total = sum(self.seg_len)

    def at(self, t):
        """(xy, unit direction, moving?) at time t after takeoff."""
        if self.total == 0 or t <= 0:
            return self.pts[0], self._dir(0), False
        s = self.speed * t
        if self.loop:
            s %= self.total
        elif s >= self.total:
            return self.pts[-1], self._dir(len(self.seg_len) - 1), False
        for i, L in enumerate(self.seg_len):
            if s <= L or i == len(self.seg_len) - 1:
                d = self._dir(i)
                return self.pts[i] + d * min(s, L), d, True
            s -= L

    def _dir(self, i):
        if not self.seg_len:
            return np.array([1.0, 0.0])
        a, b = self.pts[i], self.pts[i + 1]
        n = np.linalg.norm(b - a)
        return (b - a) / n if n > 0 else np.array([1.0, 0.0])


# ------------------------------------------------------------------- the scene
def add_objects(client):
    ids = []
    for o in OBJECTS:
        rgba = list(COLORS.get(o["color"], o["color"])) + [1]
        x, y = o["xy"]
        r = float(o.get("size", 0.15))
        if o["shape"] == "box":
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[r] * 3, physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[r] * 3, rgbaColor=rgba,
                                      physicsClientId=client)
            z, top = r, 2 * r
        elif o["shape"] == "sphere":
            col = p.createCollisionShape(p.GEOM_SPHERE, radius=r, physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=rgba,
                                      physicsClientId=client)
            z, top = r, 2 * r
        else:
            h = float(o.get("height", 0.4))
            # note the asymmetry in PyBullet's API: collision takes `height`, visual `length`
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h,
                                         physicsClientId=client)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=rgba,
                                      physicsClientId=client)
            z, top = h / 2, h
        body = p.createMultiBody(0, col, vis, [x, y, z], physicsClientId=client)
        ids.append((body, o, top))
    return ids


def draw_route(route, client):
    for a, b in zip(route.pts, route.pts[1:]):
        p.addUserDebugLine([a[0], a[1], ALTITUDE_M], [b[0], b[1], ALTITUDE_M],
                           lineColorRGB=[1, 0.5, 0], lineWidth=2, physicsClientId=client)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no GUI window, run as fast as possible")
    args = ap.parse_args()
    check_knobs()
    gui = not args.headless

    run_dir = REPO_ROOT / "results/playground" / time.strftime("%Y%m%d_%H%M%S")
    frames_dir = run_dir / "frames"
    if CAMERA_HZ:
        frames_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)

    route = Route(WAYPOINTS, SPEED_MPS, LOOP)
    start = np.array([[*route.pts[0], START_Z]])
    env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=start, initial_rpys=np.zeros((1, 3)),
                     physics=Physics.PYB, pyb_freq=PYB_HZ, ctrl_freq=CTRL_HZ,
                     gui=gui, record=False, obstacles=False, user_debug_gui=False)
    ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
    obs, _ = env.reset(seed=0)
    objects = add_objects(env.CLIENT)
    if gui:
        draw_route(route, env.CLIENT)
    cam = FrontCamera()
    drone = int(env.DRONE_IDS[0])

    steps = int(DURATION_S * CTRL_HZ)
    cam_every = CTRL_HZ // CAMERA_HZ if CAMERA_HZ else 0
    log = {k: [] for k in ("t", "pos", "vel", "target", "moving")}
    contacts = {i: 0 for i in range(len(objects))}
    min_dist = {i: np.inf for i in range(len(objects))}
    action = np.zeros((1, 4))
    yaw = np.radians(FIXED_YAW_DEG)
    wall0 = time.time()

    for i in range(steps):
        t = i / CTRL_HZ
        xy, d, moving = route.at(t - TAKEOFF_S)
        target = np.array([xy[0], xy[1], ALTITUDE_M])
        target_vel = np.array([d[0], d[1], 0.0]) * SPEED_MPS if moving else np.zeros(3)
        if YAW_MODE == "travel" and moving:
            yaw = float(np.arctan2(d[1], d[0]))

        obs, _, _, _, _ = env.step(action)
        state = obs[0]
        action[0], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP, state=state,
            target_pos=target, target_rpy=np.array([0.0, 0.0, yaw]), target_vel=target_vel)

        log["t"].append(t)
        log["pos"].append(state[0:3].copy())
        log["vel"].append(state[10:13].copy())
        log["target"].append(target)
        log["moving"].append(moving)

        for k, (body, _, _) in enumerate(objects):
            if p.getContactPoints(bodyA=drone, bodyB=body, physicsClientId=env.CLIENT):
                contacts[k] += 1
            cp = p.getClosestPoints(drone, body, distance=5.0, physicsClientId=env.CLIENT)
            if cp:
                min_dist[k] = min(min_dist[k], min(c[8] for c in cp))

        if cam_every and i % cam_every == 0:
            from PIL import Image
            rgb = cam.render(state[0:3], state[3:7], client=env.CLIENT)
            Image.fromarray(rgb).save(frames_dir / f"frame_{i // cam_every:05d}.png")

        if gui:
            if GUI_FOLLOW:
                p.resetDebugVisualizerCamera(cameraDistance=2.2, cameraYaw=-45, cameraPitch=-35,
                                             cameraTargetPosition=state[0:3].tolist(),
                                             physicsClientId=env.CLIENT)
            sync(i, wall0, env.CTRL_TIMESTEP)

    env.close()
    report(log, objects, contacts, min_dist, route, run_dir, frames_dir if CAMERA_HZ else None,
           time.time() - wall0)


# -------------------------------------------------------------------- results
def report(log, objects, contacts, min_dist, route, run_dir, frames_dir, wall_s):
    t = np.array(log["t"])
    pos, vel, tgt = np.array(log["pos"]), np.array(log["vel"]), np.array(log["target"])
    moving = np.array(log["moving"])
    hspeed = np.linalg.norm(vel[:, :2], axis=1)
    err = np.linalg.norm(pos - tgt, axis=1)
    airborne = t >= TAKEOFF_S

    print("\n================ playground result ================")
    print(f"knobs   : SPEED_MPS={SPEED_MPS}  ALTITUDE_M={ALTITUDE_M}  "
          f"{len(WAYPOINTS)} waypoints  LOOP={LOOP}  {len(OBJECTS)} objects")
    print(f"time    : {t[-1] + 1 / CTRL_HZ:.1f} sim-s in {wall_s:.1f} wall-s")
    if moving.any():
        print(f"speed   : asked {SPEED_MPS:.2f} m/s | flown mean {hspeed[moving].mean():.3f}, "
              f"max {hspeed.max():.3f} m/s (while the route was moving)")
        print(f"tracking: drone-to-target distance mean {err[moving].mean():.3f} m, "
              f"max {err[moving].max():.3f} m")
    else:
        print("speed   : the route never moved (still taking off, or one waypoint)")
    print(f"route   : {route.total:.2f} m long, needs {route.total / SPEED_MPS:.1f} s at this "
          f"speed (+{TAKEOFF_S:.0f} s takeoff); DURATION_S is {DURATION_S:.0f} s")
    if not LOOP:
        # Judge arrival by where the drone actually is, not by where the target got to:
        # a drone pinned against an obstacle still has its target reach the end.
        end_err = float(np.linalg.norm(pos[-1, :2] - route.pts[-1]))
        verdict = "arrived" if end_err < 0.15 else "did NOT arrive"
        print(f"arrival : drone ended {end_err:.3f} m from the last waypoint -> {verdict}")
    if airborne.any():
        print(f"altitude: {pos[airborne, 2].min():.3f} .. {pos[airborne, 2].max():.3f} m "
              f"(asked {ALTITUDE_M})")
    for k, (_, o, top) in enumerate(objects):
        hit = f"HIT for {contacts[k] / CTRL_HZ:.2f} s" if contacts[k] else "no contact"
        print(f"object  : {o['color']!s:>6} {o['shape']:<8} at {tuple(o['xy'])}, top z={top:.2f} m"
              f" -> closest {min_dist[k]:.3f} m, {hit}")
    if pos[-1, 2] < 0.05 and t[-1] > TAKEOFF_S:
        print("WARNING : the drone ended on the ground -- it probably crashed")

    plot(t, pos, tgt, hspeed, err, objects, route, run_dir / "summary.png")
    print(f"\nsaved   : {run_dir / 'summary.png'}")
    if frames_dir is not None:
        sheet = run_dir / "camera_sheet.png"
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts/contact_sheet.py"),
                        "--frames-dir", str(frames_dir), "--out", str(sheet)],
                       check=False, capture_output=True)
        print(f"          {sheet}")


def plot(t, pos, tgt, hspeed, err, objects, route, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig = plt.figure(figsize=(12, 5.5))
    ax = fig.add_subplot(1, 2, 1)
    pts = np.array(route.pts)
    ax.plot(pts[:, 0], pts[:, 1], "--", c="tab:orange", lw=1.5, label="planned route")
    ax.plot(pos[:, 0], pos[:, 1], c="tab:blue", lw=1.8, label="flown path")
    ax.plot(*pos[0, :2], "o", c="tab:blue", ms=6)
    for _, o, _ in objects:
        c = COLORS.get(o["color"], o["color"])
        r = float(o.get("size", 0.15))
        x, y = o["xy"]
        patch = (Rectangle((x - r, y - r), 2 * r, 2 * r, color=c, alpha=0.7) if o["shape"] == "box"
                 else Circle((x, y), r, color=c, alpha=0.7))
        ax.add_patch(patch)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Top view — speed {SPEED_MPS} m/s, altitude {ALTITUDE_M} m")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=9)

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.plot(t, hspeed, c="tab:blue", label="flown horizontal speed")
    ax1.axhline(SPEED_MPS, ls="--", c="tab:orange", label="asked")
    ax1.axvline(TAKEOFF_S, c="0.6", lw=1)
    ax1.set_ylabel("speed [m/s]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(2, 2, 4, sharex=ax1)
    ax2.plot(t, err, c="tab:red", label="distance to target")
    ax2.plot(t, pos[:, 2], c="tab:green", label="altitude z")
    ax2.axvline(TAKEOFF_S, c="0.6", lw=1)
    ax2.set_xlabel("sim time [s]"); ax2.set_ylabel("[m]")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    main()
