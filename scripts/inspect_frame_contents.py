"""What is actually in a recorded frame? Identify every pixel by body id.

The contact sheet from the 1000-frame profile shows a grey wedge along the bottom of every
frame. Guessing whether that is the drone's own airframe, its shadow, or scene geometry is
not good enough, so this renders one pose with the segmentation mask on and reports the
pixel share of each body id by name.

    python scripts/inspect_frame_contents.py
"""
import pathlib
import sys

import numpy as np
import pybullet as p

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))     # scripts/ is not the package root

from dronevla.camera import FrontCamera                                  # noqa: E402
from dronevla.profile_env import build_targets, episode_plan, waypoint   # noqa: E402
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl      # noqa: E402
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary               # noqa: E402
from gym_pybullet_drones.utils.enums import DroneModel, Physics          # noqa: E402

rng = np.random.default_rng([0, 0])
plan = episode_plan(rng, 20.0)
start_pos, start_yaw = waypoint(plan, 0.0)

env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                 initial_xyzs=start_pos.reshape(1, 3),
                 initial_rpys=np.array([[0.0, 0.0, start_yaw]]),
                 physics=Physics.PYB, pyb_freq=240, ctrl_freq=60,
                 gui=False, obstacles=False, user_debug_gui=False)
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
obs, _ = env.reset(seed=0)
scene = build_targets(env.CLIENT, rng)

# settle for 1 sim-second so the pose matches a recorded frame rather than the spawn pose
action = np.zeros((1, 4))
for _ in range(60):
    obs, _, _, _, _ = env.step(action)
    action[0], _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP, state=obs[0],
        target_pos=start_pos, target_rpy=np.array([0.0, 0.0, start_yaw]))

state = env._getDroneStateVector(0)
cam = FrontCamera(segmentation=True)
view = cam.view_matrix(state[0:3], state[3:7])
w, h = cam.width, cam.height
_, _, rgba, _, seg = p.getCameraImage(
    width=w, height=h, viewMatrix=view, projectionMatrix=cam._projection,
    shadow=0, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
    renderer=p.ER_TINY_RENDERER, physicsClientId=env.CLIENT)
seg = np.asarray(seg).reshape(h, w)

# ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX packs body and link into one int:
#   objectUniqueId = value & ((1 << 24) - 1);  linkIndex = (value >> 24) - 1
names = {
    int(env.PLANE_ID): "ground plane",
    int(env.DRONE_IDS[0]): "the drone's own airframe",
    scene["red_cube_id"]: "red cube target",
    scene["blue_sphere_id"]: "blue sphere target",
    -1: "background (sky, nothing hit)",
}

print(f"pose: xyz={np.round(state[0:3], 3).tolist()}  yaw={np.degrees(state[9]):.1f} deg")
print(f"frame {w}x{h} = {w * h} pixels, shadow off\n")
print(f"{'body id':>8}  {'link':>5}  {'pixels':>7}  {'share':>7}  what")
rows = []
for value in np.unique(seg):
    body = int(value) & ((1 << 24) - 1) if int(value) >= 0 else -1
    link = (int(value) >> 24) - 1 if int(value) >= 0 else None
    n = int((seg == value).sum())
    rows.append((n, body, link, value))
for n, body, link, value in sorted(rows, reverse=True):
    what = names.get(body, f"unknown body {body}")
    print(f"{body:>8}  {str(link):>5}  {n:>7}  {100 * n / (w * h):>6.2f}%  {what}")

env.close()
