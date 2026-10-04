"""Roadmap v3 line 302 — single-drone hover with the official DSL PID controller.

Adapted from the repo example `examples/pid.py`, which flies a *circular* trajectory with
3 drones. Here the controller and environment are identical; only the target is changed to a
constant point, so the plot shows a clean step response in z plus the steady-state hold.
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless: write a PNG, never open a window
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.enums import DroneModel, Physics

OUT = pathlib.Path.home() / "dronevla/results/phase0"
OUT.mkdir(parents=True, exist_ok=True)

PYB_HZ, CTRL_HZ, DURATION = 240, 48, 8.0
START = np.array([[0.0, 0.0, 0.10]])
TARGET = np.array([0.0, 0.0, 1.00])

env = CtrlAviary(drone_model=DroneModel.CF2X, num_drones=1,
                 initial_xyzs=START, initial_rpys=np.zeros((1, 3)),
                 physics=Physics.PYB, pyb_freq=PYB_HZ, ctrl_freq=CTRL_HZ,
                 gui=False, obstacles=False)
ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

action = np.zeros((1, 4))
log_t, log_xyz = [], []
obs, _ = env.reset(seed=0)

for i in range(int(DURATION * CTRL_HZ)):
    obs, _, _, _, _ = env.step(action)
    state = obs[0]
    action[0], _, _ = ctrl.computeControlFromState(
        control_timestep=env.CTRL_TIMESTEP, state=state,
        target_pos=TARGET, target_rpy=np.zeros(3))
    log_t.append(i / CTRL_HZ)
    log_xyz.append(state[0:3].copy())

env.close()
t = np.asarray(log_t)
xyz = np.asarray(log_xyz)
z = xyz[:, 2]

# --- step-response metrics on z ---------------------------------------------
z0, zf = START[0, 2], TARGET[2]
span = zf - z0
rise = None
above10 = np.argmax(z >= z0 + 0.1 * span)
above90 = np.argmax(z >= z0 + 0.9 * span)
if z.max() >= z0 + 0.9 * span:
    rise = t[above90] - t[above10]
overshoot = (z.max() - zf) / span * 100
tail = z[t >= DURATION - 2.0]                       # last 2 s = steady state
sse = tail.mean() - zf
band = 0.02 * span                                  # +-2 % settling band
outside = np.where(np.abs(z - zf) > band)[0]
settle = t[outside[-1]] if len(outside) else 0.0
drift_xy = np.linalg.norm(xyz[t >= DURATION - 2.0, 0:2], axis=1).max()

print(f"rise time (10-90%)      : {rise:.3f} s" if rise else "rise time: target never reached")
print(f"overshoot               : {overshoot:+.2f} %")
print(f"settling time (+-2%)    : {settle:.3f} s")
print(f"steady-state error (z)  : {sse*1000:+.1f} mm  (mean of last 2 s)")
print(f"max xy drift (last 2 s) : {drift_xy*1000:.1f} mm")

# --- plot --------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
ax[0].axhline(zf, ls="--", c="0.5", lw=1, label=f"target z = {zf:.2f} m")
ax[0].axhspan(zf - band, zf + band, color="0.85", label="±2% band")
ax[0].plot(t, z, lw=1.6, color="#1f77b4", label="measured z")
ax[0].set_ylabel("z  [m]")
ax[0].set_title("Single-drone hover — DSLPIDControl (CF2X), PyBullet 240 Hz / control 48 Hz")
ax[0].legend(loc="lower right", fontsize=9)
ax[0].grid(alpha=.3)

ax[1].plot(t, xyz[:, 0] * 1000, lw=1.4, label="x")
ax[1].plot(t, xyz[:, 1] * 1000, lw=1.4, label="y")
ax[1].set_ylabel("horizontal drift  [mm]")
ax[1].set_xlabel("time  [s]")
ax[1].legend(fontsize=9)
ax[1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUT / "hover_pid.png", dpi=130)
np.savetxt(OUT / "hover_pid.csv",
           np.column_stack([t, xyz]), delimiter=",", header="t,x,y,z", comments="")
print("saved", OUT / "hover_pid.png")
