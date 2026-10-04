"""Roadmap v3 line 302 — velocity step response with the official velocity controller.

Adapted from the repo example `examples/pid_velocity.py`, which commands 4 drones through
several switching velocity targets at once. Here it is one drone and a single clean step,
so rise time and steady-state error are actually readable off the plot.

VelocityAviary action = [dir_x, dir_y, dir_z, fraction]; the env targets
    SPEED_LIMIT * |fraction| * unit(dir)
"""
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

OUT = pathlib.Path.home() / "dronevla/results/phase0"
OUT.mkdir(parents=True, exist_ok=True)

PYB_HZ, CTRL_HZ = 240, 48
HOLD, STEP, BACK = 1.0, 3.0, 2.0          # seconds: settle, step on, step off
DURATION = HOLD + STEP + BACK
# VelocityAviary caps speed at SPEED_LIMIT = 0.03 * MAX_SPEED_KMH / 3.6 = 0.25 m/s for the CF2X.
# The action's 4th component is a *fraction* bounded to [0, 1] by the action space, so a target
# above SPEED_LIMIT would need fraction > 1 — out of bounds. The env does not clip it, so the
# assert below is what catches the mistake.
TARGET_MS = 0.20                          # commanded forward speed, m/s (80% of SPEED_LIMIT)

env = VelocityAviary(drone_model=DroneModel.CF2X, num_drones=1,
                     initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
                     initial_rpys=np.zeros((1, 3)),
                     physics=Physics.PYB, pyb_freq=PYB_HZ, ctrl_freq=CTRL_HZ,
                     gui=False, obstacles=False)

frac = TARGET_MS / env.SPEED_LIMIT        # SPEED_LIMIT is m/s inside the env
print(f"SPEED_LIMIT = {env.SPEED_LIMIT:.3f} m/s -> fraction {frac:.4f} for {TARGET_MS} m/s")
assert 0.0 <= frac <= 1.0, (
    f"fraction {frac:.3f} is outside the action space bound [0, 1]; "
    f"TARGET_MS must be <= SPEED_LIMIT = {env.SPEED_LIMIT:.3f} m/s")

obs, _ = env.reset(seed=0)
log_t, log_vx, log_cmd, log_z = [], [], [], []

for i in range(int(DURATION * CTRL_HZ)):
    t = i / CTRL_HZ
    cmd = TARGET_MS if HOLD <= t < HOLD + STEP else 0.0
    action = np.array([[1.0, 0.0, 0.0, frac if cmd else 0.0]])
    obs, _, _, _, _ = env.step(action)
    state = obs[0]
    log_t.append(t)
    log_vx.append(state[10])              # world-frame vx
    log_cmd.append(cmd)
    log_z.append(state[2])

env.close()
t = np.asarray(log_t); vx = np.asarray(log_vx)
cmd = np.asarray(log_cmd); z = np.asarray(log_z)

# --- metrics over the commanded step -----------------------------------------
on = (t >= HOLD) & (t < HOLD + STEP)
seg_t, seg_v = t[on], vx[on]
i10 = np.argmax(seg_v >= 0.1 * TARGET_MS)
i90 = np.argmax(seg_v >= 0.9 * TARGET_MS)
rise = seg_t[i90] - seg_t[i10] if seg_v.max() >= 0.9 * TARGET_MS else None
steady = vx[(t >= HOLD + STEP - 1.0) & (t < HOLD + STEP)].mean()
overshoot = (seg_v.max() - TARGET_MS) / TARGET_MS * 100
back = vx[t >= DURATION - 0.5].mean()

print(f"rise time (10-90%)     : {rise:.3f} s" if rise else "rise time: 90% never reached")
print(f"overshoot              : {overshoot:+.1f} %")
print(f"steady-state vx        : {steady:.3f} m/s  (target {TARGET_MS})  err {steady-TARGET_MS:+.3f}")
print(f"vx after step off      : {back:+.3f} m/s")
print(f"altitude hold          : {z.min():.3f} .. {z.max():.3f} m  (start 1.000)")

# --- plot ---------------------------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
ax[0].step(t, cmd, where="post", ls="--", c="0.45", lw=1.5, label="commanded vx")
ax[0].plot(t, vx, lw=1.7, color="#d62728", label="measured vx")
ax[0].axhspan(TARGET_MS * .98, TARGET_MS * 1.02, color="0.87", label="±2% of target")
ax[0].set_ylabel("forward velocity  [m/s]")
ax[0].set_title("Velocity step response — VelocityAviary (CF2X), PyBullet 240 Hz / control 48 Hz")
ax[0].legend(loc="upper right", fontsize=9); ax[0].grid(alpha=.3)

ax[1].axhline(1.0, ls="--", c="0.5", lw=1, label="initial altitude")
ax[1].plot(t, z, lw=1.5, color="#2ca02c", label="measured z")
ax[1].set_ylabel("z  [m]"); ax[1].set_xlabel("time  [s]")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUT / "velocity_step.png", dpi=130)
np.savetxt(OUT / "velocity_step.csv",
           np.column_stack([t, cmd, vx, z]), delimiter=",",
           header="t,cmd_vx,vx,z", comments="")
print("saved", OUT / "velocity_step.png")
