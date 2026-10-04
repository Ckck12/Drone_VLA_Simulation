"""Realism layers on top of gym-pybullet-drones, without modifying the pinned clone.

* `WindyCtrlAviary` -- wind. The library's rotor-drag model (`BaseAviary._drag`, from the
  Crazyflie system identification in Förster 2015) pushes against the drone's velocity
  relative to the *ground*. Real drag acts on velocity relative to the *air*. This subclass
  makes that one substitution, so with no wind it is exactly the library's drag, and with
  wind the same coefficients see `v_drone - v_wind`. It runs once per physics substep
  (240 Hz), because the library calls `_drag` inside its substep loop.

* `TiltLimitedDSLPIDControl` -- a maximum tilt, like the 35 deg a DJI Mavic 4 Pro allows.
  `DSLPIDControl` has no tilt limit: it will command any attitude its position loop asks
  for. This subclass takes the target attitude the library computed and, if its thrust axis
  leans further than the limit from vertical, rotates it back onto the limit, keeping the
  direction of the lean and the commanded heading.
"""
from __future__ import annotations

import math

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import Physics

_DRAG_PHYSICS = (Physics.PYB_DRAG, Physics.PYB_GND_DRAG_DW)


class GaussMarkov:
    """A first-order Gauss-Markov process per axis: slowly wandering randomness.

    The value wanders with time constant `tau_s`, and its long-run standard deviation is
    exactly `sigma`. Used for two things:

    * Position-estimate error: white noise at 60 Hz would be the wrong model -- real
      position estimates (GNSS, visual odometry) are wrong in a *slowly drifting* way, which
      is what makes a real drone wander while hovering instead of buzzing in place.
    * Wind gusts: gusts rise and fall over seconds, not from one control step to the next.

    `seed` may be anything `np.random.default_rng` accepts, e.g. `[snapshot_seed, 1]`, so
    independent streams can be derived from one episode seed.

    `smooth_tau_s` > 0 passes the process through a first-order low-pass. A plain
    Gauss-Markov process is continuous but jagged: at 60 Hz with sigma 0.03 m and tau 1 s it
    jumps ~5 mm per step, which a PID chases as if it were motion. A real estimator (an EKF
    fusing IMU with GNSS or vision) is smooth at that time scale. The input sigma is scaled
    by sqrt((tau + smooth) / tau), the variance ratio of a Gauss-Markov process through a
    first-order lag, so the output standard deviation stays `sigma`.
    """

    def __init__(self, sigma_xyz, tau_s, dt, seed, smooth_tau_s: float = 0.0):
        self.sigma = np.asarray(sigma_xyz, dtype=float)
        self.a = math.exp(-dt / tau_s)
        self.b = math.sqrt(1.0 - self.a ** 2)
        self.rng = np.random.default_rng(seed)
        if smooth_tau_s > 0:
            self.alpha = 1.0 - math.exp(-dt / smooth_tau_s)
            self.sigma_in = self.sigma * math.sqrt((tau_s + smooth_tau_s) / tau_s)
        else:
            self.alpha, self.sigma_in = None, self.sigma
        self.x = self.rng.normal(size=3) * self.sigma_in   # start in the stationary state
        self.e = self.x.copy()

    def step(self):
        self.x = self.a * self.x + self.b * self.sigma_in * self.rng.normal(size=3)
        self.e = self.x.copy() if self.alpha is None else self.e + self.alpha * (self.x - self.e)
        return self.e


class WindyCtrlAviary(CtrlAviary):
    """`CtrlAviary` whose rotor drag acts on airspeed. Set `wind_world` (m/s) any time."""

    def __init__(self, *args, physics=Physics.PYB_DRAG, **kwargs):
        if physics not in _DRAG_PHYSICS:
            raise ValueError(f"wind needs a physics mode that applies drag, one of {_DRAG_PHYSICS}")
        self.wind_world = np.zeros(3)
        super().__init__(*args, physics=physics, **kwargs)

    def _drag(self, rpm, nth_drone):
        # Same as BaseAviary._drag line for line, except `vel` becomes `vel - wind`.
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2 * np.pi * rpm / 60))
        airspeed = np.array(self.vel[nth_drone, :]) - self.wind_world
        drag = np.dot(base_rot.T, drag_factors * airspeed)
        p.applyExternalForce(self.DRONE_IDS[nth_drone], 4, forceObj=drag, posObj=[0, 0, 0],
                             flags=p.LINK_FRAME, physicsClientId=self.CLIENT)


def limit_tilt(target_euler, target_yaw, max_tilt_rad):
    """Clamp the lean of a target attitude. Returns (euler, was_limited, requested_tilt_rad).

    Rebuilds the attitude the same way `DSLPIDControl._dslPIDPositionControl` does: the
    z axis is the thrust direction, and x/y follow from the commanded yaw.
    """
    z = Rotation.from_euler("XYZ", target_euler).as_matrix()[:, 2]
    tilt = math.acos(max(-1.0, min(1.0, z[2])))
    if tilt <= max_tilt_rad:
        return target_euler, False, tilt
    horiz = z[:2] / np.linalg.norm(z[:2])
    z_new = np.array([math.sin(max_tilt_rad) * horiz[0], math.sin(max_tilt_rad) * horiz[1],
                      math.cos(max_tilt_rad)])
    x_c = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0])
    y_ax = np.cross(z_new, x_c)
    y_ax /= np.linalg.norm(y_ax)
    x_ax = np.cross(y_ax, z_new)
    euler = Rotation.from_matrix(np.column_stack([x_ax, y_ax, z_new])).as_euler("XYZ")
    return euler, True, tilt


class TiltLimitedDSLPIDControl(DSLPIDControl):
    """`DSLPIDControl` that never asks for more than `max_tilt_deg` of lean.

    Counts how often the limit engaged, so a run can report it. `max_tilt_deg=None`
    disables the limit and makes this identical to the library controller.
    """

    def __init__(self, *args, max_tilt_deg=35.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tilt_rad = None if max_tilt_deg is None else math.radians(max_tilt_deg)
        self.limited_steps = 0
        self.requested_tilt_max_rad = 0.0

    def reset(self):
        super().reset()
        self.limited_steps = 0
        self.requested_tilt_max_rad = 0.0

    def _dslPIDPositionControl(self, control_timestep, cur_pos, cur_quat, cur_vel,
                               target_pos, target_rpy, target_vel):
        thrust, target_euler, pos_e = super()._dslPIDPositionControl(
            control_timestep, cur_pos, cur_quat, cur_vel, target_pos, target_rpy, target_vel)
        # with the limit off, pi/2 never clamps, but the requested lean is still recorded
        cap = math.pi / 2 if self.max_tilt_rad is None else self.max_tilt_rad
        limited_euler, limited, requested = limit_tilt(target_euler, target_rpy[2], cap)
        self.requested_tilt_max_rad = max(self.requested_tilt_max_rad, requested)
        if self.max_tilt_rad is None:
            return thrust, target_euler, pos_e
        self.limited_steps += int(limited)
        return thrust, limited_euler, pos_e
