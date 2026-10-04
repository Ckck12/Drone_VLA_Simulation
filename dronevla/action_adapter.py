"""Action contract enforcement: caps, NaN rejection, body<->world velocity conversion.

    *** This whole module is the Python stand-in for C1 (`dronevla_core`). ***

Roadmap v3 §3.2 makes frame conversion, the §4.3 action caps and the watchdog a C++17
library that the Python simulator adapter must call, and forbids keeping a production
Python copy once that exists. So everything of that kind lives here and nowhere else,
and the day `_dronevla_core` lands, this module's functions are replaced by bindings and
the Python bodies are deleted -- not kept in parallel.

Conventions (§4.3):
* Policy actions are `(vx, vy, vz, yaw_rate, stop_logit)` in the **body FLU** frame:
  x forward, y left, z up; m/s and rad/s. "Body" for a velocity setpoint means the
  heading frame -- rotated by yaw only -- the usual meaning for multirotor velocity
  commands, so a level-flight command does not depend on momentary roll/pitch.
* World is PyBullet's frame, treated as ENU: x, y horizontal, z up.
* Caps: horizontal speed (vector norm) <= 0.5 m/s, |vz| <= 0.3 m/s, |yaw_rate| <= 0.5
  rad/s. Applied in a fixed order: reject non-finite -> planar zeroing -> horizontal norm
  -> vz -> yaw rate. The order is part of the contract (C1 fixtures will pin it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ActionLimits:
    horizontal_mps: float = 0.5
    vertical_mps: float = 0.3
    yaw_rate_radps: float = 0.5


@dataclass
class AdaptedAction:
    """What happened to one policy action. `raw` is kept even when the action is rejected."""

    raw: np.ndarray                      # (5,) exactly as the policy produced it
    applied: np.ndarray | None           # (4,) body FLU setpoint actually used, or None
    stop_positive: bool                  # stop_logit > 0
    rejected: str | None = None          # reason, if the action could not be applied
    flags: dict = field(default_factory=dict)   # which caps / zeroings engaged


def adapt(raw, limits: ActionLimits = ActionLimits(), planar: bool = True) -> AdaptedAction:
    """Validate and cap one raw policy action. Never raises on bad numbers -- it reports."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw.shape != (5,):
        return AdaptedAction(raw=raw, applied=None, stop_positive=False,
                             rejected=f"wrong shape {raw.shape}, expected (5,)")
    if not np.all(np.isfinite(raw)):
        return AdaptedAction(raw=raw, applied=None, stop_positive=False,
                             rejected="non-finite value in action")

    vx, vy, vz, yaw_rate, stop_logit = raw
    flags = {"planar_zeroed": False, "horizontal_capped": False,
             "vertical_capped": False, "yaw_capped": False}

    if planar and (vz != 0.0 or yaw_rate != 0.0):
        flags["planar_zeroed"] = True
    if planar:
        vz, yaw_rate = 0.0, 0.0

    speed = math.hypot(vx, vy)
    if speed > limits.horizontal_mps:
        scale = limits.horizontal_mps / speed
        vx, vy = vx * scale, vy * scale
        flags["horizontal_capped"] = True
    if abs(vz) > limits.vertical_mps:
        vz = math.copysign(limits.vertical_mps, vz)
        flags["vertical_capped"] = True
    if abs(yaw_rate) > limits.yaw_rate_radps:
        yaw_rate = math.copysign(limits.yaw_rate_radps, yaw_rate)
        flags["yaw_capped"] = True

    return AdaptedAction(raw=raw, applied=np.array([vx, vy, vz, yaw_rate]),
                         stop_positive=bool(stop_logit > 0.0), flags=flags)


def body_to_world_velocity(v_body, yaw: float) -> np.ndarray:
    """Heading-frame FLU velocity -> world ENU velocity (rotation about z by `yaw`)."""
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, vz = v_body
    return np.array([c * vx - s * vy, s * vx + c * vy, vz])


def world_to_body_velocity(v_world, yaw: float) -> np.ndarray:
    """Inverse of `body_to_world_velocity`."""
    c, s = math.cos(yaw), math.sin(yaw)
    vx, vy, vz = v_world
    return np.array([c * vx + s * vy, -s * vx + c * vy, vz])
