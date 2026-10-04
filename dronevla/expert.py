"""Scripted oracle expert for `DroneTargetPairsEnv` (roadmap v3 §4.2 item 1-2).

**Privileged inputs** (the datasheet must list these): the true pose and velocity from the
simulator, and the goal hover point. Both come from `info["privileged"]`, which the
policy never sees.

It acts through exactly the same interface as a policy: one `(vx, vy, vz, yaw_rate,
stop_logit)` action per 0.2 s, through the same adapter caps, the same PID and the same
physics. No teleporting. Its commands stay inside the caps, so the action it emits is the
action that is applied -- which is what makes it usable as a teacher label (§4.2 item 2).
"""
from __future__ import annotations

import numpy as np

from dronevla.action_adapter import world_to_body_velocity
from dronevla.task import TaskConfig


class StraightLineExpert:
    """Fly straight at the goal hover point, slow down proportionally, then Stop."""

    def __init__(self, cfg: TaskConfig = TaskConfig(), gain: float = 1.0,
                 stop_radius_m: float = 0.12, stop_speed_mps: float = 0.08,
                 goal_key: str = "goal_hover_point"):
        if stop_radius_m >= cfg.success_radius_m:
            raise ValueError("the expert must stop well inside the success radius")
        self.cfg, self.gain = cfg, gain
        self.stop_radius, self.stop_speed = stop_radius_m, stop_speed_mps
        # goal_key="other_hover_point" makes a deliberately wrong expert for evaluator tests
        self.goal_key = goal_key

    def act(self, info) -> np.ndarray:
        priv = info["privileged"]
        pos = np.asarray(priv["true_pos"])
        vel = np.asarray(priv["true_vel"])
        goal = np.asarray(priv[self.goal_key])

        d = goal[:2] - pos[:2]
        dist = float(np.linalg.norm(d))
        if dist < self.stop_radius and float(np.linalg.norm(vel[:2])) < self.stop_speed:
            return np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        v = self.gain * d
        speed = float(np.linalg.norm(v))
        cap = self.cfg.limits.horizontal_mps
        if speed > cap:
            v *= cap / speed
        vb = world_to_body_velocity([v[0], v[1], 0.0], priv["true_yaw"])
        return np.array([vb[0], vb[1], 0.0, 0.0, -1.0], dtype=np.float32)
