"""`DroneTargetPairsEnv` -- the Gymnasium environment for roadmap v3 Phase 1.

One policy step = 0.2 sim-seconds = 12 control updates = 48 physics steps (§4.3).

Observation (exactly these keys, nothing privileged -- tested):
    rgb          (96, 128, 3) uint8, front gimbal camera, rendered from the TRUE pose
    proprio      (11,) float32: body-FLU velocity (3), roll, pitch, sin(yaw), cos(yaw),
                 body angular velocity (3), altitude. **True sim state** (§4.3: "sim
                 state임을 표시"); PyBullet's angular velocity is world-frame and is
                 rotated into the body frame here.
    instruction  the fixed English instruction for this episode

Action: (vx, vy, vz, yaw_rate, stop_logit), body FLU, m/s and rad/s. Capped and
validated by `dronevla.action_adapter` (the stand-in for C1). The velocity is tracked by
integrating a position setpoint that the PID chases, with the velocity as feed-forward;
with a zero command the setpoint stands still, which is the pose-hold controller.

The controller is given the *estimated* position (truth + slowly drifting error). Every
judgement -- collision, bounds, success -- uses the truth.

Stop (§4.3): `stop_logit > 0` on `stop_consecutive` policy steps in a row requests a
terminal Stop. The env then switches to pose hold for `hold_s`, judges, and terminates.

Outcomes (§4.3 / §7.1), exactly one per finished episode:
    success           held inside the goal's success radius, speed <= stop_speed throughout
    stop_not_settled  inside the goal's radius throughout the hold, but still moving
    wrong_target      held inside the *other* target's success radius
    stop_elsewhere    stopped anywhere else
    collision         any contact (the episode starts in the air, so this includes ground)
    out_of_bounds     left the room
    timeout           `timeout_s` of episode time without a Stop  (truncated, not terminated)
    invalid_action    non-finite or wrongly shaped action (§7.1: counts as a failure)

Determinism: everything random -- layout, estimate noise, gusts -- derives from the
episode seed, never from which target the instruction names, so the two episodes of a
counterfactual pair see byte-identical first observations (§4.2 item 5; tested).
"""
from __future__ import annotations

import math
import string
import time

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces

from dronevla.action_adapter import adapt, body_to_world_velocity
from dronevla.camera import FrontCamera
from dronevla.sim import GaussMarkov, TiltLimitedDSLPIDControl, WindyCtrlAviary
from dronevla.task import COLORS, Layout, TaskConfig, make_instruction, sample_layout
from gym_pybullet_drones.utils.enums import DroneModel, Physics

INSTRUCTION_CHARSET = string.ascii_letters + string.digits + " .,'-"
OUTCOMES = ("success", "stop_not_settled", "wrong_target", "stop_elsewhere",
            "collision", "out_of_bounds", "timeout", "invalid_action")


class DroneTargetPairsEnv(gymnasium.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: TaskConfig = TaskConfig(), gui: bool = False,
                 realtime: bool = False):
        cfg = self.cfg = config
        if cfg.pyb_hz % cfg.ctrl_hz or cfg.ctrl_hz % cfg.policy_hz:
            raise ValueError("rates must divide: pyb_hz / ctrl_hz / policy_hz")
        self._ctrl_per_policy = cfg.ctrl_hz // cfg.policy_hz
        self._dt = 1.0 / cfg.ctrl_hz

        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0, 255, shape=(96, 128, 3), dtype=np.uint8),
            "proprio": spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32),
            "instruction": spaces.Text(max_length=96, charset=INSTRUCTION_CHARSET),
        })
        lim = cfg.limits
        hi = np.array([lim.horizontal_mps, lim.horizontal_mps, lim.vertical_mps,
                       lim.yaw_rate_radps, 10.0], dtype=np.float32)
        # The box is per-axis; the adapter additionally caps the horizontal *norm*.
        # Like the library's own action spaces, the box itself is not enforced -- the
        # adapter is (Phase 0 found BaseAviary ignores its bounds).
        self.action_space = spaces.Box(-hi, hi, dtype=np.float32)

        self._gui = gui
        self._realtime = realtime or gui
        self._aviary = WindyCtrlAviary(
            drone_model=DroneModel.CF2X, num_drones=1,
            initial_xyzs=np.array([[cfg.start_x, 0.0, cfg.altitude_m]]),
            initial_rpys=np.zeros((1, 3)), physics=Physics.PYB_DRAG,
            pyb_freq=cfg.pyb_hz, ctrl_freq=cfg.ctrl_hz, gui=gui, record=False,
            obstacles=False, user_debug_gui=False)
        self._client = self._aviary.CLIENT
        self._ctrl = TiltLimitedDSLPIDControl(drone_model=DroneModel.CF2X,
                                              max_tilt_deg=cfg.max_tilt_deg)
        legacy = cfg.camera_mount == "legacy"
        self._cam = FrontCamera(mount=cfg.camera_mount,
                                tilt_deg=0.0 if legacy else cfg.camera_tilt_deg,
                                fov_deg=cfg.camera_vfov_deg,
                                near=0.0397 if legacy else 0.01)
        self._layout: Layout | None = None
        self._done = True

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        if seed is None:
            seed = int(self.np_random.integers(0, 2 ** 31 - 1))
        cfg = self.cfg
        layout = options.get("layout") or sample_layout(seed, cfg)
        goal = int(options.get("goal_index", 0))
        if goal not in (0, 1):
            raise ValueError("goal_index must be 0 or 1")
        family = int(options.get("instruction_family", 0))
        instruction = options.get("instruction") or make_instruction(layout.targets[goal], family)
        if not self.observation_space["instruction"].contains(instruction):
            raise ValueError(f"instruction has characters outside the charset: {instruction!r}")

        self._seed, self._layout, self._goal = int(seed), layout, goal
        self._family, self._instruction = family, instruction

        av = self._aviary
        av.INIT_XYZS = np.array([layout.start_xyz], dtype=float)
        av.INIT_RPYS = np.array([[0.0, 0.0, layout.start_yaw]])
        av.wind_world = np.zeros(3)
        av.reset(seed=self._seed)
        self._drone = int(av.DRONE_IDS[0])
        self._names = {int(av.PLANE_ID): "ground"}
        self._target_ids = []
        for t in layout.targets:
            body = self._build_target(t)
            self._target_ids.append(body)
            self._names[body] = t.description
        self._ctrl.reset()

        # Random streams: derived from the seed only, never from the goal (pair determinism)
        self._noise = GaussMarkov((cfg.noise_xy_m, cfg.noise_xy_m, cfg.noise_z_m),
                                  cfg.noise_tau_s, self._dt, [self._seed, 1],
                                  smooth_tau_s=cfg.noise_smooth_s)
        self._gust = GaussMarkov((cfg.wind_gust_mps, cfg.wind_gust_mps, 0.0),
                                 cfg.gust_tau_s, self._dt, [self._seed, 2])
        th = math.radians(cfg.wind_toward_deg)
        self._wind_mean = cfg.wind_mps * np.array([math.cos(th), math.sin(th), 0.0])

        self._rpm = np.zeros((1, 4))
        self._sp = np.array(layout.start_xyz, dtype=float)
        self._yaw_sp = layout.start_yaw
        self._state = av._getDroneStateVector(0)
        self._est = self._state[0:3].copy()
        self._contact, self._oob = None, False
        self._last_wall = time.monotonic()

        for _ in range(int(round(cfg.settle_s * cfg.ctrl_hz))):
            self._control_step(np.zeros(3))
        settle_contact = self._contact

        self._t0 = av.step_counter / av.PYB_FREQ
        self._first_pose = (self._state[0:3].copy(), self._state[3:7].copy())
        self._stop_count, self._outcome, self._hold = 0, None, None
        self._min_d = [np.inf, np.inf]
        self._update_distances()
        self._done = False

        priv = self._privileged()
        priv["first_frame_target_px"] = self._target_pixels(self._state[0:3], self._state[3:7])
        priv["settle_contact"] = settle_contact
        return self._observe(), self._info(None, priv)

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        cfg = self.cfg
        adapted = adapt(action, cfg.limits, cfg.planar)
        if adapted.rejected:
            return self._finish("invalid_action", adapted)

        self._stop_count = self._stop_count + 1 if adapted.stop_positive else 0
        if self._stop_count >= cfg.stop_consecutive:
            return self._hold_and_judge(adapted)

        v_world = body_to_world_velocity(adapted.applied[:3], float(self._state[9]))
        for _ in range(self._ctrl_per_policy):
            self._control_step(v_world, float(adapted.applied[3]))
            if self._contact is not None:
                return self._finish("collision", adapted)
            if self._oob:
                return self._finish("out_of_bounds", adapted)
        if self._episode_t() >= cfg.timeout_s - 1e-9:
            return self._finish("timeout", adapted, truncated=True)
        return self._observe(), 0.0, False, False, self._info(adapted)

    def close(self):
        self._aviary.close()

    # ----------------------------------------------------------------- internals
    def _build_target(self, t):
        rgba = COLORS[t.color]
        c = self._client
        if t.shape == "box":
            half = [t.radius, t.radius, t.height / 2]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba,
                                      physicsClientId=c)
        else:
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=t.radius, height=t.height,
                                         physicsClientId=c)
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=t.radius, length=t.height,
                                      rgbaColor=rgba, physicsClientId=c)
        return int(p.createMultiBody(0, col, vis, [t.xy[0], t.xy[1], t.height / 2],
                                     physicsClientId=c))

    def _control_step(self, v_world, yaw_rate: float = 0.0):
        """One 60 Hz control update: 4 physics steps, then a new PID command."""
        cfg, av = self.cfg, self._aviary
        av.wind_world = self._wind_mean + self._gust.step()
        obs, *_ = av.step(self._rpm)
        state = obs[0]
        self._state = state
        est = state.copy()
        est[0:3] += self._noise.step()
        self._est = est[0:3].copy()

        self._sp[:2] += np.asarray(v_world[:2]) * self._dt
        if not cfg.planar:
            self._sp[2] += v_world[2] * self._dt
            self._yaw_sp += yaw_rate * self._dt
        lead = self._sp[:2] - est[0:2]
        n = float(np.hypot(*lead))
        if n > cfg.max_setpoint_lead_m:     # do not let the setpoint run away from the drone
            self._sp[:2] = est[0:2] + lead * (cfg.max_setpoint_lead_m / n)
        self._rpm[0], _, _ = self._ctrl.computeControlFromState(
            control_timestep=self._dt, state=est, target_pos=self._sp,
            target_rpy=np.array([0.0, 0.0, self._yaw_sp]), target_vel=np.asarray(v_world))

        # judgements use the truth
        if self._contact is None:
            cps = p.getContactPoints(bodyA=self._drone, physicsClientId=self._client)
            if cps:
                self._contact = self._names.get(int(cps[0][2]), f"body {cps[0][2]}")
        pos = state[0:3]
        if np.any(pos < np.asarray(cfg.room_min)) or np.any(pos > np.asarray(cfg.room_max)):
            self._oob = True
        if not self._done and self._layout is not None and hasattr(self, "_min_d"):
            self._update_distances()

        if self._gui:
            # BaseAviary's GUI camera looks at the origin; the drone starts 3 m away from it
            p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=-50, cameraPitch=-30,
                                         cameraTargetPosition=pos.tolist(),
                                         physicsClientId=self._client)
        if self._realtime:
            wait = self._dt - (time.monotonic() - self._last_wall)
            if wait > 0:
                time.sleep(wait)
            self._last_wall = time.monotonic()

    def _update_distances(self):
        pos = self._state[0:3]
        for i, h in enumerate(self._layout.hover_points):
            self._min_d[i] = min(self._min_d[i], float(np.linalg.norm(pos - np.asarray(h))))

    def _hold_and_judge(self, adapted):
        cfg = self.cfg
        self._sp[:2] = self._est[:2]          # pose hold at the current (estimated) position
        if not cfg.planar:
            self._sp[2] = self._est[2]
        P, V = [], []
        for _ in range(int(round(cfg.hold_s * cfg.ctrl_hz))):
            self._control_step(np.zeros(3))
            P.append(self._state[0:3].copy())
            V.append(self._state[10:13].copy())
            if self._contact is not None:
                return self._finish("collision", adapted)
            if self._oob:
                return self._finish("out_of_bounds", adapted)
        P, V = np.array(P), np.array(V)
        goal = np.asarray(self._layout.hover_points[self._goal])
        other = np.asarray(self._layout.hover_points[1 - self._goal])
        d_goal = np.linalg.norm(P - goal, axis=1)
        d_other = np.linalg.norm(P - other, axis=1)
        speed = np.linalg.norm(V, axis=1)
        R = cfg.success_radius_m
        in_goal, in_other = bool(np.all(d_goal <= R)), bool(np.all(d_other <= R))
        settled = bool(np.all(speed <= cfg.stop_speed_mps))
        self._hold = {"max_dist_to_goal": float(d_goal.max()),
                      "min_dist_to_goal": float(d_goal.min()),
                      "max_dist_to_other": float(d_other.max()),
                      "max_speed": float(speed.max()), "in_goal_throughout": in_goal,
                      "in_other_throughout": in_other, "settled": settled}
        if in_goal and settled:
            outcome = "success"
        elif in_goal:
            outcome = "stop_not_settled"
        elif in_other:
            outcome = "wrong_target"
        else:
            outcome = "stop_elsewhere"
        return self._finish(outcome, adapted)

    def _finish(self, outcome, adapted, truncated=False):
        assert outcome in OUTCOMES
        self._done, self._outcome = True, outcome
        reward = 1.0 if outcome == "success" else 0.0
        return self._observe(), reward, not truncated, truncated, self._info(adapted)

    def _episode_t(self):
        return self._aviary.step_counter / self._aviary.PYB_FREQ - self._t0

    def _observe(self):
        s = self._state
        rgb = self._cam.render(s[0:3], s[3:7], client=self._client)
        rot = np.array(p.getMatrixFromQuaternion(s[3:7])).reshape(3, 3)
        v_body = rot.T.dot(s[10:13])
        w_body = rot.T.dot(s[13:16])           # PyBullet reports angular velocity in world
        roll, pitch, yaw = s[7:10]
        proprio = np.array([*v_body, roll, pitch, math.sin(yaw), math.cos(yaw), *w_body,
                            s[2]], dtype=np.float32)
        return {"rgb": rgb, "proprio": proprio, "instruction": self._instruction}

    def _privileged(self):
        s, lay, R = self._state, self._layout, self.cfg.success_radius_m
        goal = np.asarray(lay.hover_points[self._goal])
        other = np.asarray(lay.hover_points[1 - self._goal])
        return {
            "true_pos": s[0:3].tolist(), "true_vel": s[10:13].tolist(),
            "true_yaw": float(s[9]), "estimated_pos": self._est.tolist(),
            "goal_hover_point": goal.tolist(), "other_hover_point": other.tolist(),
            "dist_to_goal": float(np.linalg.norm(s[0:3] - goal)),
            "dist_to_other": float(np.linalg.norm(s[0:3] - other)),
            "min_dist_to_goal": self._min_d[self._goal],
            "min_dist_to_other": self._min_d[1 - self._goal],
            "reached_goal": self._min_d[self._goal] <= R,
            "reached_other": self._min_d[1 - self._goal] <= R,
            "contact_with": self._contact,
            "hold": self._hold,
        }

    def _info(self, adapted, priv=None):
        lay = self._layout
        # No wall-clock time here: info must be deterministic for a given seed and action
        # sequence (gymnasium's checker enforces it). §4.3 wants wall time stored too; the
        # recorder stamps time.monotonic() itself when it receives each observation.
        return {
            "sim_t": self._aviary.step_counter / self._aviary.PYB_FREQ,
            "episode_t": self._episode_t(),
            "snapshot_seed": self._seed,
            "layout_id": lay.layout_id,
            "pair_id": f"{lay.layout_id}-{self._seed}",
            "goal_index": self._goal,
            "instruction_family": self._family,
            "action": None if adapted is None else {
                "raw": adapted.raw.tolist(),
                "applied": None if adapted.applied is None else adapted.applied.tolist(),
                "stop_positive": adapted.stop_positive, "rejected": adapted.rejected,
                "flags": adapted.flags},
            "stop_count": self._stop_count,
            "outcome": self._outcome,
            "config_hash": self.cfg.config_hash(),
            "privileged": priv if priv is not None else self._privileged(),
        }

    # ------------------------------------------------------------ layout checks
    def _target_pixels(self, pos, quat):
        """Pixels each target occupies in a segmentation render from this pose."""
        cam = self._cam
        _, _, _, _, seg = p.getCameraImage(
            cam.width, cam.height, cam.view_matrix(pos, quat), cam._projection, shadow=0,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client)
        seg = np.asarray(seg).reshape(cam.height, cam.width)
        body = np.where(seg >= 0, seg & ((1 << 24) - 1), -1)
        return [int((body == tid).sum()) for tid in self._target_ids]

    def visibility_report(self) -> dict:
        """Target pixels in the first observation's pose, and from each hover point.

        Used to reject layouts whose targets the policy could not see at the start, or whose
        goal would be invisible where the policy has to decide to stop.

        The start is rendered from the pose the drone actually had after the settle -- the
        pose of the first observation -- and *not* from the nominal `layout.start_xyz`. The
        drone's body is in the scene during these renders. Estimate noise leaves the real
        drone a few cm off the nominal start, so a camera placed at the nominal start sat
        inside the drone's own arm and propeller, which hid a target completely (seed 3006:
        346 px in the real first frame, 0 px from the nominal pose). The hover points are
        metres away from the drone, so the same problem cannot occur there.
        """
        lay = self._layout
        quat = p.getQuaternionFromEuler([0.0, 0.0, lay.start_yaw])
        start_px = self._target_pixels(*self._first_pose)
        hover_px = [self._target_pixels(h, quat) for h in lay.hover_points]
        return {"start_px": start_px,
                "own_target_px_at_hover": [hover_px[i][i] for i in range(2)],
                "other_target_px_at_hover": [hover_px[i][1 - i] for i in range(2)]}
