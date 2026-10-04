"""`DroneTargetPairs` task definition: config, scene layouts, instructions.

Roadmap v3 §4.1 (task and scene generator) and §4.2 (instructions), Phase 1 scope:
two targets, no obstacles, altitude 1 m, yaw fixed, planar motion. The action schema is
still 4-D; `planar=True` records that vz and yaw_rate are forced to zero in this phase.

Everything here is pure Python -- no simulator -- so layouts can be generated, hashed and
rejected without starting PyBullet.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

import numpy as np

from dronevla.action_adapter import ActionLimits

COLORS = {
    "red": (1.0, 0.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0, 1.0),
    "green": (0.0, 0.7, 0.0, 1.0),
    "yellow": (1.0, 0.85, 0.0, 1.0),
}
SHAPES = ("box", "cylinder")

# Paraphrase families (§4.2 item 6). Human-written, no LLM. Phase 1 uses family 0 by
# default; the split-by-family design belongs to Phase 2.
INSTRUCTION_FAMILIES = (
    "Go to the {color} {shape} and stop.",
    "Approach the {color} {shape}, then hold position.",
)


@dataclass(frozen=True)
class TaskConfig:
    """Every number that defines an episode. Recorded (and hashed) into each episode's info.

    Values marked (§x.y) come from the roadmap contract; the rest are Phase 1 choices.
    """

    # room (§4.1 design: 8 x 8 x 3 m)
    room_min: tuple = (-4.0, -4.0, 0.0)
    room_max: tuple = (4.0, 4.0, 3.0)
    # start pose: fixed yaw, small lateral jitter
    start_x: float = -3.0
    start_y_jitter: float = 0.5
    altitude_m: float = 1.0               # (§4.1 Phase 1)
    start_yaw: float = 0.0                # (§4.1 Phase 1: yaw fixed)
    # targets: pillars tall enough to stay in frame from the hover point (see docs)
    target_radius_m: float = 0.2          # cylinder radius / box half-width
    target_height_m: float = 1.2
    target_x_range: tuple = (1.0, 3.0)
    target_y_range: tuple = (-1.6, 1.6)
    max_bearing_deg: float = 22.0         # inside the +-30 deg half-FOV with margin
    require_distinct_colors: bool = True
    min_hover_separation_m: float = 1.5
    path_clearance_m: float = 0.35        # straight expert path vs the other target
    # goal and termination (§4.3)
    standoff_m: float = 0.75              # hover point this far in front of the target
    success_radius_m: float = 0.4
    stop_speed_mps: float = 0.1
    hold_s: float = 1.0
    timeout_s: float = 30.0
    stop_consecutive: int = 3
    # rates (§4.3)
    pyb_hz: int = 240
    ctrl_hz: int = 60
    policy_hz: int = 5
    # action contract (§4.3)
    limits: ActionLimits = field(default_factory=ActionLimits)
    planar: bool = True
    max_setpoint_lead_m: float = 0.3      # velocity setpoint may not run further ahead
    # camera (docs/realism_mapping.md)
    camera_mount: str = "gimbal"
    camera_tilt_deg: float = -20.0
    camera_vfov_deg: float = 47.0
    # realism (docs/realism_mapping.md)
    noise_xy_m: float = 0.03
    noise_z_m: float = 0.01
    # 1 s (the first playground choice) made the expert fail 2/20 Stops on drift alone;
    # 3 s gave 20/20. Measured in docs/phase1_env.md.
    noise_tau_s: float = 3.0
    noise_smooth_s: float = 0.3           # estimator smoothing; see sim.GaussMarkov
    wind_mps: float = 0.0                 # off for v0.1: the drag model is optimistic
    wind_toward_deg: float = 0.0
    wind_gust_mps: float = 0.0
    gust_tau_s: float = 2.0
    max_tilt_deg: float | None = 35.0
    settle_s: float = 1.0                 # hover at the start before the first obs
    min_first_frame_target_px: int = 30

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Target:
    color: str
    shape: str
    xy: tuple
    radius: float
    height: float

    @property
    def footprint_radius(self) -> float:
        """Radius of the smallest vertical cylinder that contains the target."""
        return self.radius * math.sqrt(2.0) if self.shape == "box" else self.radius

    @property
    def description(self) -> str:
        return f"{self.color} {self.shape}"


@dataclass(frozen=True)
class Layout:
    start_xyz: tuple
    start_yaw: float
    targets: tuple          # (Target, Target)
    hover_points: tuple     # ((x, y, z), (x, y, z)) -- evaluator/expert only (§4.1)
    sample_seed: int
    rejected_before: int    # how many candidates the sampler threw away first

    @property
    def layout_id(self) -> str:
        """Hash of the geometry only (§4.1): independent of seed and of target order."""
        canon = {
            "start_xyz": [round(v, 4) for v in self.start_xyz],
            "start_yaw": round(self.start_yaw, 4),
            "targets": sorted(
                [{"color": t.color, "shape": t.shape, "xy": [round(v, 4) for v in t.xy],
                  "radius": t.radius, "height": t.height} for t in self.targets],
                key=lambda d: (d["color"], d["shape"])),
        }
        blob = json.dumps(canon, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "layout_id": self.layout_id,
            "start_xyz": list(self.start_xyz),
            "start_yaw": self.start_yaw,
            "targets": [dataclasses.asdict(t) for t in self.targets],
            "hover_points": [list(h) for h in self.hover_points],
            "sample_seed": self.sample_seed,
            "rejected_before": self.rejected_before,
        }


def hover_point(target: Target, start_xy, cfg: TaskConfig) -> tuple:
    """`standoff_m` in front of the target's footprint, on the line toward the start.

    "In front" is defined this way so it is deterministic and so the straight expert path
    from the start never has to pass its own target.
    """
    c = np.asarray(target.xy, dtype=float)
    to_start = np.asarray(start_xy, dtype=float) - c
    to_start /= np.linalg.norm(to_start)
    xy = c + to_start * (target.footprint_radius + cfg.standoff_m)
    return (float(xy[0]), float(xy[1]), cfg.altitude_m)


def _segment_point_distance(a, b, p) -> float:
    a, b, p = (np.asarray(v, dtype=float) for v in (a, b, p))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(a + t * ab - p))


def check_geometry(targets, start_xyz, hovers, cfg: TaskConfig) -> str | None:
    """Return why a candidate layout is unusable, or None if it passes. Simulator-free."""
    sx, sy = start_xyz[0], start_xyz[1]
    for t in targets:
        bearing = math.degrees(math.atan2(t.xy[1] - sy, t.xy[0] - sx)) - math.degrees(cfg.start_yaw)
        if abs(bearing) > cfg.max_bearing_deg:
            return f"{t.description} at bearing {bearing:.1f} deg, outside +-{cfg.max_bearing_deg}"
    h0, h1 = (np.asarray(h[:2]) for h in hovers)
    if np.linalg.norm(h0 - h1) < cfg.min_hover_separation_m:
        return "hover points closer than min_hover_separation_m"
    for i, h in enumerate(hovers):
        other = targets[1 - i]
        # §4.1: reject if a success region overlaps a target mesh
        if np.linalg.norm(np.asarray(h[:2]) - other.xy) < other.footprint_radius + cfg.success_radius_m + 0.1:
            return f"success region {i} overlaps the {other.description}"
        if _segment_point_distance(start_xyz[:2], h[:2], other.xy) < other.footprint_radius + cfg.path_clearance_m:
            return f"straight path to goal {i} passes the {other.description}"
        lo, hi = np.asarray(cfg.room_min[:2]) + 0.5, np.asarray(cfg.room_max[:2]) - 0.5
        if np.any(np.asarray(h[:2]) < lo) or np.any(np.asarray(h[:2]) > hi):
            return f"hover point {i} too close to the room boundary"
    return None


def sample_layout(seed: int, cfg: TaskConfig = TaskConfig(), max_tries: int = 1000) -> Layout:
    """Rejection-sample a valid two-target layout. Deterministic in `seed`."""
    rng = np.random.default_rng([int(seed), 0xA11])
    for tries in range(max_tries):
        colors = list(COLORS)
        c0 = colors[rng.integers(len(colors))]
        if cfg.require_distinct_colors:
            colors.remove(c0)
        c1 = colors[rng.integers(len(colors))]
        s0, s1 = (SHAPES[rng.integers(len(SHAPES))] for _ in range(2))
        if (c0, s0) == (c1, s1):
            continue                      # §4.1: each target needs a unique description
        start = (cfg.start_x, float(rng.uniform(-cfg.start_y_jitter, cfg.start_y_jitter)),
                 cfg.altitude_m)
        targets = tuple(
            Target(color=c, shape=s,
                   xy=(float(rng.uniform(*cfg.target_x_range)),
                       float(rng.uniform(*cfg.target_y_range))),
                   radius=cfg.target_radius_m, height=cfg.target_height_m)
            for c, s in ((c0, s0), (c1, s1)))
        hovers = tuple(hover_point(t, start[:2], cfg) for t in targets)
        if check_geometry(targets, start, hovers, cfg) is None:
            return Layout(start_xyz=start, start_yaw=cfg.start_yaw, targets=targets,
                          hover_points=hovers, sample_seed=int(seed), rejected_before=tries)
    raise RuntimeError(f"no valid layout after {max_tries} tries for seed {seed}")


def make_instruction(target: Target, family: int = 0) -> str:
    return INSTRUCTION_FAMILIES[family].format(color=target.color, shape=target.shape)
