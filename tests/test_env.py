"""DroneTargetPairsEnv: API contract, pair determinism, frame convention, and one fixture
per outcome (roadmap v3 Phase 1 item 2: deliberately produce each outcome and check that
the evaluator names it correctly).

The fixture policies read `info["privileged"]` or the layout. That is allowed here --
they are test drivers for the evaluator, not policies.
"""
import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from dronevla.env import OUTCOMES, DroneTargetPairsEnv
from dronevla.expert import StraightLineExpert
from dronevla.task import TaskConfig, sample_layout

CFG = TaskConfig()


@pytest.fixture(scope="module")
def env():
    e = DroneTargetPairsEnv(CFG)
    yield e
    e.close()


def run(env, policy, seed=0, goal=0, max_steps=400):
    obs, info = env.reset(seed=seed, options={"goal_index": goal})
    for n in range(max_steps):
        obs, reward, term, trunc, info = env.step(policy(obs, info))
        if term or trunc:
            return info, term, trunc, n + 1
    raise AssertionError("episode did not finish")


def toward(info, point, speed=0.5, stop=-1.0):
    """Body-frame command flying at `speed` toward a world point (yaw is fixed at 0)."""
    pos = np.asarray(info["privileged"]["true_pos"])
    d = np.asarray(point[:2]) - pos[:2]
    n = np.linalg.norm(d)
    v = d / n * speed if n > 1e-6 else np.zeros(2)
    return np.array([v[0], v[1], 0.0, 0.0, stop], dtype=np.float32)


# --------------------------------------------------------------------- API contract
def test_gymnasium_check_env():
    e = DroneTargetPairsEnv(CFG)
    check_env(e, skip_render_check=True)


def test_observation_has_exactly_the_contract_keys(env):
    obs, info = env.reset(seed=0)
    assert set(obs) == {"rgb", "proprio", "instruction"}
    assert obs["rgb"].shape == (96, 128, 3) and obs["rgb"].dtype == np.uint8
    assert obs["proprio"].shape == (11,) and obs["proprio"].dtype == np.float32
    assert "privileged" in info          # goal coordinates live here, never in obs


def test_proprio_layout_while_hovering(env):
    obs, _ = env.reset(seed=0)
    p = obs["proprio"]
    assert np.all(np.abs(p[0:3]) < 0.2)               # body velocity, settled
    assert abs(p[5]) < 0.05 and p[6] > 0.99           # sin/cos yaw at yaw 0
    assert p[10] == pytest.approx(CFG.altitude_m, abs=0.05)


@pytest.mark.parametrize("seed", range(5))
def test_pair_members_see_identical_first_observations(env, seed):
    o0, i0 = env.reset(seed=seed, options={"goal_index": 0})
    o1, i1 = env.reset(seed=seed, options={"goal_index": 1})
    assert np.array_equal(o0["rgb"], o1["rgb"])
    assert np.array_equal(o0["proprio"], o1["proprio"])
    assert o0["instruction"] != o1["instruction"]
    assert i0["pair_id"] == i1["pair_id"]


def test_same_seed_and_goal_reproduce(env):
    a, _ = env.reset(seed=3, options={"goal_index": 1})
    b, _ = env.reset(seed=3, options={"goal_index": 1})
    assert np.array_equal(a["rgb"], b["rgb"]) and np.array_equal(a["proprio"], b["proprio"])


def test_instruction_names_the_goal(env):
    lay = sample_layout(0, CFG)
    for goal in (0, 1):
        obs, _ = env.reset(seed=0, options={"goal_index": goal})
        assert lay.targets[goal].description in obs["instruction"]


def test_step_after_episode_end_is_an_error(env):
    env.reset(seed=0)
    env.step(np.array([np.nan, 0, 0, 0, 0], dtype=np.float32))
    with pytest.raises(RuntimeError):
        env.step(np.zeros(5, dtype=np.float32))


# ------------------------------------------------------------------ frame convention
@pytest.mark.parametrize("cmd, axis", [((0.3, 0.0), 0), ((0.0, 0.3), 1), ((0.0, -0.3), 1)])
def test_body_velocity_moves_the_drone_the_right_way(env, cmd, axis):
    """At yaw 0, body FLU and world ENU agree: +vx -> +x, +vy (left) -> +y."""
    obs, info = env.reset(seed=0)
    start = np.asarray(info["privileged"]["true_pos"])
    for _ in range(10):                                  # 2 s
        obs, _, term, trunc, info = env.step(np.array([*cmd, 0, 0, -1], dtype=np.float32))
        assert not (term or trunc)
    moved = np.asarray(info["privileged"]["true_pos"]) - start
    sign = np.sign(cmd[axis])
    assert sign * moved[axis] > 0.35                     # ~0.6 m commanded, minus spin-up
    assert abs(moved[1 - axis]) < 0.1


# --------------------------------------------------------- one fixture per outcome
def test_outcome_success(env):
    expert = StraightLineExpert(CFG)
    info, term, trunc, _ = run(env, lambda o, i: expert.act(i), seed=0, goal=0)
    assert info["outcome"] == "success" and term and not trunc
    assert info["privileged"]["reached_goal"] and not info["privileged"]["reached_other"]


def test_outcome_wrong_target(env):
    wrong = StraightLineExpert(CFG, goal_key="other_hover_point")
    info, *_ = run(env, lambda o, i: wrong.act(i), seed=0, goal=0)
    assert info["outcome"] == "wrong_target"
    assert info["privileged"]["reached_other"] and not info["privileged"]["reached_goal"]


def test_outcome_stop_elsewhere(env):
    stop_now = np.array([0, 0, 0, 0, 1.0], dtype=np.float32)
    info, term, _, steps = run(env, lambda o, i: stop_now, seed=0)
    assert info["outcome"] == "stop_elsewhere" and term
    assert steps == CFG.stop_consecutive


def test_outcome_stop_not_settled(env):
    """Arrive at full speed and Stop on the goal: inside the radius, but still moving."""
    def policy(o, i):
        goal = i["privileged"]["goal_hover_point"]
        close = i["privileged"]["dist_to_goal"] < 0.25
        return toward(i, goal, speed=0.5, stop=1.0 if close else -1.0)
    info, *_ = run(env, policy, seed=0)
    assert info["outcome"] == "stop_not_settled"
    assert info["privileged"]["hold"]["in_goal_throughout"]
    assert info["privileged"]["hold"]["max_speed"] > CFG.stop_speed_mps


def test_outcome_collision(env):
    target_xy = sample_layout(0, CFG).targets[0].xy
    info, term, *_ = run(env, lambda o, i: toward(i, target_xy), seed=0)
    assert info["outcome"] == "collision" and term
    assert info["privileged"]["contact_with"] == sample_layout(0, CFG).targets[0].description


def test_outcome_out_of_bounds(env):
    backward = np.array([-0.5, 0, 0, 0, -1.0], dtype=np.float32)
    info, term, *_ = run(env, lambda o, i: backward, seed=0)
    assert info["outcome"] == "out_of_bounds" and term


def test_outcome_timeout_is_truncation():
    e = DroneTargetPairsEnv(dataclasses.replace(CFG, timeout_s=2.0))
    try:
        hover = np.array([0, 0, 0, 0, -1.0], dtype=np.float32)
        info, term, trunc, steps = run(e, lambda o, i: hover, seed=0)
        assert info["outcome"] == "timeout" and trunc and not term
        assert steps == 10                                # 2 s at 5 Hz
    finally:
        e.close()


def test_outcome_invalid_action(env):
    nan = np.array([np.nan, 0, 0, 0, 0], dtype=np.float32)
    info, term, *_ = run(env, lambda o, i: nan, seed=0)
    assert info["outcome"] == "invalid_action" and term
    assert info["action"]["rejected"]


def test_every_outcome_has_a_fixture():
    tested = {"success", "wrong_target", "stop_elsewhere", "stop_not_settled", "collision",
              "out_of_bounds", "timeout", "invalid_action"}
    assert tested == set(OUTCOMES)


# ------------------------------------------------------------------- task design
@pytest.mark.parametrize("seed", range(10))
def test_both_targets_visible_at_start_and_goal_visible_at_hover(env, seed):
    _, info = env.reset(seed=seed)
    assert min(info["privileged"]["first_frame_target_px"]) >= CFG.min_first_frame_target_px
    vis = env.visibility_report()
    assert min(vis["own_target_px_at_hover"]) > 0.2 * 128 * 96


@pytest.mark.parametrize("seed", range(5))
def test_expert_succeeds_on_both_members_of_a_pair(env, seed):
    expert = StraightLineExpert(CFG)
    for goal in (0, 1):
        info, *_ = run(env, lambda o, i: expert.act(i), seed=seed, goal=goal)
        assert info["outcome"] == "success", (seed, goal, info["privileged"]["hold"])
