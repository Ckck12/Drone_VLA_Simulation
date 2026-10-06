"""World-model planner (stages 2-3) and the BC auxiliary-offset control. No simulator needed."""
import numpy as np
import torch

from dronevla.model import TinyBC, Vocab, pad_batch
from dronevla.planner import (GoalSelector, PlannerConfig, _Planner, cap_norm,
                              integrator_displacement)
from dronevla.world_model import COLOUR_ORDER, DT, body_offsets


def test_cap_norm_keeps_direction_and_limits_speed():
    a = np.array([[3.0, 4.0], [0.1, 0.0]])
    out = cap_norm(a, 0.5)
    np.testing.assert_allclose(out[0], [0.3, 0.4])
    np.testing.assert_allclose(out[1], [0.1, 0.0])


def test_integrator_displacement_moves_the_offset_against_the_command():
    A = np.full((1, 3, 2), [0.5, 0.0])
    np.testing.assert_allclose(integrator_displacement(A)[0, :, 0], [-0.1, -0.2, -0.3])


def test_planner_heads_for_the_goal_and_stops_inside_the_radius():
    p = _Planner(PlannerConfig())
    p.reset(seed=0)
    a = p.plan(np.array([2.0, -1.0]), integrator_displacement)
    assert a[0] > 0.2 and a[1] < -0.1                # toward (+x, -y)
    assert np.linalg.norm(a) <= 0.5
    act = p.finish(np.array([2.0, -1.0]), 0.3, a)
    assert act[4] < 0 and np.allclose(act[:2], a)
    act = p.finish(np.array([0.05, 0.0]), 0.02, a)    # inside the radius and slow
    assert act[4] > 0 and np.allclose(act[:4], 0)
    act = p.finish(np.array([0.05, 0.0]), 0.3, a)     # inside the radius but still moving
    assert act[4] < 0


def test_planner_is_reproducible_per_seed():
    def run(seed):
        p = _Planner(PlannerConfig())
        p.reset(seed=seed)
        return p.plan(np.array([1.0, 0.5]), integrator_displacement)
    np.testing.assert_array_equal(run(3), run(3))


def test_goal_selector_shapes():
    v = Vocab.from_texts(["go to the red box", "go to the blue cylinder"])
    m = GoalSelector(len(v))
    out = m(pad_batch([v.encode("go to the red box"), v.encode("blue")]))
    assert out.shape == (2, len(COLOUR_ORDER))


def test_body_offsets_rotate_with_yaw():
    lay = {"targets": [{"color": "red"}], "hover_points": [[1.0, 0.0, 1.0]]}
    off, mask = body_offsets(lay, np.array([[0.0, 0.0, 1.0]]), np.array([np.pi / 2]))
    # facing +y, a point at +x world is to the right: body (0, -1)
    np.testing.assert_allclose(off[0, COLOUR_ORDER.index("red")], [0.0, -1.0], atol=1e-6)
    assert mask[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_bc_aux_head_is_optional_and_trains():
    plain = TinyBC(10)
    assert plain.aux_head is None
    m = TinyBC(10, aux_offsets=True)
    rgb = torch.zeros(2, 96, 128, 3, dtype=torch.uint8)
    prop, text = torch.zeros(2, 11), m.encode_text(torch.ones(2, 3, dtype=torch.long))
    motion, logit, aux = m(rgb, prop, text, return_aux=True)
    assert motion.shape == (2, 4) and logit.shape == (2,) and aux.shape == (2, 8)
    aux.sum().backward()
    # the policy output path is unchanged by the aux head
    motion2, logit2 = m(rgb, prop, text)
    torch.testing.assert_close(motion, motion2)
    assert DT == 0.2
