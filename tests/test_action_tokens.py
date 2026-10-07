"""OpenVLA-style action tokens in TinyBC: binning, round trip, shapes, checkpoint round trip."""
import io

import torch

from dronevla.model import TinyBC, count_parameters


def _model(bins=256, img_dim=104):
    m = TinyBC(vocab_size=17, action_bins=bins, img_dim=img_dim)
    g = torch.Generator().manual_seed(0)
    acts = torch.zeros(500, 4)
    acts[:, 0] = 0.6 + 0.4 * torch.rand(500, generator=g)        # mostly forward, like the expert
    acts[:, 1] = 2 * torch.rand(500, generator=g) - 1
    m.set_action_bins(acts)
    return m, acts


def test_token_round_trip_within_half_a_bin():
    m, acts = _model()
    toks = m.tokenize_actions(acts)
    assert toks.min() >= 0 and toks.max() <= 255
    logits = torch.full((len(acts), 2, 256), -10.0)
    logits.scatter_(2, toks.unsqueeze(-1), 10.0)
    dec = m.detokenize(logits)
    width = (m.bin_edges[:, 1] - m.bin_edges[:, 0])
    inside = (acts[:, 0] >= m.bin_edges[0, 0]) & (acts[:, 0] <= m.bin_edges[0, -1])
    assert torch.all((dec[inside, 0] - acts[inside, 0]).abs() <= width[0] / 2 + 1e-6)
    assert torch.all(dec[:, 2:] == 0)                                # vz, yaw stay zero


def test_forward_shapes_and_logits():
    m, _ = _model()
    rgb = torch.randint(0, 256, (3, 96, 128, 3), dtype=torch.uint8)
    motion, stop, logits = m(rgb, torch.randn(3, 11), m.encode_text(torch.tensor([[2, 3]] * 3)),
                             return_logits=True)
    assert motion.shape == (3, 4) and stop.shape == (3,) and logits.shape == (3, 2, 256)


def test_bin_edges_survive_a_checkpoint():
    m, _ = _model()
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    buf.seek(0)
    m2 = TinyBC(vocab_size=17, action_bins=256, img_dim=104)
    m2.load_state_dict(torch.load(buf, weights_only=True))
    assert torch.equal(m.bin_edges, m2.bin_edges)


def test_token_model_parameter_count_is_comparable():
    ratio = count_parameters(TinyBC(17, action_bins=256, img_dim=104)) / count_parameters(TinyBC(17))
    assert 0.8 <= ratio <= 1.0, ratio


def test_chunk_head_shapes_and_decode():
    m = TinyBC(vocab_size=17, action_bins=256, img_dim=104, chunk=8)
    m.set_action_bins(torch.rand(100, 4))
    rgb = torch.randint(0, 256, (2, 96, 128, 3), dtype=torch.uint8)
    motion, stop, (logits, stops) = m(rgb, torch.randn(2, 11), m.encode_text(torch.tensor([[2, 3]] * 2)),
                                      return_logits=True)
    assert motion.shape == (2, 4) and stop.shape == (2,)
    assert logits.shape == (2, 8, 2, 256) and stops.shape == (2, 8)
    assert torch.allclose(m.decode_chunk(logits)[:, 0], motion)


def test_paired_goal_head_and_parameter_counts():
    base = count_parameters(TinyBC(17))
    for kw in (dict(action_bins=256, img_dim=104, chunk=8),
               dict(action_bins=256, img_dim=104, paired_goal=True),
               dict(action_bins=64, img_dim=120),
               dict(action_bins=1024, img_dim=88, head_hidden=64)):
        assert 0.8 <= count_parameters(TinyBC(17, **kw)) / base <= 1.0, kw
    m = TinyBC(17, action_bins=256, img_dim=104, paired_goal=True)
    feats = torch.randn(3, 104 + 32 + 32)
    assert m.goal_head(feats).shape == (3, 2)
