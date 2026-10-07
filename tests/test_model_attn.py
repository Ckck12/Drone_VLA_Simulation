"""The attention variant of TinyBC: shapes, attention weights, and that the words reach it."""
import torch

from dronevla.model import TinyBC, count_parameters


def _batch(b=3):
    g = torch.Generator().manual_seed(0)
    rgb = torch.randint(0, 256, (b, 96, 128, 3), dtype=torch.uint8, generator=g)
    prop = torch.randn(b, 11, generator=g)
    return rgb, prop


def test_attn_shapes_and_budget():
    m = TinyBC(vocab_size=17, attn=True)
    rgb, prop = _batch()
    text = m.encode_text(torch.tensor([[2, 3, 4], [5, 6, 0], [7, 0, 0]]))
    motion, logit = m(rgb, prop, text)
    assert motion.shape == (3, 4) and logit.shape == (3,)
    assert count_parameters(m) <= 2_000_000
    w = m.last_attn
    assert w.shape == (3, TinyBC.ATTN_HEADS, TinyBC.PATCHES)
    assert torch.allclose(w.sum(-1), torch.ones(3, TinyBC.ATTN_HEADS), atol=1e-5)


def test_attn_weights_depend_on_the_sentence():
    torch.manual_seed(0)
    m = TinyBC(vocab_size=17, attn=True).eval()
    rgb, prop = _batch(1)
    with torch.no_grad():
        m(rgb, prop, m.encode_text(torch.tensor([[11, 5]])))      # "red box"
        w_red = m.last_attn.clone()
        m(rgb, prop, m.encode_text(torch.tensor([[4, 6]])))       # "blue cylinder"
        w_blue = m.last_attn.clone()
    assert not torch.allclose(w_red, w_blue)


def test_attn_and_film_are_exclusive():
    try:
        TinyBC(vocab_size=17, attn=True, film=True)
    except ValueError:
        return
    raise AssertionError("film + attn should be rejected")


def test_default_model_unchanged():
    """The plain model keeps its parameter count and state-dict keys (old checkpoints load)."""
    m = TinyBC(vocab_size=17)
    assert count_parameters(m) == 480_901
    assert not any(k.startswith(("xattn", "query", "pos", "patch_proj")) for k in m.state_dict())


def test_attn_parameter_count_is_comparable():
    """A fair comparison needs a similar size: 80-100% of the plain model's parameters."""
    ratio = count_parameters(TinyBC(17, attn=True)) / count_parameters(TinyBC(17))
    assert 0.8 <= ratio <= 1.0, ratio
