"""The tiny BC policy: tokenizer, model budget, Stop threshold, obs-only interface, and one
end-to-end smoke run (record -> train -> closed-loop evaluate)."""
import inspect
import json

import pytest
import torch

from dronevla import evaluate, record, train
from dronevla.evaluate import LearnedPolicy
from dronevla.model import MAX_TOKENS, TinyBC, Vocab, count_parameters, upgrade_state_dict
from dronevla.train import best_threshold


def test_vocab_maps_unseen_words_to_unk_and_counts_them():
    v = Vocab.from_texts(["Go to the red box and stop."])
    ids = v.encode("Go to the purple box and stop.")
    assert ids.count(1) == 1                      # "purple" -> <unk>
    assert v.unknown_rate("Go to the purple box") == pytest.approx(1 / 5)


def test_overlong_instruction_is_rejected_not_truncated():
    v = Vocab.from_texts(["go"])
    with pytest.raises(ValueError):
        v.encode(" ".join(["go"] * (MAX_TOKENS + 1)))


def test_model_fits_the_roadmap_budget_and_shapes():
    m = TinyBC(vocab_size=20)
    assert count_parameters(m) <= 2_000_000                       # §3.1
    motion, logit = m(torch.zeros(3, 96, 128, 3, dtype=torch.uint8), torch.zeros(3, 11),
                      m.encode_text(torch.tensor([[2, 3, 0], [4, 0, 0], [1, 1, 1]])))
    assert motion.shape == (3, 4) and logit.shape == (3,)
    assert motion.abs().max() <= 1.0                               # tanh, then scaled by caps


def test_film_starts_as_exactly_the_unconditioned_network():
    torch.manual_seed(0)
    plain = TinyBC(vocab_size=20)
    film = TinyBC(vocab_size=20, film=True)
    film.load_state_dict(plain.state_dict(), strict=False)      # everything but the FiLM layer
    rgb = torch.randint(0, 255, (2, 96, 128, 3), dtype=torch.uint8)
    prop, ids = torch.randn(2, 11), torch.tensor([[2, 3], [4, 5]])
    a = plain(rgb, prop, plain.encode_text(ids))
    b = film(rgb, prop, film.encode_text(ids))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_pre_film_checkpoints_still_load():
    old = {"cnn.0.weight": 1, "cnn.6.bias": 2, "head.0.weight": 3}
    assert upgrade_state_dict(old) == {"convs.0.weight": 1, "convs.3.bias": 2, "head.0.weight": 3}


def test_stop_threshold_sits_in_the_middle_of_the_gap():
    """Regression: the first version returned the smallest positive logit, i.e. the edge of
    the Stop class, and a memorised policy missed its Stop in closed loop."""
    logits = torch.tensor([-6.0, -5.9, -3.0, 8.7, 9.0, 9.2])
    labels = torch.tensor([0.0, 0, 0, 1, 1, 1])
    th, f1, _ = best_threshold(logits, labels)
    assert f1 == 1.0 and th == pytest.approx((-3.0 + 8.7) / 2)


def test_learned_policy_acts_on_the_observation_only():
    params = list(inspect.signature(LearnedPolicy.act).parameters)
    assert params == ["self", "obs"]                               # no info, no privileged state


def test_record_train_evaluate_end_to_end(tmp_path):
    data = tmp_path / "data"
    assert record.main(["--out", str(data), "--pairs", "1", "1", "1"]) == 0
    run = tmp_path / "run"
    assert train.main(["--data", str(data), "--out", str(run), "--epochs", "1"]) == 0
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["parameters"] <= 2_000_000 and "stop_threshold" in cfg
    out = tmp_path / "eval.json"
    assert evaluate.main(["--data", str(data), "--split", "val", "--policy", str(run),
                          "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    (summary,) = [v["summary"] for v in rep["policies"].values()]
    assert summary["episodes"] == 2 and sum(summary["outcomes"].values()) == 2
