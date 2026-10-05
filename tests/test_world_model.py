"""World model: training windows, offset sign convention, shapes. No simulator needed."""
import numpy as np
import torch

from dronevla.world_model import COLOUR_ORDER, Norm, WorldModel, losses, windows


def fake_episode(n=8, flown_until=6, stop_trigger_at=None):
    flown = np.zeros(n, dtype=bool)
    flown[:flown_until] = True
    if stop_trigger_at is not None:
        flown[stop_trigger_at] = False
    return {"rgb": np.zeros((n, 96, 128, 3), np.uint8), "prop": np.zeros((n, 11), np.float32),
            "act": np.zeros((n, 2), np.float32), "flown": flown,
            "off": np.zeros((n, 2 * len(COLOUR_ORDER)), np.float32),
            "mask": np.zeros((n, len(COLOUR_ORDER)), np.float32)}


def test_windows_only_use_actions_that_were_flown():
    # rows 0..5 flown, row 6 is the Stop trigger (not flown), row 7 terminal
    ep = fake_episode(n=8, flown_until=7, stop_trigger_at=6)
    w = windows([ep], horizon=3)
    assert w == [(0, 0), (0, 1), (0, 2), (0, 3)]          # never spans row 6


def test_offsets_are_in_the_body_frame(tmp_path):
    """A hover point straight ahead (+x) and to the left (+y) must come out positive."""
    import json
    import pyarrow as pa
    from dronevla.dataset import STEP_SCHEMA, write_table
    from dronevla.world_model import load_episode_arrays
    from PIL import Image

    root = tmp_path
    (root / "layouts").mkdir()
    (root / "rgb" / "e").mkdir(parents=True)
    Image.fromarray(np.zeros((96, 128, 3), np.uint8)).save(root / "rgb/e/00000.png")
    lay = {"targets": [{"color": "red", "shape": "box", "xy": [3, 2], "radius": 0.2, "height": 1.2},
                       {"color": "blue", "shape": "box", "xy": [3, -2], "radius": 0.2, "height": 1.2}],
           "hover_points": [[2.0, 1.0, 1.0], [2.0, -1.0, 1.0]]}
    (root / "layouts" / "L.json").write_text(json.dumps(lay))
    row = {k: None for k in STEP_SCHEMA.names}
    row.update(episode_id="e", t=0, sim_t=0.0, episode_t=0.0, wall_t_monotonic=0.0,
               rgb_path="rgb/e/00000.png", rgb_sha256="x",
               proprio=[0, 0, 0, 0, 0, 0.0, 1.0, 0, 0, 0, 1.0],     # yaw 0
               action_mask=False, stop_trigger=False, terminal=True,
               priv_true_pos=[0.0, 0.0, 1.0], priv_dist_to_goal=0.0, priv_dist_to_other=0.0)
    write_table([row], STEP_SCHEMA, root / "steps" / "e.parquet")
    ep = load_episode_arrays(root, {"episode_id": "e", "layout_id": "L"})
    off = ep["off"].reshape(len(COLOUR_ORDER), 2)
    red, blue = COLOUR_ORDER.index("red"), COLOUR_ORDER.index("blue")
    np.testing.assert_allclose(off[red], [2.0, 1.0], atol=1e-6)     # ahead and to the left
    np.testing.assert_allclose(off[blue], [2.0, -1.0], atol=1e-6)   # ahead and to the right
    assert ep["mask"][0, red] == 1 and ep["mask"][0, COLOUR_ORDER.index("green")] == 0


def test_model_shapes_and_loss_runs():
    m = WorldModel(z_dim=16)
    norm = Norm(torch.zeros(11), torch.ones(11))
    B, H = 3, 2
    batch = (torch.zeros(B, H + 1, 96, 128, 3, dtype=torch.uint8), torch.zeros(B, H + 1, 11),
             torch.zeros(B, H, 2), torch.zeros(B, H + 1, 8), torch.ones(B, H + 1, 4))
    l = losses(m, norm, batch, H)
    assert set(l) == {"latent", "prop", "off", "sum"} and torch.isfinite(l["sum"])
    l["sum"].backward()
    z = m.encode(batch[0][:, 0], batch[1][:, 0])
    assert m.step(z, torch.zeros(B, 2)).shape == (B, 16)
