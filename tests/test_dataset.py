"""The dataset recorder and its validator.

A validator that always passes is worse than none, so most of these tests break a copy of
a valid dataset in one specific way and check that the matching rule -- and not just "some"
rule -- catches it.
"""
import json
import shutil

import pyarrow.parquet as pq
import pytest

from dronevla import record
from dronevla.dataset import (EPISODE_SCHEMA, STEP_SCHEMA, failed, load_episodes, validate,
                              write_table)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds") / "tiny"
    assert record.main(["--out", str(root), "--pairs", "1", "1", "1"]) == 0
    return root


@pytest.fixture
def copy(tiny, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(tiny, dst)
    return dst


def rewrite_episodes(root, edit):
    rows = load_episodes(root)
    edit(rows)
    write_table(rows, EPISODE_SCHEMA, root / "episodes.parquet")


def test_recorded_dataset_passes_every_check(tiny):
    report = validate(tiny)
    assert failed(report) == [], report
    manifest = json.loads((tiny / "manifest.json").read_text())
    assert manifest["counts"]["episodes"] == 6
    sheet = (tiny / "datasheet.md").read_text()
    assert "| train | 1 | 2 | 1001-1001 |" in sheet       # split table filled from the data
    assert "- 6 episodes," in sheet


def test_missing_image_is_caught(copy):
    victim = next((copy / "rgb").rglob("*.png"))
    victim.unlink()
    bad = failed(validate(copy))
    assert "images_present_and_hashes_match" in bad
    assert "rgb_tree_hash_matches_manifest" in bad


def test_modified_image_is_caught(copy):
    victim = next((copy / "rgb").rglob("*.png"))
    other = sorted((copy / "rgb").rglob("*.png"))[-1]
    victim.write_bytes(other.read_bytes())
    assert "images_present_and_hashes_match" in failed(validate(copy))


def test_pair_with_different_first_frames_is_caught(copy):
    def edit(rows):
        rows[1]["first_rgb_sha256"] = "0" * 64
    rewrite_episodes(copy, edit)
    assert "pairs_complete_and_identical_at_start" in failed(validate(copy))


def test_layout_leaking_across_splits_is_caught(copy):
    def edit(rows):
        train = next(r for r in rows if r["split"] == "train")
        for r in rows:
            if r["split"] == "val":
                r["layout_id"] = r["family_id"] = train["layout_id"]
    rewrite_episodes(copy, edit)
    assert "splits_disjoint_by_layout_and_family" in failed(validate(copy))


def test_action_on_the_terminal_row_is_caught(copy):
    ep = load_episodes(copy)[0]["episode_id"]
    path = copy / "steps" / f"{ep}.parquet"
    rows = pq.read_table(path).to_pylist()
    rows[-1]["action_mask"] = True
    rows[-1]["action_raw"] = [0.0, 0.0, 0.0, 0.0, -1.0]
    write_table(rows, STEP_SCHEMA, path)
    assert "steps_alignment_and_actions" in failed(validate(copy))


def test_tampering_with_any_tracked_file_breaks_the_manifest(copy):
    (copy / "splits.json").write_text((copy / "splits.json").read_text() + " ")
    assert "manifest_file_hashes" in failed(validate(copy))


def test_same_seeds_reproduce_byte_identical_images(tiny, tmp_path):
    again = tmp_path / "again"
    assert record.main(["--out", str(again), "--pairs", "1", "1", "1"]) == 0
    a = json.loads((tiny / "manifest.json").read_text())["rgb"]["tree_sha256"]
    b = json.loads((again / "manifest.json").read_text())["rgb"]["tree_sha256"]
    assert a == b
