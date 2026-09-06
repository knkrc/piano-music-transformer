from __future__ import annotations

import pytest

from pmt.config import DataConfig, load_config


def test_defaults_are_usable_without_a_file():
    cfg = load_config(DataConfig)
    assert cfg == DataConfig()


def test_yaml_lists_become_tuples(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("pitch_range: [40, 90]\nbeat_res: [[0, 4, 8]]\nblock_size: 256\n")

    cfg = load_config(DataConfig, path)

    assert cfg.pitch_range == (40, 90)
    assert cfg.beat_res == ((0, 4, 8),)
    assert cfg.block_size == 256


def test_keyword_overrides_beat_the_file(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("block_size: 256\n")

    cfg = load_config(DataConfig, path, block_size=512)

    assert cfg.block_size == 512


def test_unknown_keys_are_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("blok_size: 256\n")

    with pytest.raises(ValueError, match="blok_size"):
        load_config(DataConfig, path)


def test_beat_res_map_matches_miditok_shape():
    cfg = DataConfig(beat_res=((0, 4, 16), (4, 12, 4)))

    assert cfg.beat_res_map == {(0, 4): 16, (4, 12): 4}
