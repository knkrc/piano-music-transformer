from __future__ import annotations

import numpy as np
import pytest
import torch

from pmt.data.dataset import TokenWindowDataset


def write_stream(tmp_path, length: int):
    path = tmp_path / "stream.npy"
    np.save(path, np.arange(length, dtype=np.uint16))
    return path


def test_targets_are_inputs_shifted_by_one(tmp_path):
    dataset = TokenWindowDataset(write_stream(tmp_path, 100), block_size=8)

    inputs, targets = dataset[0]

    assert torch.equal(inputs[1:], targets[:-1])
    assert inputs.dtype == torch.int64


def test_windows_tile_the_stream_without_overlap(tmp_path):
    dataset = TokenWindowDataset(write_stream(tmp_path, 100), block_size=8)

    first, _ = dataset[0]
    second, _ = dataset[1]

    assert int(second[0]) - int(first[0]) == 8
    assert len(dataset) == (100 - 8 - 1) // 8 + 1


def test_stride_controls_overlap(tmp_path):
    dataset = TokenWindowDataset(write_stream(tmp_path, 100), block_size=8, stride=4)

    first, _ = dataset[0]
    second, _ = dataset[1]

    assert int(second[0]) - int(first[0]) == 4


def test_last_window_stays_in_bounds(tmp_path):
    dataset = TokenWindowDataset(write_stream(tmp_path, 100), block_size=8)

    inputs, targets = dataset[len(dataset) - 1]

    assert len(inputs) == 8
    assert len(targets) == 8


def test_stream_shorter_than_one_block_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="block_size"):
        TokenWindowDataset(write_stream(tmp_path, 4), block_size=8)
