"""Published weights must reconstruct the model they came from, exactly."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("safetensors", reason="install with --extra hub")

from safetensors.torch import load_model

from pmt.export import export_model
from pmt.models.lstm import LSTMConfig, build_lstm

TINY = LSTMConfig(vocab_size=64, d_model=16, hidden_size=16, num_layers=1, dropout=0.0)


@pytest.fixture
def checkpoint(tmp_path):
    """A training checkpoint, complete with the state a published model should shed."""
    model = build_lstm(TINY)
    path = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {"heavy": torch.zeros(100_000)},
            "rng": {"torch": torch.get_rng_state()},
            "step": 12000,
            "best_val": 4.0853,
            "model_config": {
                "vocab_size": 64,
                "d_model": 16,
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "tie_weights": True,
            },
            "train_config": {"model": "lstm", "block_size": 1024, "precision": "fp32"},
        },
        path,
    )
    return path, model


def test_exported_weights_reproduce_the_original_exactly(checkpoint, tmp_path):
    path, original = checkpoint
    original.eval()
    tokens = torch.randint(0, TINY.vocab_size, (1, 8))

    export_model(path, tmp_path / "out")

    restored = build_lstm(TINY)
    load_model(restored, str(tmp_path / "out" / "model.safetensors"))
    restored.eval()
    with torch.no_grad():
        assert torch.equal(original(tokens)[0], restored(tokens)[0])


def test_the_optimizer_state_is_left_behind(checkpoint, tmp_path):
    """More than half a training checkpoint is state nobody downloading it needs."""
    path, _ = checkpoint

    export_model(path, tmp_path / "out")

    assert (tmp_path / "out" / "model.safetensors").stat().st_size < path.stat().st_size / 2


def test_the_config_records_what_produced_the_weights(checkpoint, tmp_path):
    path, _ = checkpoint

    export_model(path, tmp_path / "out")

    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["model"] == "lstm"
    assert config["step"] == 12000
    assert config["validation_loss"] == pytest.approx(4.0853)
    assert "optimizer" not in config
