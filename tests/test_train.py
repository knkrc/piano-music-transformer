"""The training loop's contract: resumable, deterministic, and honest about it."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from pmt.data.dataset import TokenWindowDataset, deterministic_batch, evaluation_batches
from pmt.models.lstm import LSTMConfig
from pmt.train import TrainConfig, autocast_for, learning_rate_at, train

TINY_TRAIN = TrainConfig(
    batch_size=2,
    grad_accum=1,
    block_size=32,
    max_steps=4,
    warmup_steps=1,
    log_every=100,
    eval_every=100,
    eval_batches=1,
    save_every=1,
    device="cpu",
    seed=7,
)
TINY_MODEL = LSTMConfig(vocab_size=64, d_model=16, hidden_size=16, num_layers=1, dropout=0.0)


@pytest.fixture
def token_data(tmp_path):
    rng = np.random.default_rng(0)
    for split in ("train", "validation"):
        np.save(tmp_path / f"{split}.npy", rng.integers(4, 64, 4000).astype(np.uint16))
    (tmp_path / "meta.json").write_text(json.dumps({"vocab_size": TINY_MODEL.vocab_size}))
    return tmp_path


def test_resume_reproduces_an_uninterrupted_run(token_data, tmp_path):
    """A run stopped at step 2 and resumed must land exactly where a 4-step run does.

    This is the property that makes overnight training on one machine safe.
    """
    straight = tmp_path / "straight"
    train(TINY_TRAIN, TINY_MODEL, token_data, straight)

    interrupted = tmp_path / "interrupted"
    train(replace(TINY_TRAIN, max_steps=2), TINY_MODEL, token_data, interrupted)
    train(TINY_TRAIN, TINY_MODEL, token_data, interrupted, resume=interrupted / "last.pt")

    expected = torch.load(straight / "last.pt", weights_only=False)
    actual = torch.load(interrupted / "last.pt", weights_only=False)
    assert actual["step"] == expected["step"] == 4
    for key, tensor in expected["model"].items():
        assert torch.equal(actual["model"][key], tensor), f"diverged at {key}"


def test_training_writes_metrics_and_config(token_data, tmp_path):
    out = tmp_path / "run"

    train(TINY_TRAIN, TINY_MODEL, token_data, out)

    assert (out / "config.json").exists()
    assert (out / "metrics.jsonl").read_text().strip()
    assert (out / "last.pt").exists()


def test_batches_depend_only_on_seed_and_step(token_data):
    dataset = TokenWindowDataset(token_data / "train.npy", block_size=32)

    first, _ = deterministic_batch(dataset, step=3, batch_size=2, seed=7)
    again, _ = deterministic_batch(dataset, step=3, batch_size=2, seed=7)
    later, _ = deterministic_batch(dataset, step=4, batch_size=2, seed=7)

    assert torch.equal(first, again)
    assert not torch.equal(first, later)


def test_evaluation_batches_are_stable_across_calls(token_data):
    dataset = TokenWindowDataset(token_data / "train.npy", block_size=32)

    first = evaluation_batches(dataset, batch_size=2, num_batches=3)
    second = evaluation_batches(dataset, batch_size=2, num_batches=3)

    assert len(first) == 3
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second, strict=True))


def test_schedule_warms_up_then_decays_to_the_floor():
    cfg = TrainConfig(lr=1e-3, warmup_steps=10, max_steps=100, min_lr_ratio=0.1)

    assert learning_rate_at(0, cfg) == pytest.approx(1e-4)
    assert learning_rate_at(9, cfg) == pytest.approx(1e-3)
    assert learning_rate_at(99, cfg) == pytest.approx(1e-4, rel=0.02)
    assert learning_rate_at(50, cfg) < learning_rate_at(10, cfg)


def test_vocabulary_size_is_taken_from_the_data(token_data, tmp_path):
    """A model whose vocabulary disagrees with the data emits undecodable ids.

    The failure surfaces only at generation time as a tokenizer crash, so the two
    are tied together here instead of being left to agree by convention.
    """
    out = tmp_path / "run"

    train(TINY_TRAIN, replace(TINY_MODEL, vocab_size=4096), token_data, out)

    saved = json.loads((out / "config.json").read_text())
    assert saved["model"]["vocab_size"] == TINY_MODEL.vocab_size


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_every_configured_precision_resolves(precision):
    """A wrong precision name used to surface only on the first non-fp32 run.

    `getattr(torch, "bf16")` reads as if it works and does not: torch spells it
    `bfloat16`. The names live in a table now, and this walks all of them.
    """
    with autocast_for(torch.device("cpu"), precision):
        pass


def test_an_unknown_precision_is_rejected_immediately():
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_for(torch.device("cpu"), "float8")
