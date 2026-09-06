from __future__ import annotations

import torch

from pmt.models.base import LanguageModel
from pmt.sample import generate, pick_next

VOCAB = 8


class ConstantModel(LanguageModel):
    """Always predicts one token, so generation behaviour can be tested exactly."""

    def __init__(self, token: int, context_limit: int | None = None) -> None:
        super().__init__()
        self.token = token
        self.context_limit = context_limit

    @property
    def max_context(self) -> int | None:
        return self.context_limit

    def forward(self, tokens, state=None, use_cache=False):
        logits = torch.full((tokens.size(0), tokens.size(1), VOCAB), -10.0)
        logits[..., self.token] = 10.0
        return logits, state


def test_zero_temperature_is_greedy():
    logits = torch.tensor([0.1, 5.0, 0.2, 0.3])

    assert pick_next(logits, temperature=0.0, top_k=0, banned=[]) == 1


def test_banned_tokens_are_never_produced():
    logits = torch.tensor([9.0, 1.0, 0.5, 0.2])

    for _ in range(20):
        assert pick_next(logits, temperature=1.0, top_k=0, banned=[0]) != 0


def test_top_k_of_one_collapses_to_the_argmax():
    logits = torch.tensor([0.1, 0.2, 7.0, 0.3])

    for _ in range(10):
        assert pick_next(logits, temperature=2.0, top_k=1, banned=[]) == 2


def test_generation_stops_at_end_of_sequence():
    model = ConstantModel(token=2)

    produced = generate(
        model,
        [1],
        max_new_tokens=50,
        temperature=0.0,
        top_k=0,
        banned=[],
        eos_id=2,
        device=torch.device("cpu"),
    )

    assert produced == []


def test_generation_respects_the_token_budget():
    model = ConstantModel(token=5)

    produced = generate(
        model,
        [1],
        max_new_tokens=12,
        temperature=0.0,
        top_k=0,
        banned=[],
        eos_id=2,
        device=torch.device("cpu"),
    )

    assert produced == [5] * 12


def test_generation_stops_at_the_context_limit():
    """A Transformer cannot see past its positional encoding, so sampling must stop."""
    model = ConstantModel(token=5, context_limit=10)

    produced = generate(
        model,
        [1, 1, 1],
        max_new_tokens=100,
        temperature=0.0,
        top_k=0,
        banned=[],
        eos_id=2,
        device=torch.device("cpu"),
    )

    assert len(produced) == 7  # 3 prompt tokens + 7 generated = the 10-token limit
