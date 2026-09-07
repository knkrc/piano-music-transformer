from __future__ import annotations

import torch
from torch import nn

from pmt.sample import generate, pick_next

VOCAB = 8


class ConstantModel(nn.Module):
    """Always predicts one token, so generation behaviour can be tested exactly."""

    def __init__(self, token: int) -> None:
        super().__init__()
        self.token = token

    def forward(self, tokens, state=None):
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
