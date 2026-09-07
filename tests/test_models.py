from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from pmt.models.lstm import LSTMConfig, build_lstm

SMALL = LSTMConfig(vocab_size=64, d_model=32, hidden_size=32, num_layers=2, dropout=0.0)


def test_forward_returns_logits_over_the_vocabulary():
    model = build_lstm(SMALL)

    logits, _ = model(torch.randint(0, SMALL.vocab_size, (3, 7)))

    assert logits.shape == (3, 7, SMALL.vocab_size)


def test_state_lets_generation_advance_one_token_at_a_time():
    """Feeding a sequence whole must match feeding it token by token."""
    model = build_lstm(SMALL).eval()
    tokens = torch.randint(0, SMALL.vocab_size, (1, 6))

    with torch.no_grad():
        whole, _ = model(tokens)
        state = None
        stepwise = []
        for index in range(tokens.size(1)):
            logits, state = model(tokens[:, index : index + 1], state)
            stepwise.append(logits)

    assert torch.allclose(whole, torch.cat(stepwise, dim=1), atol=1e-5)


def test_weight_tying_shares_one_tensor_and_is_counted_once():
    tied = build_lstm(replace(SMALL, tie_weights=True))
    untied = build_lstm(replace(SMALL, tie_weights=False))

    assert tied.head.weight is tied.embedding.weight
    assert tied.num_parameters() < untied.num_parameters()


def test_tying_requires_matching_widths():
    with pytest.raises(ValueError, match="tie_weights"):
        LSTMConfig(d_model=32, hidden_size=64, tie_weights=True)
