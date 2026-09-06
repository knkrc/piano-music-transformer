"""The Transformer's load-bearing properties: causality, cache exactness, fairness."""

from __future__ import annotations

import pytest
import torch

from pmt.models.lstm import LSTMConfig, build_lstm
from pmt.models.transformer import TransformerConfig, apply_rope, build_transformer

SMALL = TransformerConfig(
    vocab_size=64,
    d_model=64,
    n_heads=4,
    n_layers=3,
    ffn_hidden=128,
    max_seq_len=32,
    dropout=0.0,
)


@pytest.fixture
def model():
    return build_transformer(SMALL).eval()


@pytest.fixture
def tokens():
    torch.manual_seed(0)
    return torch.randint(0, SMALL.vocab_size, (2, 12))


def test_incremental_decoding_matches_a_full_forward(model, tokens):
    """The KV cache is an optimisation; if it changes the output it is a bug."""
    with torch.no_grad():
        whole, _ = model(tokens)

        logits, state = model(tokens[:, :5], use_cache=True)
        pieces = [logits]
        for index in range(5, tokens.size(1)):
            logits, state = model(tokens[:, index : index + 1], state, use_cache=True)
            pieces.append(logits)

    assert torch.allclose(whole, torch.cat(pieces, dim=1), atol=1e-4)


def test_a_later_token_cannot_change_earlier_logits(model, tokens):
    altered = tokens.clone()
    altered[:, -1] = (altered[:, -1] + 7) % SMALL.vocab_size

    with torch.no_grad():
        original, _ = model(tokens)
        changed, _ = model(altered)

    assert torch.allclose(original[:, :-1], changed[:, :-1], atol=1e-6)


def test_rope_scores_depend_only_on_relative_position(model):
    """RoPE's defining property, and the reason it was chosen over learned positions.

    The attention score between a query at position m and a key at position n must
    depend on m - n alone. Music is full of material repeated at a distance, and
    that is the structure this makes visible to the model.
    """
    torch.manual_seed(0)
    query = torch.randn(1, 1, 1, SMALL.head_dim)
    key = torch.randn(1, 1, 1, SMALL.head_dim)

    def score(at_query: int, at_key: int) -> float:
        rotated_query = apply_rope(
            query, model.rope_cos[at_query : at_query + 1], model.rope_sin[at_query : at_query + 1]
        )
        rotated_key = apply_rope(
            key, model.rope_cos[at_key : at_key + 1], model.rope_sin[at_key : at_key + 1]
        )
        return float((rotated_query * rotated_key).sum())

    assert score(3, 1) == pytest.approx(score(9, 7), abs=1e-5)
    assert score(3, 1) != pytest.approx(score(3, 2), abs=1e-5)


def test_rope_is_a_rotation(model):
    """It must not rescale anything - only turn it."""
    torch.manual_seed(0)
    vector = torch.randn(1, 1, 1, SMALL.head_dim)

    rotated = apply_rope(vector, model.rope_cos[5:6], model.rope_sin[5:6])

    assert torch.allclose(rotated.norm(), vector.norm(), atol=1e-5)
    assert torch.allclose(
        apply_rope(vector, model.rope_cos[0:1], model.rope_sin[0:1]), vector, atol=1e-6
    )


def test_no_cache_is_returned_unless_asked_for(model, tokens):
    with torch.no_grad():
        _, state = model(tokens)

    assert state is None


def test_sequences_beyond_the_positional_encoding_are_rejected(model):
    too_long = torch.zeros((1, SMALL.max_seq_len + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="max_seq_len"):
        model(too_long)


def test_heads_must_divide_the_width():
    with pytest.raises(ValueError, match="n_heads"):
        TransformerConfig(d_model=100, n_heads=6)


def test_the_two_models_are_matched_on_parameters():
    """The comparison is only meaningful if the budgets are.

    This test is the fairness claim in the README, made enforceable.
    """
    transformer = build_transformer(TransformerConfig())
    lstm = build_lstm(LSTMConfig())

    ratio = transformer.num_parameters() / lstm.num_parameters()
    assert 0.95 <= ratio <= 1.05, f"parameter budgets diverged: {ratio:.2f}x"
