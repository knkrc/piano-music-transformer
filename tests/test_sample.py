from __future__ import annotations

import pytest
import torch

from pmt.config import DataConfig
from pmt.data.tokenizer import build_tokenizer, decode_to_score
from pmt.models.base import LanguageModel
from pmt.models.transformer import TransformerConfig, build_transformer
from pmt.sample import SamplingSettings, generate, pick_next, prompt_from_midi

VOCAB = 8
NEVER = -1  # an eos id no token can equal, for tests about length rather than stopping


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

    assert pick_next(logits, SamplingSettings(temperature=0.0), banned=[]) == 1


def test_banned_tokens_are_never_produced():
    logits = torch.tensor([9.0, 1.0, 0.5, 0.2])

    for _ in range(20):
        assert pick_next(logits, SamplingSettings(), banned=[0]) != 0


def test_top_k_of_one_collapses_to_the_argmax():
    logits = torch.tensor([0.1, 0.2, 7.0, 0.3])

    for _ in range(10):
        assert pick_next(logits, SamplingSettings(temperature=2.0, top_k=1), banned=[]) == 2


def test_top_p_discards_the_tail():
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])
    settings = SamplingSettings(top_k=0, top_p=0.5)

    for _ in range(20):
        assert pick_next(logits, settings, banned=[]) == 0


def test_top_p_always_keeps_at_least_one_token():
    """A head that alone exceeds top_p must not leave an empty distribution."""
    logits = torch.tensor([20.0, 1.0, 1.0, 1.0])
    settings = SamplingSettings(top_k=0, top_p=0.01)

    assert pick_next(logits, settings, banned=[]) == 0


def test_repetition_penalty_is_off_by_default():
    """Music is repetition; the default must not fight it."""
    logits = torch.tensor([1.0, 9.0, 0.5, 0.2])

    assert SamplingSettings().repetition_penalty == 1.0
    assert pick_next(logits, SamplingSettings(temperature=0.0), banned=[], recent=[1]) == 1


def test_repetition_penalty_demotes_a_recent_token():
    logits = torch.tensor([1.0, 9.0, 8.0, 0.2])
    settings = SamplingSettings(temperature=0.0, repetition_penalty=2.0)

    assert pick_next(logits, settings, banned=[], recent=[1]) == 2


def test_the_penalty_only_looks_back_over_its_window():
    logits = torch.tensor([1.0, 9.0, 8.0, 0.2])
    settings = SamplingSettings(temperature=0.0, repetition_penalty=2.0, repetition_window=2)
    long_ago = [1, 3, 3]  # token 1 has fallen out of the last two

    assert pick_next(logits, settings, banned=[], recent=long_ago) == 1


def test_generation_stops_at_end_of_sequence():
    produced = generate(
        ConstantModel(token=2),
        [1],
        50,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=2,
        device=torch.device("cpu"),
    )

    assert produced == []


def test_generation_respects_the_token_budget():
    produced = generate(
        ConstantModel(token=5),
        [1],
        12,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=NEVER,
        device=torch.device("cpu"),
    )

    assert produced == [5] * 12


def test_generation_slides_past_the_attention_window():
    """RoPE is what makes this safe: a sliding window keeps every query-key
    distance inside the trained range, however far absolute positions travel."""
    model = build_transformer(
        TransformerConfig(
            vocab_size=VOCAB,
            d_model=32,
            n_heads=2,
            n_layers=2,
            ffn_hidden=64,
            max_seq_len=16,
            dropout=0.0,
        )
    ).eval()

    produced = generate(
        model,
        [1, 1, 1, 1],
        40,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=NEVER,
        device=torch.device("cpu"),
    )

    assert len(produced) == 40  # more than twice the 16-token window


def test_a_prompt_longer_than_the_window_is_trimmed():
    model = build_transformer(
        TransformerConfig(
            vocab_size=VOCAB,
            d_model=32,
            n_heads=2,
            n_layers=2,
            ffn_hidden=64,
            max_seq_len=8,
            dropout=0.0,
        )
    ).eval()

    produced = generate(
        model,
        [1] * 50,
        5,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=NEVER,
        device=torch.device("cpu"),
    )

    assert len(produced) == 5


@pytest.mark.parametrize("bars", [2, 4])
def test_a_midi_prompt_is_cut_at_a_barline(make_score, tmp_path, bars):
    """Whole bars, not a token count - the model picks up on a barline as a player would."""
    eight_bars = make_score([60, 62, 64, 65, 67, 69, 71, 72] * 8)  # 64 eighths = 8 bars
    path = tmp_path / "prompt.mid"
    eight_bars.dump_midi(str(path))
    tokenizer = build_tokenizer(DataConfig())

    ids = prompt_from_midi(tokenizer, path, bars=bars)

    notes = decode_to_score(tokenizer, ids).tracks[0].notes
    assert len(notes) == bars * 8  # eight eighth-notes to the bar


def test_sampling_ignores_dropout_left_switched_on():
    """Generation must not depend on the mode the caller happened to leave behind."""
    config = TransformerConfig(
        vocab_size=VOCAB,
        d_model=32,
        n_heads=2,
        n_layers=2,
        ffn_hidden=64,
        max_seq_len=32,
        dropout=0.5,
    )
    model = build_transformer(config)

    model.eval()
    from_eval = generate(
        model,
        [1],
        8,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=NEVER,
        device=torch.device("cpu"),
    )
    model.train()
    from_train = generate(
        model,
        [1],
        8,
        SamplingSettings(temperature=0.0),
        banned=[],
        eos_id=NEVER,
        device=torch.device("cpu"),
    )

    assert from_eval == from_train
    assert model.training is True  # and the caller's mode is handed back untouched
