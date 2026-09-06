"""LSTM baseline - the original Coursera architecture, brought up to date.

This exists to be beaten. It shares the tokenizer, the data pipeline, the
training loop and the evaluation protocol with the Transformer that follows, so
any difference between the two is attributable to the architecture and not to
the hundred other things that usually differ between two repositories.

What is modernised relative to the course version: a REMI token vocabulary
instead of a 78-way one-hot, weight tying, dropout between layers, and training
on windows of a continuous stream rather than fixed 30-step excerpts.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from pmt.models.base import LanguageModel

State = tuple[Tensor, Tensor]


@dataclass(slots=True)
class LSTMConfig:
    """~10M parameters at the defaults, which suits a 12.6M-token corpus."""

    vocab_size: int = 4096
    d_model: int = 512
    hidden_size: int = 512
    num_layers: int = 3
    dropout: float = 0.2
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.tie_weights and self.hidden_size != self.d_model:
            raise ValueError(
                f"tie_weights needs hidden_size == d_model, "
                f"got {self.hidden_size} != {self.d_model}"
            )


class LSTMLanguageModel(LanguageModel):
    """Next-token model over REMI tokens."""

    def __init__(self, cfg: LSTMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.lstm = nn.LSTM(
            cfg.d_model,
            cfg.hidden_size,
            cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        if cfg.tie_weights:
            self.head.weight = self.embedding.weight

    def forward(
        self, tokens: Tensor, state: State | None = None, use_cache: bool = False
    ) -> tuple[Tensor, State]:
        """Return ``(logits, state)``.

        ``use_cache`` is accepted for interface parity and ignored: recurrent state
        is a fixed-size tensor the LSTM produces anyway, so there is nothing to save
        by not returning it.
        """
        hidden, state = self.lstm(self.dropout(self.embedding(tokens)), state)
        return self.head(self.dropout(hidden)), state


def build_lstm(cfg: LSTMConfig) -> LSTMLanguageModel:
    model = LSTMLanguageModel(cfg)
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
    return model
