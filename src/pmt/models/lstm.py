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


class LSTMLanguageModel(nn.Module):
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

    def forward(self, tokens: Tensor, state: State | None = None) -> tuple[Tensor, State]:
        """Return ``(logits, state)``.

        The returned state lets generation advance one token at a time without
        replaying the whole prefix, which is what makes sampling cheap for an RNN.
        """
        hidden, state = self.lstm(self.dropout(self.embedding(tokens)), state)
        return self.head(self.dropout(hidden)), state

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (p for p in params if p.requires_grad)
        seen: dict[int, Tensor] = {}
        for param in params:
            seen[id(param)] = param  # weight tying shares one tensor between two modules
        return sum(param.numel() for param in seen.values())


def build_lstm(cfg: LSTMConfig) -> LSTMLanguageModel:
    model = LSTMLanguageModel(cfg)
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
    return model
