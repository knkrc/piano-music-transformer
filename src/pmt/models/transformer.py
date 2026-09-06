"""Decoder-only Transformer - the modern half of the comparison.

Sized to match the LSTM baseline in parameters (~8.6M against ~8.4M) rather than
in width or depth. Same budget, same data, same training loop: the question this
project asks is which architecture spends that budget better, and a comparison
between models of different sizes could not answer it.

Neither model gets hyperparameter tuning. Tuning one and not the other would be
worse than tuning neither, and tuning both properly is a bigger project than this.

Choices worth naming:
- **RoPE** instead of learned positional embeddings. Music is full of transposed
  and repeated material, and relative positions are what carry rhythm.
- **Pre-norm with RMSNorm**, which trains stably without a warmup ritual.
- **SwiGLU** feed-forward, better quality per parameter than a ReLU MLP.
- **``F.scaled_dot_product_attention``** rather than a hand-written attention.
  Reading one is educational; shipping one is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pmt.models.base import LanguageModel

LayerCache = tuple[Tensor, Tensor]
Cache = tuple[LayerCache, ...]


@dataclass(slots=True)
class TransformerConfig:
    """~8.6M parameters at the defaults, matching the LSTM baseline."""

    vocab_size: int = 4096
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 4
    ffn_hidden: int = 1024
    max_seq_len: int = 1024
    dropout: float = 0.2
    tie_weights: bool = True
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model must divide by n_heads, got {self.d_model} / {self.n_heads}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the query/key pairs of ``x`` by their position.

    ``cos``/``sin`` are cast to the input dtype so this stays correct under
    autocast, where q and k arrive as bfloat16 but the buffers are float32.
    """
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    left, right = x.chunk(2, dim=-1)
    return torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCache | None]:
        batch, length, width = x.shape

        queries, keys, values = self.qkv(x).split(width, dim=2)
        shape = (batch, length, self.n_heads, self.head_dim)
        queries = queries.view(shape).transpose(1, 2)
        keys = keys.view(shape).transpose(1, 2)
        values = values.view(shape).transpose(1, 2)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        if cache is not None:
            past_keys, past_values = cache
            keys = torch.cat((past_keys, keys), dim=2)
            values = torch.cat((past_values, values), dim=2)

        context = keys.size(2)
        if context == length:
            attended = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # With a cache the query block sits at the end of the context, so
            # `is_causal` - which aligns the mask top-left - would be wrong. Shifting
            # the triangle by the cached length lets each query see the whole past
            # and nothing ahead of it.
            mask = torch.ones(length, context, dtype=torch.bool, device=x.device).tril(
                context - length
            )
            attended = F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=mask, dropout_p=0.0
            )

        attended = attended.transpose(1, 2).reshape(batch, length, width)
        return self.proj(attended), ((keys, values) if use_cache else None)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(cfg.d_model)
        self.attention = CausalSelfAttention(cfg)
        self.feedforward_norm = nn.RMSNorm(cfg.d_model)
        self.feedforward = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.residual_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cache: LayerCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerCache | None]:
        attended, new_cache = self.attention(self.attention_norm(x), cos, sin, cache, use_cache)
        x = x + self.residual_dropout(attended)
        x = x + self.residual_dropout(self.feedforward(self.feedforward_norm(x)))
        return x, new_cache


class TransformerLanguageModel(LanguageModel):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.embedding_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.embedding.weight

        frequencies = 1.0 / (
            cfg.rope_base ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim)
        )
        angles = torch.outer(torch.arange(cfg.max_seq_len).float(), frequencies)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)

    @property
    def max_context(self) -> int:
        return self.cfg.max_seq_len

    def forward(
        self, tokens: Tensor, state: Cache | None = None, use_cache: bool = False
    ) -> tuple[Tensor, Cache | None]:
        _, length = tokens.shape
        offset = 0 if state is None else state[0][0].size(2)
        if offset + length > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence of {offset + length} exceeds max_seq_len={self.cfg.max_seq_len}"
            )

        cos = self.rope_cos[offset : offset + length]
        sin = self.rope_sin[offset : offset + length]

        x = self.embedding_dropout(self.embedding(tokens))
        caches: list[LayerCache | None] = []
        for index, block in enumerate(self.blocks):
            x, layer_cache = block(x, cos, sin, None if state is None else state[index], use_cache)
            caches.append(layer_cache)

        logits = self.head(self.norm(x))
        return logits, (tuple(caches) if use_cache else None)


def build_transformer(cfg: TransformerConfig) -> TransformerLanguageModel:
    model = TransformerLanguageModel(cfg)

    def initialise(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    model.apply(initialise)

    # Residual branches accumulate across depth, so their output projections start
    # smaller to keep the variance of the residual stream roughly constant.
    scale = (2 * cfg.n_layers) ** -0.5
    for block in model.blocks:
        nn.init.normal_(block.attention.proj.weight, mean=0.0, std=0.02 * scale)
        nn.init.normal_(block.feedforward.down.weight, mean=0.0, std=0.02 * scale)
    return model
