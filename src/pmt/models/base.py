"""The interface every model in this project shares.

The training loop and the sampler are written once and used by both models. That
only works if the models agree on a contract, and this is it.
"""

from __future__ import annotations

from torch import Tensor, nn


class LanguageModel(nn.Module):
    """Next-token model over REMI tokens.

    ``forward`` returns ``(logits, state)``. ``state`` is whatever the model needs
    to continue generation one token at a time without replaying the prefix:
    recurrent state for the LSTM, a key/value cache for the Transformer. Sampling
    therefore looks identical for both.

    ``use_cache`` exists because that state is free for an RNN and expensive for a
    Transformer - a cache over a full training batch is hundreds of megabytes that
    would be discarded immediately. Training leaves it off; generation turns it on.
    """

    def forward(self, tokens: Tensor, state=None, use_cache: bool = False) -> tuple[Tensor, object]:
        raise NotImplementedError

    @property
    def max_context(self) -> int | None:
        """Longest sequence the model can attend over, or ``None`` if unbounded.

        An RNN carries a fixed-size state and has no limit; a Transformer is capped
        by its positional encoding. Sliding that window is Phase 3 work.
        """
        return None

    def trim_state(self, state, window: int):
        """Shorten the state so the attention window stays within ``window``.

        An RNN's state is a fixed-size tensor with nothing to trim, so the default
        returns it untouched.
        """
        return state

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            params = (param for param in params if param.requires_grad)
        # Weight tying shares one tensor between two modules; count it once.
        unique = {id(param): param for param in params}
        return sum(param.numel() for param in unique.values())
