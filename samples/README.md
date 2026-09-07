# Samples

Matched pairs: the same seed, the same prompt, the same sampling settings
(temperature 1.0, top-k 32), both models after 12,000 training steps.

| | Unprompted | Continuing four real bars |
|---|---|---|
| LSTM (perplexity 58.2) | `lstm_unprompted.*` | `lstm_continuation.*` |
| Transformer (perplexity 31.9) | `transformer_unprompted.*` | `transformer_continuation.*` |

Not cherry-picked from many runs — these are the first sample of each batch.

The continuations begin with four bars of a real MAESTRO performance, so the first
few seconds are human playing and everything after is the model.

MP3s are rendered with FluidSynth and the MIT-licensed MuseScore_General soundfont,
peak-normalised by a constant gain so the dynamics survive intact.
