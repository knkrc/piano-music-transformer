---
title: Piano Music Transformer
emoji: 🎹
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
license: mit
short_description: An LSTM and a Transformer of the same size, side by side
---

# Piano Music Transformer

An **LSTM** and a **Transformer** of the same size — 8.40M against 8.65M
parameters — trained on the same MAESTRO piano data for the same 12,000 steps,
generating from the same prompt with the same seed.

The point is not that either is good. It is that the only thing differing between
the two columns is the architecture.

| | LSTM | Transformer |
|---|---|---|
| Test perplexity | 58.2 | **31.9** |

45% lower perplexity on an identical budget — and yet the two sound far more alike
than that gap suggests. Both drift the same way from real playing: more diatonic,
more rhythmically regular, roughly a third of the note density, a narrower
keyboard. Better next-token prediction is not the same thing as better music.

Each column shows its metrics beside the same measurements taken on real MAESTRO
performances, because "scale consistency 0.85" means nothing until you know the
corpus scores 0.81.

- **Code:** [github.com/knkrc/piano-music-transformer](https://github.com/knkrc/piano-music-transformer)
- **Weights:** [huggingface.co/knkrc26/piano-music-transformer](https://huggingface.co/knkrc26/piano-music-transformer)

Running on free CPU, so generation takes a little while — the models decode one
token at a time. Start from scratch, or upload a MIDI file and hear both models
carry on from its first few bars.

Training data is [MAESTRO v3](https://magenta.tensorflow.org/datasets/maestro),
CC BY-NC-SA 4.0. The tokenised dataset is not redistributed here; the reference
figures are pre-measured aggregates.
