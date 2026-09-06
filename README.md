# piano-music-transformer

Symbolic piano music generation, built from scratch in PyTorch — and benchmarked
against the LSTM approach it replaces.

This started as the LSTM jazz-solo exercise from the Coursera *Sequence Models*
course. The interesting question was not "can a Transformer do it better", but
**where the original approach actually loses information** — and the answer turned
out to be the representation, not the architecture.

> **Status: the baseline is trained; the Transformer is training now.** Everything
> through Phase 5 is built. The head-to-head table lands when the second run finishes.

---

## The idea

The original exercise encodes music as a 78-way one-hot vector on a fixed time
grid: no velocity, no sustain pedal, no expressive timing. Any model trained on
that representation is capped by it, however good the architecture is.

This project replaces the representation first, then the architecture:

```
MIDI ──► REMI tokens ──► decoder-only Transformer ──► sampling ──► MIDI ──► audio
                    └──► LSTM baseline ──────────────────┘
```

Both models share one pipeline, one tokenizer and one evaluation protocol, so the
comparison between them means something.

## The baseline, trained

12,000 steps (~3 epochs of the augmented corpus, ~10 hours on an Apple M5).

| | |
|---|---|
| Validation loss | **4.085** (perplexity 59.5) |
| Test perplexity | **58.2** |
| Parameters | 8.40M |

Validation loss was still falling at the last step and the train/validation gap
stayed small (3.95 vs 4.09), so this model is limited by its budget rather than by
overfitting. Listen: [`samples/`](samples/).

### What the notes look like

| Metric | LSTM | Real MAESTRO |
|---|---|---|
| Pitch-class entropy | 2.89 | 3.24 |
| Scale consistency | **0.87** | 0.81 |
| Groove consistency | **0.72** | 0.63 |
| Note density | 2.03 | 5.08 |
| Pitch range | 47.2 | 63.1 |

The interesting part is the direction of the errors. The model is *more* diatonic
and *more* rhythmically regular than the music it learned from, while being far
sparser and narrower. It has found the safe middle of the distribution: a
plausible, well-behaved average of a MAESTRO performance, with the chromaticism,
rubato and density that make an actual performance interesting sanded off. That is
what a small model underfitting a large corpus looks like, and no amount of
sampling tuning fixes it.

### It does not always know how to start

Nine of twenty unprompted samples stop early — four of them within 90 tokens.
Trained on a stream where pieces are separated by begin- and end-of-sequence
tokens, the model learned that pieces end, and from a cold start it reaches for
that too eagerly. Given four bars to continue, this does not happen: prompted
generation ran the full budget every time. Worth knowing before judging the
unprompted samples.

## What is verified so far

The tokenizer is the foundation, so it is measured rather than assumed. Every
number below comes from `python -m pmt.data.prepare` on MAESTRO v3:

| Property | Result |
|---|---|
| Note retention (MIDI → tokens → MIDI) | **1.0000** — no note is ever dropped |
| Pitch fidelity | **1.0000** — every pitch returns intact |
| Onset error | **0.0156 beats** ≈ 8 ms at 120 bpm |
| Musical context per 1024 tokens | **~18 bars** |

Time resolution was chosen by measurement, not taste. Across 4, 8, 16 and 32 steps
per beat, retention and pitch fidelity stayed at 1.0 while onset error halved at
each step — for only ~12% more tokens across the entire range. 16 steps/beat puts
timing error below the threshold where rhythmic displacement is audible; 32 buys
inaudible precision at the cost of sparser position statistics.

## Dataset

[MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) — ~200 hours of
competition piano performances with real velocity, timing and pedal, using the
dataset's own train/validation/test split.

| Split | Files | Tokens |
|---|---|---|
| train | 962 | 12,622,711 |
| train, augmented | 10,637 sequences | 131,601,371 |
| validation | 137 | 1,448,862 |
| test | 177 | 1,626,439 |

REMI base vocabulary 519, extended to 4096 by BPE. Only the 58 MB MIDI archive is
downloaded; the ~120 GB audio release is never touched.

**Only the training split is augmented.** Each piece is transposed up to six semitones
either way, giving an 11.1x effective multiplier — of 12,506 possible variants, 1,869
would push notes off the 88-key range and are dropped whole rather than clipped, since
clipping would silently rewrite the music. Validation and test stay pristine, or the
comparison they exist to support would mean nothing.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
git clone https://github.com/knkrc/piano-music-transformer
cd piano-music-transformer
uv sync
```

Validate the pipeline in seconds on a tiny subset:

```bash
uv run python -m pmt.data.prepare --smoke
```

Build the real thing (downloads MAESTRO on first run, a few minutes total):

```bash
uv run python -m pmt.data.prepare
```

Train the baseline (~10 h on an Apple M5; resumable at any point):

```bash
uv run python -m pmt.train --config configs/lstm.yaml
```

```bash
uv run python -m pmt.train --config configs/lstm.yaml --resume outputs/lstm/last.pt
```

Generate from a checkpoint:

```bash
uv run python -m pmt.sample --checkpoint outputs/lstm/best.pt --num 5
```

Compare the trained models on held-out data and on what they generate:

```bash
uv run python -m pmt.evaluate --checkpoint outputs/lstm/best.pt --checkpoint outputs/transformer/best.pt
```

Hear them side by side in the browser:

```bash
uv run --extra demo python app.py
```

Audio needs FluidSynth and a soundfont. Install them once:

```bash
brew install fluid-synth && uv run python -m pmt.render --install-soundfont
```

Run the tests — they build their MIDI in memory and need no dataset:

```bash
uv run pytest
```

## Layout

```
src/pmt/
├── config.py          # dataclass configs; the defaults here are the source of truth
├── train.py           # one training loop, shared by every model
├── sample.py          # generation, with diagnostics for degenerate output
├── models/
│   └── lstm.py        # the baseline: ~8.4M parameters with tied weights
└── data/
    ├── download.py    # MAESTRO, MIDI only
    ├── tokenizer.py   # REMI build / encode / decode / track merge
    ├── augment.py     # transposition, training split only
    ├── prepare.py     # entry point: download → BPE → shards → fidelity report
    └── dataset.py     # strided windows + deterministic batching
```

### Reproducibility

Batches are a pure function of `(seed, step)` rather than a shuffled `DataLoader`, so
a run stopped and resumed continues bit-for-bit where an uninterrupted run would have
gone. That property has a test, because overnight training on a single machine is only
safe if interruption is free.

## Roadmap

- [x] **Phase 0** — Data pipeline: download, REMI+BPE tokenization, token shards, round-trip verification
- [x] **Phase 1** — LSTM baseline, rewritten in PyTorch *(training underway)*
- [x] **Phase 2** — Decoder-only Transformer (RoPE, pre-norm RMSNorm, SwiGLU, SDPA)
- [x] **Phase 3** — Sampling: top-k / top-p / repetition penalty, KV cache, prompt continuation, sliding context
- [x] **Phase 4** — Evaluation: perplexity plus musical metrics, head-to-head table, rendered audio *(code complete, awaiting trained models)*
- [x] **Phase 5** — Gradio demo, side by side under identical conditions *(code complete, awaiting trained models)*

## Constraints and limitations

This is trained entirely on a single Apple M5 with 16 GB of unified memory. That is
a real constraint and it shapes the results:

- The baseline is ~8.4M parameters at a 1024-token context; the Transformer will be
  sized to match, and both get an identical 12,000-step budget
- Even at 131.6M augmented tokens, expect a model that is **locally coherent but
  structurally weak** — convincing phrases, no long-range musical form
- Solo piano only, no stylistic range
- Augmentation is transposition alone. It multiplies the data without adding a single
  new musical idea, which is a real ceiling on what any of this can learn

These limits will be reported with numbers, not hidden, once models exist.

## Licence

Code is MIT (see [LICENSE](LICENSE)).

The MAESTRO dataset is distributed by Google Magenta under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is **not**
covered by this repository's licence. It is downloaded at runtime, never vendored.

> Curtis Hawthorne et al. *Enabling Factorized Piano Music Modeling and Generation
> with the MAESTRO Dataset.* ICLR 2019.
