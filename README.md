# piano-music-transformer

Symbolic piano music generation, built from scratch in PyTorch — and benchmarked
against the LSTM approach it replaces.

This started as the LSTM jazz-solo exercise from the Coursera *Sequence Models*
course. The interesting question was not "can a Transformer do it better", but
**where the original approach actually loses information** — and the answer turned
out to be the representation, not the architecture.

> **Status: Phase 1 of 5 in progress.** The data pipeline is built and verified and
> the LSTM baseline trains end to end; its full run is underway. No results are
> claimed here until they exist. Roadmap below.

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
- [ ] **Phase 1** — LSTM baseline, rewritten in PyTorch *(code complete, training underway)*
- [ ] **Phase 2** — Decoder-only Transformer (RoPE, pre-norm, SDPA)
- [ ] **Phase 3** — Sampling: top-k / top-p / repetition penalty, KV cache, prompt continuation
- [ ] **Phase 4** — Evaluation: perplexity plus musical metrics, head-to-head table, rendered audio
- [ ] **Phase 5** — Gradio demo and published weights

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
