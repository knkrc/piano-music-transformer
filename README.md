# piano-music-transformer

Symbolic piano music generation, built from scratch in PyTorch — and benchmarked
against the LSTM approach it replaces.

This started as the LSTM jazz-solo exercise from the Coursera *Sequence Models*
course. The interesting question was not "can a Transformer do it better", but
**where the original approach actually loses information** — and the answer turned
out to be the representation, not the architecture.

> **Status: Phase 0 of 5 complete.** The data pipeline is built and verified.
> No model has been trained yet. Roadmap below.

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
| validation | 137 | 1,448,862 |
| test | 177 | 1,626,439 |

REMI base vocabulary 519, extended to 4096 by BPE. Only the 58 MB MIDI archive is
downloaded; the ~120 GB audio release is never touched.

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

Run the tests — they build their MIDI in memory and need no dataset:

```bash
uv run pytest
```

## Layout

```
src/pmt/
├── config.py          # dataclass configs; the defaults here are the source of truth
└── data/
    ├── download.py    # MAESTRO, MIDI only
    ├── tokenizer.py   # REMI build / encode / decode / track merge
    ├── prepare.py     # entry point: download → BPE → shards → fidelity report
    └── dataset.py     # strided windows over the token stream
```

## Roadmap

- [x] **Phase 0** — Data pipeline: download, REMI+BPE tokenization, token shards, round-trip verification
- [ ] **Phase 1** — LSTM baseline, rewritten in PyTorch
- [ ] **Phase 2** — Decoder-only Transformer (RoPE, pre-norm, SDPA)
- [ ] **Phase 3** — Sampling: top-k / top-p / repetition penalty, KV cache, prompt continuation
- [ ] **Phase 4** — Evaluation: perplexity plus musical metrics, head-to-head table, rendered audio
- [ ] **Phase 5** — Gradio demo and published weights

## Constraints and limitations

This is trained entirely on a single Apple M5 with 16 GB of unified memory. That is
a real constraint and it shapes the results:

- Model budget is ~20-30M parameters at a 1024-token context
- 12.6M training tokens is small. Expect a model that is **locally coherent but
  structurally weak** — convincing phrases, no long-range musical form
- Solo piano only, no stylistic range
- Data augmentation (pitch transposition, tempo jitter) will do more for quality
  here than adding parameters would

These limits will be reported with numbers, not hidden, once models exist.

## Licence

Code is MIT (see [LICENSE](LICENSE)).

The MAESTRO dataset is distributed by Google Magenta under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is **not**
covered by this repository's licence. It is downloaded at runtime, never vendored.

> Curtis Hawthorne et al. *Enabling Factorized Piano Music Modeling and Generation
> with the MAESTRO Dataset.* ICLR 2019.
