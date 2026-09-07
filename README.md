# piano-music-transformer

Symbolic piano music generation, built from scratch in PyTorch — and benchmarked
against the LSTM approach it replaces.

This started as the LSTM jazz-solo exercise from the Coursera *Sequence Models*
course. The interesting question was not "can a Transformer do it better", but
**where the original approach actually loses information** — and the answer turned
out to be the representation, not the architecture.

> **Status: complete.** Both models trained, compared and rendered. Results below.

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

## Results

Both models trained on the same MAESTRO data, the same tokenizer, the same 12,000-step
budget, the same data order and the same evaluation. Three things are deliberately not
tied: the architecture, the learning rate (each architecture's conventional default) and
the numerical precision. All three are spelled out under [Limitations](#constraints-and-limitations).

### Prediction

| | LSTM | Transformer |
|---|---|---|
| Parameters | 8.40M | 8.65M |
| Validation loss | 4.085 | **3.550** |
| Test perplexity | 58.2 | **31.9** |

**45% lower perplexity on the same budget.** The Transformer passed the LSTM's *final*
score at around step 2,000 — one sixth of the way through its own training.

Both models were still improving when the budget ran out, and neither overfitted: the
train/validation gap stayed at 0.14 (LSTM) and 0.06 (Transformer). These are numbers
about a 10-hour laptop run, not about the architectures at scale.

### What the notes look like

| Metric | LSTM | Transformer | Real MAESTRO |
|---|---|---|---|
| Pitch-class entropy | 2.86 | 2.86 | 3.24 |
| Scale consistency | 0.91 | **0.85** | 0.81 |
| Groove consistency | 0.75 | **0.71** | 0.63 |
| Note density | **2.30** | 1.69 | 5.08 |
| Pitch range | **47.4** | 46.0 | 63.1 |

Bold marks whichever model sits closer to the corpus. It is a split decision: the
Transformer is closer on tonality and rhythmic regularity, the LSTM on density and
range, and they are indistinguishable on pitch-class entropy.

**This is the finding worth sitting with.** A 45% cut in perplexity did not buy a
matching improvement in what the output looks like. Both models drift the same way from
the corpus — more diatonic, more rhythmically regular, roughly a third of the note
density, a narrower keyboard. Both have found the safe middle of the distribution: a
plausible, well-behaved average of a MAESTRO performance, with the chromaticism, rubato
and density that make a real one interesting sanded off. Better prediction of the next
token is not the same thing as better music, and at this scale the gap between them is
wide.

### Starting from nothing is hard for both

| | Reached the full 1024 tokens |
|---|---|
| LSTM | 11 / 20 |
| Transformer | 14 / 20 |

Trained on a stream where pieces are separated by begin- and end-of-sequence tokens,
both models learned that pieces end and reach for it too eagerly from a cold start — the
LSTM produced four samples under 90 tokens. Given four bars to continue, neither model
did this: prompted generation ran the full budget every time. Worth knowing before
judging the unprompted samples.

### Listen

[`samples/`](samples/) holds matched pairs — the same seed, the same prompt, both models:

| | Unprompted | Continuing four real bars |
|---|---|---|
| LSTM | `lstm_unprompted.mp3` | `lstm_continuation.mp3` |
| Transformer | `transformer_unprompted.mp3` | `transformer_continuation.mp3` |

Not cherry-picked: they are the first sample of each batch.

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

Package the trained weights for publication (inference-only, safetensors):

```bash
uv run --extra hub python -m pmt.export
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
- [x] **Phase 4** — Evaluation: perplexity plus musical metrics, head-to-head table, rendered audio
- [x] **Phase 5** — Gradio demo, side by side under identical conditions

## Constraints and limitations

This is trained entirely on a single Apple M5 with 16 GB of unified memory. That is
a real constraint and it shapes the results:

- **One run per model, no error bars.** Every number here comes from a single training
  run at a single seed. The perplexity gap is far too large to be seed noise; the
  musical-metric differences (0.91 vs 0.85) may well not be. Read them as suggestive,
  not settled.
- **Neither model got a hyperparameter sweep.** Learning rate is each architecture's
  conventional default — 1e-3 for the LSTM, 6e-4 for the Transformer. Forcing a shared
  value would suit one of them; tuning one and not the other would be worse. Tuning both
  properly is a larger project than this.
- **The Transformer trains in bf16, the LSTM in fp32.** Measured at +22% for the
  Transformer against +7% for the LSTM, with loss agreeing to 0.0002 over the same
  steps. Autocast keeps parameters in fp32 and reduces only the matmuls, so if anything
  this costs the Transformer a little precision rather than flattering it.
- Both models are **locally coherent and structurally weak** — convincing phrases, no
  long-range musical form. 8.65M parameters on a laptop is the reason.
- Solo piano only, no stylistic range.
- Augmentation is transposition alone. It multiplies the data without adding a single
  new musical idea, which is a real ceiling on what any of this can learn.

These limits will be reported with numbers, not hidden, once models exist.

## Licence

Code is MIT (see [LICENSE](LICENSE)).

The MAESTRO dataset is distributed by Google Magenta under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is **not**
covered by this repository's licence. It is downloaded at runtime, never vendored.

> Curtis Hawthorne et al. *Enabling Factorized Piano Music Modeling and Generation
> with the MAESTRO Dataset.* ICLR 2019.
