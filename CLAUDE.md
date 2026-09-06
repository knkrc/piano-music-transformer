# CLAUDE.md

> The working agreement for this project: decisions, constraints, current state.
> Not a code explanation — *why* and *where we are* live here.
> Updated at the end of every phase.

## 1. What this is

A modern rebuild of the LSTM jazz-solo exercise from the Coursera Sequence Models
course. Symbolic (MIDI) music generation:

```
MIDI -> REMI tokens -> decoder-only Transformer -> sampling -> MIDI -> audio
```

An LSTM baseline, rewritten in PyTorch, runs on the same data pipeline as the
comparison anchor. That comparison is the project's actual story: *same data,
same metrics, old approach vs modern approach.*

**Key insight:** the bottleneck in the original exercise was not the architecture,
it was the representation (78-way one-hot, fixed grid, no velocity, no pedal).
The largest gain comes from tokenization. This is why Phase 0 is the most
important phase even though it looks like the most boring one.

## 2. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Domain | Symbolic / MIDI | Trainable from scratch on an M5; audio-domain training is infeasible here |
| Dataset | MAESTRO v3 (MIDI only) | ~200 h of clean human performance with real velocity and pedal; single instrument keeps the problem tractable |
| Hardware | This Mac only (M5, 16 GB, MPS) | No Colab, no rented GPU. The model budget follows from this |
| Engineering level | Plain and tested | No Docker, no CI, no config framework. Clean, readable, working |
| Framework | PyTorch | — |
| Tokenizer | MidiTok REMI + BPE | Not hand-rolled, but verified by a round-trip test |
| Metrics | Implemented in-repo | `muspy` is a fragile dependency; the 3-4 metrics we need are ~60 lines |
| Package name | `pmt` | Matches the repo name `piano-music-transformer` |
| Language | English everywhere | Public portfolio repo — code, comments, docs, commits |
| Augmentation | Offline, transposition, train split only | BPE merges several events into one token, so pitch cannot be shifted on ids after the fact |
| Batching | Step-indexed deterministic sampling, no `DataLoader` | Makes resume *exact* rather than approximate; memmap reads are microseconds, so there is nothing to prefetch |
| Training budget | 12,000 steps (~3 epochs, ~5.5 h) | Phase 2 gets the identical budget — an under-trained baseline would flatter the Transformer |
| Model comparison | Matched on **parameters** (8.65M vs 8.40M), not width or depth | The question is which architecture spends the same budget better; different sizes cannot answer it. There is a test enforcing this |
| Learning rate | Each architecture's conventional default (LSTM 1e-3, Transformer 6e-4) | Forcing a shared value would suit one of them; tuning one and not the other is worse. Neither got a sweep, and the README says so |
| `vocab_size` | Derived from the data's `meta.json`, never configured | A model and tokenizer that disagree produce ids that cannot be decoded, and it only surfaces at generation time |

**The audio release is never downloaded.** MAESTRO with audio is ~120 GB; only the
58 MB MIDI archive is fetched.

## 3. Hardware reality and what follows from it

Apple M5, 16 GB unified memory, MPS backend.

- Model budget: **~20-30M parameters**, context **1024 tokens**, 8 layers / d_model 512
- Effective batch size comes from gradient accumulation; physical batch follows memory
- Try `torch.autocast("mps", bfloat16)`; **fall back to fp32** on NaN or kernel errors, and record it here
- `torch.compile` stays **off** by default (flaky on MPS); enabling it is opt-in
- Every training run must be **resumable** (optimizer, scheduler and RNG state included) — overnight training is the working model
- Expect training time in hours, not minutes. Every entry point keeps a `--smoke` mode for fast iteration

## 4. Deliberately out of scope

Docker · GitHub Actions CI · FastAPI service · multi-GPU DDP · custom CUDA kernels ·
MLflow server · Hydra · Lakh MIDI (for now) · scraping copyrighted MIDI ·
shipping a hand-written attention implementation (`F.scaled_dot_product_attention` is used)

These were discussed and rejected. If one comes back, it comes back with a reason.

## 5. Repository layout

```
piano-music-transformer/
├── CLAUDE.md              # this file
├── README.md              # English, the shop window
├── pyproject.toml
├── configs/
│   └── data.yaml          # visible copy of the data defaults
├── src/pmt/
│   ├── config.py          # dataclasses + YAML loader
│   ├── data/
│   │   ├── download.py   # MAESTRO, MIDI only
│   │   ├── tokenizer.py  # REMI build / encode / decode / merge
│   │   ├── augment.py    # transposition, train split only
│   │   ├── prepare.py    # entry point: download -> BPE -> shards -> report
│   │   └── dataset.py    # windows + deterministic batching
│   ├── models/
│   │   └── lstm.py       # the baseline, ~8.4M params with tied weights
│   ├── train.py          # one loop, shared by every model
│   └── sample.py         # generation + degenerate-output diagnostics
└── tests/
```

## 6. Phases and status

- [x] **Phase 0 — Skeleton and data pipeline.** Packaging, MAESTRO download, REMI+BPE
      tokenization, `.npy` shards per split, **round-trip test**. *Done when
      `python -m pmt.data.prepare` runs and the suite is green.*
- [~] **Phase 1 — LSTM baseline.** Augmentation, model, training loop with exact
      resume, sampling. Code complete and tested; the 12,000-step run is in flight.
      *Done when it produces a MIDI file worth listening to.*
- [~] **Phase 2 — Transformer.** Decoder-only, RoPE, pre-norm RMSNorm, SwiGLU, SDPA,
      KV cache. Code complete and tested; waiting for the GPU. *Done when it beats
      the baseline on validation NLL.*
- [ ] **Phase 3 — Sampling.** top-p and repetition penalty, **prompt continuation**
      (give it 4 bars, it continues), context sliding past `max_seq_len`. The KV
      cache moved to Phase 2 - see the log. The demo lives or dies here.
- [ ] **Phase 4 — Evaluation.** Perplexity plus musical metrics, LSTM vs
      Transformer table, rendered audio. *Done when the README has numbers and sound.*
- [ ] **Phase 5 — Shop window.** Gradio demo, weights on the HF Hub, README with a
      Limitations section.
- [ ] **Phase 6 (optional) — Control.** Chord conditioning or infilling.

Phases 0-4 make a finished project. Phase 5 makes it a visible one.

## 7. Code conventions

- Python 3.13. If a dependency lacks 3.13 wheels, pin to 3.12 and note it here
- `src/` layout, package `pmt`
- `ruff format` + `ruff check`, line length 100
- Type hints required on public functions. No mypy/pyright
- Config: `dataclass` + YAML. Defaults in `config.py` are the source of truth
- Tests: `pytest`, fast and deterministic, no dataset dependency. Test shapes,
  round-trips and determinism — not training convergence
- Seed every entry point and print the seed used
- Generated files live under `outputs/` or `data/` and are never committed
- Commits: conventional commits (`feat:`, `fix:`, `docs:`)

## 8. Evaluation protocol

A comparison only means something under these conditions, so they hold every time:

- Same tokenizer, same split (MAESTRO's own official train/validation/test)
- Validation NLL / perplexity, per token
- Musical metrics: pitch-class entropy, scale consistency, groove consistency, note density
- 20 samples per model, **same seed and same sampling settings**
- Listening does not replace metrics and metrics do not replace listening — the README carries both

## 9. Known limits (these go in the README too)

- At this scale the model is **locally coherent, structurally weak**; it drifts after ~30 s
- Solo piano only, no stylistic range
- 12.6M training tokens: the corpus, not the parameter count, is the binding constraint.
  Augmentation matters more than model size here
- Stating these plainly makes the project more credible, not less

## 10. How to update this file

At the end of each phase: tick the status boxes, add new decisions to the table,
record traps hit along the way (especially MPS-specific ones). **Do not explain
code here** — record decisions and state. If the file grows, delete what went
stale rather than archiving it.

## 11. Session log

- **2026-09-06 — Phase 0 complete.**
  - Repo scaffolded: `uv`, `pyproject.toml`, MIT licence, `pmt` package, 16 tests green.
  - MAESTRO v3 MIDI (58 MB) downloaded; official split honoured.
  - **Time resolution decided by measurement**, not by taste. At 4/8/16/32 steps per
    beat, note retention and pitch match were 1.0000 everywhere — nothing is ever
    lost — while onset error halved with each doubling (0.063 / 0.031 / 0.016 /
    0.008 beats) for only ~12% more tokens across the whole range. Chose **16
    steps/beat** (~8 ms at 120 bpm, below the audible-displacement threshold);
    32 buys inaudible precision at the cost of sparser position statistics.
  - **Trap found by a test:** a note-less MIDI still encodes to ~4 bar/tempo/time-signature
    tokens, so `prepare` filters by `MIN_TOKENS_PER_FILE`, not by emptiness.
  - **Round-trip metric rewritten.** The first version compared notes index by index
    and reported 0.585 pitch accuracy — an artefact, since quantisation reorders
    simultaneous notes. Comparisons are now multiset-based and order-insensitive.
  - **Corpus is smaller than the model budget assumed.** 12.6M training tokens against
    a planned 20-30M parameters is heavily data-bound. Two consequences for Phase 2:
    start nearer **10-15M parameters** and treat **data augmentation** (pitch
    transposition +/-6 semitones, tempo and velocity jitter) as a first-class part of
    the pipeline rather than a nice-to-have — it multiplies effective data far more
    cheaply than parameters buy quality here.
  - A 1024-token context covers **~18 bars** (~58 tokens/bar after BPE), so the context
    length is musically meaningful rather than a fragment.
  - Next: Phase 1, LSTM baseline.

- **2026-09-06 — Phase 1 code complete, training in flight.**
  - **MPS is worth it for the LSTM:** 0.37 s/step vs 2.13 s on CPU at 8x1024 — 5.7x.
    Real end-to-end throughput with `grad_accum=4` is ~20k tokens/s (1.64 s/step).
  - **bf16 buys only ~7% for the LSTM** (0.350 vs 0.374 s/step); the recurrent path is
    not matmul-bound. Baseline stays fp32. Re-measure for the Transformer, where the
    gain should be much larger — that is the one place this decision may flip.
  - **Augmentation is 11.1x, not 13x.** 1,869 of 12,506 transposed variants push notes
    off the 88-key range and are dropped whole rather than clipped (clipping would
    silently rewrite the music). Corpus: 12.6M -> **131.6M training tokens**, which
    turns the data-bound problem from Phase 0 into a healthy ~15 tokens/parameter.
  - **`DataLoader` dropped for step-indexed sampling.** Each batch is a pure function of
    `(seed, step)`, so a run stopped and resumed lands bit-for-bit where an
    uninterrupted run would. There is a test for exactly this.
  - **Trap:** `bpe_vocab_size` below the REMI base vocabulary makes MidiTok skip BPE
    with only a warning. Raising the time resolution in Phase 0 grew the base vocab
    519 and silently broke smoke runs. Now it raises.
  - **Trap:** `tokenizer[id]` raises `KeyError` for BPE-merged ids — a merge covers
    several events and has no single event name.
  - **An undertrained model emits almost only `Pitch` tokens** and never completes a
    Pitch/Velocity/Duration triplet, so nothing decodes to a note. The sampler now
    prints the event mix when a sample yields zero notes, which separates
    "undertrained" from "broken pipeline" at a glance.
  - Next: results from the 12,000-step run, then Phase 2.

- **2026-09-06 — Phase 2 code complete, written while the baseline trains.**
  - **The KV cache moved up from Phase 3, by force.** Both models share one
    `generate()`, which advances one token at a time carrying a `state`. For an RNN
    that state is free; for a Transformer it is the KV cache. Without it the
    Transformer simply cannot sample, so it is not optional and not Phase 3 work.
    A test asserts cached decoding matches a full forward pass to 1e-4.
  - **`is_causal` is wrong once a cache exists.** It aligns the mask top-left, but a
    cached query block sits at the *end* of the context. The attention shifts the
    triangle by the cached length instead when the two lengths differ.
  - **Bug with real reach: `vocab_size` was never tied to the data.** Model configs
    defaulted to 4096 while the smoke dataset's tokenizer had 1024, so the model
    emitted ids that did not exist and MidiTok died with `KeyError: None` deep in
    BPE decoding. The LSTM had the same defect and hid it by collapsing onto frequent
    tokens. `vocab_size` is now read from `meta.json` at training time and validated
    at sampling time.
  - **A test that measured nothing.** "Position changes the prediction", fed a
    sequence of identical tokens, could never fail: identical tokens give identical
    value vectors, and any weighted average of identical vectors is that vector -
    with or without RoPE. Replaced with RoPE's actual defining property, that the
    query-key score depends only on the distance between positions.
  - **Do not run anything on the GPU during a long training run.** A CPU smoke test
    plus one sampling call dropped throughput from 21k to 13k tokens/s.
  - Next: train the Transformer on the same 12,000-step budget once the GPU frees up,
    and measure bf16 for it - the one decision from Phase 1 likely to flip.
