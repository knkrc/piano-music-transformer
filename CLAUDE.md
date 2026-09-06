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
│   └── data/
│       ├── download.py    # MAESTRO, MIDI only
│       ├── tokenizer.py   # REMI build / encode / decode / merge
│       ├── prepare.py     # entry point: download -> BPE -> shards -> report
│       └── dataset.py     # strided windows over the token stream
└── tests/
```

## 6. Phases and status

- [x] **Phase 0 — Skeleton and data pipeline.** Packaging, MAESTRO download, REMI+BPE
      tokenization, `.npy` shards per split, **round-trip test**. *Done when
      `python -m pmt.data.prepare` runs and the suite is green.*
- [ ] **Phase 1 — LSTM baseline.** Training loop, checkpoint/resume, generation.
      *Done when it produces a MIDI file worth listening to.*
- [ ] **Phase 2 — Transformer.** Decoder-only, RoPE, pre-norm, SDPA, cosine LR,
      gradient accumulation. Same pipeline, same CLI. *Done when it beats the
      baseline on validation NLL.*
- [ ] **Phase 3 — Sampling.** temperature / top-k / top-p / repetition penalty,
      KV cache, **prompt continuation** (give it 4 bars, it continues). The demo
      lives or dies here.
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
