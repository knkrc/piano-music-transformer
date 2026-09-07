"""Generate music from a trained checkpoint.

    uv run python -m pmt.sample --checkpoint outputs/lstm/best.pt --num 5
    uv run python -m pmt.sample --checkpoint outputs/lstm/best.pt \
        --prompt-midi some.mid --prompt-bars 4

Both models are driven through the same interface: ``forward`` returns a state,
and generation advances one token at a time carrying it. For the LSTM that state
is recurrent; for the Transformer it is a key/value cache.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from symusic import Score

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.data.dataset import TokenWindowDataset
from pmt.data.tokenizer import (
    TOKENIZER_FILENAME,
    decode_to_score,
    encode_score,
    load_tokenizer,
    merge_to_single_track,
)
from pmt.models.base import LanguageModel
from pmt.train import MODELS, resolve_device, seed_everything


@dataclass(slots=True)
class SamplingSettings:
    """How a distribution over next events becomes one event.

    ``repetition_penalty`` defaults to 1.0, which is off, and that default is a
    judgement about music rather than an oversight. Music *is* repetition -
    motifs, sequences, ostinati, an accompaniment figure held for sixteen bars.
    The penalty that stops a language model looping suppresses exactly the
    structure this model is meant to learn. It is here for rescuing a degenerate
    model that has collapsed onto one note, not for routine use.
    """

    temperature: float = 1.0
    top_k: int = 32
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    repetition_window: int = 64


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[LanguageModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    name = checkpoint["train_config"]["model"]
    config_cls, builder = MODELS[name]
    model = builder(config_cls(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def event_kind(tokenizer, token: int) -> str:
    """Coarse label for a token id, for diagnostics only.

    Ids above the REMI base vocabulary are BPE merges covering several events at
    once, so they have no single event type and are reported as ``BPE``.
    """
    try:
        return str(tokenizer[token]).split("_")[0]
    except (KeyError, IndexError):
        return "BPE"


def _apply_repetition_penalty(
    scores: torch.Tensor, recent: Sequence[int], settings: SamplingSettings
) -> None:
    if settings.repetition_penalty == 1.0 or not recent:
        return
    window = recent[-settings.repetition_window :]
    seen = torch.tensor(sorted(set(window)), device=scores.device, dtype=torch.long)
    values = scores[seen]
    # Divide positive scores and multiply negative ones, so the penalty always
    # pushes towards zero rather than flipping a token's sign.
    scores[seen] = torch.where(
        values > 0, values / settings.repetition_penalty, values * settings.repetition_penalty
    )


def _keep_nucleus(scores: torch.Tensor, top_p: float) -> torch.Tensor:
    """Mask out the tail beyond cumulative probability ``top_p``."""
    ordered, order = torch.sort(scores, descending=True)
    probabilities = torch.softmax(ordered, dim=-1)
    # Subtracting each probability keeps the first token even when it alone
    # already exceeds top_p, which otherwise leaves nothing to sample from.
    keep = (probabilities.cumsum(dim=-1) - probabilities) < top_p
    ordered = ordered.masked_fill(~keep, float("-inf"))
    return torch.full_like(scores, float("-inf")).scatter(0, order, ordered)


def pick_next(
    logits: torch.Tensor,
    settings: SamplingSettings,
    banned: Sequence[int],
    recent: Sequence[int] = (),
) -> int:
    """One token from the model's distribution over the next event."""
    scores = logits.float().clone()
    if len(banned):
        scores[list(banned)] = float("-inf")

    _apply_repetition_penalty(scores, recent, settings)

    if settings.temperature <= 0:
        return int(scores.argmax())
    scores = scores / settings.temperature

    if settings.top_k > 0:
        cutoff = torch.topk(scores, min(settings.top_k, scores.size(-1))).values[-1]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))

    if settings.top_p < 1.0:
        scores = _keep_nucleus(scores, settings.top_p)

    return int(torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1))


@torch.no_grad()
def generate(
    model: LanguageModel,
    prompt: list[int],
    max_new_tokens: int,
    settings: SamplingSettings,
    banned: Sequence[int],
    eos_id: int,
    device: torch.device,
) -> list[int]:
    """Continue ``prompt``, one token at a time, carrying the model's state.

    When the model has a bounded attention window the oldest cache entries are
    dropped as generation runs past it. This works precisely because of RoPE:
    scores depend on the *distance* between positions, so a sliding window keeps
    every distance inside the range the model was trained on, however far the
    absolute positions travel.
    """
    was_training = model.training
    model.eval()  # dropout during sampling is never wanted, whatever the caller left on
    try:
        return _generate(model, prompt, max_new_tokens, settings, banned, eos_id, device)
    finally:
        model.train(was_training)


def _generate(
    model: LanguageModel,
    prompt: list[int],
    max_new_tokens: int,
    settings: SamplingSettings,
    banned: Sequence[int],
    eos_id: int,
    device: torch.device,
) -> list[int]:
    window = model.max_context
    if window is not None and len(prompt) > window:
        prompt = prompt[-window:]

    logits, state = model(torch.tensor([prompt], device=device), use_cache=True)
    next_logits = logits[0, -1]

    history = list(prompt)
    produced: list[int] = []
    for _ in range(max_new_tokens):
        token = pick_next(next_logits, settings, banned, history)
        if token == eos_id:
            break
        produced.append(token)
        history.append(token)

        if window is not None:
            state = model.trim_state(state, window - 1)
        logits, state = model(torch.tensor([[token]], device=device), state, use_cache=True)
        next_logits = logits[0, -1]
    return produced


def prompt_from_midi(tokenizer, path: Path, bars: int) -> list[int]:
    """Tokenize the first ``bars`` bars of a MIDI file.

    The score is clipped at a downbeat rather than after a token count, so the
    model is handed whole bars and picks up on a barline like a musician would.
    """
    score = merge_to_single_track(Score.from_file(str(path)))
    downbeats = score.get_downbeats()
    if bars > 0 and len(downbeats) > bars:
        score = score.clip(0, int(downbeats[bars]))
    return encode_score(tokenizer, score)


def build_prompts(args, tokenizer, bos: int) -> list[list[int]]:
    if args.prompt_midi is not None:
        return [prompt_from_midi(tokenizer, args.prompt_midi, args.prompt_bars)] * args.num

    if args.prompt_tokens > 0:
        windows = TokenWindowDataset(args.data / "validation.npy", args.prompt_tokens)
        stride = max(1, len(windows) // args.num)
        return [
            [int(token) for token in windows[(index * stride) % len(windows)][0]]
            for index in range(args.num)
        ]

    return [[bos] for _ in range(args.num)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "samples")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1024, help="tokens to generate")
    parser.add_argument("--seed", type=int, default=1337)

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--temperature", type=float, default=1.0)
    sampling.add_argument("--top-k", type=int, default=32)
    sampling.add_argument("--top-p", type=float, default=1.0)
    sampling.add_argument("--repetition-penalty", type=float, default=1.0)
    sampling.add_argument("--repetition-window", type=int, default=64)

    priming = parser.add_argument_group("priming")
    priming.add_argument("--prompt-midi", type=Path, default=None, help="continue this file")
    priming.add_argument("--prompt-bars", type=int, default=4, help="bars to take from it")
    priming.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="instead, prime with this many tokens from the validation split",
    )
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = resolve_device("auto")
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data / TOKENIZER_FILENAME)

    trained_vocab = checkpoint["model_config"]["vocab_size"]
    if trained_vocab != len(tokenizer):
        raise ValueError(
            f"checkpoint was trained on a {trained_vocab}-token vocabulary but "
            f"{args.data.name} has {len(tokenizer)}. Generated ids would not decode - "
            f"point --data at the dataset this checkpoint was trained on."
        )

    special = json.loads((args.data / "meta.json").read_text())["special_tokens"]
    bos, eos = special["bos"], special["eos"]
    banned = sorted({special["pad"], bos, 3})  # PAD, BOS and MASK are never musical output

    settings = SamplingSettings(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
    )
    prompts = build_prompts(args, tokenizer, bos)

    args.out.mkdir(parents=True, exist_ok=True)
    priming_note = (
        f"{args.prompt_midi.name} ({args.prompt_bars} bars)"
        if args.prompt_midi
        else (f"{args.prompt_tokens} validation tokens" if args.prompt_tokens else "none")
    )
    print(
        f"checkpoint : {args.checkpoint} (step {checkpoint['step']}, "
        f"val loss {checkpoint.get('best_val', float('nan')):.4f})\n"
        f"device     : {device} | seed {args.seed}\n"
        f"sampling   : temperature {settings.temperature}, top-k {settings.top_k}, "
        f"top-p {settings.top_p}, repetition {settings.repetition_penalty}\n"
        f"priming    : {priming_note}\n"
        f"generating : {args.tokens} tokens x {args.num}\n"
    )

    for index, prompt in enumerate(prompts):
        produced = generate(model, prompt, args.tokens, settings, banned, eos, device)
        ids = [token for token in (*prompt, *produced) if token not in banned and token != eos]
        score = decode_to_score(tokenizer, ids)
        notes = score.tracks[0].notes if score.tracks else []

        path = args.out / f"sample_{index + 1:02d}.mid"
        score.dump_midi(str(path))

        if notes:
            span = f"{min(n.pitch for n in notes)}-{max(n.pitch for n in notes)}"
            print(
                f"  {path.name}: {len(produced)} tokens -> {len(notes)} notes, "
                f"{score.end() / score.tpq / 4:.1f} bars, pitches {span}"
            )
        else:
            # A model that has only learned the marginal distribution emits the most
            # common event class and never completes a Pitch/Velocity/Duration triplet,
            # so nothing decodes to a note. Showing the mix separates "undertrained"
            # from "broken pipeline" at a glance.
            mix = Counter(event_kind(tokenizer, token) for token in produced)
            print(
                f"  {path.name}: {len(produced)} tokens -> NO NOTES. "
                f"event mix {dict(mix.most_common(4))} - model is undertrained"
            )

    print(f"\nwrote {args.num} file(s) to {args.out}")


if __name__ == "__main__":
    main()
