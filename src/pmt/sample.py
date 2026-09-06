"""Generate music from a trained checkpoint.

    uv run python -m pmt.sample --checkpoint outputs/lstm/best.pt --num 5

Sampling here is deliberately basic: temperature and top-k. Nucleus sampling,
repetition penalty and a proper KV cache belong to Phase 3, where they can be
compared against each other rather than chosen by taste.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor, nn

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.data.dataset import TokenWindowDataset
from pmt.data.tokenizer import TOKENIZER_FILENAME, decode_to_score, load_tokenizer
from pmt.train import MODELS, resolve_device, seed_everything


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict]:
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


def pick_next(logits: Tensor, temperature: float, top_k: int, banned: list[int]) -> int:
    """One token from the model's distribution over the next event."""
    scores = logits.float().clone()
    scores[banned] = float("-inf")

    if temperature <= 0:
        return int(scores.argmax())

    scores = scores / temperature
    if top_k > 0:
        cutoff = torch.topk(scores, min(top_k, scores.size(-1))).values[-1]
        scores[scores < cutoff] = float("-inf")
    return int(torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1))


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    banned: list[int],
    eos_id: int,
    device: torch.device,
) -> list[int]:
    """Continue ``prompt`` one token at a time, carrying the recurrent state.

    The state is what makes this cheap: each new token costs one LSTM step rather
    than a re-read of the whole prefix.
    """
    logits, state = model(torch.tensor([prompt], device=device))
    next_logits = logits[0, -1]

    produced: list[int] = []
    for _ in range(max_new_tokens):
        token = pick_next(next_logits, temperature, top_k, banned)
        if token == eos_id:
            break
        produced.append(token)
        logits, state = model(torch.tensor([[token]], device=device), state)
        next_logits = logits[0, -1]
    return produced


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate MIDI from a checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "samples")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=1024, help="tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="prime with this many tokens from the validation split (0 = start from BOS)",
    )
    args = parser.parse_args(argv)

    seed_everything(args.seed)
    device = resolve_device("auto")
    model, checkpoint = load_model(args.checkpoint, device)
    tokenizer = load_tokenizer(args.data / TOKENIZER_FILENAME)

    special = json.loads((args.data / "meta.json").read_text())["special_tokens"]
    bos, eos = special["bos"], special["eos"]
    banned = sorted({special["pad"], bos, 3})  # PAD, BOS and MASK are never musical output

    prompts: list[list[int]] = []
    if args.prompt_tokens > 0:
        windows = TokenWindowDataset(args.data / "validation.npy", args.prompt_tokens)
        stride = max(1, len(windows) // args.num)
        for index in range(args.num):
            inputs, _ = windows[(index * stride) % len(windows)]
            prompts.append([int(token) for token in inputs])
    else:
        prompts = [[bos] for _ in range(args.num)]

    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"checkpoint : {args.checkpoint} (step {checkpoint['step']}, "
        f"val loss {checkpoint.get('best_val', float('nan')):.4f})\n"
        f"device     : {device} | seed {args.seed}\n"
        f"sampling   : temperature {args.temperature}, top-k {args.top_k}, "
        f"{args.tokens} tokens\n"
    )

    for index, prompt in enumerate(prompts):
        produced = generate(
            model, prompt, args.tokens, args.temperature, args.top_k, banned, eos, device
        )
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
