"""Compare models on held-out data and on what they actually produce.

    uv run python -m pmt.evaluate \
        --checkpoint outputs/lstm/best.pt --checkpoint outputs/transformer/best.pt

Two kinds of number, because neither is sufficient alone:

- **Test perplexity** measures prediction on held-out performances. It is the
  honest headline for a language model, and it says nothing about whether the
  output is listenable.
- **Musical metrics** describe the notes actually generated. They are reported
  next to the same metrics computed on real MAESTRO excerpts of the same length,
  because a number like "scale consistency 0.86" means nothing until you know the
  corpus scores 0.88.

Every model is generated from with the same seed and the same sampling settings.
Changing those between models would make the table meaningless.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.data.dataset import TokenWindowDataset, evaluation_batches
from pmt.data.tokenizer import TOKENIZER_FILENAME, decode_to_score, load_tokenizer
from pmt.metrics import METRIC_NAMES, aggregate, summarise
from pmt.sample import SamplingSettings, generate, load_model
from pmt.train import evaluate as evaluate_loss
from pmt.train import resolve_device, seed_everything


def reference_summaries(
    data_dir: Path, tokenizer, count: int, length: int, banned: list[int]
) -> list[dict]:
    """Metrics for real excerpts, so the generated numbers have a scale.

    The excerpts come from the same token stream at the same length as the
    samples, so the comparison is not confounded by duration or tokenizer.
    """
    windows = TokenWindowDataset(data_dir / "validation.npy", length)
    stride = max(1, len(windows) // count)
    summaries = []
    for index in range(count):
        tokens = windows[(index * stride) % len(windows)][0]
        ids = [int(token) for token in tokens if int(token) not in banned]
        summaries.append(summarise(decode_to_score(tokenizer, ids)))
    return summaries


def sample_summaries(
    model,
    tokenizer,
    args,
    settings: SamplingSettings,
    bos: int,
    eos: int,
    banned: list[int],
    device: torch.device,
) -> list[dict]:
    summaries = []
    for _ in range(args.samples):
        produced = generate(model, [bos], args.tokens, settings, banned, eos, device)
        ids = [token for token in produced if token not in banned and token != eos]
        summaries.append(summarise(decode_to_score(tokenizer, ids)))
    return summaries


def format_table(columns: dict[str, dict]) -> str:
    names = list(columns)
    width = max(len(name) for name in [*METRIC_NAMES, "test perplexity"]) + 2
    header = f"{'metric':<{width}}" + "".join(f"{name:>16}" for name in names)
    lines = [header, "-" * len(header)]

    for metric in METRIC_NAMES:
        row = f"{metric:<{width}}"
        for name in names:
            value = columns[name].get(metric)
            row += f"{value:>16.3f}" if value is not None else f"{'-':>16}"
        lines.append(row)

    row = f"{'test perplexity':<{width}}"
    for name in names:
        value = columns[name].get("test_perplexity")
        row += f"{value:>16.2f}" if value is not None else f"{'-':>16}"
    lines.append(row)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate and compare checkpoints")
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "evaluation.json")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)

    device = resolve_device("auto")
    tokenizer = load_tokenizer(args.data / TOKENIZER_FILENAME)
    meta = json.loads((args.data / "meta.json").read_text())
    special = meta["special_tokens"]
    bos, eos = special["bos"], special["eos"]
    banned = sorted({special["pad"], bos, 3})

    settings = SamplingSettings(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(
        f"data     : {args.data} (vocab {meta['vocab_size']})\n"
        f"device   : {device} | seed {args.seed}\n"
        f"sampling : temperature {settings.temperature}, top-k {settings.top_k}, "
        f"top-p {settings.top_p}\n"
        f"samples  : {args.samples} x {args.tokens} tokens per model\n"
    )

    print("measuring real excerpts for reference...")
    columns: dict[str, dict] = {
        "reference": aggregate(
            reference_summaries(args.data, tokenizer, args.samples, args.tokens, banned)
        )
    }

    for checkpoint_path in args.checkpoint:
        seed_everything(args.seed)  # identical conditions for every model
        model, checkpoint = load_model(checkpoint_path, device)
        name = checkpoint["train_config"]["model"]
        block_size = checkpoint["train_config"]["block_size"]
        precision = checkpoint["train_config"]["precision"]
        print(f"evaluating {name} ({checkpoint_path})...")

        test_data = TokenWindowDataset(args.data / "test.npy", block_size)
        batches = evaluation_batches(
            test_data, checkpoint["train_config"]["batch_size"], args.eval_batches
        )
        loss = evaluate_loss(model, batches, device, precision)

        column = aggregate(
            sample_summaries(model, tokenizer, args, settings, bos, eos, banned, device)
        )
        column["test_loss"] = loss
        column["test_perplexity"] = math.exp(min(loss, 20))
        column["parameters"] = model.num_parameters()
        column["step"] = checkpoint["step"]
        columns[name] = column

    print("\n" + format_table(columns))
    print(
        "\nReference is real MAESTRO, not a target to hit exactly - it is the scale that\n"
        "makes the other columns readable. Perplexity is the headline; the rest describe\n"
        "the notes. Neither replaces listening."
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "settings": {
                    "samples": args.samples,
                    "tokens": args.tokens,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                },
                "columns": columns,
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
