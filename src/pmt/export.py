"""Package trained checkpoints for publication.

    uv run --extra hub python -m pmt.export

A training checkpoint carries optimizer and RNG state - roughly 55 MB of its
96 MB - which nobody downloading the model needs. This writes an inference-only
artifact per model: weights as safetensors, the config beside them, and the
tokenizer without which the weights are unusable.

safetensors rather than a pickled ``.pt``: it cannot execute code on load, and a
published model is exactly the case where that matters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_model

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.data.tokenizer import TOKENIZER_FILENAME
from pmt.train import MODELS

MODEL_NAMES = ("lstm", "transformer")


def export_model(checkpoint_path: Path, destination: Path) -> dict:
    """Write ``model.safetensors`` and ``config.json`` for one checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    name = checkpoint["train_config"]["model"]
    config_cls, builder = MODELS[name]
    model = builder(config_cls(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    destination.mkdir(parents=True, exist_ok=True)
    # save_model, not save_file: tied embedding and output weights are one tensor
    # stored twice in the state dict, and safetensors refuses duplicate storage.
    save_model(model, str(destination / "model.safetensors"))

    summary = {
        "model": name,
        "parameters": model.num_parameters(),
        "step": checkpoint["step"],
        "validation_loss": checkpoint["best_val"],
        "model_config": checkpoint["model_config"],
        "train_config": checkpoint["train_config"],
    }
    (destination / "config.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package checkpoints for the Hub")
    parser.add_argument("--runs", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "hub")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for name in MODEL_NAMES:
        checkpoint = args.runs / name / "best.pt"
        if not checkpoint.exists():
            print(f"  {name}: no checkpoint at {checkpoint}, skipping")
            continue
        summary = export_model(checkpoint, args.out / name)
        size = (args.out / name / "model.safetensors").stat().st_size / 1e6
        print(
            f"  {name}: {summary['parameters'] / 1e6:.2f}M params, "
            f"val loss {summary['validation_loss']:.4f} -> {size:.1f} MB"
        )

    # The weights are unusable without the tokenizer that produced their vocabulary.
    for filename in (TOKENIZER_FILENAME, "meta.json"):
        (args.out / filename).write_bytes((args.data / filename).read_bytes())
    print(f"  tokenizer and metadata copied\n\nwrote {args.out}")


if __name__ == "__main__":
    main()
