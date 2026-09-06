"""Training loop shared by every model in this project.

    uv run python -m pmt.train --config configs/lstm.yaml
    uv run python -m pmt.train --config configs/lstm.yaml --resume outputs/lstm/last.pt
    uv run python -m pmt.train --smoke

The Transformer in Phase 2 will use this same loop unchanged. That is the point:
if the two models share the optimiser, the schedule, the data order and the
evaluation, then whatever separates them is the architecture.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR, load_config
from pmt.data.dataset import TokenWindowDataset, deterministic_batch, evaluation_batches
from pmt.models.lstm import LSTMConfig, build_lstm

MODELS = {"lstm": (LSTMConfig, build_lstm)}


@dataclass(slots=True)
class TrainConfig:
    model: str = "lstm"

    # Optimisation. Effective batch = batch_size * grad_accum.
    batch_size: int = 8
    grad_accum: int = 4
    max_steps: int = 6000
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    block_size: int = 1024

    log_every: int = 25
    eval_every: int = 500
    eval_batches: int = 40
    save_every: int = 500

    seed: int = 1337
    device: str = "auto"
    # bf16 buys ~7% for the LSTM on MPS - the recurrent path is not matmul-bound.
    # The Transformer should gain considerably more, hence the switch.
    precision: str = "fp32"


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup into cosine decay down to ``min_lr_ratio * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = max(1, cfg.max_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / span)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine)


def autocast_for(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=getattr(torch, precision))


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1)
    )


@torch.no_grad()
def evaluate(model: nn.Module, batches, device: torch.device, precision: str) -> float:
    """Mean loss per token over a fixed set of windows, in nats."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)
        with autocast_for(device, precision):
            logits, _ = model(inputs)
        total_loss += cross_entropy(logits, targets).item() * targets.numel()
        total_tokens += targets.numel()
    model.train()
    return total_loss / max(1, total_tokens)


def capture_rng() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def save_checkpoint(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train(
    cfg: TrainConfig,
    model_cfg,
    data_dir: Path,
    out_dir: Path,
    resume: Path | None = None,
) -> dict:
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = TokenWindowDataset(data_dir / "train.npy", cfg.block_size)
    val_data = TokenWindowDataset(data_dir / "validation.npy", cfg.block_size)
    val_batches = evaluation_batches(val_data, cfg.batch_size, cfg.eval_batches)

    _, builder = MODELS[cfg.model]
    model = builder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )

    start_step = 0
    best_val = float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_val = checkpoint.get("best_val", float("inf"))
        restore_rng(checkpoint["rng"])
        print(f"resumed from {resume} at step {start_step}")

    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size
    print(
        f"model      : {cfg.model} ({model.num_parameters() / 1e6:.2f}M params)\n"
        f"device     : {device} ({cfg.precision})\n"
        f"seed       : {cfg.seed}\n"
        f"train      : {len(train_data):,} windows | val: {len(val_data):,} windows\n"
        f"batch      : {cfg.batch_size} x {cfg.grad_accum} accum x {cfg.block_size} "
        f"= {tokens_per_step:,} tokens/step\n"
        f"steps      : {start_step} -> {cfg.max_steps}\n"
    )

    metrics_path = out_dir / "metrics.jsonl"
    (out_dir / "config.json").write_text(
        json.dumps({"train": asdict(cfg), "model": asdict(model_cfg)}, indent=2)
    )

    def log(record: dict) -> None:
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

    model.train()
    running_loss = 0.0
    window_start = time.perf_counter()

    for step in range(start_step, cfg.max_steps):
        lr = learning_rate_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for micro in range(cfg.grad_accum):
            inputs, targets = deterministic_batch(
                train_data, step * cfg.grad_accum + micro, cfg.batch_size, cfg.seed
            )
            inputs, targets = inputs.to(device), targets.to(device)
            with autocast_for(device, cfg.precision):
                logits, _ = model(inputs)
            loss = cross_entropy(logits, targets) / cfg.grad_accum
            loss.backward()
            step_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        running_loss += step_loss

        if (step + 1) % cfg.log_every == 0:
            elapsed = time.perf_counter() - window_start
            mean_loss = running_loss / cfg.log_every
            throughput = cfg.log_every * tokens_per_step / elapsed
            print(
                f"step {step + 1:>6}/{cfg.max_steps} | loss {mean_loss:.4f} "
                f"| ppl {math.exp(min(mean_loss, 20)):>8.1f} | lr {lr:.2e} "
                f"| gnorm {grad_norm:.2f} | {throughput:,.0f} tok/s",
                flush=True,
            )
            log(
                {
                    "step": step + 1,
                    "train_loss": mean_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": throughput,
                }
            )
            running_loss = 0.0
            window_start = time.perf_counter()

        is_last = step + 1 == cfg.max_steps
        if (step + 1) % cfg.eval_every == 0 or is_last:
            val_loss = evaluate(model, val_batches, device, cfg.precision)
            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                marker = "  <- best"
                save_checkpoint(
                    out_dir / "best.pt",
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step + 1,
                    best_val=best_val,
                    rng=capture_rng(),
                    train_config=asdict(cfg),
                    model_config=asdict(model_cfg),
                )
            print(
                f"  eval @ {step + 1}: val loss {val_loss:.4f} "
                f"| ppl {math.exp(min(val_loss, 20)):.1f}{marker}",
                flush=True,
            )
            log({"step": step + 1, "val_loss": val_loss})
            window_start = time.perf_counter()

        if (step + 1) % cfg.save_every == 0 or is_last:
            save_checkpoint(
                out_dir / "last.pt",
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                step=step + 1,
                best_val=best_val,
                rng=capture_rng(),
                train_config=asdict(cfg),
                model_config=asdict(model_cfg),
            )

    print(f"\ndone. best val loss {best_val:.4f} (ppl {math.exp(min(best_val, 20)):.1f})")
    print(f"checkpoints in {out_dir}")
    return {"best_val_loss": best_val}


def load_configs(path: Path | None, **overrides) -> tuple[TrainConfig, object]:
    raw = yaml.safe_load(path.read_text()) if path else {}
    raw = raw or {}
    train_cfg = load_config(TrainConfig, None, **{**raw.get("train", {}), **overrides})
    model_cls, _ = MODELS[train_cfg.model]
    return train_cfg, load_config(model_cls, None, **raw.get("model", {}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a music model")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="a few steps on the smoke dataset")
    args = parser.parse_args(argv)

    overrides = {"max_steps": args.max_steps, "device": args.device}
    overrides = {key: value for key, value in overrides.items() if value is not None}

    data_dir = args.data
    if args.smoke:
        overrides |= {
            "max_steps": overrides.get("max_steps", 20),
            "batch_size": 2,
            "grad_accum": 1,
            "block_size": 128,
            "log_every": 5,
            "eval_every": 10,
            "eval_batches": 2,
            "save_every": 10,
            "warmup_steps": 5,
        }
        if data_dir == PROCESSED_DIR:
            data_dir = data_dir.parent / "processed_smoke"

    cfg, model_cfg = load_configs(args.config, **overrides)
    out_dir = args.out or OUTPUTS_DIR / (f"{cfg.model}-smoke" if args.smoke else cfg.model)
    train(cfg, model_cfg, data_dir, out_dir, args.resume)


if __name__ == "__main__":
    main()
