"""Gradio demo: the same prompt, the same seed, both models, side by side.

    uv run --extra demo python app.py

The interface lives in the package so it can be imported and tested; ``app.py``
at the repository root is the three-line entry point Hugging Face Spaces expects.

The layout is the argument. A single-model generator would be a toy; putting the
LSTM and the Transformer next to each other under identical conditions is the
thing this project is actually about, and it lets anyone hear the difference
rather than read a table about it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import gradio as gr
import torch

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.data.dataset import TokenWindowDataset
from pmt.data.tokenizer import TOKENIZER_FILENAME, decode_to_score, load_tokenizer
from pmt.metrics import aggregate, summarise
from pmt.render import find_soundfont, render_midi
from pmt.sample import SamplingSettings, generate, load_model, prompt_from_midi
from pmt.train import resolve_device, seed_everything

MODEL_LABELS = {"lstm": "LSTM baseline", "transformer": "Transformer"}
REFERENCE_SAMPLES = 20
FROM_SCRATCH = "From scratch"
FROM_UPLOAD = "Continue an uploaded MIDI"
FROM_CORPUS = "Continue a MAESTRO excerpt"


def discover_checkpoints(root: Path = OUTPUTS_DIR) -> dict[str, Path]:
    """Find the best checkpoint of each trained model, ordered oldest phase first."""
    found = {}
    for name in ("lstm", "transformer"):
        candidate = root / name / "best.pt"
        if candidate.exists():
            found[name] = candidate
    return found


class Demo:
    """Holds the models and the reference metrics, loaded once and reused."""

    def __init__(self, data_dir: Path = PROCESSED_DIR, root: Path = OUTPUTS_DIR) -> None:
        self.data_dir = data_dir
        self.device = resolve_device("auto")
        self.checkpoints = discover_checkpoints(root)
        self.models: dict[str, object] = {}
        self.tokenizer = load_tokenizer(data_dir / TOKENIZER_FILENAME)

        special = json.loads((data_dir / "meta.json").read_text())["special_tokens"]
        self.bos, self.eos = special["bos"], special["eos"]
        self.banned = sorted({special["pad"], self.bos, 3})
        self._reference: dict[str, float | None] | None = None

    def model(self, name: str):
        if name not in self.models:
            self.models[name] = load_model(self.checkpoints[name], self.device)[0]
        return self.models[name]

    def reference(self, length: int) -> dict[str, float | None]:
        """Metrics for real MAESTRO, so the generated numbers have a scale."""
        if self._reference is None:
            windows = TokenWindowDataset(self.data_dir / "validation.npy", length)
            stride = max(1, len(windows) // REFERENCE_SAMPLES)
            self._reference = aggregate(
                [
                    summarise(
                        decode_to_score(
                            self.tokenizer,
                            [
                                int(token)
                                for token in windows[(index * stride) % len(windows)][0]
                                if int(token) not in self.banned
                            ],
                        )
                    )
                    for index in range(REFERENCE_SAMPLES)
                ]
            )
        return self._reference

    def build_prompt(self, mode: str, upload, bars: int, length: int) -> list[int]:
        if mode == FROM_UPLOAD and upload is not None:
            return prompt_from_midi(self.tokenizer, Path(upload), bars)
        if mode == FROM_CORPUS:
            windows = TokenWindowDataset(self.data_dir / "validation.npy", length)
            index = int(torch.randint(len(windows), (1,)))
            return [int(token) for token in windows[index][0]]
        return [self.bos]

    def compose(
        self,
        name: str,
        mode: str,
        upload,
        bars: int,
        tokens: int,
        settings: SamplingSettings,
        seed: int,
        workspace: Path,
    ) -> tuple[str | None, str, str]:
        """Generate one piece and return ``(audio, midi, metrics markdown)``.

        Reseeding here rather than once per run is what makes the columns
        comparable: every model sees the same prompt and draws from the same
        random stream.
        """
        seed_everything(seed)
        prompt_length = min(tokens, 512)
        prompt = self.build_prompt(mode, upload, bars, prompt_length)
        produced = generate(
            self.model(name), prompt, tokens, settings, self.banned, self.eos, self.device
        )
        ids = [
            token
            for token in (*prompt, *produced)
            if token not in self.banned and token != self.eos
        ]
        score = decode_to_score(self.tokenizer, ids)

        midi_path = workspace / f"{name}.mid"
        score.dump_midi(str(midi_path))

        audio_path = None
        try:
            audio_path = str(render_midi(midi_path, workspace / f"{name}.mp3", find_soundfont()))
        except (FileNotFoundError, RuntimeError) as error:
            gr.Warning(f"Audio unavailable ({error}); the MIDI is still downloadable.")

        return (
            audio_path,
            str(midi_path),
            metrics_markdown(summarise(score), self.reference(prompt_length)),
        )


def metrics_markdown(summary: dict, reference: dict) -> str:
    rows = [
        "| metric | this sample | real MAESTRO |",
        "| --- | --- | --- |",
    ]
    for name, value in summary.items():
        expected = reference.get(name)
        rows.append(
            f"| {name.replace('_', ' ')} | {value:.2f} | {expected:.2f} |"
            if value is not None and expected is not None
            else f"| {name.replace('_', ' ')} | - | - |"
        )
    return "\n".join(rows)


def build_interface(demo: Demo) -> gr.Blocks:
    available = list(demo.checkpoints)

    def run(mode, upload, bars, tokens, temperature, top_k, top_p, penalty, seed):
        settings = SamplingSettings(
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=penalty,
        )
        workspace = Path(tempfile.mkdtemp())
        outputs: list = []
        for name in available:
            outputs.extend(
                demo.compose(
                    name, mode, upload, int(bars), int(tokens), settings, int(seed), workspace
                )
            )
        return outputs

    with gr.Blocks(title="Piano Music Transformer") as interface:
        gr.Markdown(
            "# Piano Music Transformer\n"
            "An LSTM and a Transformer of **the same size** (8.4M vs 8.65M parameters), "
            "trained on the same MAESTRO piano data for the same number of steps, "
            "generating from the same prompt with the same seed.\n\n"
            "The point is not that either is good. It is that the only thing differing "
            "between the two columns is the architecture."
        )

        if not available:
            gr.Markdown(
                "**No trained checkpoints found.** Train one first:\n"
                "```\nuv run python -m pmt.train --config configs/lstm.yaml\n```"
            )
            return interface

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    [FROM_SCRATCH, FROM_UPLOAD, FROM_CORPUS],
                    value=FROM_SCRATCH,
                    label="Starting point",
                )
                upload = gr.File(label="MIDI file", file_types=[".mid", ".midi"])
                bars = gr.Slider(1, 16, value=4, step=1, label="Bars to take from it")
                tokens = gr.Slider(128, 2048, value=768, step=64, label="Tokens to generate")

                gr.Markdown("### Sampling")
                temperature = gr.Slider(0.1, 1.5, value=1.0, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 128, value=32, step=1, label="Top-k (0 = off)")
                top_p = gr.Slider(0.1, 1.0, value=1.0, step=0.01, label="Top-p (1.0 = off)")
                penalty = gr.Slider(
                    1.0,
                    2.0,
                    value=1.0,
                    step=0.05,
                    label="Repetition penalty (1.0 = off — music is repetition)",
                )
                seed = gr.Number(value=1337, precision=0, label="Seed")
                go = gr.Button("Generate", variant="primary")

            with gr.Column(scale=2):
                slots = []
                for name in available:
                    gr.Markdown(f"### {MODEL_LABELS.get(name, name)}")
                    slots.extend(
                        [
                            gr.Audio(label="Audio", type="filepath"),
                            gr.File(label="MIDI"),
                            gr.Markdown(),
                        ]
                    )

        go.click(
            run,
            inputs=[mode, upload, bars, tokens, temperature, top_k, top_p, penalty, seed],
            outputs=slots,
        )

    return interface


def main() -> None:
    build_interface(Demo()).launch()


if __name__ == "__main__":
    main()
