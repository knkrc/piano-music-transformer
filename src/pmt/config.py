"""Project paths and the configuration objects that drive the pipeline.

Configs are plain dataclasses with sane defaults. A YAML file may override any
field; anything the file does not mention keeps its default. There is no config
framework here on purpose - the defaults in this file are the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(slots=True)
class DataConfig:
    """Controls how raw MIDI becomes a flat stream of token ids.

    Defaults target MAESTRO: solo piano, 88 keys, expressive timing and sustain
    pedal preserved. Velocity and pedal are what separate this representation
    from the fixed-grid one-hot encoding used in the original LSTM exercise.
    """

    # Tokenizer vocabulary
    pitch_range: tuple[int, int] = (21, 109)
    # 16 steps/beat for the first 4 beats: ~8 ms timing error at 120 bpm, below the
    # threshold where displacement is audible, for +3% sequence length. Long notes
    # (4-12 beats) do not need that precision.
    beat_res: tuple[tuple[int, int, int], ...] = ((0, 4, 16), (4, 12, 4))
    num_velocities: int = 32
    use_tempos: bool = True
    num_tempos: int = 32
    tempo_range: tuple[int, int] = (40, 250)
    use_time_signatures: bool = True
    use_sustain_pedals: bool = True
    use_rests: bool = False
    use_chords: bool = False

    # Subword (BPE) training
    bpe_vocab_size: int = 4096
    bpe_training_files: int = 400

    # Augmentation (training split only)
    # 6 semitones each way -> 13 variants per piece. The corpus, not the model,
    # is the binding constraint, so this is part of the pipeline rather than an extra.
    augment_semitones: int = 6

    # Sequence assembly
    block_size: int = 1024
    seed: int = 1337

    @property
    def beat_res_map(self) -> dict[tuple[int, int], int]:
        """MidiTok wants ``{(start_beat, end_beat): resolution}``."""
        return {(start, end): res for start, end, res in self.beat_res}


def _to_tuples(value: Any) -> Any:
    """YAML gives lists; dataclass fields here expect tuples."""
    if isinstance(value, list):
        return tuple(_to_tuples(item) for item in value)
    return value


def load_config[T](cls: type[T], path: Path | str | None = None, **overrides: Any) -> T:
    """Build a config dataclass from an optional YAML file plus keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    values.update({k: v for k, v in overrides.items() if v is not None})

    known = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")

    return cls(**{key: _to_tuples(value) for key, value in values.items()})
