"""Render MIDI to audio with FluidSynth.

    uv run python -m pmt.render --input outputs/samples --out outputs/audio

A soundfont is tens of megabytes of sampled instruments and is never vendored
into this repository. The module looks in the usual per-platform locations and,
finding none, says exactly what to install rather than failing obscurely.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pmt.config import OUTPUTS_DIR

SOUNDFONT_DIRS = (
    Path.home() / "Library/Audio/Sounds/Banks",  # macOS
    Path("/usr/share/sounds/sf2"),  # Debian / Ubuntu
    Path("/opt/homebrew/share/soundfonts"),
)
SOUNDFONT_URL = (
    "https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf3"
)
INSTALL_HINT = (
    "No soundfont found. Install one, for example:\n"
    f"  mkdir -p ~/Library/Audio/Sounds/Banks && curl -L -o "
    f"~/Library/Audio/Sounds/Banks/MuseScore_General.sf3 {SOUNDFONT_URL}\n"
    "Or point PMT_SOUNDFONT at an existing .sf2/.sf3 file."
)


def find_soundfont(explicit: Path | None = None) -> Path:
    """Locate a soundfont, preferring an explicit path, then the environment."""
    candidates = [explicit] if explicit else []
    if from_env := os.environ.get("PMT_SOUNDFONT"):
        candidates.append(Path(from_env))
    for candidate in candidates:
        if not candidate.exists():
            raise FileNotFoundError(f"soundfont not found: {candidate}")
        return candidate

    for directory in SOUNDFONT_DIRS:
        for pattern in ("*.sf3", "*.sf2"):
            found = sorted(directory.glob(pattern)) if directory.is_dir() else []
            if found:
                return found[0]
    raise FileNotFoundError(INSTALL_HINT)


def _peak_dbfs(path: Path) -> float | None:
    """Peak level of an audio file, or None if ffmpeg cannot report it."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stderr.splitlines():
        if "max_volume:" in line:
            return float(line.split("max_volume:")[1].strip().split()[0])
    return None


def render_midi(
    midi_path: Path,
    output_path: Path,
    soundfont: Path,
    sample_rate: int = 44100,
    gain: float = 0.8,
    peak_dbfs: float = -1.0,
) -> Path:
    """Synthesise ``midi_path`` to ``output_path``.

    FluidSynth writes WAV; if ffmpeg is present and an ``.mp3`` was asked for, the
    WAV is encoded and discarded. A minute of piano is ~10 MB as WAV and ~1 MB as
    MP3, which is the difference between committing samples to the repository and
    not being able to.

    Output is brought up to ``peak_dbfs`` by a **constant** gain, measured from the
    render's own peak. Loudness filters like ``loudnorm`` would sound better on a
    podcast and would be wrong here: they compress dynamic range, and preserved
    velocity is the whole point of this representation.
    """
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth is not installed (brew install fluid-synth)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    wants_mp3 = output_path.suffix.lower() == ".mp3"
    if wants_mp3 and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for mp3 output (brew install ffmpeg)")

    with tempfile.TemporaryDirectory() as workspace:
        wav_path = Path(workspace) / "render.wav" if wants_mp3 else output_path
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-g",
                str(gain),
                "-r",
                str(sample_rate),
                "-F",
                str(wav_path),
                str(soundfont),
                str(midi_path),
            ],
            check=True,
            capture_output=True,
        )
        if not wants_mp3:
            return output_path

        peak = _peak_dbfs(wav_path)
        adjust = ["-af", f"volume={peak_dbfs - peak:.2f}dB"] if peak is not None else []
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(wav_path),
                *adjust,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(output_path),
            ],
            check=True,
            capture_output=True,
        )

    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Render MIDI files to audio")
    parser.add_argument("--input", type=Path, default=OUTPUTS_DIR / "samples")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR / "audio")
    parser.add_argument("--soundfont", type=Path, default=None)
    parser.add_argument("--format", choices=("mp3", "wav"), default="mp3")
    parser.add_argument("--gain", type=float, default=0.8)
    args = parser.parse_args(argv)

    soundfont = find_soundfont(args.soundfont)
    sources = sorted(args.input.glob("*.mid")) if args.input.is_dir() else [args.input]
    if not sources:
        raise SystemExit(f"no MIDI files in {args.input}")

    print(f"soundfont: {soundfont.name}\n")
    for source in sources:
        target = args.out / f"{source.stem}.{args.format}"
        render_midi(source, target, soundfont, gain=args.gain)
        size_kb = target.stat().st_size / 1024
        print(f"  {source.name} -> {target.name} ({size_kb:,.0f} KB)")

    print(f"\nwrote {len(sources)} file(s) to {args.out}")


if __name__ == "__main__":
    main()
