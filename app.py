"""Entry point for Hugging Face Spaces, which expects app.py at the repository root.

The interface itself lives in ``pmt.demo`` so that it is importable and testable.

    uv run --extra demo python app.py

Locally this uses whatever you have trained. On a hosted Space there is nothing
trained and no dataset, so it falls back to the published weights and to
pre-measured reference numbers - the tokenised MAESTRO split is a derivative of a
CC BY-NC-SA dataset and is never redistributed.
"""

from __future__ import annotations

from pathlib import Path

from pmt.config import OUTPUTS_DIR, PROCESSED_DIR
from pmt.demo import Demo, build_interface, discover_checkpoints
from pmt.render import download_soundfont, find_soundfont

HUB_REPO = "knkrc26/piano-music-transformer"


def resolve_assets() -> tuple[Path, Path]:
    """Return ``(data_dir, models_root)``, fetching the published weights if needed."""
    if discover_checkpoints(OUTPUTS_DIR) and (PROCESSED_DIR / "meta.json").exists():
        return PROCESSED_DIR, OUTPUTS_DIR

    from huggingface_hub import snapshot_download

    published = Path(snapshot_download(HUB_REPO))
    return published, published


def main() -> None:
    try:
        find_soundfont()
    except FileNotFoundError:
        # A hosted Space starts with nothing; without this the page loads and every
        # generation comes back silent.
        download_soundfont()

    data_dir, models_root = resolve_assets()
    build_interface(Demo(data_dir=data_dir, root=models_root)).launch()


if __name__ == "__main__":
    main()
