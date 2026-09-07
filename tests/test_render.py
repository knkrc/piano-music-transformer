from __future__ import annotations

import pytest

from pmt import render


def test_an_explicit_soundfont_wins(tmp_path):
    soundfont = tmp_path / "custom.sf2"
    soundfont.touch()

    assert render.find_soundfont(soundfont) == soundfont


def test_an_explicit_path_that_is_missing_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"custom\.sf2"):
        render.find_soundfont(tmp_path / "custom.sf2")


def test_the_environment_is_consulted_next(tmp_path, monkeypatch):
    soundfont = tmp_path / "from_env.sf3"
    soundfont.touch()
    monkeypatch.setenv("PMT_SOUNDFONT", str(soundfont))

    assert render.find_soundfont() == soundfont


def test_known_directories_are_searched(tmp_path, monkeypatch):
    soundfont = tmp_path / "installed.sf3"
    soundfont.touch()
    monkeypatch.delenv("PMT_SOUNDFONT", raising=False)
    monkeypatch.setattr(render, "SOUNDFONT_DIRS", (tmp_path,))

    assert render.find_soundfont() == soundfont


def test_missing_soundfont_explains_how_to_install_one(tmp_path, monkeypatch):
    """A soundfont is tens of megabytes and cannot be vendored, so the error has to teach."""
    monkeypatch.delenv("PMT_SOUNDFONT", raising=False)
    monkeypatch.setattr(render, "SOUNDFONT_DIRS", (tmp_path,))

    with pytest.raises(FileNotFoundError, match="curl"):
        render.find_soundfont()
