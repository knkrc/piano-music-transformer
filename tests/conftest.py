"""Shared fixtures.

Tests build their MIDI in memory. Nothing here touches the downloaded dataset,
so the suite stays fast and runs on a clean checkout.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from symusic import Note, Score, Tempo, TimeSignature, Track

TICKS_PER_QUARTER = 480


@pytest.fixture
def tpq() -> int:
    return TICKS_PER_QUARTER


@pytest.fixture
def make_score() -> Callable[..., Score]:
    """Factory for a single-track piano score with one note per grid step."""

    def _make(pitches: list[int], step_ticks: int = TICKS_PER_QUARTER // 2) -> Score:
        score = Score(TICKS_PER_QUARTER)
        track = Track(name="piano", program=0, is_drum=False)
        for index, pitch in enumerate(pitches):
            track.notes.append(
                Note(time=index * step_ticks, duration=step_ticks - 10, pitch=pitch, velocity=80)
            )
        score.tracks.append(track)
        score.tempos.append(Tempo(0, 120.0))
        score.time_signatures.append(TimeSignature(0, 4, 4))
        return score

    return _make


@pytest.fixture
def scale_score(make_score: Callable[..., Score]) -> Score:
    return make_score([60, 62, 64, 65, 67, 69, 71, 72] * 4)
