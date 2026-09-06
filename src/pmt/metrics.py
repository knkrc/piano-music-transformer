"""Objective metrics for generated music.

Perplexity says how well a model predicts held-out data. It says nothing about
whether the result is music: a model can score well and still produce something
nobody would listen to. These look at the notes themselves.

None of them is a verdict, and a higher number is not automatically better. They
are computed on real MAESTRO performances alongside the generated samples, and
those reference values are the only scale that makes the generated ones mean
anything. A model whose numbers sit near the corpus is behaving like the corpus,
which is the most this kind of measurement can tell you. Whether it is *good* is
still decided by listening.

Implemented here rather than taken from muspy, which brings a large dependency
tree for four functions of a few lines each.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from symusic import Score

# Semitone offsets of a major scale. The twelve rotations of this set also cover
# every natural minor scale, since a natural minor is its relative major's notes.
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)

METRIC_NAMES = (
    "pitch_class_entropy",
    "scale_consistency",
    "groove_consistency",
    "note_density",
    "pitch_range",
)


def _notes(score: Score) -> list:
    return [note for track in score.tracks for note in track.notes]


def pitch_class_entropy(score: Score) -> float | None:
    """Shannon entropy of the pitch-class histogram, in bits (0 to log2(12) = 3.58).

    Near zero means one or two pitch classes dominate; near 3.58 means all twelve
    are used equally, which in practice means the model is not committing to a key.
    """
    notes = _notes(score)
    if not notes:
        return None
    counts = np.bincount([note.pitch % 12 for note in notes], minlength=12).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def scale_consistency(score: Score) -> float | None:
    """Largest fraction of notes that fit a single diatonic scale.

    A chromatic run scores 7/12 = 0.583 by chance, so that is the floor worth
    comparing against rather than zero.
    """
    notes = _notes(score)
    if not notes:
        return None
    classes = np.array([note.pitch % 12 for note in notes])
    return max(
        float(np.isin(classes, [(root + step) % 12 for step in MAJOR_SCALE]).mean())
        for root in range(12)
    )


def groove_consistency(score: Score, grid: int = 16) -> float | None:
    """How steadily the rhythmic pattern repeats from bar to bar, 0 to 1.

    Each bar becomes a binary onset pattern on a ``grid``-step grid; the score is
    one minus the mean disagreement between consecutive bars. Music with a steady
    accompaniment scores high; rubato and free playing score lower, which is a
    description rather than a criticism.
    """
    notes = _notes(score)
    downbeats = np.asarray(score.get_downbeats(), dtype=np.int64)
    if len(notes) == 0 or len(downbeats) < 3:
        return None

    onsets = np.sort(np.array([note.time for note in notes], dtype=np.int64))
    patterns = []
    for start, end in pairwise(downbeats):
        if end <= start:
            continue
        low, high = np.searchsorted(onsets, (start, end))
        pattern = np.zeros(grid, dtype=bool)
        if high > low:
            positions = (onsets[low:high] - start) * grid // (end - start)
            pattern[np.clip(positions, 0, grid - 1)] = True
        patterns.append(pattern)

    if len(patterns) < 2:
        return None
    stacked = np.stack(patterns)
    return float(1.0 - np.mean(stacked[:-1] != stacked[1:]))


def note_density(score: Score) -> float | None:
    """Notes per beat."""
    notes = _notes(score)
    if not notes:
        return None
    beats = score.end() / score.ticks_per_quarter
    return float(len(notes) / beats) if beats > 0 else None


def pitch_range(score: Score) -> float | None:
    """Distance in semitones from the lowest note to the highest."""
    notes = _notes(score)
    if not notes:
        return None
    pitches = [note.pitch for note in notes]
    return float(max(pitches) - min(pitches))


def summarise(score: Score) -> dict[str, float | None]:
    return {
        "pitch_class_entropy": pitch_class_entropy(score),
        "scale_consistency": scale_consistency(score),
        "groove_consistency": groove_consistency(score),
        "note_density": note_density(score),
        "pitch_range": pitch_range(score),
    }


def aggregate(summaries: list[dict[str, float | None]]) -> dict[str, float | None]:
    """Mean of each metric over the summaries where it is defined.

    Metrics are undefined rather than zero for a score too short or too empty to
    measure - averaging those in as zeros would quietly flatter a model that
    produced nothing.
    """
    aggregated: dict[str, float | None] = {}
    for name in METRIC_NAMES:
        values = [s[name] for s in summaries if s.get(name) is not None]
        aggregated[name] = float(np.mean(values)) if values else None
    return aggregated
