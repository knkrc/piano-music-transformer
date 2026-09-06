"""Metrics are only useful if the numbers mean what the names claim."""

from __future__ import annotations

import math

import pytest
from symusic import Note, Score, Tempo, TimeSignature, Track

from pmt.metrics import (
    aggregate,
    groove_consistency,
    note_density,
    pitch_class_entropy,
    pitch_range,
    scale_consistency,
    summarise,
)

C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]
CHROMATIC = list(range(60, 72))


def test_one_repeated_pitch_has_no_entropy(make_score):
    assert pitch_class_entropy(make_score([60] * 8)) == pytest.approx(0.0)


def test_every_pitch_class_used_equally_is_maximum_entropy(make_score):
    assert pitch_class_entropy(make_score(CHROMATIC)) == pytest.approx(math.log2(12))


def test_a_diatonic_melody_is_fully_scale_consistent(make_score):
    assert scale_consistency(make_score(C_MAJOR)) == pytest.approx(1.0)


def test_a_chromatic_run_scores_the_chance_floor(make_score):
    """Seven of twelve pitch classes fit any major scale, so 0.583 is the floor."""
    assert scale_consistency(make_score(CHROMATIC)) == pytest.approx(7 / 12)


def test_transposition_does_not_change_scale_consistency(make_score):
    """The metric looks for *a* key, not for C."""
    original = scale_consistency(make_score(C_MAJOR))
    transposed = scale_consistency(make_score([pitch + 5 for pitch in C_MAJOR]))

    assert original == pytest.approx(transposed)


def test_identical_bars_are_perfectly_groove_consistent(make_score):
    four_identical_bars = make_score(C_MAJOR * 4)  # eight eighth-notes to the bar

    assert groove_consistency(four_identical_bars) == pytest.approx(1.0)


def test_an_irregular_rhythm_is_less_groove_consistent(tpq):
    score = Score(tpq)
    track = Track(program=0, is_drum=False)
    bar = 4 * tpq
    for index in range(8):  # a busy first bar
        track.notes.append(Note(time=index * tpq // 2, duration=100, pitch=60, velocity=80))
    for index in range(2):  # a sparse second bar, off the first bar's grid
        track.notes.append(
            Note(time=bar + index * tpq * 3 // 2, duration=100, pitch=60, velocity=80)
        )
    track.notes.append(Note(time=2 * bar, duration=100, pitch=60, velocity=80))
    score.tracks.append(track)
    score.tempos.append(Tempo(0, 120.0))
    score.time_signatures.append(TimeSignature(0, 4, 4))

    assert groove_consistency(score) < 0.9


def test_groove_needs_at_least_two_complete_bars(make_score):
    assert groove_consistency(make_score([60, 62])) is None


def test_note_density_counts_notes_per_beat(make_score):
    """Eight eighth-notes span four beats, so two notes to the beat."""
    density = note_density(make_score(C_MAJOR))

    assert density == pytest.approx(2.0, rel=0.05)


def test_pitch_range_is_measured_in_semitones(make_score):
    assert pitch_range(make_score([60, 72])) == 12


def test_an_empty_score_leaves_every_metric_undefined(make_score):
    """Undefined, not zero - a silent sample must not average in as a good one."""
    assert all(value is None for value in summarise(make_score([])).values())


def test_aggregation_ignores_undefined_metrics_rather_than_zeroing_them():
    summaries = [
        {"pitch_class_entropy": 3.0, "scale_consistency": None},
        {"pitch_class_entropy": 1.0, "scale_consistency": 0.8},
    ]

    aggregated = aggregate(summaries)

    assert aggregated["pitch_class_entropy"] == pytest.approx(2.0)
    assert aggregated["scale_consistency"] == pytest.approx(0.8)
    assert aggregated["groove_consistency"] is None
