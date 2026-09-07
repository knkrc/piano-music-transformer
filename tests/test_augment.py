from __future__ import annotations

import pytest

from pmt.data.augment import fits_pitch_range, transpose_score, transposition_offsets


def test_offsets_start_at_zero_and_fan_out():
    assert transposition_offsets(2) == (0, 1, -1, 2, -2)


def test_zero_semitones_means_no_augmentation():
    assert transposition_offsets(0) == (0,)


def test_negative_semitones_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        transposition_offsets(-1)


def test_transpose_shifts_every_pitch(make_score):
    score = make_score([60, 64, 67])

    shifted = transpose_score(score, 3)

    assert [note.pitch for note in shifted.tracks[0].notes] == [63, 67, 70]


def test_transpose_leaves_the_original_untouched(make_score):
    score = make_score([60, 64, 67])

    transpose_score(score, 3)

    assert [note.pitch for note in score.tracks[0].notes] == [60, 64, 67]


def test_zero_offset_is_a_no_op(make_score):
    score = make_score([60])

    assert transpose_score(score, 0) is score


@pytest.mark.parametrize(
    ("pitches", "offset", "expected"),
    [
        ([60], 6, True),
        ([22], -1, True),
        ([22], -2, False),  # would fall below A0
        ([107], 1, True),
        ([107], 2, False),  # would climb past C8
    ],
)
def test_range_check_drops_variants_that_leave_the_keyboard(make_score, pitches, offset, expected):
    assert fits_pitch_range(make_score(pitches), offset, (21, 109)) is expected


def test_a_score_with_no_notes_never_fits(make_score):
    assert fits_pitch_range(make_score([]), 0, (21, 109)) is False
