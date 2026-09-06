"""The round-trip guarantee: whatever is lost here, the model can never learn."""

from __future__ import annotations

from symusic import Note, Score, Track

from pmt.config import DataConfig
from pmt.data.prepare import MIN_TOKENS_PER_FILE
from pmt.data.tokenizer import (
    build_tokenizer,
    decode_to_score,
    encode_score,
    merge_to_single_track,
    note_summary,
)


def test_roundtrip_keeps_every_pitch(scale_score):
    tokenizer = build_tokenizer(DataConfig())

    recovered = decode_to_score(tokenizer, encode_score(tokenizer, scale_score))

    pitches_in, _ = note_summary(scale_score)
    pitches_out, _ = note_summary(recovered)
    assert sorted(pitches_out) == sorted(pitches_in)


def test_roundtrip_onsets_stay_within_the_quantisation_grid(scale_score):
    cfg = DataConfig()
    tokenizer = build_tokenizer(cfg)
    grid = cfg.beat_res_map[(0, 4)]

    recovered = decode_to_score(tokenizer, encode_score(tokenizer, scale_score))

    _, onsets_in = note_summary(scale_score)
    _, onsets_out = note_summary(recovered)
    assert len(onsets_out) == len(onsets_in)
    for expected, actual in zip(sorted(onsets_in), sorted(onsets_out), strict=True):
        assert abs(expected - actual) <= 1 / grid


def test_encoding_is_deterministic(scale_score):
    tokenizer = build_tokenizer(DataConfig())

    assert encode_score(tokenizer, scale_score) == encode_score(tokenizer, scale_score)


def test_merge_collapses_tracks_without_losing_notes(tpq):
    score = Score(tpq)
    for pitches in ([60, 64], [67, 72]):
        track = Track(program=0, is_drum=False)
        for index, pitch in enumerate(pitches):
            track.notes.append(Note(time=index * tpq, duration=tpq, pitch=pitch, velocity=80))
        score.tracks.append(track)

    merged = merge_to_single_track(score)

    assert len(merged.tracks) == 1
    assert sorted(note.pitch for note in merged.tracks[0].notes) == [60, 64, 67, 72]


def test_multi_track_input_yields_one_stream(tpq):
    """REMI emits one sequence per track; the merge is what keeps a piece single."""
    score = Score(tpq)
    for offset in (0, tpq):
        track = Track(program=0, is_drum=False)
        track.notes.append(Note(time=offset, duration=tpq, pitch=60 + offset // tpq, velocity=80))
        score.tracks.append(track)

    ids = encode_score(tokenizer=build_tokenizer(DataConfig()), score=score)
    recovered = decode_to_score(build_tokenizer(DataConfig()), ids)

    assert len(recovered.tracks) == 1
    assert len(recovered.tracks[0].notes) == 2


def test_note_less_score_yields_only_metadata(make_score):
    """A score with no notes still emits bar/tempo/time-signature tokens.

    Such a file carries no music, so ``prepare`` filters it out by length rather
    than by emptiness - this test pins the behaviour that makes that necessary.
    """
    tokenizer = build_tokenizer(DataConfig())

    ids = encode_score(tokenizer, make_score([]))

    assert len(ids) < MIN_TOKENS_PER_FILE
    assert decode_to_score(tokenizer, ids).note_num() == 0
