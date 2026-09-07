"""The demo's logic, minus the interface.

Gradio is an optional extra, so these skip rather than fail on a lean install.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gradio", reason="install with --extra demo")

from pmt.demo import discover_checkpoints, metrics_markdown


def test_no_checkpoints_in_an_empty_tree(tmp_path):
    assert discover_checkpoints(tmp_path) == {}


def test_both_models_are_found_when_trained(tmp_path):
    for name in ("lstm", "transformer"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "best.pt").touch()

    assert list(discover_checkpoints(tmp_path)) == ["lstm", "transformer"]


def test_a_run_directory_without_a_best_checkpoint_is_skipped(tmp_path):
    """A training run that has not reached its first evaluation is not a model yet."""
    (tmp_path / "lstm").mkdir()
    (tmp_path / "lstm" / "last.pt").touch()

    assert discover_checkpoints(tmp_path) == {}


def test_the_metrics_table_pairs_each_value_with_the_corpus():
    table = metrics_markdown(
        {"scale_consistency": 0.78, "note_density": 1.33},
        {"scale_consistency": 0.84, "note_density": 5.26},
    )

    assert "| scale consistency | 0.78 | 0.84 |" in table
    assert "| note density | 1.33 | 5.26 |" in table


def test_undefined_metrics_are_shown_as_missing_rather_than_zero():
    table = metrics_markdown({"scale_consistency": None}, {"scale_consistency": 0.84})

    assert "| scale consistency | - | - |" in table
