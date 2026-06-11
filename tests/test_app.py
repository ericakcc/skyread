"""Tests for the Gradio glue layer (no model download, no network)."""

from pathlib import Path

import app


def test_analyze_surfaces_compute_errors_as_message(tmp_path: Path) -> None:
    # Parses fine (all columns present) but is physically unusable: the
    # failure happens in compute_indices, not in loading.
    bad = tmp_path / "empty.csv"
    bad.write_text("pressure,temperature,dewpoint,direction,speed\n")
    fig, pro, grandma, badge = app.analyze(
        app.SOURCE_UPLOAD, "", "", str(bad), use_llm=False
    )
    assert fig is None
    assert pro.startswith("⚠️")


def test_analyze_example_rule_based_returns_cards() -> None:
    fig, pro, grandma, badge = app.analyze(
        app.SOURCE_EXAMPLE, "", next(iter(app.EXAMPLES)), None, use_llm=False
    )
    assert fig is not None
    assert pro.startswith("【同行版")
    assert grandma.startswith("【生活版】")
    assert "MetPy" in badge
