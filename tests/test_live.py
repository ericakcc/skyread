"""Tests for live-sounding time logic (network calls are not tested here)."""

from datetime import datetime, timezone

from skyread.live import _latest_synoptic


def test_latest_synoptic_morning_rounds_to_00z() -> None:
    now = datetime(2026, 6, 11, 3, 30, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 0, 0, tzinfo=timezone.utc)


def test_latest_synoptic_afternoon_rounds_to_12z() -> None:
    now = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def test_latest_synoptic_exactly_noon_is_12z() -> None:
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
