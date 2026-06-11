"""Fetch the latest real sounding from the University of Wyoming archive.

Network access happens only here. Any failure should be caught by the caller
(the app falls back to bundled examples), so a dead upstream never kills a demo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from siphon.simplewebservice.wyoming import WyomingUpperAir

from skyread.sounding import Sounding, from_wyoming_dataframe

BANQIAO = "46692"  # Taipei / Banqiao upper-air station

_COLUMNS = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]


def _latest_synoptic(now: datetime) -> datetime:
    """Round ``now`` down to the most recent 00Z/12Z synoptic hour."""
    base = now.replace(minute=0, second=0, microsecond=0)
    return base.replace(hour=12) if base.hour >= 12 else base.replace(hour=0)


def latest_sounding(station: str = BANQIAO, max_lookback: int = 4) -> Sounding:
    """Fetch the most recent sounding, stepping back 12 h per attempt.

    Args:
        station: WMO station identifier.
        max_lookback: How many 12-hourly synoptic times to try.

    Returns:
        The parsed :class:`Sounding`, named ``"<station> <time>Z"``.

    Raises:
        RuntimeError: If no sounding exists within the lookback window.
    """
    candidate = _latest_synoptic(datetime.now(timezone.utc))
    for _ in range(max_lookback):
        try:
            df = WyomingUpperAir.request_data(candidate.replace(tzinfo=None), station)
        except ValueError:  # Wyoming returns this when the hour has no data yet
            candidate -= timedelta(hours=12)
            continue
        name = f"{station} {candidate:%Y-%m-%d %H}Z"
        return from_wyoming_dataframe(df[_COLUMNS], name=name)
    raise RuntimeError(
        f"No sounding for station {station} in the last {max_lookback * 12} hours"
    )
