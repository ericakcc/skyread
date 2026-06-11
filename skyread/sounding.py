"""Load radiosonde sounding data into a units-aware structure.

This module is pure I/O + parsing. It returns MetPy/pint quantities ready to be
fed into :mod:`skyread.indices`. Two sources are supported for the spike:

* MetPy's bundled sample soundings (``get_test_data``) — zero network, perfect
  for demos that must never break.
* University-of-Wyoming-style fixed-width text (also the IGRA2-export shape) —
  the format users upload or fetch online.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from metpy.calc import wind_components
from metpy.cbook import get_test_data
from metpy.units import units
from pint import Quantity


@dataclass
class Sounding:
    """A parsed atmospheric sounding with units attached.

    Attributes:
        pressure: Pressure profile (hPa), decreasing upward.
        temperature: Environmental temperature profile (degC).
        dewpoint: Dewpoint temperature profile (degC).
        u_wind: Zonal wind component (knots).
        v_wind: Meridional wind component (knots).
        name: Human-readable label for the sounding.
    """

    pressure: Quantity
    temperature: Quantity
    dewpoint: Quantity
    u_wind: Quantity
    v_wind: Quantity
    name: str


def _from_wyoming_dataframe(df: pd.DataFrame, name: str) -> Sounding:
    """Build a :class:`Sounding` from a Wyoming-style dataframe.

    Args:
        df: Columns ``pressure, height, temperature, dewpoint, direction, speed``.
        name: Label for the sounding.

    Returns:
        A units-aware :class:`Sounding`.
    """
    df = df.dropna(
        subset=("temperature", "dewpoint", "direction", "speed"), how="all"
    ).reset_index(drop=True)

    pressure = df["pressure"].to_numpy() * units.hPa
    temperature = df["temperature"].to_numpy() * units.degC
    dewpoint = df["dewpoint"].to_numpy() * units.degC
    u_wind, v_wind = wind_components(
        df["speed"].to_numpy() * units.knots,
        df["direction"].to_numpy() * units.deg,
    )
    return Sounding(pressure, temperature, dewpoint, u_wind, v_wind, name)


def load_csv(path: str, name: str = "uploaded") -> Sounding:
    """Load a user-uploaded CSV sounding.

    Expected columns (header row, case-insensitive): ``pressure, temperature,
    dewpoint, direction, speed``. Pressure in hPa, temperatures in degC,
    direction in degrees, speed in knots.

    Args:
        path: Path to the CSV file.
        name: Label for the sounding.

    Returns:
        The parsed :class:`Sounding`.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"pressure", "temperature", "dewpoint", "direction", "speed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    return _from_wyoming_dataframe(df[list(required)], name=name)


def load_sample(name: str = "may4_sounding.txt") -> Sounding:
    """Load a MetPy bundled sample sounding (no network at runtime).

    Args:
        name: One of ``may4_sounding.txt``, ``jan20_sounding.txt``,
            ``nov11_sounding.txt``.

    Returns:
        The parsed :class:`Sounding`.
    """
    path = get_test_data(name, as_file_obj=False)
    col_names = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]
    df = pd.read_fwf(path, skiprows=5, usecols=[0, 1, 2, 3, 6, 7], names=col_names)
    return _from_wyoming_dataframe(df, name=name.replace("_sounding.txt", ""))
