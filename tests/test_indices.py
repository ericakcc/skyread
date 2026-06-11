"""Baseline tests for deterministic index computation (synthetic profile)."""

import numpy as np
from metpy.units import units

from skyread.indices import compute_indices
from skyread.sounding import Sounding

EXPECTED_KEYS = {
    "cape_jkg",
    "cin_jkg",
    "lcl_hpa",
    "lfc_hpa",
    "el_hpa",
    "k_index",
    "lifted_index",
    "total_totals",
    "pwat_mm",
}


def _synthetic_sounding() -> Sounding:
    """A hand-made conditionally-unstable profile (9 levels)."""
    pressure = (
        np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0])
        * units.hPa
    )
    temperature = (
        np.array([30.0, 24.0, 18.0, 8.0, -10.0, -22.0, -38.0, -48.0, -55.0])
        * units.degC
    )
    dewpoint = (
        np.array([24.0, 20.0, 14.0, 2.0, -20.0, -35.0, -55.0, -65.0, -70.0])
        * units.degC
    )
    zeros = np.zeros(9) * units.knots
    return Sounding(pressure, temperature, dewpoint, zeros, zeros, "synthetic")


def test_compute_indices_returns_all_expected_keys() -> None:
    assert set(compute_indices(_synthetic_sounding())) == EXPECTED_KEYS


def test_compute_indices_unstable_profile_has_positive_cape() -> None:
    assert compute_indices(_synthetic_sounding())["cape_jkg"] > 0


def test_compute_indices_values_are_plain_floats() -> None:
    assert all(
        isinstance(v, float) for v in compute_indices(_synthetic_sounding()).values()
    )
