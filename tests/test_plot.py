"""Tests for the Skew-T renderer (headless, no display)."""

import numpy as np
from matplotlib.figure import Figure
from metpy.units import units

from skyread.plot import make_skewt
from skyread.sounding import Sounding


def _sounding() -> Sounding:
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


def test_make_skewt_returns_figure_outside_pyplot_registry() -> None:
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    fig = make_skewt(_sounding())
    assert isinstance(fig, Figure)
    # Figures must not accumulate in pyplot's global manager (memory leak on
    # a long-lived Space).
    assert plt.get_fignums() == before
