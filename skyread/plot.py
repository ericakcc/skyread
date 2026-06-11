"""Render a Skew-T / Log-P diagram from a sounding.

Uses MetPy's :class:`~metpy.plots.SkewT` so the plotted curves come straight
from the data — no chart-image reading involved. A Matplotlib ``Figure`` is
returned so Gradio's ``gr.Plot`` can display it directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for server-side rendering

import matplotlib.pyplot as plt  # noqa: E402
import metpy.calc as mpcalc  # noqa: E402
from metpy.plots import SkewT  # noqa: E402

from skyread.sounding import Sounding  # noqa: E402


def make_skewt(snd: Sounding) -> plt.Figure:
    """Build a Skew-T figure (temperature, dewpoint, parcel path, CAPE/CIN).

    Args:
        snd: A parsed sounding.

    Returns:
        A Matplotlib figure ready for display or saving.
    """
    fig = plt.figure(figsize=(7, 8))
    skew = SkewT(fig, rotation=45)

    skew.plot(
        snd.pressure, snd.temperature, "tab:red", linewidth=2, label="Temperature"
    )
    skew.plot(snd.pressure, snd.dewpoint, "tab:green", linewidth=2, label="Dewpoint")
    skew.plot_barbs(snd.pressure[::3], snd.u_wind[::3], snd.v_wind[::3])

    parcel = mpcalc.parcel_profile(
        snd.pressure, snd.temperature[0], snd.dewpoint[0]
    ).to("degC")
    skew.plot(
        snd.pressure, parcel, "black", linewidth=1.5, linestyle="--", label="Parcel"
    )
    skew.shade_cape(snd.pressure, snd.temperature, parcel)
    skew.shade_cin(snd.pressure, snd.temperature, parcel)

    skew.plot_dry_adiabats(alpha=0.3)
    skew.plot_moist_adiabats(alpha=0.3)
    skew.plot_mixing_lines(alpha=0.3)

    skew.ax.set_xlim(-40, 50)
    skew.ax.set_ylim(1050, 100)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.ax.set_title(f"Skew-T / Log-P — {snd.name}")
    skew.ax.legend(loc="upper right", fontsize=8)
    return fig
