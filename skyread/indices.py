"""Deterministic stability-index computation via MetPy.

This is the *non-AI* core of SkyRead: every number here is computed exactly by
MetPy, not estimated by a model. The LLM layer (:mod:`skyread.interpret`) only
turns these numbers into plain language.
"""

from __future__ import annotations

import metpy.calc as mpcalc

from skyread.sounding import Sounding


def compute_indices(snd: Sounding) -> dict[str, float]:
    """Compute the standard convective stability indices for a sounding.

    Args:
        snd: A parsed sounding.

    Returns:
        Mapping of index name to a plain float (SI-stripped, rounded), e.g.
        ``cape_jkg``, ``cin_jkg``, ``lcl_hpa``, ``lfc_hpa``, ``el_hpa``,
        ``k_index``, ``lifted_index``, ``total_totals``, ``pwat_mm``. Values
        that cannot be computed (e.g. no LFC) are ``float('nan')``.
    """
    p, t, td = snd.pressure, snd.temperature, snd.dewpoint

    parcel = mpcalc.parcel_profile(p, t[0], td[0]).to("degC")
    cape, cin = mpcalc.surface_based_cape_cin(p, t, td)

    lcl_p, _ = mpcalc.lcl(p[0], t[0], td[0])
    lfc_p, _ = mpcalc.lfc(p, t, td)
    el_p, _ = mpcalc.el(p, t, td)

    def _hpa(q) -> float:
        return round(float(q.to("hPa").magnitude), 1)

    def _scalar(q, unit: str) -> float:
        return round(float(q.to(unit).magnitude), 1)

    def _index(q) -> float:
        """Index values (K, LI, TT) are reported in their native degree unit."""
        return round(float(q.magnitude), 1)

    return {
        "cape_jkg": _scalar(cape, "joule/kilogram"),
        "cin_jkg": _scalar(cin, "joule/kilogram"),
        "lcl_hpa": _hpa(lcl_p),
        "lfc_hpa": _hpa(lfc_p),
        "el_hpa": _hpa(el_p),
        "k_index": _index(mpcalc.k_index(p, t, td)),
        "lifted_index": _index(mpcalc.lifted_index(p, t, parcel)[0]),
        "total_totals": _index(mpcalc.total_totals_index(p, t, td)),
        "pwat_mm": _scalar(mpcalc.precipitable_water(p, td), "mm"),
    }
