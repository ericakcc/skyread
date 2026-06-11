"""Turn computed indices into plain-language, dual-layer interpretation.

Architecture note — this is where (and the *only* place) AI is load-bearing:

* The numbers come from MetPy (exact, deterministic).
* :func:`build_llm_prompt` is what we hand to a small LLM (e.g. OpenBMB MiniCPM)
  to produce natural, two-audience interpretation.
* :func:`interpret_rule_based` is a deterministic stand-in that runs today with
  no model download. It proves the data->language shape end-to-end and doubles
  as a few-shot example / safe fallback for the LLM.

Thresholds follow the standard convective-parameter references (K-index, Lifted
Index, Total Totals, CAPE/CIN).
"""

from __future__ import annotations

import math

# (lower_bound_inclusive, label, pro_phrase, grandma_phrase)
_CAPE_BANDS = (
    (4000, "extreme", "極端不穩定", "空氣非常不穩定，會有劇烈雷雨"),
    (2500, "strong", "強烈不穩定", "很可能有強雷雨"),
    (1000, "moderate", "中等不穩定", "下午容易有雷陣雨"),
    (1, "marginal", "弱不穩定", "可能有零星短暫雷雨"),
    (-math.inf, "stable", "穩定", "天氣大致穩定，不太會有雷雨"),
)


def _cape_band(cape_jkg: float) -> tuple[str, str, str]:
    """Return (label, pro_phrase, grandma_phrase) for a CAPE value."""
    value = 0.0 if math.isnan(cape_jkg) else cape_jkg
    for lower, label, pro, grandma in _CAPE_BANDS:
        if value >= lower:
            return label, pro, grandma
    return "stable", "穩定", "天氣大致穩定"


def assess(indices: dict[str, float]) -> dict[str, str]:
    """Derive a qualitative severity assessment from raw indices.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.

    Returns:
        Mapping with ``label``, ``pro_phrase``, ``grandma_phrase`` and a
        ``cap_note`` describing the convective inhibition (CIN) barrier.
    """
    label, pro, grandma = _cape_band(indices["cape_jkg"])

    cin = indices["cin_jkg"]
    if math.isnan(cin) or cin >= -25:
        cap_note = "幾乎沒有對流抑制，對流容易啟動"
    elif cin >= -100:
        cap_note = "有中等的對流抑制（蓋子），需要日照加熱才會爆發"
    else:
        cap_note = "對流抑制很強，除非有強迫抬升，否則不易發展"

    return {
        "label": label,
        "pro_phrase": pro,
        "grandma_phrase": grandma,
        "cap_note": cap_note,
    }


def build_llm_prompt(indices: dict[str, float], name: str) -> str:
    """Build the prompt handed to a small LLM (e.g. MiniCPM) for interpretation.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding (station / case name).

    Returns:
        A ready-to-send prompt string requesting two-audience output.
    """
    facts = "\n".join(f"- {k}: {v}" for k, v in indices.items())
    return (
        "你是一位大氣科學家。以下是某筆探空計算出的對流穩定度指數"
        f"（個案：{name}）：\n{facts}\n\n"
        "請用繁體中文輸出兩段判讀，嚴格依據上面的數值，不要捏造：\n"
        "【同行版】給氣象專業人員，2-3 句，點出潛勢與關鍵指數。\n"
        "【生活版】給完全不懂氣象的長輩，1-2 句，只講今天要不要帶傘、"
        "會不會打雷、能不能曬棉被。\n"
        "閾值參考：K>35 雷雨頻繁；LI<-5 非常強；TT>52 多severe；"
        "CAPE>2500 強、>4000 極端；CIN 越負代表抑制越強。"
    )


def interpret_rule_based(indices: dict[str, float], name: str) -> dict[str, str]:
    """Produce dual-layer cards deterministically (no model required).

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding.

    Returns:
        Mapping with ``pro`` and ``grandma`` card text.
    """
    a = assess(indices)

    def _lvl(key: str) -> str:
        value = indices[key]
        return "資料未及" if math.isnan(value) else f"{value:.0f} hPa"

    pro = (
        f"【同行版 · {name}】"
        f"CAPE {indices['cape_jkg']:.0f} J/kg、"
        f"CIN {indices['cin_jkg']:.0f} J/kg、"
        f"LI {indices['lifted_index']:.0f}、"
        f"K {indices['k_index']:.0f}、"
        f"TT {indices['total_totals']:.0f}。"
        f"大氣呈{a['pro_phrase']}，{a['cap_note']}。"
        f"LFC≈{_lvl('lfc_hpa')}、EL≈{_lvl('el_hpa')}，"
        f"可降水量 {indices['pwat_mm']:.0f} mm。"
    )
    grandma = f"【生活版】{a['grandma_phrase']}。"
    if a["label"] in ("moderate", "strong", "extreme"):
        grandma += "出門記得帶傘，棉被先別曬，午後盡量避免在空曠處。☔"
    else:
        grandma += "今天適合外出，棉被可以放心曬。☀️"
    return {"pro": pro, "grandma": grandma}
