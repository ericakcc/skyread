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


def build_grandma_prompt(indices: dict[str, float], name: str) -> str:
    """Build the rewrite prompt handed to a small LLM for the grandma card.

    The pro card is pure numbers and stays rule-based; only the layperson
    sentence benefits from a natural-language touch. The rule-based grandma
    line is embedded as a factually-correct draft, so the model only rewrites
    tone — a task small models handle far more reliably than free generation.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding (station / case name).

    Returns:
        A ready-to-send prompt string requesting a single rewritten sentence.
    """
    draft = interpret_rule_based(indices, name)["grandma"].removeprefix("【生活版】")
    return (
        "把這句天氣提醒改寫成更口語、更親切的說法"
        "（繁體中文，講給長輩聽，一到兩句）：\n"
        f"「{draft}」\n"
        "保留原本的結論與建議，不要新增資訊。只輸出改寫後的句子。"
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
