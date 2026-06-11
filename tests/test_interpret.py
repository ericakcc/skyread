"""Baseline tests for the rule-based interpretation layer."""

from skyread.interpret import assess, build_grandma_prompt, interpret_rule_based


def _indices(cape: float, cin: float) -> dict[str, float]:
    return {
        "cape_jkg": cape,
        "cin_jkg": cin,
        "lcl_hpa": 900.0,
        "lfc_hpa": 850.0,
        "el_hpa": 200.0,
        "k_index": 36.0,
        "lifted_index": -6.0,
        "total_totals": 53.0,
        "pwat_mm": 40.0,
    }


def test_assess_extreme_cape_returns_extreme_label() -> None:
    assert assess(_indices(4500.0, -10.0))["label"] == "extreme"


def test_assess_zero_cape_returns_stable_label() -> None:
    assert assess(_indices(0.0, 0.0))["label"] == "stable"


def test_assess_nan_cape_treated_as_stable() -> None:
    assert assess(_indices(float("nan"), 0.0))["label"] == "stable"


def test_assess_stable_cap_note_does_not_claim_easy_initiation() -> None:
    # CAPE 0 + CIN 0 means "no convection at all", not "convection starts easily".
    note = assess(_indices(0.0, 0.0))["cap_note"]
    assert "容易啟動" not in note


def test_interpret_rule_based_marginal_advises_umbrella_not_carefree_sunning() -> None:
    # "可能有雷雨" must not be followed by "棉被可以放心曬".
    cards = interpret_rule_based(_indices(300.0, -200.0), "test")
    assert "傘" in cards["grandma"]
    assert "放心曬" not in cards["grandma"]


def test_interpret_rule_based_unstable_advises_umbrella() -> None:
    cards = interpret_rule_based(_indices(2000.0, -50.0), "test")
    assert "帶傘" in cards["grandma"]


def test_interpret_rule_based_stable_allows_sunbathing_quilt() -> None:
    cards = interpret_rule_based(_indices(0.0, 0.0), "test")
    assert "曬" in cards["grandma"]


def test_build_grandma_prompt_embeds_rule_based_draft() -> None:
    prompt = build_grandma_prompt(_indices(2000.0, -50.0), "test")
    draft = interpret_rule_based(_indices(2000.0, -50.0), "test")
    assert draft["grandma"].removeprefix("【生活版】") in prompt


def test_build_grandma_prompt_requests_rewrite_only() -> None:
    prompt = build_grandma_prompt(_indices(1500.0, -30.0), "test")
    assert "繁體中文" in prompt
    assert "只輸出改寫後的句子" in prompt
