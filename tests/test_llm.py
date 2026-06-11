"""Tests for the MiniCPM interpretation layer (no model download needed)."""

import pytest

from skyread import llm

_INDICES = {
    "cape_jkg": 1500.0,
    "cin_jkg": -30.0,
    "lcl_hpa": 900.0,
    "lfc_hpa": 850.0,
    "el_hpa": 200.0,
    "k_index": 36.0,
    "lifted_index": -4.0,
    "total_totals": 50.0,
    "pwat_mm": 45.0,
}


def test_parse_cards_extracts_both_sections() -> None:
    text = (
        "【同行版】CAPE 中等，午後有對流潛勢。\n【生活版】下午可能下雷陣雨，記得帶傘。"
    )
    cards = llm._parse_cards(text)
    assert cards is not None
    assert cards["pro"].startswith("【同行版】")
    assert "帶傘" in cards["grandma"]


def test_parse_cards_returns_none_when_sections_missing() -> None:
    assert llm._parse_cards("今天天氣不錯") is None


def test_parse_cards_returns_none_when_section_empty() -> None:
    assert llm._parse_cards("【同行版】【生活版】帶傘。") is None


def test_interpret_llm_falls_back_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(llm, "_generate", boom)
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "rule-based"
    assert cards["pro"].startswith("【同行版")


def test_interpret_llm_falls_back_when_output_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_generate", lambda prompt: "嗯嗯今天天氣不錯喔")
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "rule-based"


def test_interpret_llm_uses_model_output_when_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm, "_generate", lambda prompt: "【同行版】不穩定。【生活版】帶傘。"
    )
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "minicpm"
    assert cards["grandma"] == "【生活版】帶傘。"
