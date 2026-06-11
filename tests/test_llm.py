"""Tests for the LLM rewrite layer (no model download needed)."""

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


def test_clean_rewrite_accepts_normal_sentence() -> None:
    text = "「今天下午會打雷，出門帶把傘卡安心。」"
    assert llm._clean_rewrite(text) == "今天下午會打雷，出門帶把傘卡安心。"


def test_clean_rewrite_rejects_instruction_echo() -> None:
    assert llm._clean_rewrite("好的，以下是改寫後的句子：") is None


def test_clean_rewrite_rejects_empty_and_non_chinese() -> None:
    assert llm._clean_rewrite("   ") is None
    assert llm._clean_rewrite("Sure! Here is the sentence.") is None


def test_clean_rewrite_rejects_overlong_output() -> None:
    assert llm._clean_rewrite("雨" * 300) is None


def test_interpret_llm_falls_back_when_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("model exploded")

    monkeypatch.setattr(llm, "_generate", boom)
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "rule-based"
    assert cards["pro"].startswith("【同行版")
    assert cards["grandma"].startswith("【生活版】")


def test_interpret_llm_falls_back_when_output_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_generate", lambda prompt: "Here you go!")
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "rule-based"


def test_interpret_llm_rewrites_only_grandma_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_generate", lambda prompt: "下午會打雷，帶傘較妥當。")
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "minicpm"
    assert cards["grandma"] == "【生活版】下午會打雷，帶傘較妥當。"
    assert cards["pro"].startswith("【同行版")  # untouched, rule-based numbers
