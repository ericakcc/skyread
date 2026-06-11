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


def test_clean_rewrite_rejects_simplified_chinese() -> None:
    # Real failure modes observed from MiniCPM3-4B on the GPU box.
    assert llm._clean_rewrite("下午有機會打雷下雨，记得帶把伞。") is None
    assert llm._clean_rewrite("今天天氣挺稳当的，不太會打雷下雨。") is None
    assert llm._clean_rewrite("棉被可以拿出来晒太陽。") is None
    assert llm._clean_rewrite("可能会有小小滴雨滴哦！") is None
    assert llm._clean_rewrite("今天温度舒适，适合外出。") is None


def test_clean_rewrite_accepts_pure_traditional_sentence() -> None:
    text = "下午可能會打雷下雨，記得帶把傘，棉被先別曬喔。"
    assert llm._clean_rewrite(text) == text


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


def test_interpret_llm_retries_until_a_usable_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = iter(["可能会有雨。", "下午可能下雨，帶把傘較安心。"])
    monkeypatch.setattr(llm, "_generate", lambda prompt: next(outputs))
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "llm"
    assert cards["grandma"] == "【生活版】下午可能下雨，帶把傘較安心。"


def test_interpret_llm_gives_up_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def always_simplified(prompt: str) -> str:
        calls.append(prompt)
        return "可能会有雨。"

    monkeypatch.setattr(llm, "_generate", always_simplified)
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "rule-based"
    assert len(calls) == llm._MAX_ATTEMPTS


def test_interpret_llm_rewrites_only_grandma_card(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_generate", lambda prompt: "下午會打雷，帶傘較妥當。")
    cards, engine = llm.interpret_llm(_INDICES, "test")
    assert engine == "llm"
    assert cards["grandma"] == "【生活版】下午會打雷，帶傘較妥當。"
    assert cards["pro"].startswith("【同行版")  # untouched, rule-based numbers
