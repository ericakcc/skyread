"""Small-LLM rewriting of the grandma card, with deterministic fallback.

Model layering (the hackathon's "honest small-model fit" story):

* MetPy computes every number exactly (:mod:`skyread.indices`).
* The pro card is pure numbers, so it stays rule-based — exact by design.
* A small LLM only *rewrites* the layperson sentence from a factually-correct
  draft, the one place natural language genuinely matters.
* Any failure (load, generation, malformed output) silently falls back to
  the rule-based cards, so the app never breaks on stage.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from functools import lru_cache

from skyread.interpret import build_grandma_prompt, interpret_rule_based

logger = logging.getLogger(__name__)

# Qwen3-0.6B: validated 100% Traditional-Chinese-clean on the GPU box
# (MiniCPM3-4B kept slipping into Simplified on unstable-weather wording and
# needs the old transformers 4.46 pin — see PROGRESS notes, 2026-06-11).
MODEL_ID = os.environ.get("SKYREAD_MODEL_ID", "Qwen/Qwen3-0.6B")

_MAX_REWRITE_CHARS = 180
_MAX_ATTEMPTS = 3

# High-frequency simplified-only characters: one hit means the model slipped
# out of Traditional Chinese, so the rewrite is rejected. Shared forms that
# are also standard in Taiwan (e.g. 后 in 皇后, 台, 干, 呆) are deliberately
# excluded only when ambiguity is likely; the gate is biased toward rejecting,
# since the fallback is graceful.
_SIMPLIFIED_CHARS = frozenset(
    "记伞来这为时说对让们个无气电视见车东转动书长门点云飞应过头实发现别样"
    "认师问题难岁热闹风阴湿预报员变坏轻紧稳鲜盖旷阵处带备凉润闷强从众传写"
    "决刚务医华单压历双叶号听响围国图块坚执扩扫护担拥挂损换据断显晓暂术机"
    "杂权条极标树桥梦检楼归录忆怀态总惊惯愿凭"
    "会还没几开关边儿学间阳雾闪温适当满离远进节随虽谢请"
)


def _pick_device() -> str:  # pragma: no cover - hardware dependent
    """Best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_model_once():  # pragma: no cover - exercised manually / on the Space
    """Load tokenizer and model (call via :func:`_load_model`)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto")
    model.to(_pick_device())
    model.eval()
    return tokenizer, model


def _load_model():  # pragma: no cover - thin thread-safety wrapper
    """Thread-safe single load: the warm-up thread may race the first request.

    ``lru_cache`` alone does not serialise concurrent first calls — two
    threads can both miss the cache and load the model twice.
    """
    with _LOAD_LOCK:
        return _load_model_once()


def _generate(prompt: str) -> str:  # pragma: no cover - needs model weights
    """Run one chat-formatted sampled generation and return the new text only.

    Sampling (not greedy) on purpose: a rejected output would otherwise be
    deterministic, making the retry loop in :func:`interpret_llm` useless.
    """
    import torch

    tokenizer, model = _load_model()
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        enable_thinking=False,  # Qwen3: skip <think> blocks; no-op elsewhere
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=96,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
    new_tokens = output[0][encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _clean_rewrite(text: str) -> str | None:
    """Validate and normalise a rewrite; ``None`` if it is not usable."""
    line = text.strip().strip("「」\"' \n")
    if not line or len(line) > _MAX_REWRITE_CHARS:
        return None
    if any(marker in line for marker in ("改寫", "原句", "輸出")):
        return None  # instruction echo, not a rewrite
    if not re.search(r"[一-鿿]", line):
        return None
    if any(char in _SIMPLIFIED_CHARS for char in line):
        return None  # slipped into Simplified Chinese
    return line


def interpret_llm(indices: dict[str, float], name: str) -> tuple[dict[str, str], str]:
    """Interpret indices, rewriting the grandma card with a small LLM.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding.

    Returns:
        ``(cards, engine)`` where ``engine`` is ``"llm"`` or ``"rule-based"``.
    """
    cards = interpret_rule_based(indices, name)
    prompt = build_grandma_prompt(indices, name)
    try:
        for attempt in range(_MAX_ATTEMPTS):
            raw = _generate(prompt)
            rewritten = _clean_rewrite(raw)
            if rewritten is not None:
                return {**cards, "grandma": "【生活版】" + rewritten}, "llm"
            logger.warning(
                "LLM rewrite unusable (attempt %d/%d): %r",
                attempt + 1,
                _MAX_ATTEMPTS,
                raw[:200],
            )
    except Exception:
        logger.exception("LLM generation failed, falling back")
    return cards, "rule-based"


def warm_up() -> None:
    """Eagerly load the model (call from a background thread at app start)."""
    try:
        _load_model()
    except Exception:  # pragma: no cover
        logger.exception("Model warm-up failed; rule-based fallback will be used")
