"""MiniCPM-backed interpretation with deterministic fallback.

Model layering (the hackathon's "honest small-model fit" story):

* MetPy computes every number exactly (:mod:`skyread.indices`).
* MiniCPM4-0.5B only *rewrites* a factually-correct draft into natural
  language — a task a 0.5B model handles reliably on free CPU hardware.
* Any failure (load, generation, malformed output) silently falls back to
  the rule-based cards, so the app never breaks on stage.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache

from skyread.interpret import build_llm_prompt, interpret_rule_based

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("SKYREAD_MODEL_ID", "openbmb/MiniCPM4-0.5B")


@lru_cache(maxsize=1)
def _load_model():  # pragma: no cover - exercised manually / on the Space
    """Load tokenizer and model once per process (CPU, fp32)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    )
    model.eval()
    return tokenizer, model


def _generate(prompt: str) -> str:  # pragma: no cover - needs model weights
    """Run one chat-formatted generation and return the new text only."""
    import torch

    tokenizer, model = _load_model()
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
        )
    return tokenizer.decode(output[0][inputs.shape[1] :], skip_special_tokens=True)


def _parse_cards(text: str) -> dict[str, str] | None:
    """Extract 【同行版】/【生活版】 sections; ``None`` if the shape is wrong."""
    pro = re.search(r"【同行版】(.*?)(?=【生活版】|$)", text, re.S)
    grandma = re.search(r"【生活版】(.+)", text, re.S)
    if not (pro and grandma):
        return None
    pro_text = pro.group(1).strip()
    grandma_text = grandma.group(1).strip()
    if not pro_text or not grandma_text:
        return None
    return {
        "pro": "【同行版】" + pro_text,
        "grandma": "【生活版】" + grandma_text,
    }


def interpret_llm(indices: dict[str, float], name: str) -> tuple[dict[str, str], str]:
    """Interpret indices with MiniCPM, falling back to rule-based on failure.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding.

    Returns:
        ``(cards, engine)`` where ``engine`` is ``"minicpm"`` or ``"rule-based"``.
    """
    try:
        raw = _generate(build_llm_prompt(indices, name))
        cards = _parse_cards(raw)
        if cards is not None:
            return cards, "minicpm"
        logger.warning("MiniCPM output unparseable, falling back: %r", raw[:200])
    except Exception:
        logger.exception("MiniCPM generation failed, falling back")
    return interpret_rule_based(indices, name), "rule-based"


def warm_up() -> None:
    """Eagerly load the model (call from a background thread at app start)."""
    try:
        _load_model()
    except Exception:  # pragma: no cover
        logger.exception("MiniCPM warm-up failed; rule-based fallback will be used")
