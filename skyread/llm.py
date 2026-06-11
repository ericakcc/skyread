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
from functools import lru_cache

from skyread.interpret import build_grandma_prompt, interpret_rule_based

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("SKYREAD_MODEL_ID", "openbmb/MiniCPM3-4B")

_MAX_REWRITE_CHARS = 180


def _pick_device() -> str:  # pragma: no cover - hardware dependent
    """Best available device: CUDA, then Apple MPS, then CPU."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def _load_model():  # pragma: no cover - exercised manually / on the Space
    """Load tokenizer and model once per process."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype="auto"
    )
    model.to(_pick_device())
    model.eval()
    return tokenizer, model


def _generate(prompt: str) -> str:  # pragma: no cover - needs model weights
    """Run one chat-formatted greedy generation and return the new text only."""
    import torch

    tokenizer, model = _load_model()
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**encoded, max_new_tokens=96, do_sample=False)
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
    return line


def interpret_llm(indices: dict[str, float], name: str) -> tuple[dict[str, str], str]:
    """Interpret indices, rewriting the grandma card with a small LLM.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding.

    Returns:
        ``(cards, engine)`` where ``engine`` is ``"minicpm"`` or ``"rule-based"``.
    """
    cards = interpret_rule_based(indices, name)
    try:
        raw = _generate(build_grandma_prompt(indices, name))
        rewritten = _clean_rewrite(raw)
        if rewritten is not None:
            return {**cards, "grandma": "【生活版】" + rewritten}, "minicpm"
        logger.warning("LLM rewrite unusable, falling back: %r", raw[:200])
    except Exception:
        logger.exception("LLM generation failed, falling back")
    return cards, "rule-based"


def warm_up() -> None:
    """Eagerly load the model (call from a background thread at app start)."""
    try:
        _load_model()
    except Exception:  # pragma: no cover
        logger.exception("Model warm-up failed; rule-based fallback will be used")
