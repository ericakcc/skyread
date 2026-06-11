# SkyRead — Build Small Hackathon 4-Day Sprint Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 SkyRead 從本機 spike 變成可提交的黑客松作品：MiniCPM 真實推論、即時板橋探空、部署 HF Space、demo 影片與提交物，2026-06-15 截止（目標 06-14 完成，留一天緩衝）。

**Architecture:** 三層不變——MetPy 確定性計算數值、小模型只負責把數字講成人話、規則式判讀降級為 fallback 兼 few-shot 草稿。LLM 採「草稿改寫」策略：規則式輸出當草稿餵給 MiniCPM4-0.5B 潤飾，0.5B 做改寫遠比自由生成可靠，且可在免費 CPU Space 上跑（瞄準 OpenBMB 贊助獎 + Off the Grid badge）。

**Tech Stack:** Python 3.12 / uv / MetPy / Gradio 6.17.3 / transformers + torch（MiniCPM4-0.5B, CPU）/ siphon（Wyoming 即時探空）/ pytest + ruff / HF Spaces。

**黑客松對照（Backyard AI track 評審標準）:**
- 問題真實具體 → 即時板橋探空 +「阿嬤版」(Task 6-8)
- 真實使用者採用 → Task 10 使用者證據
- 與小模型限制誠實契合 → 「MetPy 算數字、0.5B 只改寫」敘事 (Task 2-4)
- Gradio 完成度 → Task 7, 9

**硬性規則檢查:** ≤32B ✅ (0.5B)、Gradio ✅、HF Space (Task 8)、demo 影片 (Task 11)、社群貼文 (Task 12)。

---

## Day 1（6/11）— 讓 AI 真的上線

### Task 1: 開發基礎建設與測試基線

**Files:**
- Modify: `pyproject.toml`（描述、dev deps）
- Delete: `main.py`（uv init 殘留）
- Create: `tests/__init__.py`, `tests/test_interpret.py`, `tests/test_indices.py`

- [ ] **Step 1: 開 sprint 分支（遵守不直接 push main）**

```bash
git checkout -b feat/day1-minicpm
```

- [ ] **Step 2: 加 dev 依賴、刪殘留檔**

```bash
uv add --dev ruff pytest
git rm main.py
```

- [ ] **Step 3: 修 pyproject 描述**

把 `pyproject.toml` 的 `description = "Add your description here"` 改成：

```toml
description = "Turn Skew-T soundings into plain-language weather advice — MetPy computes, a 0.5B LLM narrates"
```

- [ ] **Step 4: 寫 interpret 基線測試**

建立 `tests/__init__.py`（空檔）與 `tests/test_interpret.py`：

```python
"""Baseline tests for the rule-based interpretation layer."""

from skyread.interpret import assess, build_llm_prompt, interpret_rule_based


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


def test_interpret_rule_based_unstable_advises_umbrella() -> None:
    cards = interpret_rule_based(_indices(2000.0, -50.0), "test")
    assert "帶傘" in cards["grandma"]


def test_interpret_rule_based_stable_allows_sunbathing_quilt() -> None:
    cards = interpret_rule_based(_indices(0.0, 0.0), "test")
    assert "曬" in cards["grandma"]


def test_build_llm_prompt_contains_values_and_format_markers() -> None:
    prompt = build_llm_prompt(_indices(1500.0, -30.0), "test")
    assert "1500" in prompt
    assert "【同行版】" in prompt
    assert "【生活版】" in prompt
```

- [ ] **Step 5: 寫 indices 基線測試（合成探空，不碰網路）**

建立 `tests/test_indices.py`：

```python
"""Baseline tests for deterministic index computation (synthetic profile)."""

import numpy as np
from metpy.units import units

from skyread.indices import compute_indices
from skyread.sounding import Sounding

EXPECTED_KEYS = {
    "cape_jkg",
    "cin_jkg",
    "lcl_hpa",
    "lfc_hpa",
    "el_hpa",
    "k_index",
    "lifted_index",
    "total_totals",
    "pwat_mm",
}


def _synthetic_sounding() -> Sounding:
    """A hand-made conditionally-unstable profile (9 levels)."""
    pressure = (
        np.array([1000.0, 925.0, 850.0, 700.0, 500.0, 400.0, 300.0, 250.0, 200.0])
        * units.hPa
    )
    temperature = (
        np.array([30.0, 24.0, 18.0, 8.0, -10.0, -22.0, -38.0, -48.0, -55.0])
        * units.degC
    )
    dewpoint = (
        np.array([24.0, 20.0, 14.0, 2.0, -20.0, -35.0, -55.0, -65.0, -70.0])
        * units.degC
    )
    zeros = np.zeros(9) * units.knots
    return Sounding(pressure, temperature, dewpoint, zeros, zeros, "synthetic")


def test_compute_indices_returns_all_expected_keys() -> None:
    assert set(compute_indices(_synthetic_sounding())) == EXPECTED_KEYS


def test_compute_indices_unstable_profile_has_positive_cape() -> None:
    assert compute_indices(_synthetic_sounding())["cape_jkg"] > 0


def test_compute_indices_values_are_plain_floats() -> None:
    assert all(isinstance(v, float) for v in compute_indices(_synthetic_sounding()).values())
```

- [ ] **Step 6: 跑測試確認全綠**

Run: `uv run pytest tests/ -v`
Expected: 9 passed

- [ ] **Step 7: lint + commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add -A
git commit -m "test: add baseline tests for indices and interpretation; project housekeeping"
```

---

### Task 2: build_llm_prompt 改為「草稿改寫」策略

0.5B 模型自由生成容易亂編；給它正確草稿要求「改寫得自然」可靠得多，也正好實現「rule-based 同時是 LLM 的 few-shot 範例」的既有設計。

**Files:**
- Modify: `skyread/interpret.py:67-87`（`build_llm_prompt`）
- Test: `tests/test_interpret.py`

- [ ] **Step 1: 寫失敗測試（草稿須嵌入 prompt）**

在 `tests/test_interpret.py` 加：

```python
def test_build_llm_prompt_embeds_rule_based_draft() -> None:
    prompt = build_llm_prompt(_indices(2000.0, -50.0), "test")
    draft = interpret_rule_based(_indices(2000.0, -50.0), "test")
    assert draft["grandma"] in prompt
```

- [ ] **Step 2: 跑測試確認失敗**

Run: `uv run pytest tests/test_interpret.py -v -k draft`
Expected: FAIL（現在的 prompt 沒嵌草稿）

- [ ] **Step 3: 改寫 build_llm_prompt**

把 `skyread/interpret.py` 的 `build_llm_prompt` 整個換成：

```python
def build_llm_prompt(indices: dict[str, float], name: str) -> str:
    """Build the rewrite-style prompt handed to a small LLM (MiniCPM).

    The rule-based output is embedded as a factually-correct draft; the model
    only rewrites tone, which a 0.5B model does far more reliably than free
    generation.

    Args:
        indices: Output of :func:`skyread.indices.compute_indices`.
        name: Label of the sounding (station / case name).

    Returns:
        A ready-to-send prompt string requesting two-audience output.
    """
    facts = "\n".join(f"- {k}: {v}" for k, v in indices.items())
    draft = interpret_rule_based(indices, name)
    return (
        "你是一位大氣科學家。以下是某筆探空計算出的對流穩定度指數"
        f"（個案：{name}）：\n{facts}\n\n"
        "下面是一份機器草稿，數值與結論正確，但語氣生硬：\n"
        f"{draft['pro']}\n{draft['grandma']}\n\n"
        "請把草稿改寫得自然、口語，保留所有數值與結論，"
        "不要新增草稿沒有的數字。嚴格依照此格式輸出（繁體中文）：\n"
        "【同行版】2-3 句，給氣象專業人員，點出潛勢與關鍵指數。\n"
        "【生活版】1-2 句，給完全不懂氣象的長輩，只講要不要帶傘、"
        "會不會打雷、能不能曬棉被。"
    )
```

注意：`interpret_rule_based` 定義在檔案較後面，呼叫發生在執行期，不需搬動。

- [ ] **Step 4: 跑全部測試**

Run: `uv run pytest tests/ -v`
Expected: 10 passed（含原本的 format markers 測試仍綠）

- [ ] **Step 5: commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add skyread/interpret.py tests/test_interpret.py
git commit -m "feat(interpret): switch LLM prompt to draft-rewrite strategy for 0.5B reliability"
```

---

### Task 3: skyread/llm.py — MiniCPM 推論 + fallback

**Files:**
- Create: `skyread/llm.py`
- Test: `tests/test_llm.py`
- Modify: `pyproject.toml`（依賴）

- [ ] **Step 1: 加依賴**

```bash
uv add torch transformers
```

- [ ] **Step 2: 寫失敗測試**

建立 `tests/test_llm.py`：

```python
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
    text = "【同行版】CAPE 中等，午後有對流潛勢。\n【生活版】下午可能下雷陣雨，記得帶傘。"
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
    assert "【同行版】" in cards["pro"]


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
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: skyread.llm` 或 import error

- [ ] **Step 4: 實作 skyread/llm.py**

```python
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
    pro = re.search(r"【同行版】(.+?)(?=【生活版】|$)", text, re.S)
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
```

- [ ] **Step 5: 跑測試確認通過**

Run: `uv run pytest tests/ -v`
Expected: 16 passed

- [ ] **Step 6: 真實煙霧測試（會下載 ~1GB 模型，跑一次就好）**

```bash
uv run python -c "
from skyread.sounding import load_sample
from skyread.indices import compute_indices
from skyread.llm import interpret_llm
snd = load_sample('may4_sounding.txt')
cards, engine = interpret_llm(compute_indices(snd), snd.name)
print('engine =', engine)
print(cards['pro'])
print(cards['grandma'])
"
```

Expected: `engine = minicpm`，兩段繁中輸出、數值與草稿一致。若品質差（簡體、亂編數字），調整 `temperature`（試 0.2）或在 prompt 加「必須使用繁體中文」再試。**這一步是 Day 1 的品質關卡，值得花 30 分鐘調。**

- [ ] **Step 7: commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add skyread/llm.py tests/test_llm.py pyproject.toml uv.lock
git commit -m "feat(llm): wire MiniCPM4-0.5B interpretation with rule-based fallback"
```

---

### Task 4: app.py 接上 LLM（含既有 bug 修正）

既有 bug：`build_ui().launch(theme=gr.themes.Soft())` —— `launch()` 不收 `theme`，要搬進 `gr.Blocks(...)`。

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 改寫 app.py 的 analyze 與 UI**

`app.py` 全文換成（Day 2 還會再改資料來源部分）：

```python
"""SkyRead — Gradio app: sounding -> Skew-T plot + dual-layer interpretation.

Run locally:
    uv run python app.py
"""

from __future__ import annotations

import threading

import gradio as gr

from skyread.indices import compute_indices
from skyread.interpret import interpret_rule_based
from skyread.llm import interpret_llm, warm_up
from skyread.plot import make_skewt
from skyread.sounding import Sounding, load_csv, load_sample

# Curated, demo-safe example soundings bundled with MetPy (zero network).
EXAMPLES: dict[str, str] = {
    "1999-05-04 Oklahoma (強對流 / tornado outbreak)": "may4_sounding.txt",
    "2010-01-20 winter case": "jan20_sounding.txt",
    "2011-11-11 case": "nov11_sounding.txt",
}

_BADGE_LLM = "🧠 判讀由 **MiniCPM4-0.5B**（本機推論）改寫；所有數值由 MetPy 確定性計算。"
_BADGE_RULE = "📐 規則式判讀（fallback）；所有數值由 MetPy 確定性計算。"


def analyze(example_label: str, uploaded: str | None, use_llm: bool):
    """Run the full chain and return (figure, pro_md, grandma_md, badge_md)."""
    try:
        snd: Sounding = (
            load_csv(uploaded, name="uploaded")
            if uploaded
            else load_sample(EXAMPLES[example_label])
        )
    except Exception as exc:  # surface parse errors to the user, don't crash
        return None, f"⚠️ 讀取失敗：{exc}", "", ""

    indices = compute_indices(snd)
    if use_llm:
        cards, engine = interpret_llm(indices, snd.name)
    else:
        cards, engine = interpret_rule_based(indices, snd.name), "rule-based"
    badge = _BADGE_LLM if engine == "minicpm" else _BADGE_RULE
    return make_skewt(snd), cards["pro"], cards["grandma"], badge


def _analyze_fast(example_label: str, uploaded: str | None):
    """Instant first paint on page load: skip the LLM, show rule-based cards."""
    return analyze(example_label, uploaded, use_llm=False)


def build_ui() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="SkyRead 探空白話判讀器", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🌤️ SkyRead — 探空白話判讀器\n"
            "把艱深的 Skew-T 探空圖，翻成**同行看的指數**與**阿嬤看的帶傘建議**。\n"
            "_數值由 MetPy 精確計算，AI 只負責把數字講成人話。_"
        )
        with gr.Row():
            with gr.Column(scale=1):
                example = gr.Dropdown(
                    choices=list(EXAMPLES), value=list(EXAMPLES)[0], label="範例探空"
                )
                upload = gr.File(
                    label="或上傳你的探空 CSV (pressure,temperature,dewpoint,direction,speed)",
                    file_types=[".csv"],
                    type="filepath",
                )
                use_llm = gr.Checkbox(value=True, label="🧠 用 MiniCPM 白話判讀（慢幾秒，但更像人話）")
                btn = gr.Button("判讀 ☁️", variant="primary")
            with gr.Column(scale=1):
                plot = gr.Plot(label="Skew-T / Log-P")
        pro = gr.Markdown()
        grandma = gr.Markdown()
        badge = gr.Markdown()

        btn.click(analyze, inputs=[example, upload, use_llm], outputs=[plot, pro, grandma, badge])
        demo.load(_analyze_fast, inputs=[example, upload], outputs=[plot, pro, grandma, badge])
    return demo


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    build_ui().launch()
```

- [ ] **Step 2: 手動驗證**

Run: `uv run python app.py`，瀏覽器開 http://127.0.0.1:7860
Expected: 載入即顯示規則式卡片（秒回）；勾選 MiniCPM 按「判讀」，等數十秒後出現 AI 版卡片與 🧠 badge。

- [ ] **Step 3: 跑全部測試 + commit**

```bash
uv run pytest tests/ -v && uv run ruff check . --fix && uv run ruff format .
git add app.py
git commit -m "feat(app): LLM interpretation toggle, model warm-up, fix launch theme kwarg"
```

- [ ] **Step 4: 開 Day 1 PR 並合併**

```bash
git push -u origin feat/day1-minicpm
gh pr create --fill && gh pr merge --squash --delete-branch
git checkout main && git pull
```

---

## Day 2（6/12）— 即時資料 + 部署上線

### Task 5: sounding.py 公開 Wyoming dataframe 入口

**Files:**
- Modify: `skyread/sounding.py:44`（`_from_wyoming_dataframe` 改名公開）

- [ ] **Step 1: 開分支、改名**

```bash
git checkout -b feat/day2-live-deploy
```

`skyread/sounding.py` 中 `_from_wyoming_dataframe` 改名為 `from_wyoming_dataframe`（拿掉底線），並更新檔內兩個呼叫點（`load_csv`、`load_sample`）。docstring 不變。

- [ ] **Step 2: 跑測試 + commit**

Run: `uv run pytest tests/ -v`
Expected: 全綠

```bash
git add skyread/sounding.py
git commit -m "refactor(sounding): expose from_wyoming_dataframe for live-data module"
```

---

### Task 6: skyread/live.py — 板橋即時探空

**Files:**
- Create: `skyread/live.py`
- Test: `tests/test_live.py`
- Modify: `pyproject.toml`（siphon）

- [ ] **Step 1: 加依賴**

```bash
uv add siphon
```

- [ ] **Step 2: 寫失敗測試（純時間邏輯，不碰網路）**

建立 `tests/test_live.py`：

```python
"""Tests for live-sounding time logic (network calls are not tested here)."""

from datetime import datetime, timezone

from skyread.live import _latest_synoptic


def test_latest_synoptic_morning_rounds_to_00z() -> None:
    now = datetime(2026, 6, 11, 3, 30, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 0, 0, tzinfo=timezone.utc)


def test_latest_synoptic_afternoon_rounds_to_12z() -> None:
    now = datetime(2026, 6, 11, 15, 0, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)


def test_latest_synoptic_exactly_noon_is_12z() -> None:
    now = datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
    assert _latest_synoptic(now) == datetime(2026, 6, 11, 12, 0, tzinfo=timezone.utc)
```

- [ ] **Step 3: 跑測試確認失敗**

Run: `uv run pytest tests/test_live.py -v`
Expected: FAIL（模組不存在）

- [ ] **Step 4: 實作 skyread/live.py**

```python
"""Fetch the latest real sounding from the University of Wyoming archive.

Network access happens only here. Any failure should be caught by the caller
(the app falls back to bundled examples), so a dead upstream never kills a demo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from siphon.simplewebservice.wyoming import WyomingUpperAir

from skyread.sounding import Sounding, from_wyoming_dataframe

BANQIAO = "46692"  # Taipei / Banqiao upper-air station

_COLUMNS = ["pressure", "height", "temperature", "dewpoint", "direction", "speed"]


def _latest_synoptic(now: datetime) -> datetime:
    """Round ``now`` down to the most recent 00Z/12Z synoptic hour."""
    base = now.replace(minute=0, second=0, microsecond=0)
    return base.replace(hour=12) if base.hour >= 12 else base.replace(hour=0)


def latest_sounding(station: str = BANQIAO, max_lookback: int = 4) -> Sounding:
    """Fetch the most recent sounding, stepping back 12 h per attempt.

    Args:
        station: WMO station identifier.
        max_lookback: How many 12-hourly synoptic times to try.

    Returns:
        The parsed :class:`Sounding`, named ``"<station> <time>Z"``.

    Raises:
        RuntimeError: If no sounding exists within the lookback window.
    """
    candidate = _latest_synoptic(datetime.now(timezone.utc))
    for _ in range(max_lookback):
        try:
            df = WyomingUpperAir.request_data(
                candidate.replace(tzinfo=None), station
            )
        except ValueError:  # Wyoming returns this when the hour has no data yet
            candidate -= timedelta(hours=12)
            continue
        name = f"{station} {candidate:%Y-%m-%d %H}Z"
        return from_wyoming_dataframe(df[_COLUMNS], name=name)
    raise RuntimeError(
        f"No sounding for station {station} in the last {max_lookback * 12} hours"
    )
```

- [ ] **Step 5: 跑測試確認通過**

Run: `uv run pytest tests/ -v`
Expected: 全綠（19 passed）

- [ ] **Step 6: 真實網路煙霧測試**

```bash
uv run python -c "
from skyread.live import latest_sounding
from skyread.indices import compute_indices
snd = latest_sounding()
print(snd.name, len(snd.pressure), 'levels')
print(compute_indices(snd))
"
```

Expected: 印出 `46692 2026-06-1x ..Z`、合理層數與指數。若 Wyoming 暫時掛掉，重試一次；持續失敗也沒關係——app 層會 fallback，繼續往下做。

- [ ] **Step 7: commit**

```bash
uv run ruff check . --fix && uv run ruff format .
git add skyread/live.py tests/test_live.py pyproject.toml uv.lock
git commit -m "feat(live): fetch latest Banqiao 46692 sounding from Wyoming archive"
```

---

### Task 7: app.py 加入「即時探空」資料來源

**Files:**
- Modify: `app.py`

- [ ] **Step 1: 改 analyze 與 UI 支援三種來源**

`app.py` 中 `EXAMPLES` 之後加上常數，並改寫 `analyze` 上半部與 `build_ui` 的輸入區：

```python
SOURCE_LIVE = "🛰️ 即時探空（台北板橋 46692）"
SOURCE_EXAMPLE = "📚 經典個案"
SOURCE_UPLOAD = "📄 上傳 CSV"


def _load_sounding(source: str, example_label: str, uploaded: str | None) -> Sounding:
    """Resolve the selected data source into a parsed Sounding."""
    if source == SOURCE_LIVE:
        return latest_sounding()
    if source == SOURCE_UPLOAD:
        if not uploaded:
            raise ValueError("請先上傳 CSV 檔")
        return load_csv(uploaded, name="uploaded")
    return load_sample(EXAMPLES[example_label])


def analyze(source: str, example_label: str, uploaded: str | None, use_llm: bool):
    """Run the full chain and return (figure, pro_md, grandma_md, badge_md)."""
    try:
        snd = _load_sounding(source, example_label, uploaded)
    except Exception as exc:  # network/parse errors surface to the user
        return None, f"⚠️ 讀取失敗：{exc}（可改選經典個案）", "", ""

    indices = compute_indices(snd)
    if use_llm:
        cards, engine = interpret_llm(indices, snd.name)
    else:
        cards, engine = interpret_rule_based(indices, snd.name), "rule-based"
    badge = _BADGE_LLM if engine == "minicpm" else _BADGE_RULE
    return make_skewt(snd), cards["pro"], cards["grandma"], badge


def _analyze_fast(source: str, example_label: str, uploaded: str | None):
    """Instant first paint on page load: skip the LLM, show rule-based cards."""
    return analyze(source, example_label, uploaded, use_llm=False)
```

`build_ui` 輸入欄改為：

```python
source = gr.Radio(
    choices=[SOURCE_LIVE, SOURCE_EXAMPLE, SOURCE_UPLOAD],
    value=SOURCE_EXAMPLE,
    label="資料來源",
)
example = gr.Dropdown(
    choices=list(EXAMPLES), value=list(EXAMPLES)[0], label="範例探空"
)
upload = gr.File(
    label="探空 CSV (pressure,temperature,dewpoint,direction,speed)",
    file_types=[".csv"],
    type="filepath",
)
```

事件綁定改為：

```python
btn.click(
    analyze,
    inputs=[source, example, upload, use_llm],
    outputs=[plot, pro, grandma, badge],
)
demo.load(_analyze_fast, inputs=[source, example, upload], outputs=[plot, pro, grandma, badge])
```

`_analyze_fast` 簽名同步改成 `(source, example_label, uploaded)` 並轉呼叫 `analyze(source, example_label, uploaded, use_llm=False)`。記得 import：`from skyread.live import latest_sounding`。

- [ ] **Step 2: 手動驗證三條路**

Run: `uv run python app.py`
Expected: 預設經典個案秒回；切「即時探空」按判讀，出現板橋當日資料（標題含 `46692`）；上傳壞 CSV 顯示 ⚠️ 而不崩。

- [ ] **Step 3: 測試 + commit**

```bash
uv run pytest tests/ -v && uv run ruff check . --fix && uv run ruff format .
git add app.py
git commit -m "feat(app): add live Banqiao sounding as a data source with graceful fallback"
```

---

### Task 8: 部署到 HF Space（hackathon org 底下）

**Files:**
- Create: `requirements.txt`
- Modify: `README.md`（加 Space YAML frontmatter）

- [ ] **Step 1: 產出 requirements.txt**

```bash
uv export --format requirements-txt --no-hashes --no-dev --no-emit-project -o requirements.txt
```

- [ ] **Step 2: README 加 Space frontmatter**

`README.md` 開頭加（正文 Day 3 補）：

```markdown
---
title: SkyRead 探空白話判讀器
emoji: 🌤️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
---

# 🌤️ SkyRead — 探空白話判讀器

Turn Skew-T soundings into plain-language weather advice.
MetPy computes every number; a 0.5B MiniCPM only narrates.
```

- [ ] **Step 3: commit + PR 合併**

```bash
git add requirements.txt README.md
git commit -m "chore(deploy): requirements export and HF Space metadata"
git push -u origin feat/day2-live-deploy
gh pr create --fill && gh pr merge --squash --delete-branch
git checkout main && git pull
```

- [ ] **Step 4: 建立 Space 並上傳**

先確認登入與 org 名（hackathon org 的確切 slug 到 https://huggingface.co/build-small-hackathon 確認，下面以 `build-small-hackathon` 為例）：

```bash
hf auth whoami   # 未登入則 hf auth login
hf repo create build-small-hackathon/skyread --repo-type space --space-sdk gradio
hf upload build-small-hackathon/skyread . . --repo-type space \
  --exclude ".git/*" --exclude ".venv/*" --exclude "docs/*" --exclude "uv.lock" --exclude ".python-version"
```

**人工檢查點（需要 Eric）：** 若 org 內無建 repo 權限，改到個人帳號下建 Space，提交時再轉移或依大會指示處理。

- [ ] **Step 5: Space 煙霧測試**

開 `https://huggingface.co/spaces/<org>/skyread`，等 build 完成。
Expected: 頁面載入秒出規則式卡片；勾 MiniCPM 判讀，首次因下載模型較慢（warm-up thread 會先跑），之後約 30-60 秒內回。免費 CPU 太慢的備案：Space Settings 把硬體升級到 CPU Upgrade（用大會發的 $20 HF credits）。

---

## Day 3（6/13）— 打磨門面 + 真實使用者證據

### Task 9: README 正文 + 截圖

**Files:**
- Modify: `README.md`
- Create: `docs/screenshot.png`

- [ ] **Step 1: 開分支、本機跑 app 截圖**

```bash
git checkout -b feat/day3-polish
uv run python app.py
```

瀏覽器截一張「Skew-T + 雙層卡片」全景圖，存成 `docs/screenshot.png`。

- [ ] **Step 2: README 正文（frontmatter 之後全部換成）**

```markdown
# 🌤️ SkyRead — 探空白話判讀器

> 把艱深的 Skew-T 探空圖，翻成**同行看的指數**與**阿嬤看的帶傘建議**。

![SkyRead screenshot](docs/screenshot.png)

## Why

每天全球施放上千顆探空氣球，但讀懂一張 Skew-T 需要多年訓練。
SkyRead 把它變成兩張卡片：給氣象同行的指數摘要，和給長輩的
「要不要帶傘、能不能曬棉被」。

## The honest small-model architecture

| 層 | 負責 | 由誰做 |
|----|------|--------|
| 數值 | CAPE/CIN、LCL/LFC/EL、K、LI、TT、PWAT | **MetPy**（確定性計算，AI 不碰數字） |
| 語言 | 把數字改寫成兩種受眾的人話 | **MiniCPM4-0.5B**（本機 CPU 推論） |
| 保險 | 模型失敗時的判讀 | 規則式 fallback（同時是 LLM 的草稿） |

0.5B 模型算不準 CAPE——所以我們不讓它算。它只做小模型真正擅長的事：
把一份數值正確的草稿改寫成自然的人話。

## Data sources

- 🛰️ 即時探空：台北板橋 46692（University of Wyoming archive）
- 📚 經典個案：MetPy 內建（含 1999-05-04 Oklahoma tornado outbreak）
- 📄 上傳 CSV：`pressure,temperature,dewpoint,direction,speed`（hPa/°C/deg/kt）

## Run locally

```bash
uv sync
uv run python app.py        # Gradio UI at http://127.0.0.1:7860
uv run python -m skyread.spike   # CLI end-to-end demo
uv run pytest tests/ -v
```

## Built for

Hugging Face **Build Small Hackathon 2026** — Backyard AI track.
```

- [ ] **Step 3: commit**

```bash
git add README.md docs/screenshot.png
git commit -m "docs: README with architecture story and screenshot"
```

---

### Task 10: 真實使用者證據 + Field Notes（merit badges）

**Files:**
- Modify: `README.md`（加 user 段落）

- [ ] **Step 1: 真實使用者測試（需要 Eric 本人，30 分鐘）**

把 Space 連結傳給至少一位非氣象背景的家人/朋友（最好真的是長輩），請對方看「生活版」卡片回答：「今天會不會下雷雨？要帶傘嗎？」記錄：(a) 對方原話引述、(b) 一張使用畫面照片（徵得同意）、(c) 看不懂的地方——當天就修。

- [ ] **Step 2: README 加使用者段落（引述換成真實內容）**

```markdown
## Real users

> 「喔～這個我看得懂，下午會打雷，棉被要收。」— Eric 的家人，第一次使用

SkyRead 的生活版卡片由真實長輩驗收：看一眼就知道要不要帶傘，才算通過。
```

- [ ] **Step 3: Field Notes 文章（badge 加分）**

在 HF 個人頁發一篇短文（Community blog / hackathon 指定處），結構照抄：

1. **為什麼選這個題目** — 家裡長輩看不懂氣象 App 的雷達圖，但聽得懂「帶傘」
2. **小模型的誠實用法** — 0.5B 不會算 CAPE，所以 MetPy 算、模型只改寫草稿；附 fallback 設計
3. **翻車記錄** — 0.5B 調教過程（溫度、繁中、格式遵循）真實記錄
4. **學到什麼** — 「draft-rewrite 比 free-generation 可靠」一段

- [ ] **Step 4: commit + PR 合併 + 重新部署**

```bash
git add README.md
git commit -m "docs: add real-user evidence section"
git push -u origin feat/day3-polish
gh pr create --fill && gh pr merge --squash --delete-branch
git checkout main && git pull
hf upload build-small-hackathon/skyread . . --repo-type space \
  --exclude ".git/*" --exclude ".venv/*" --exclude "docs/*" --exclude "uv.lock" --exclude ".python-version"
```

（注意 `docs/*` 被 exclude，screenshot 不會上 Space——Space README 的圖改用 HF 上傳後的絕對 URL，或把截圖也傳上去：`hf upload build-small-hackathon/skyread docs/screenshot.png docs/screenshot.png --repo-type space`。）

---

## Day 4（6/14）— Demo 影片 + 提交（6/15 留作緩衝）

### Task 11: Demo 影片（60–90 秒）

- [ ] **Step 1: 照腳本錄製（螢幕錄影 + 旁白，QuickTime 即可）**

| 秒數 | 畫面 | 旁白（中文錄製，影片加英文字幕） |
|------|------|------|
| 0-10 | 一張密密麻麻的 Skew-T 圖 | 「這張圖，氣象系要學三年。我阿嬤只想知道：今天要不要帶傘？」 |
| 10-25 | 打開 Space，按「即時探空」判讀 | 「SkyRead 抓下今天台北的探空氣球資料，MetPy 算出所有對流指數。」 |
| 25-45 | 鏡頭停在兩張卡片，badge 入鏡 | 「然後一顆 0.5B 的 MiniCPM——不算任何數字，只把數字講成人話。同行版給專業的，生活版給阿嬤。」 |
| 45-60 | 真實長輩看手機的畫面（Task 10 素材） | 「這是我家人第一次用：看一眼，就知道棉被要收。」 |
| 60-75 | 切到 1999 Oklahoma 個案，CAPE 爆表畫面 | 「換成 1999 年龍捲風爆發那天的探空——連阿嬤版都會叫你別出門。」 |
| 75-90 | README 架構表 | 「小模型算不準大氣物理？沒關係，我們從沒讓它算。Built small, honestly.」 |

- [ ] **Step 2: 上傳影片**（YouTube unlisted 或大會指定平台），連結記下來。

### Task 12: 社群貼文 + 正式提交

- [ ] **Step 1: 社群貼文（X / LinkedIn / HF，內容直接用）**

> My grandma can't read a Skew-T diagram. Neither can a 0.5B model — so I never ask it to. 🌤️
>
> SkyRead: MetPy computes every convective index exactly; MiniCPM4-0.5B only rewrites the numbers into plain language — one card for meteorologists, one for grandma ("bring an umbrella, don't sun the quilt").
>
> Live soundings from Taipei, runs on free CPU, falls back gracefully.
> Built for @huggingface #BuildSmallHackathon 🔗 [Space 連結]

- [ ] **Step 2: 對照大會提交表單送出**

提交 checklist：
- [ ] Space 在 hackathon org 下、public、能跑
- [ ] Demo 影片連結
- [ ] 社群貼文連結
- [ ] Track 選 **Backyard AI**
- [ ] Merit badges 自評：Off the Grid（模型在 Space 本機推論、無外部 API）、Field Notes（Task 10 文章）、Sharing is Caring（依大會定義檢查，可能是開源 + 貼文即符合）

- [ ] **Step 3: 提交後在本 repo 打 tag**

```bash
git tag -a hackathon-submission -m "Build Small Hackathon submission (2026-06-14)"
git push origin hackathon-submission
```

---

## 風險與備案

| 風險 | 機率 | 備案 |
|------|------|------|
| MiniCPM4-0.5B 繁中品質不行 | 中 | 換 `openbmb/MiniCPM3-4B`（仍 ≪32B；CPU 慢 → 升級 Space 硬體或量化），改 `SKYREAD_MODEL_ID` 環境變數即可，程式不用動 |
| Wyoming 資料源掛掉 | 低 | app 已 graceful fallback 到經典個案；demo 影片提前錄好即時段 |
| 免費 CPU Space 推論太慢（>90s） | 中 | $20 HF credits 升級 CPU Upgrade；再不行 demo 用本機錄影 |
| org 內無建 Space 權限 | 低 | 個人帳號建，依大會指示轉移 |
| 未完成報名（6/3 已截止） | — | **開工前先確認**，否則整個計畫作廢 |
