"""SkyRead — Gradio app: sounding -> Skew-T plot + dual-layer interpretation.

Run locally:
    uv run python app.py
"""

from __future__ import annotations

import threading

import gradio as gr

from skyread.indices import compute_indices
from skyread.interpret import interpret_rule_based
from skyread.live import STATIONS, latest_sounding
from skyread.llm import MODEL_ID, interpret_llm, warm_up
from skyread.plot import make_skewt
from skyread.sounding import Sounding, load_csv, load_sample

# Curated, demo-safe example soundings bundled with MetPy (zero network).
EXAMPLES: dict[str, str] = {
    "1999-05-04 Oklahoma (強對流 / tornado outbreak)": "may4_sounding.txt",
    "2010-01-20 winter case": "jan20_sounding.txt",
    "2011-11-11 case": "nov11_sounding.txt",
}

SOURCE_LIVE = "🛰️ 即時探空（鄰近測站）"
SOURCE_EXAMPLE = "📚 經典個案"
SOURCE_UPLOAD = "📄 上傳 CSV"

_MODEL_NAME = MODEL_ID.split("/")[-1]
_BADGE_LLM = (
    f"🧠 生活版由 **{_MODEL_NAME}**（本機推論）改寫；"
    "同行版與所有數值由 MetPy 確定性計算。"
)
_BADGE_RULE = "📐 規則式判讀（fallback）；所有數值由 MetPy 確定性計算。"


def _load_sounding(
    source: str, station_label: str, example_label: str, uploaded: str | None
) -> Sounding:
    """Resolve the selected data source into a parsed Sounding."""
    if source == SOURCE_LIVE:
        return latest_sounding(STATIONS[station_label])
    if source == SOURCE_UPLOAD:
        if not uploaded:
            raise ValueError("請先上傳 CSV 檔")
        return load_csv(uploaded, name="uploaded")
    return load_sample(EXAMPLES[example_label])


def analyze(
    source: str,
    station_label: str,
    example_label: str,
    uploaded: str | None,
    use_llm: bool,
):
    """Run the full chain and return (figure, pro_md, grandma_md, badge_md)."""
    try:
        snd = _load_sounding(source, station_label, example_label, uploaded)
    except Exception as exc:  # network/parse errors surface to the user
        return None, f"⚠️ 讀取失敗：{exc}（可改選經典個案）", "", ""

    indices = compute_indices(snd)
    if use_llm:
        cards, engine = interpret_llm(indices, snd.name)
    else:
        cards, engine = interpret_rule_based(indices, snd.name), "rule-based"
    badge = _BADGE_LLM if engine == "llm" else _BADGE_RULE
    return make_skewt(snd), cards["pro"], cards["grandma"], badge


def _analyze_fast(
    source: str, station_label: str, example_label: str, uploaded: str | None
):
    """Instant first paint on page load: skip the LLM, show rule-based cards."""
    return analyze(source, station_label, example_label, uploaded, use_llm=False)


def build_ui() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="SkyRead 探空白話判讀器") as demo:
        gr.Markdown(
            "# 🌤️ SkyRead — 探空白話判讀器\n"
            "把艱深的 Skew-T 探空圖，翻成**同行看的指數**與**阿嬤看的帶傘建議**。\n"
            "_數值由 MetPy 精確計算，AI 只負責把數字講成人話。_"
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(
                    choices=[SOURCE_LIVE, SOURCE_EXAMPLE, SOURCE_UPLOAD],
                    value=SOURCE_EXAMPLE,
                    label="資料來源",
                )
                station = gr.Dropdown(
                    choices=list(STATIONS),
                    value=list(STATIONS)[0],
                    label="即時測站（台灣探空未開放於 Wyoming 資料庫，取最近測站）",
                )
                example = gr.Dropdown(
                    choices=list(EXAMPLES), value=list(EXAMPLES)[0], label="範例探空"
                )
                upload = gr.File(
                    label="探空 CSV (pressure,temperature,dewpoint,direction,speed)",
                    file_types=[".csv"],
                    type="filepath",
                )
                use_llm = gr.Checkbox(
                    value=True,
                    label=f"🧠 用 {_MODEL_NAME} 潤飾生活版（慢幾秒，但更像人話）",
                )
                btn = gr.Button("判讀 ☁️", variant="primary")
            with gr.Column(scale=1):
                plot = gr.Plot(label="Skew-T / Log-P")
        pro = gr.Markdown()
        grandma = gr.Markdown()
        badge = gr.Markdown()

        btn.click(
            analyze,
            inputs=[source, station, example, upload, use_llm],
            outputs=[plot, pro, grandma, badge],
        )
        demo.load(
            _analyze_fast,
            inputs=[source, station, example, upload],
            outputs=[plot, pro, grandma, badge],
        )
    return demo


if __name__ == "__main__":
    threading.Thread(target=warm_up, daemon=True).start()
    build_ui().launch(theme=gr.themes.Soft())
