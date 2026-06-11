"""SkyRead — Gradio app: sounding -> Skew-T plot + dual-layer interpretation.

Run locally:
    uv run python app.py

The interpretation currently uses the deterministic rule-based stand-in
(:func:`skyread.interpret.interpret_rule_based`). Swap in a small LLM
(e.g. OpenBMB MiniCPM) by replacing that one call.
"""

from __future__ import annotations

import gradio as gr

from skyread.indices import compute_indices
from skyread.interpret import interpret_rule_based
from skyread.plot import make_skewt
from skyread.sounding import Sounding, load_csv, load_sample

# Curated, demo-safe example soundings bundled with MetPy (zero network).
EXAMPLES: dict[str, str] = {
    "1999-05-04 Oklahoma (強對流 / tornado outbreak)": "may4_sounding.txt",
    "2010-01-20 winter case": "jan20_sounding.txt",
    "2011-11-11 case": "nov11_sounding.txt",
}


def analyze(example_label: str, uploaded: str | None):
    """Run the full chain and return (figure, pro_markdown, grandma_markdown).

    Args:
        example_label: Key into :data:`EXAMPLES`.
        uploaded: Path to an uploaded CSV, or ``None`` to use the example.

    Returns:
        Tuple of a Matplotlib figure and two Markdown strings.
    """
    try:
        snd: Sounding = (
            load_csv(uploaded, name="你的探空")
            if uploaded
            else load_sample(EXAMPLES[example_label])
        )
    except Exception as exc:  # surface parse errors to the user, don't crash
        return None, f"⚠️ 讀取失敗：{exc}", ""

    indices = compute_indices(snd)
    cards = interpret_rule_based(indices, snd.name)
    fig = make_skewt(snd)
    return fig, cards["pro"], cards["grandma"]


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
                example = gr.Dropdown(
                    choices=list(EXAMPLES), value=list(EXAMPLES)[0], label="範例探空"
                )
                upload = gr.File(
                    label="或上傳你的探空 CSV (pressure,temperature,dewpoint,direction,speed)",
                    file_types=[".csv"],
                    type="filepath",
                )
                btn = gr.Button("判讀 ☁️", variant="primary")
            with gr.Column(scale=1):
                plot = gr.Plot(label="Skew-T / Log-P")
        pro = gr.Markdown(label="同行版")
        grandma = gr.Markdown(label="生活版")

        btn.click(analyze, inputs=[example, upload], outputs=[plot, pro, grandma])
        demo.load(analyze, inputs=[example, upload], outputs=[plot, pro, grandma])
    return demo


if __name__ == "__main__":
    build_ui().launch(theme=gr.themes.Soft())
