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

> 把艱深的 Skew-T 探空圖，翻成**同行看的指數**與**阿嬤看的帶傘建議**。

![SkyRead Skew-T](docs/screenshot.png)

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

- 🛰️ 即時探空：石垣島 47918 / 香港 45004 等鄰近測站（University of Wyoming
  archive；台灣本島測站未開放於該資料庫，故取距離最近者）
- 📚 經典個案：MetPy 內建（含 1999-05-04 Oklahoma tornado outbreak）
- 📄 上傳 CSV：`pressure,temperature,dewpoint,direction,speed`（hPa/°C/deg/kt）

## Run locally

```bash
uv sync
uv run python app.py            # Gradio UI at http://127.0.0.1:7860
uv run python -m skyread.spike  # CLI end-to-end demo
uv run pytest tests/ -v
```

## Built for

Hugging Face **Build Small Hackathon 2026** — Backyard AI track.
