# 提交 Checklist（截止 2026-06-15）

Track: **Backyard AI**

## 硬性要求
- [ ] Space 建在 hackathon org 下、public、能跑
      （`hf auth login` → `hf repo create <org>/skyread --repo-type space --space-sdk gradio` → `hf upload <org>/skyread . . --repo-type space --exclude ".git/*" --exclude ".venv/*" --exclude "docs/*" --exclude "uv.lock" --exclude ".python-version"`）
- [ ] 模型 ≤ 32B ✅（Qwen3-0.6B）
- [ ] Gradio app ✅
- [ ] Demo 影片連結（腳本：video-script.md）
- [ ] 社群貼文連結（文案：social-post.md）
- [ ] 提交表單送出

## 加分項（merit badges 自評）
- [ ] Off the Grid — 模型在 Space 本機推論、無外部 API（已符合，提交時勾選）
- [ ] Field Notes — 開發筆記文章（草稿：docs/field-notes.md，發到 HF blog）
- [ ] Sharing is Caring — 依大會定義確認（開源 repo + 貼文可能即符合）

## 真實使用者證據（評審標準「actual user adoption」）
- [ ] 至少一位非氣象背景家人/朋友實際使用
- [ ] 記錄：原話引述 + 使用畫面照片（徵得同意）
- [ ] 寫進 README「Real users」段落，重新部署

## 提交後
- [ ] `git tag -a hackathon-submission -m "Build Small Hackathon submission" && git push origin hackathon-submission`

## 部署備註
- 預設模型已是 Qwen3-0.6B（2026-06-11 GPU 驗證後定案,MiniCPM3-4B
  繁中品質不過關——詳見 PROGRESS.md）,免費 CPU Space 預期可跑。
  若實測仍慢,備案：
  1. Space 設定升級 CPU Upgrade（用大會發的 $20 HF credits）
  2. 申請 ZeroGPU（org 可能有 grant）
