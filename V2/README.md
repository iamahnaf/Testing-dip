# V2 — Text-as-Image Efficiency Study (Kaggle-notebook-first)

Clean restart. V1 pilot (`src/`, root `*.ipynb`, `results/`) is untouched legacy —
do not build on its numbers (unfair prompts, `prompt_eval_count` accounting,
fake visual-budget control). See root `AGENTS.md` for the study spec.

## Layout

```
V2/
  README.md                  # this file — how to run on Kaggle
  requirements-kaggle.txt    # Kaggle pip installs (also installed in-notebook)
  notebooks/
    00_setup_env.ipynb           # Env log. Run first. No weights, no HF.
    01_renderer_validation.ipynb # Ref-port renderer + no-crop checks. Run second.
    03_openrouter_qwen38.ipynb   # Qwen3.8-27B-free via OpenRouter (retrieval + CNN/DM). Needs OPENROUTER_API_KEY.
    04_ollama_gemma4.ipynb       # Gemma4-31B via Ollama Cloud API (retrieval + CNN/DM). Needs OLLAMA_API_KEY.
  modules/                   # .py mirrors for git diff only (never imported on Kaggle)
  results/raw/               # JSONL logs (Kaggle: /kaggle/working/results/raw/)
```

## How to run on Kaggle (you have no local GPU — and you don't need one)

1. Create a Kaggle notebook with Internet ON (GPU optional — nothing downloads
   or runs locally; both models are hosted APIs).
2. Clone the repo (branch `Sayok`, where the V2 work lives) — or skip this,
   every notebook's Cell 0 does it for you and pulls latest on re-run:
   `!git clone -b Sayok https://github.com/iamahnaf/Testing-dip.git`
   (lands in `/kaggle/working/Testing-dip`; alternatively upload a single
   `V2/notebooks/*.ipynb` via File → Upload — notebooks run standalone).
3. Open `V2/notebooks/00_setup_env.ipynb` → Run All.
   - Set Kaggle Secrets `OPENROUTER_API_KEY` (for `03`) and `OLLAMA_API_KEY`
     (ollama.com → Settings → API keys, for `04`); attach both (Add-ons → Secrets).
4. Run `01`, then `03` and `04` in any order. Each notebook is **standalone top-to-bottom**.
5. After each run: Kaggle → Save Version (persists `/kaggle/working/` outputs).
   Copy `/kaggle/working/results/raw/*.jsonl` back to `V2/results/raw/`
   for analysis — never hand-edit them.

## Model policy (cloud-only since 2026-10-03; local arms removed/parked)

- Qwen arm: `qwen/qwen3.8-27b:free` via OpenRouter (`03`,
  verified 2026-10-03: $0 in/out, 262K ctx, vision in, ~200 req/day free tier;
  default run = ~18 calls). No paid fallback.
- Gemma arm: `gemma4:31b` via Ollama Cloud API (`04`, dense 31B, 256K ctx,
  Text+Image in — verified visible via `/api/tags` 2026-10-03). Key from
  Kaggle Secret `OLLAMA_API_KEY`. Same A/B harness as `03` (same seeds,
  same questions) for a fair cross-model comparison.
- Both arms are accuracy/quality-only: latency/VRAM columns are null by design.
  Parked: local Ollama Gemma and the deleted local-HF Qwen3.5 arm.
- Reference method: https://github.com/yanhong-lbh/text_or_pixels
  (EMNLP 2025 Findings 558, arXiv 2510.18279). LaTeX→PDF→png pipeline
  ported in `01`; PIL path kept as Kaggle fallback (no apt in some images).

## Fairness / accounting rules (enforced in notebooks)

- SAME document, SAME question (`+ " Output the answer directly."`), SAME
  `temperature=0 / max_tokens=256` both arms. Only representation changes.
- Token accounting (OpenRouter-only): provider `usage.prompt_tokens` per arm,
  logged in every row. `m/q/k` and `CR_*` stay null by design — never estimated.
- Raw JSONL per Sec 21 schema first; aggregates/figures derived only from logs.
