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
    00_setup_env.ipynb           # GPU/env log + HF auth. Run first. No weights.
    01_renderer_validation.ipynb # Ref-port renderer + no-crop checks. Run second.
    02_qwen35_validation.ipynb   # Qwen3.5-9B-4bit local A/B gate. Run third.
    03_openrouter_qwen38.ipynb   # Qwen3.8-27B-free via OpenRouter (retrieval + CNN/DM). Needs OPENROUTER_API_KEY.
    04_ollama_gemma4.ipynb       # Gemma4-e4b via local Ollama server (retrieval). Needs GPU.
  modules/                   # .py mirrors for git diff only (never imported on Kaggle)
  results/raw/               # JSONL logs (Kaggle: /kaggle/working/results/raw/)
```

## How to run on Kaggle (you have no local GPU)

1. Create a Kaggle notebook with GPU enabled (T4 free tier OK) + Internet ON.
2. Clone the repo (branch `Sayok`, where the V2 work lives) — or skip this,
   every notebook's Cell 0 does it for you and pulls latest on re-run:
   `!git clone -b Sayok https://github.com/iamahnaf/Testing-dip.git`
   (lands in `/kaggle/working/Testing-dip`; alternatively upload a single
   `V2/notebooks/*.ipynb` via File → Upload — notebooks run standalone).
3. Open `V2/notebooks/00_setup_env.ipynb` → Run All.
   - Set Kaggle Secret `HF_TOKEN` (Add-ons → Secrets, then attach it) for gated models.
   - For `03`, also set Secret `OPENROUTER_API_KEY` (Add-ons → Secrets, then attach it).
4. Run `01`, then `02`/`03`/`04` in any order. Each notebook is **standalone top-to-bottom**.
5. After each run: Kaggle → Save Version (persists `/kaggle/working/` outputs;
   `/tmp` weights are intentionally NOT persisted).
   Copy `/kaggle/working/results/raw/*.jsonl` back to `V2/results/raw/`
   for analysis — never hand-edit them.

## Model policy

- Local/reproducible arm: `Qwen/Qwen3.5-9B` 4-bit (biggest Qwen3.5 that fits T4 16GB).
  No fallback. Weights download to `/tmp/hf_cache`, not the working dir.
- Local Gemma arm: `gemma4:e4b` via Ollama (`04`, 9.6GB, fits T4).
  No fallback. Weights live under `/tmp/ollama_models` (`OLLAMA_MODELS`).
  Token instrument is Ollama's `prompt_eval_count` — label `CR_decoder_ollama`,
  never mix silently with HF-processor `m/k`.
- Hosted-API arm (accuracy-only): `qwen/qwen3.8-27b:free` via OpenRouter (`03`,
  verified 2026-10-03: $0 in/out, 262K ctx, vision in, ~200 req/day free tier;
  default run = ~18 calls). No paid fallback. Never compare its latency/VRAM
  against local runs.
- Reference method: https://github.com/yanhong-lbh/text_or_pixels
  (EMNLP 2025 Findings 558, arXiv 2510.18279). LaTeX→PDF→png pipeline
  ported in `01`; PIL path kept as Kaggle fallback (no apt in some images).

## Fairness / accounting rules (enforced in notebooks)

- SAME document, SAME question (`+ " Output the answer directly."`), SAME
  `temperature=0 / max_new_tokens=128` both arms. Only representation changes.
- Token split from processor: `m` = context tokens, `q` = query/system tokens,
  `k` = decoder-visible tokens in image arm (`input_ids.shape[-1] - q`).
  Report `CR_context = m/k` AND `CR_decoder = (m+q)/(k+q)` — always label which.
- Raw JSONL per Sec 21 schema first; aggregates/figures derived only from logs.
