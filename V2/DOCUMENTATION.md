# V2 Documentation — Text-as-Image Efficiency Study

Kaggle-notebook-first clean restart. Implements root `AGENTS.md`
(§1 research question, §4 A/B design, §9 token accounting, §13 env logging,
§21 JSONL schema) for open-weight MLLMs. You have no local GPU — all model
runs happen on Kaggle; this repo only stores code + result logs.

## 1. Why V2 exists (vs V1)

V1 pilot (`src/`, root `*.ipynb`, `results/`) is untouched legacy. Do not
build on its numbers: unfair prompts between arms, `prompt_eval_count`
accounting, fake visual-budget control. V2 enforces SAME document /
SAME question / SAME decoding, real processor-based `m/q/k` accounting,
and labels the renderer used in every log row.

## 2. Layout

```
V2/
  DOCUMENTATION.md          # this file
  README.md                 # short run-book (how to run on Kaggle)
  requirements-kaggle.txt   # version pins for local linting; in-notebook pip cells are authoritative
  notebooks/
    00_setup_env.ipynb          # Env log. Run first. No weights, no HF.
    01_renderer_validation.ipynb# ref-port renderer + no-crop checks. Run second.
    03_openrouter_qwen38.ipynb  # Qwen3.8-27B-free via OpenRouter, retrieval + CNN/DM (accuracy-only)
    04_ollama_gemma4.ipynb      # Gemma4-31B via Ollama Cloud API, retrieval + CNN/DM (accuracy-only)
  modules/
    renderer.py             # .py mirror of 01's PIL renderer (git diff only, never imported on Kaggle)
  results/raw/              # JSONL logs copied back from /kaggle/working/results/raw/. Never hand-edit.
```

Notebooks are **standalone top-to-bottom** — each redefines what it needs
so Kaggle "Run All" works without imports from `modules/`.

## 3. Notebooks

### 00 — Environment setup (no weights, no HF)
1. System deps: `poppler-utils` for `pdf2image`; `tectonic` optional.
   Missing tectonic → 01 uses the PIL fallback (labeled in logs).
2. Python deps: `openai pillow pandas matplotlib rouge-score datasets`.
   No torch/transformers — nothing runs locally.
3. §13 env header: timestamp, python/platform, run-mode flag,
   `nvidia-smi`. Paste into every result log.

### 01 — Renderer validation (no model)
Ports `yanhong-lbh/text_or_pixels:text_to_image.py` (EMNLP 2025 Findings
558, arXiv 2510.18279): NFKC normalize → LaTeX-escape → Tectonic PDF →
`pdf2image` @300 DPI → fixed W×H + font-size search to fill_ratio≈0.85
with a single-page assert. PIL direct-draw (`render_pil`) is the Kaggle
fallback when tectonic/poppler are absent.
Validation renders short (~350w) / med (~1500w) / long (~3000w) synthetic
S-NIAH contexts at 2 canvas widths, asserts content bbox doesn't touch
the image edge (margin ≥ 4px, fill_ratio > 0.05), saves proof PNGs to
`/kaggle/working/results/raw/val_*.png`. Gate passes only if all renders
pass — fix the renderer before running 03.

### 03 — OpenRouter Qwen3.8-27B-free (accuracy-only, retrieval + summarization)
Config: `MODEL_ID="qwen/qwen3.8-27b:free"`, 2 budgets (canvas 600/1100)
and 3 retrieval contexts (short/med/long), plus
`N_SUMM=4` CNN/DailyMail articles truncated to 800 words. Cell 2 loads the
key from Secret `OPENROUTER_API_KEY` and burns 1 cheap text-only call as a
reachability check. Thinking is OFF (`reasoning: {"enabled": False}` —
verified 2026-10-03: with thinking on, tiny `max_tokens` go entirely to the
`reasoning` field and `content` comes back null). Images go as JPEG-72 base64 `image_url` parts
(downscaled to max_h 2600, sent size logged). Retrieval logs 6 paired rows
(hit/F1, API usage tokens both arms; `m/q/k` and `CR_*` stay null by design). Summarization logs
`N_SUMM` rows with ROUGE-1/2/L text-vs-image. `chat()` backs off on
free-tier 429/502. Latency/VRAM columns are `null` by design — never compare
them with local runs.

### 04 — Gemma4-31B via Ollama Cloud API (accuracy-only, retrieval + summarization)
No download, no server, no GPU. Calls `https://ollama.com/api/chat` with
`OLLAMA_MODEL="gemma4:31b"` (dense 31B, 256K ctx — verified visible via
`/api/tags` 2026-10-03). Cell 2 reads `OLLAMA_API_KEY` and burns 1 cheap
text-only call as reachability check. Same A/B harness as `03` (same seeds →
same documents, same questions, `temperature=0 / num_predict=256`, JPEG-72
payloads): 6 paired retrieval rows (hit/F1, provider `prompt_eval_count` per
arm) + `N_SUMM=4` CNN/DM rows (ROUGE-1/2/L). `ollama_chat()` backs off on
transient errors. Latency/VRAM columns are `null` by design.

## 4. Fairness, token accounting, metrics

- Fairness: SAME document, SAME question, SAME `temperature=0 /
  max_tokens=256`. Only the context representation changes.
  Summarization uses one fixed instruction (`QSUMM`) for both arms.
- Accounting (cloud-only): provider-reported tokens per arm, logged in every
  row — OpenRouter `usage.prompt/completion_tokens` (`03`), Ollama Cloud
  `prompt_eval_count` / `eval_count` (`04`). `m/q/k` and `CR_*` stay null by
  design — never estimated, never compared across runs. Tall renders are
  JPEG-72 + downscaled (max_h 2600) — sent size logged.
- Retrieval metrics: exact hit (`norm(gt) in norm(pred)`) + token-F1
  (`norm` = lowercase, strip punctuation, collapse whitespace).
- Summarization metrics: ROUGE-1/2/L (F-measure, `rouge-score` + stemmer)
  text-vs-image on the same articles.

## 5. Model policy (cloud-only since 2026-10-03; local arms removed/parked)

- Qwen arm: `qwen/qwen3.8-27b:free` via OpenRouter (`03`, accuracy-only) —
  verified live 2026-10-03 (`GET /api/v1/models`: price $0/$0, 262144 ctx,
  `text+image+video->text`). Key comes from Kaggle Secret
  `OPENROUTER_API_KEY`, never hardcoded. Free tier ≈ 200 req/day;
  default notebook = ~18 calls (1 verify + 9 retrieval + 8 summ).
  No paid fallback.
- Gemma arm: `gemma4:31b` via Ollama Cloud API (`04`, accuracy-only) —
  dense 31B, 256K ctx, Text+Image in; verified visible via `/api/tags`
  2026-10-03. Key comes from Kaggle Secret `OLLAMA_API_KEY` (ollama.com →
  Settings → API keys), never hardcoded. Same harness/seeds as `03` for a
  fair cross-model comparison. Cloud quota unverified — if limited, lower
  `N_SUMM`; fallback is `google/gemma-4-31b-it:free` on OpenRouter
  (verified $0) with zero harness changes.
- No weights are downloaded anywhere in the current path. Parked/removed:
  local Ollama Gemma and the deleted local-HF Qwen3.5 arm (`02` notebook +
  `modules/qwen_backend.py` removed 2026-10-03). Per AGENTS.md, this scoping
  change is recorded here rather than made silently.

## 6. How to run on Kaggle

1. New Kaggle notebook, internet ON (GPU optional — nothing downloads or runs
   locally; both models are hosted APIs).
   Add-ons → Secrets: `OPENROUTER_API_KEY` (for `03`), `OLLAMA_API_KEY`
   (for `04`). Attach both to the notebook.
2. Clone the repo on the `Sayok` branch (or skip — every notebook's Cell 0
   clones to `/kaggle/working/Testing-dip` if missing, else `git pull`):
   `!git clone -b Sayok https://github.com/iamahnaf/Testing-dip.git`
   Alternative: upload a single `V2/notebooks/*.ipynb` via File → Upload —
   notebooks are standalone and run without the rest of the tree.
3. Run `00` → `01` → `03` + `04` in any order, each Run-All top-to-bottom.
4. After each run: Save Version (persists `/kaggle/working/`), then copy
   `/kaggle/working/results/raw/*.jsonl` (+ `val_*.png` proofs) back to
   `V2/results/raw/`. Raw logs are append-only source of truth —
   aggregates/figures derive from them, never hand-edits.

## 7. Current status and what's missing

Done: 00 env check (HF-free), 01 renderer+no-crop gate, 03 OpenRouter
Qwen3.8-free retrieval (6 paired rows) + CNN/DM summarization (`N_SUMM=4`
default, ROUGE-1/2/L), 04 Ollama Cloud Gemma4-31B retrieval + summarization
(same harness/seeds as 03). Helper logic smoke-tested locally; all code
cells compile. `V2/results/raw/` is empty — no Kaggle run logged yet.
Removed 2026-10-03: local-HF Qwen3.5 arm (02 notebook + mirror deleted).
Not yet built: analysis script (Tables 3/4, Figs 2–4 from raw JSONL only),
BABILong reasoning.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `tectonic missing` | Expected on locked images; PIL fallback runs, rows log `renderer=pil-fallback` |
| 429 / retries exhausted (03) | Free-tier quota — lower `N_SUMM`, retry tomorrow; no paid fallback |
| 502/503 on `:free` (03) | Flaky free upstream — notebook already backs off + retries |
| Cloud chat 401/403 (04) | Bad/revoked key — recheck `OLLAMA_API_KEY` at ollama.com → Settings → API keys |
| Cloud chat retries exhausted (04) | Quota or model busy — lower `N_SUMM`, retry later |
| Lost `/kaggle/working/` outputs | Save Version before closing; copy JSONL back to repo |
