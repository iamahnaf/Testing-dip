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
    00_setup_env.ipynb          # GPU/env log + HF auth. Run first. No weights.
    01_renderer_validation.ipynb# ref-port renderer + no-crop checks. Run second.
    02_qwen35_validation.ipynb  # Qwen3.5-9B-4bit text-vs-image gate (6 paired rows). Run third.
    03_openrouter_qwen38.ipynb  # Qwen3.8-27B-free via OpenRouter, retrieval + CNN/DM (accuracy-only)
    04_ollama_gemma4.ipynb      # Gemma4-e4b via local Ollama server, retrieval gate
  modules/
    renderer.py             # .py mirror of 01's PIL renderer (git diff only, never imported on Kaggle)
    qwen_backend.py         # .py mirror of 02's loader + norm/tok_f1 (git diff only)
  results/raw/              # JSONL logs copied back from /kaggle/working/results/raw/. Never hand-edit.
```

Notebooks are **standalone top-to-bottom** — each redefines what it needs
so Kaggle "Run All" works without imports from `modules/`.

## 3. Notebooks

### 00 — Environment setup (no weights)
1. System deps: `poppler-utils` for `pdf2image`; `tectonic` optional.
   Missing tectonic → 01/02 use the PIL fallback (labeled in logs).
2. Python deps from `transformers@main` + `accelerate bitsandbytes torchao
   qwen-vl-utils pillow pdf2image pandas matplotlib rouge-score huggingface_hub`.
3. §13 env header: timestamp, python/platform, torch/CUDA, transformers
   version, `nvidia-smi`, CUDA version. Paste into every result log.
4. HF auth via Kaggle Secret `HF_TOKEN` (graceful skip if absent; gated
   models then fail in 02 with a clear message).
5. Model-ID check: `Qwen/Qwen3.5-9B` only — prints `sha`/pipeline tag.
   No fallback: if PRIMARY is missing/renamed, stop and record it.

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
pass — fix the renderer before running 02.

### 02 — Qwen3.5-9B validation gate (downloads the model to /tmp)
Config: `MODEL_ID="Qwen/Qwen3.5-9B"`, no fallback,
2 visual budgets (`low`: max_pixels=512·28²/canvas 600px,
`high`: 1024·28²/1100px), 3 contexts (short 350w / med 1500w / long
3000w), `temperature=0, max_new_tokens=128`.
Loader **is** the downloader: `from_pretrained` pulls weights to
`/tmp/hf_cache` (kept out of the working dir; re-downloads each session). T4 rules: `fp16` (Turing has no bf16),
`sdpa` (no flash-attn-2), `TorchAoConfig int4_weight_only`,
`device_map="auto"`, batch=1. `max_pixels` cap is the visual-budget knob.
A/B loop per context: Arm A text (`Context:\n…\n\nQuestion: …`) vs Arm B
image (rendered context PNG + `Question: …` text). SAME question string
(+ `" Output the answer directly."`), SAME decoding — only the context
representation changes. Logs 6 paired rows → `$OUT =
/kaggle/working/results/raw/qwen35_val.jsonl`, then aggregates
accuracy/F1/CR/latency per budget from the JSONL. Gate passes iff 6 rows
logged, images uncropped (see 01), `k/m/CR` columns non-null.

### 03 — OpenRouter Qwen3.8-27B-free (accuracy-only, retrieval + summarization)
Config: `MODEL_ID="qwen/qwen3.8-27b:free"`, same 2 budgets (canvas 600/1100)
and 3 retrieval contexts (seeds 100+ci — SAME documents as 02/04), plus
`N_SUMM=4` CNN/DailyMail articles truncated to 800 words. Cell 2 loads the
key from Secret `OPENROUTER_API_KEY` and burns 1 cheap text-only call as a
reachability check. Images go as JPEG-72 base64 `image_url` parts
(downscaled to max_h 2600, sent size logged). Retrieval logs 6 paired rows
(hit/F1, API usage tokens both arms, local `m/q/k` when the CPU-only HF
tokenizer/processor download works, else `CR_* = null`). Summarization logs
`N_SUMM` rows with ROUGE-1/2/L text-vs-image. `chat()` backs off on
free-tier 429/502. Latency/VRAM columns are `null` by design — never compare
them with local runs.

### 04 — Gemma4 via Ollama (local GPU, retrieval gate)
Cell 2 installs the Ollama binary, starts `ollama serve` with
`OLLAMA_MODELS=/tmp/ollama_models` (log at `/tmp/ollama.log`), pulls
`gemma4:e4b` (~10GB, once, into `/tmp`). Cell 3 warm-up call
excludes model load from steady-state timing. A/B loop mirrors 02 with the
same seeds; image goes as base64 PNG in `/api/chat`. Each row logs
`prompt_eval_count` per arm, `eval_count`, server `total_duration` +
client `end_to_end_s`, and `nvidia-smi` used-memory (constant/resident is
expected). Gate: 6 rows, hit/f1 non-null; `CR_decoder_ollama` labeled with
its instrument, never silently pooled with 02's processor-based CR.

## 4. Fairness, token accounting, metrics

- Fairness: SAME document, SAME question, SAME `temperature=0 /
  max_new_tokens=128` (API: `max_tokens`; Ollama: `num_predict=128`).
  Only the context representation changes. Summarization uses one fixed
  instruction (`QSUMM`) for both arms.
- Accounting per instrument (never mix without labeling):
  - `02` (HF processor): `m` = context tokens, `q` = question tokens,
    `k` = decoder-visible image-arm tokens (`input_ids.shape[-1] − q`).
    `CR_context = m/k`, `CR_decoder = (m+q)/(k+q)`.
  - `03` (OpenRouter API): provider `usage.prompt_tokens` per arm always
    logged; local HF tokenizer/processor `m/q/k` filled when the
    CPU-only download succeeds, else `CR_* = null` (never estimated).
    Tall renders are JPEG-72 + downscaled (max_h 2600) — sent size logged.
  - `04` (Ollama): `prompt_eval_count` per arm from `/api/chat`
    (decoder-visible input length as reported by the runtime),
    `CR_decoder_ollama = prompt_eval_text / prompt_eval_image`.
    VRAM via `nvidia-smi` used-memory (server owns the GPU; expect resident).
- Retrieval metrics: exact hit (`norm(gt) in norm(pred)`) + token-F1
  (`norm` = lowercase, strip punctuation, collapse whitespace).
- Summarization deps (`rouge-score`, `bert-score`) are pre-installed for
  the planned CNN/DailyMail extension; no summarization notebook yet.

## 5. Model policy

- Local/reproducible arm: `Qwen/Qwen3.5-9B` 4-bit (biggest Qwen3.5 that
  fits T4 16GB). No fallback. Weights go to `/tmp/hf_cache`.
- Local Gemma arm (`04`): `gemma4:e4b` via Ollama (9.6GB, 128K ctx,
  Text+Image in — from the `ollama.com/library/gemma4` catalog).
  No fallback. Weights go to `/tmp/ollama_models` (`OLLAMA_MODELS`);
  Kaggle installs the Ollama binary, runs `ollama serve`, pulls once (~10GB).
- Hosted-API arm (`03`, accuracy-only): `qwen/qwen3.8-27b:free` via
  OpenRouter — verified live 2026-10-03 through the local OpenRouter
  key (`GET /api/v1/models`: price $0/$0, 262144 ctx,
  `text+image+video->text`). Key comes from Kaggle Secret
  `OPENROUTER_API_KEY`, never hardcoded. Free tier ≈ 200 req/day;
  default notebook = ~18 calls (1 verify + 9 retrieval + 8 summ).
  No paid fallback.
- No HF-transformers Gemma notebook in V2 — Gemma needs its own
  processor/loader and its real visual-token knob (do not assume Qwen's
  `max_pixels` transfers). The Ollama arm sidesteps this by using the
  runtime's own counters, labeled accordingly.

## 6. How to run on Kaggle

1. New Kaggle notebook, GPU ON (T4 suffices), internet ON.
   Add-ons → Secrets: `HF_TOKEN` (for gated weights in `02`),
   `OPENROUTER_API_KEY` (for `03`). Attach each secret to the notebook.
2. Clone the repo on the `Sayok` branch (or skip — every notebook's Cell 0
   clones to `/kaggle/working/Testing-dip` if missing, else `git pull`):
   `!git clone -b Sayok https://github.com/iamahnaf/Testing-dip.git`
   Alternative: upload a single `V2/notebooks/*.ipynb` via File → Upload —
   notebooks are standalone and run without the rest of the tree.
3. Run `00` → `01`, then `02`/`03`/`04` in any order, each Run-All top-to-bottom.
   For report numbers fast: `00 → 01 → 03` (no heavy downloads).
   Overnight local numbers: `04` (~10GB Ollama pull) and `02` (9B 4-bit pull).
4. After each run: Save Version (persists `/kaggle/working/`; `/tmp` weights
   are intentionally excluded), then copy
   `/kaggle/working/results/raw/*.jsonl` (+ `val_*.png` proofs) back to
   `V2/results/raw/`. Raw logs are append-only source of truth —
   aggregates/figures derive from them, never hand-edits.

## 7. Current status and what's missing

Done: 00 env/auth/ID-check, 01 renderer+no-crop gate, 02 six-row local
retrieval gate, 03 OpenRouter Qwen3.8-free retrieval (6 paired rows) +
CNN/DM summarization (`N_SUMM=4` default, ROUGE-1/2/L), 04 Ollama Gemma4
retrieval gate (Ollama counters + nvidia-smi VRAM). All new-notebook code
cells compile; helper logic (render/samples/metrics/JPEG payload)
smoke-tested locally. `V2/results/raw/` is empty — no Kaggle run logged yet.
Not yet built: analysis script (Tables 3/4, Figs 2–4 from raw JSONL only),
Qwen3.5 summarization arm, BABILong reasoning.
Known V2 limitation: `k` in 02 is a decoder-visible proxy
(`input_ids.shape[-1] − q` incl. chat template), not the internal vision
patch count — label it as such; inspect processor/model config for the
exact patch mechanism when writing up.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| MISS in 00 cell 5 | Model ID drifted — stop and record it, no fallback exists |
| OOM in 02 cell 2/4 | No fallback — reduce context sizes or stop and record it |
| `tectonic missing` | Expected on locked images; PIL fallback runs, rows log `renderer=pil-fallback` |
| Gated-model 401 | Set `HF_TOKEN` secret, re-run 00 cell 4 |
| `bf16` / flash-attn errors | Use fp16 + sdpa (T4/Turing, sm75) as coded |
| 429 / retries exhausted (03) | Free-tier quota — lower `N_SUMM`, retry tomorrow; no paid fallback |
| 502/503 on `:free` (03) | Flaky free upstream — notebook already backs off + retries |
| `ollama chat failed` (04) | `ollama serve` died — restart cell 2, check `/tmp/ollama.log` |
| Lost `/kaggle/working/` outputs | Save Version before closing; copy JSONL back to repo |
