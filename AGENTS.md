# AGENTS.md --- Text-as-Image LLM Efficiency Study

## 0. Mission

You are the research agent for a reproducible empirical study
investigating whether long textual context can be represented as an
image and processed by a multimodal large language model (MLLM) using
substantially fewer decoder-visible tokens, while preserving task
performance.

The primary models are:

-   **Gemma 4** --- open-weight multimodal model family
-   **Qwen3.5** --- open-weight multimodal model family

The study will run primarily on **Kaggle GPU infrastructure**.

The attached reference paper is the methodological starting point. Its
core idea is to render textual context as an image, feed that image to a
multimodal model together with the query, and study the trade-off
between visual-token budget, task performance, and efficiency. The
reference paper uses a controlled text-to-image pipeline and evaluates
retrieval and summarization, with additional long-context reasoning
experiments. Treat that paper as the baseline methodology, not as
evidence that our models will produce the same results.

**Do not silently change the research question, models, datasets,
metrics, or experimental definitions. If a methodological change becomes
necessary, document it explicitly.**

------------------------------------------------------------------------

# 1. Research Objective

Main question:

> Can textual context be converted into an image and processed by
> open-weight multimodal LLMs with substantially fewer decoder-visible
> tokens while retaining comparable task performance?

Secondary questions:

1.  How does the text-as-image approach compare with ordinary text
    input?
2.  How does the trade-off differ between Gemma 4 and Qwen3.5?
3.  How does performance change as the visual-token budget is reduced?
4.  Is there a model/task-dependent visual-token threshold below which
    performance degrades substantially?
5.  Do reductions in decoder-visible tokens translate into lower latency
    or GPU-memory usage, given the additional cost of image processing?
6.  Does the usefulness of visual compression differ between retrieval,
    summarization, and reasoning tasks?

------------------------------------------------------------------------

# 2. Hypotheses

Treat these as hypotheses to test, not conclusions.

### H1 --- Efficiency

Text-as-image input can reduce the number of decoder-visible context
tokens while maintaining comparable performance on at least some
long-context tasks.

### H2 --- Model dependence

The performance/efficiency trade-off differs between Gemma 4 and Qwen3.5
because their multimodal architectures and visual-token processing
differ.

### H3 --- Visual-token threshold

There is a model- and task-dependent minimum visual-token budget below
which performance begins to degrade.

### H4 --- Nonlinear systems benefit

A reduction in decoder-visible token count will not necessarily produce
a proportional reduction in end-to-end latency because image
preprocessing and vision encoding introduce overhead.

------------------------------------------------------------------------

# 3. Reference Paper

Reference file:

`2025.findings-emnlp.558(1).pdf`

Use it as the primary methodological reference for:

-   text-to-image conversion
-   visual-token accounting
-   experimental conditions
-   RULER / S-NIAH evaluation
-   CNN/DailyMail summarization
-   long-context reasoning experiments
-   latency analysis
-   compression-ratio definitions
-   controlled inference settings

The reference paper's text-to-image pipeline should be treated as the
starting point. In particular, preserve the distinction between:

-   the original textual token count;
-   the visual embeddings/tokens presented to the decoder;
-   the question/query tokens; and
-   generated output tokens.

Do not describe visual embeddings as if they were literally ordinary
tokenizer tokens.

If the reference paper and current model documentation differ, follow
the actual implementation/documentation of the model and explicitly
record the difference.

------------------------------------------------------------------------

# 4. Core Experimental Design

Every experiment should compare at least:

## Condition A --- Text baseline

Original context is supplied as ordinary text.

``` text
Document/context
    ↓
Tokenizer
    ↓
Text tokens
    ↓
Gemma 4 / Qwen3.5
    ↓
Answer
```

## Condition B --- Text-as-image

The exact same textual content is rendered to an image. The image is
supplied as the context, while the query remains textual.

``` text
Document/context
    ↓
Controlled renderer
    ↓
Image
    ↓
Vision processor / encoder
    ↓
Visual representations
    ↓
Gemma 4 / Qwen3.5
    ↓
Answer
```

## Condition C --- Hybrid

Optional and only run if compute permits.

Some context is text and some is represented visually. This is
exploratory and must not replace the primary A/B experiment.

------------------------------------------------------------------------

# 5. Primary Experimental Matrix

The central experiment is a **context-length × visual-token-budget
sweep**.

Suggested original text lengths:

-   500 tokens
-   1,000 tokens
-   2,000 tokens
-   4,000 tokens
-   8,000 tokens
-   16,000 tokens

Adjust upward/downward based on actual model context limits and Kaggle
GPU capacity.

For Gemma 4, use supported visual-token budgets where available,
beginning with:

-   70
-   140
-   280
-   560
-   1120

Do not assume Qwen3.5 exposes exactly the same visual-token controls.
Determine its actual image-token behavior from its implementation and
processor, then document the mechanism precisely.

The conceptual matrix is:

  -----------------------------------------------------------------------
      Original text     Lowest visual               ...    Highest visual
             tokens            budget                              budget
  ----------------- ----------------- ----------------- -----------------
                500                 ✓               ...                 ✓

              1,000                 ✓               ...                 ✓

              2,000                 ✓               ...                 ✓

              4,000                 ✓               ...                 ✓

              8,000                 ✓               ...                 ✓

             16,000                 ✓               ...                 ✓
  -----------------------------------------------------------------------

Run the text baseline for every relevant context length.

------------------------------------------------------------------------

# 6. Datasets / Tasks

The preferred core benchmark suite is:

## 6.1 RULER

Use RULER, especially the Single Needle-in-a-Haystack (S-NIAH) task, as
the primary long-context retrieval benchmark.

Purpose:

-   controlled retrieval;
-   systematic context-length scaling;
-   easy comparison between text and image representations.

Where feasible, use the same sampling/generation protocol as the
reference paper so results are comparable.

## 6.2 CNN/DailyMail

Use CNN/DailyMail for summarization.

Purpose:

-   tests whether the model can understand the entire document rather
    than retrieve one hidden fact;
-   provides a qualitatively different task from S-NIAH.

Preferred metrics:

-   ROUGE-1
-   ROUGE-2
-   ROUGE-L
-   BERTScore

## 6.3 BABILong

Use BABILong, or a clearly documented equivalent long-context reasoning
benchmark, if compute permits.

Purpose:

-   multi-step reasoning over long contexts;
-   tests whether visual representation preserves relationships rather
    than only exact retrieval.

If BABILong is omitted because of compute constraints, record the reason
in the experiment log.

------------------------------------------------------------------------

# 7. Text-to-Image Rendering Pipeline

The rendering pipeline must be deterministic and content-preserving.

Preferred pipeline:

``` text
Raw text
  ↓
Normalization
  ↓
Escape special characters
  ↓
Controlled typesetting / rendering
  ↓
PDF or equivalent intermediate representation
  ↓
Rasterization
  ↓
Image
  ↓
Model image processor
```

The exact rendering implementation may use the reference paper's
ConTexImage-style approach.

Rules:

1.  Do not rewrite, summarize, paraphrase, reorder, or semantically
    alter the source text.
2.  Do not manually highlight the answer.
3.  Do not use different text for the image and text baseline.
4.  Preserve punctuation, whitespace where meaningful, headings, and
    paragraph boundaries as consistently as possible.
5.  Record image dimensions, DPI/resolution, font size, line spacing,
    and rendering parameters.
6.  Ensure the entire intended context is present in the image.
7.  Do not accidentally crop text.
8.  Validate rendered images automatically where possible.

------------------------------------------------------------------------

# 8. Critical Fairness Controls

The comparison is invalid if the image condition receives information
that the text condition does not, or vice versa.

For every sample:

``` text
SAME DOCUMENT
SAME QUESTION
SAME ANSWER TARGET
SAME SYSTEM PROMPT
SAME GENERATION PARAMETERS
```

Only the **representation of the context** should change.

Do not:

-   alter the question between conditions;
-   give image inputs additional instructions;
-   add OCR output to the image condition;
-   manually crop around relevant evidence;
-   highlight answer spans;
-   use a different document version;
-   use different decoding settings.

If a model requires different prompt syntax for text versus image input,
make the smallest necessary formatting change and document it.

------------------------------------------------------------------------

# 9. Token Accounting

This is one of the most important parts of the study.

## 9.1 Text tokens

Measure using the actual tokenizer/processor associated with the model.

Record:

``` text
context_text_tokens
question_tokens
system_prompt_tokens
total_text_input_tokens
output_tokens
```

Do not estimate token counts using character length.

## 9.2 Visual tokens

Do NOT claim that the image contains an equivalent number of ordinary
text tokens.

Instead, define:

> visual-token count = the number of visual representations/tokens
> passed from the vision processing pathway into the language-model
> portion of the multimodal model, according to the model
> implementation.

For Gemma 4, use the supported visual-token budget/configuration.

For Qwen3.5, inspect the processor/model implementation and determine
how image patches/visual representations are transformed before entering
the language model.

Record the exact mechanism.

## 9.3 Output tokens

Record generated tokens separately.

Do not mix output tokens with input/context compression.

------------------------------------------------------------------------

# 10. Compression Metrics

Define at least two related quantities.

### Context compression ratio

Let:

-   `m` = number of original context text tokens
-   `k` = number of visual tokens/visual representations

Then:

`CR_context = m / k`

### Decoder-input compression ratio

If the query and other textual prompt tokens are included:

`CR_decoder = (m + q) / (k + q)`

where `q` is the number of textual query/prompt tokens that remain
present in the image condition.

Always state which definition is being reported.

------------------------------------------------------------------------

# 11. Performance Metrics

Use task-appropriate metrics.

### Retrieval

-   Exact Match / accuracy

### Summarization

-   ROUGE-1
-   ROUGE-2
-   ROUGE-L
-   BERTScore

### Reasoning

-   exact-match accuracy or the benchmark's official scoring method

Do not invent a custom metric merely to make one representation look
better.

------------------------------------------------------------------------

# 12. Efficiency Metrics

Measure all of the following where technically available:

-   input/context token count
-   visual-token count
-   output token count
-   time to first token, if available
-   prefill/input-processing time, if available
-   generation time
-   end-to-end latency
-   peak GPU memory
-   GPU type
-   inference batch size
-   precision/quantization
-   image dimensions
-   image preprocessing time
-   vision encoding time, if measurable

Important:

**Token reduction is not equivalent to latency reduction.**

The image condition has additional work:

``` text
image decode
→ preprocessing
→ vision encoding
→ multimodal projection
→ language-model inference
```

Therefore report both token-level and system-level efficiency.

------------------------------------------------------------------------

# 13. Kaggle Infrastructure

Primary execution environment:

-   Kaggle notebook
-   Kaggle GPU
-   Hugging Face Transformers or the model's official open-source
    implementation
-   Python

The agent should detect the actual Kaggle GPU rather than assuming a
particular GPU model.

At the beginning of every experiment, log:

``` text
GPU model
GPU memory
CUDA version
Python version
PyTorch version
Transformers version
model revision / commit
dataset version
experiment git/repository commit if available
```

Do not silently upgrade or downgrade packages during a reproducibility
run.

Pin versions where practical.

------------------------------------------------------------------------

# 14. Open-Weight Model Policy

The study is specifically intended to use models that can be executed
locally rather than relying on closed API billing.

This is desirable because it allows direct inspection of:

-   tokenizer counts;
-   processor outputs;
-   visual-token representations where exposed;
-   model configuration;
-   inference latency;
-   GPU memory;
-   intermediate tensor shapes.

Do not claim that local token counts are identical to a provider's
billing token counts unless the provider's accounting specification
confirms it.

------------------------------------------------------------------------

# 15. Model Selection

Primary models:

1.  Gemma 4
2.  Qwen3.5

Start with model sizes that are realistically executable on the
available Kaggle GPU.

A practical first pass is:

-   Gemma 4 E4B or another manageable Gemma 4 variant;
-   Qwen3.5-9B or another manageable Qwen3.5 variant.

Larger variants can be added if resources allow.

Model size must be reported in the final paper.

Do not compare a quantized model against a full-precision model without
clearly labeling the difference.

For the cleanest primary comparison, keep:

-   precision/quantization;
-   decoding settings;
-   batch size;
-   context lengths;
-   dataset samples

as consistent as technically possible.

------------------------------------------------------------------------

# 16. Inference Settings

Default starting point:

``` text
temperature = 0
```

Use deterministic decoding where supported.

Keep fixed:

-   temperature
-   top-p / sampling configuration
-   max_new_tokens
-   stop conditions
-   system prompt
-   question format

If a model requires different generation settings for valid operation,
document them rather than hiding the difference.

------------------------------------------------------------------------

# 17. Repeated Runs and Randomness

For deterministic settings, one run may be sufficient for
generation-level reproducibility, but system measurements such as
latency can fluctuate.

For latency/resource measurements:

-   use warm-up runs;
-   exclude model loading from steady-state inference latency unless
    explicitly reporting cold-start latency;
-   perform multiple timed runs;
-   report mean and variability;
-   keep the environment as stable as possible.

For stochastic experiments, record random seeds.

------------------------------------------------------------------------

# 18. Statistical Analysis

Because the same underlying sample can be evaluated under multiple
representation conditions, prefer **paired comparisons**.

For binary matched outcomes:

-   consider McNemar's test where appropriate.

For continuous per-example metrics:

-   use an appropriate paired statistical test;
-   report effect sizes where appropriate.

Do not run many significance tests without accounting for multiple
comparisons.

Report uncertainty rather than only point estimates.

The key comparison is not simply:

> Model A score vs Model B score

but:

> Text baseline vs image representation at a specified visual-token
> budget, on the same examples.

------------------------------------------------------------------------

# 19. Main Figures to Produce

At minimum, generate:

### Figure 1 --- Experimental pipeline

``` text
Text → Renderer → Image → Vision encoder → MLLM → Answer
Text → Tokenizer → MLLM → Answer
```

### Figure 2 --- Performance vs visual-token budget

X-axis:

-   visual-token budget

Y-axis:

-   task performance

Separate lines for:

-   Gemma 4
-   Qwen3.5

### Figure 3 --- Compression ratio vs performance

Show the trade-off between:

-   compression ratio
-   task performance

### Figure 4 --- Context length vs performance

Compare:

-   text baseline
-   image at several visual-token budgets

### Figure 5 --- Latency decomposition

Where measurable:

-   image preprocessing
-   vision encoding
-   language-model inference
-   total latency

### Figure 6 --- GPU memory

Compare text and image conditions.

------------------------------------------------------------------------

# 20. Main Tables

### Table 1 --- Models

  ----------------------------------------------------------------------------------
  Model                Size        Context Vision         Visual-token   Precision
                                           architecture   mechanism      
  ---------- -------------- -------------- -------------- -------------- -----------

  ----------------------------------------------------------------------------------

### Table 2 --- Datasets

  Dataset   Task     Samples Context lengths   Metric
  --------- ------ --------- ----------------- --------

### Table 3 --- Main results

  Model   Task   Input     Visual tokens   Compression   Performance
  ------- ------ ------- --------------- ------------- -------------

### Table 4 --- Efficiency

  Model   Input     Input tokens   Visual tokens   Latency   Peak VRAM
  ------- ------- -------------- --------------- --------- -----------

------------------------------------------------------------------------

# 21. Experiment Tracking

Every inference batch must generate machine-readable metadata.

Recommended JSON/JSONL schema:

``` json
{
  "experiment_id": "",
  "timestamp": "",
  "model": "",
  "model_revision": "",
  "dataset": "",
  "sample_id": "",
  "task": "",
  "input_mode": "text|image|hybrid",
  "context_text_tokens": 0,
  "question_tokens": 0,
  "system_prompt_tokens": 0,
  "visual_tokens": 0,
  "output_tokens": 0,
  "compression_ratio_context": 0.0,
  "compression_ratio_decoder": 0.0,
  "image_width": 0,
  "image_height": 0,
  "image_dpi": 0,
  "temperature": 0.0,
  "max_new_tokens": 0,
  "precision": "",
  "quantization": "",
  "gpu": "",
  "peak_gpu_memory_mb": 0,
  "prefill_time_seconds": null,
  "generation_time_seconds": null,
  "end_to_end_time_seconds": null,
  "score": null,
  "error": null
}
```

Never overwrite raw experiment logs.

------------------------------------------------------------------------

# 22. Directory Structure

Use a structure similar to:

``` text
project/
├── AGENTS.md
├── README.md
├── configs/
│   ├── gemma4.yaml
│   └── qwen35.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── rendering/
│   ├── renderer.py
│   └── validation.py
├── models/
│   ├── gemma4.py
│   └── qwen35.py
├── inference/
│   ├── run_text.py
│   ├── run_image.py
│   └── run_experiment.py
├── evaluation/
│   ├── retrieval.py
│   ├── summarization.py
│   └── reasoning.py
├── analysis/
│   ├── token_analysis.py
│   ├── latency_analysis.py
│   └── statistics.py
├── results/
│   ├── raw/
│   ├── processed/
│   └── figures/
├── logs/
└── notebooks/
```

------------------------------------------------------------------------

# 23. Reproducibility Rules

Every result must be traceable to:

1.  a model revision;
2.  a dataset version/sample;
3.  an input representation;
4.  a rendering configuration;
5.  a visual-token configuration;
6.  an inference configuration;
7.  a hardware/software environment;
8.  a raw output.

Never manually edit a result table without retaining the underlying raw
data.

Never report a number that cannot be regenerated from stored logs.

------------------------------------------------------------------------

# 24. Validation Before Large Runs

Before launching a large Kaggle experiment, run a tiny validation suite.

Use approximately:

-   3--5 samples per task;
-   1 short context;
-   1 medium context;
-   1 longer context;
-   at least two visual-token budgets.

Verify:

-   model loads;
-   image is correctly rendered;
-   no text is cropped;
-   processor accepts the image;
-   generation works;
-   text and image conditions use the same underlying content;
-   token counts are captured;
-   visual-token accounting is captured or defensibly estimated from
    model internals;
-   latency is measured correctly;
-   GPU memory is logged;
-   outputs are saved;
-   evaluation script produces expected metrics.

Do not launch the full experiment until this validation passes.

------------------------------------------------------------------------

# 25. Common Methodological Mistakes to Avoid

## Mistake 1 --- Comparing different information

Never allow the image to contain a modified version of the text.

## Mistake 2 --- Counting pixels as tokens

Pixels, image patches, visual embeddings, and language-model tokens are
not interchangeable.

## Mistake 3 --- Treating fewer tokens as automatically cheaper

Measure actual latency and GPU memory.

## Mistake 4 --- Ignoring OCR difficulty

Text rendered into an image is an OCR/document-understanding problem.
Performance losses may arise from visual recognition rather than context
reasoning.

Discuss this explicitly.

## Mistake 5 --- Using only one context length

A single context length cannot establish a compression curve.

## Mistake 6 --- Using only retrieval

A method can succeed at finding one exact fact while failing to
understand an entire document.

## Mistake 7 --- Changing prompts between conditions

Keep the semantic prompt identical.

## Mistake 8 --- Reporting only aggregate averages

Preserve per-example results.

## Mistake 9 --- Mixing model sizes or quantization settings without labeling them

Model size and numerical precision can materially affect the result.

## Mistake 10 --- Claiming general superiority

The study is intended to characterize a trade-off, not establish that
images are universally better than text.

------------------------------------------------------------------------

# 26. Interpretation Rules

When analyzing results, distinguish:

### Observed fact

Example:

> "At 560 visual tokens, Gemma 4 achieved X% accuracy on the evaluated
> S-NIAH subset."

### Derived measurement

Example:

> "This corresponds to a context compression ratio of Y×."

### Interpretation

Example:

> "This suggests that the model preserved retrieval performance at this
> compression level."

Do not turn an observation into a universal claim.

Avoid statements such as:

-   "images are better than text";
-   "visual tokens are always cheaper";
-   "this proves multimodal models understand compressed text";
-   "Gemma is better than Qwen."

Instead specify:

-   model;
-   task;
-   context length;
-   visual-token budget;
-   hardware;
-   metric;
-   uncertainty.

------------------------------------------------------------------------

# 27. Expected Core Result

The study should ultimately produce a **Pareto-style analysis**, not a
single winner.

For each model/task pair, characterize:

``` text
performance
     ↑
     │             ●
     │          ●
     │       ●
     │    ●
     │ ●
     └────────────────────→
          compression
```

The important question is:

> At what compression level can performance remain acceptably close to
> the text baseline, and what computational cost is associated with
> achieving that representation?

Do not select a "best model" or "best method" as the paper's conclusion.
Report the observed trade-offs.

------------------------------------------------------------------------

# 28. Research Paper Framing

The paper should be framed as an empirical study of:

> **visual representation as an alternative context encoding mechanism
> for long-context multimodal LLM inference.**

The contribution should not be claimed as invention of the general
text-as-image idea, because the attached reference paper already
establishes that direction.

Potential contributions:

1.  Evaluation of the approach on modern open-weight multimodal models.
2.  Systematic visual-token-budget sweep.
3.  Cross-model comparison between Gemma 4 and Qwen3.5.
4.  Analysis across retrieval, summarization, and reasoning.
5.  Token-efficiency versus end-to-end latency analysis.
6.  Reproducible open-model/Kaggle evaluation pipeline.

------------------------------------------------------------------------

# 29. What the Agent Should Do First

When beginning implementation:

### Step 1

Inspect the reference paper and extract the exact methodology,
definitions, datasets, and experimental details that are relevant.

### Step 2

Verify the current official model documentation/configuration for Gemma
4 and Qwen3.5 before implementing model-specific code.

### Step 3

Build the text baseline.

### Step 4

Build and validate the text-to-image renderer.

### Step 5

Build Gemma 4 image inference.

### Step 6

Build Qwen3.5 image inference.

### Step 7

Implement token/visual-token accounting.

### Step 8

Run the small validation suite.

### Step 9

Run the full benchmark matrix.

### Step 10

Generate raw results first, then derived metrics and figures.

### Step 11

Perform statistical analysis.

### Step 12

Write the paper only after the raw experiment logs are complete.

------------------------------------------------------------------------

# 30. Source and Citation Discipline

When implementing or writing the paper:

-   cite the attached reference paper for its methodology and reported
    findings;
-   cite official model documentation/model cards for model architecture
    and token-processing behavior;
-   cite original dataset/benchmark papers;
-   distinguish implementation facts from experimental observations;
-   do not invent model specifications;
-   if documentation is ambiguous, inspect the actual source
    code/configuration and report the uncertainty.

For current model behavior, use current official sources rather than
relying on memory.

------------------------------------------------------------------------

# 31. Final Research Principle

The central experimental question is not:

> "Can an image contain text?"

It obviously can.

The scientific question is:

> **How much textual context can a multimodal LLM reliably process
> through a visual representation, how many visual representations are
> required, and does that representation provide a meaningful efficiency
> advantage over conventional text input?**

Every experiment, metric, implementation choice, and conclusion should
serve that question.
