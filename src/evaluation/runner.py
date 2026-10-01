"""
Benchmark Runner Pipeline
Executes automated comparative evaluation across context token lengths
(500, 1000, 2000) comparing Condition A (Text) vs Condition B (Image).
"""

import time
import json
import os
import pandas as pd
from ollama import chat

from src.datasets.ruler_loader import generate_ruler_sample
from src.renderer.rasterizer import TextRasterizer
from src.evaluation.metrics import compute_metrics

MODEL_NAME = "gemma4:31b"

def run_experiment(
    token_lengths=[500, 1000, 1500],
    visual_budget=560,
    output_csv="results/ruler_benchmark_results.csv"
):
    print("=" * 80)
    print(f"STARTING TEXT-AS-IMAGE BENCHMARK ON {MODEL_NAME}")
    print(f"Token Lengths: {token_lengths} | Target Visual Token Budget: {visual_budget}")
    print("=" * 80)

    rasterizer = TextRasterizer()
    results = []

    for length in token_lengths:
        print(f"\n>>> Running Evaluation for Context Length: ~{length} tokens")
        sample = generate_ruler_sample(target_tokens=length, needle_depth_ratio=0.5, seed=length)
        context = sample["context"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        # Condition A: Text Query
        t0 = time.time()
        res_text = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            options={"temperature": 0.0}
        )
        t_text = time.time() - t0
        pred_text = res_text.message.content
        tok_text = res_text.prompt_eval_count

        # Condition B: Image Query
        img = rasterizer.render(context, token_budget=visual_budget)
        img_bytes = rasterizer.to_bytes(img)

        t0 = time.time()
        res_image = chat(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": f"Carefully inspect the document image and answer: {question}",
                "images": [img_bytes]
            }],
            options={"temperature": 0.0}
        )
        t_image = time.time() - t0
        pred_image = res_image.message.content
        tok_image = res_image.prompt_eval_count

        # Compute Comparative Metrics
        metrics = compute_metrics(
            pred_text=pred_text,
            pred_image=pred_image,
            ground_truth=ground_truth,
            tokens_text=tok_text,
            tokens_image=tok_image,
            latency_text=t_text,
            latency_image=t_image
        )
        metrics["target_context_tokens"] = length
        metrics["ground_truth"] = ground_truth
        metrics["pred_text"] = pred_text.strip()
        metrics["pred_image"] = pred_image.strip()

        print(f"  [Text]  Acc: {metrics['accuracy_text']} | Tokens: {tok_text} | Latency: {t_text:.2f}s")
        print(f"  [Image] Acc: {metrics['accuracy_image']} | Tokens: {tok_image} | Latency: {t_image:.2f}s")
        print(f"  => Compression Ratio (CR): {metrics['compression_ratio_cr']} | Token Savings: {metrics['token_savings_pct']}%")

        results.append(metrics)

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[DONE] Results successfully exported to {output_csv}")
    return df

if __name__ == "__main__":
    run_experiment(token_lengths=[500, 1000])
