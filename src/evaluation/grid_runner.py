"""
Full Grid Benchmark Runner
Tests all combinations:
Context Lengths: 500, 1000, 1500 tokens
Visual Token Budgets: 70, 140, 280, 560, 1120
"""

import time
import os
import pandas as pd
from ollama import chat

from src.datasets.ruler_loader import generate_ruler_sample
from src.renderer.rasterizer import TextRasterizer
from src.evaluation.metrics import compute_metrics

MODEL_NAME = "gemma4:31b"
CONTEXT_LENGTHS = [500, 1000, 1500]
TOKEN_BUDGETS = [70, 140, 280, 560, 1120]

def run_grid_benchmark(
    context_lengths=CONTEXT_LENGTHS,
    token_budgets=TOKEN_BUDGETS,
    output_csv="results/full_grid_benchmark_results.csv"
):
    print("=" * 85)
    print("STARTING FULL GRID EXPERIMENT: CONTEXT LENGTHS x VISUAL TOKEN BUDGETS")
    print(f"Context Lengths (m) : {context_lengths}")
    print(f"Visual Budgets  (k) : {token_budgets}")
    print("=" * 85)

    rasterizer = TextRasterizer()
    results = []

    # First, run Condition A (Text baseline) for each context length once
    text_baselines = {}
    for length in context_lengths:
        print(f"\n[Baseline] Running Condition A (Text) for Context ~{length} tokens...")
        sample = generate_ruler_sample(target_tokens=length, needle_depth_ratio=0.5, seed=length)
        
        t0 = time.time()
        res_text = chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": f"Context:\n{sample['context']}\n\nQuestion: {sample['question']}"}],
            options={"temperature": 0.0}
        )
        t_text = time.time() - t0
        text_baselines[length] = {
            "sample": sample,
            "pred_text": res_text.message.content.strip(),
            "tokens_text": res_text.prompt_eval_count,
            "latency_text": t_text
        }
        print(f"  => Prompt Tokens: {res_text.prompt_eval_count} | Answer: {res_text.message.content.strip()[:60]}...")

    # Now run Condition B (Image) across all (context_length, visual_budget) pairs
    total_runs = len(context_lengths) * len(token_budgets)
    current_run = 0

    for length in context_lengths:
        baseline_info = text_baselines[length]
        sample = baseline_info["sample"]
        context = sample["context"]
        question = sample["question"]
        ground_truth = sample["ground_truth"]
        tok_text = baseline_info["tokens_text"]
        t_text = baseline_info["latency_text"]
        pred_text = baseline_info["pred_text"]

        for budget in token_budgets:
            current_run += 1
            print(f"\n[{current_run}/{total_runs}] Context: ~{length} | Visual Budget: {budget} tokens")
            
            # Rasterize image targeting discrete visual budget
            img = rasterizer.render(context, token_budget=budget)
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
            pred_image = res_image.message.content.strip()
            tok_image = res_image.prompt_eval_count

            metrics = compute_metrics(
                pred_text=pred_text,
                pred_image=pred_image,
                ground_truth=ground_truth,
                tokens_text=tok_text,
                tokens_image=tok_image,
                latency_text=t_text,
                latency_image=t_image
            )
            metrics["context_length"] = length
            metrics["visual_budget_setting"] = budget
            metrics["image_dimensions"] = f"{img.size[0]}x{img.size[1]}"
            metrics["ground_truth"] = ground_truth
            metrics["pred_text"] = pred_text
            metrics["pred_image"] = pred_image

            print(f"  [Image] Acc: {metrics['accuracy_image']} | Prompt Tokens: {tok_image} | Latency: {t_image:.2f}s")
            print(f"  => CR: {metrics['compression_ratio_cr']}x | Answer: {pred_image[:50]}...")

            results.append(metrics)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n[COMPLETE] Full grid benchmark saved to: {output_csv}")
    return df

if __name__ == "__main__":
    run_grid_benchmark()
