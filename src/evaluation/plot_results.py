"""
Plotting & Analysis Script
Visualizes:
1. Compression Ratio (CR) vs Context Tokens
2. Accuracy Retention vs Compression Ratio (Degradation Threshold)
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_benchmark_plots(csv_path="results/ruler_benchmark_results.csv", output_dir="results"):
    if not os.path.exists(csv_path):
        print(f"Results file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Plot Tokens & CR
    color = "tab:blue"
    ax1.set_xlabel("Target Context Length (Tokens)")
    ax1.set_ylabel("Compression Ratio (CR)", color=color)
    line1 = ax1.plot(df["target_context_tokens"], df["compression_ratio_cr"], marker="o", color=color, linewidth=2, label="Compression Ratio (CR)")
    ax1.tick_params(axis="y", labelcolor=color)

    # Secondary Axis for Accuracy
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Condition B (Image) Accuracy", color=color)
    line2 = ax2.plot(df["target_context_tokens"], df["accuracy_image"], marker="s", linestyle="--", color=color, linewidth=2, label="Image Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(-0.1, 1.1)

    plt.title("Text-as-Image Compression: CR & Accuracy Degradation", fontsize=12, fontweight="bold")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "cr_vs_accuracy_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Analysis plot saved to {out_path}")

if __name__ == "__main__":
    generate_benchmark_plots()
