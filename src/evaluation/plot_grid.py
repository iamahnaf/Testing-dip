"""
Comprehensive Visualizer for Full Grid Benchmark Results
Generates:
1. Accuracy Heatmap: Context Length vs Visual Token Budget
2. Compression Ratio (CR) vs Accuracy Curve
3. Latency comparison bar chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_full_grid_analysis(csv_path="results/full_grid_benchmark_results.csv", output_dir="results"):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # -------------------------------------------------------------
    # Plot 1: Heatmap of Retrieval Accuracy (Context Length vs Token Budget)
    # -------------------------------------------------------------
    pivot_acc = df.pivot(index="context_length", columns="visual_budget_setting", values="accuracy_image")
    im = axes[0].imshow(pivot_acc.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    
    axes[0].set_xticks(np.arange(len(pivot_acc.columns)))
    axes[0].set_yticks(np.arange(len(pivot_acc.index)))
    axes[0].set_xticklabels(pivot_acc.columns)
    axes[0].set_yticklabels(pivot_acc.index)
    axes[0].set_xlabel("Visual Token Budget (k)", fontweight="bold")
    axes[0].set_ylabel("Context Length (m tokens)", fontweight="bold")
    axes[0].set_title("1. Accuracy Heatmap (Degradation Matrix)", fontweight="bold")

    # Add text labels inside heatmap cells
    for i in range(len(pivot_acc.index)):
        for j in range(len(pivot_acc.columns)):
            val = pivot_acc.values[i, j]
            txt = "100%" if val == 1.0 else "0%"
            color = "white" if val < 0.5 else "black"
            axes[0].text(j, i, txt, ha="center", va="center", color=color, fontweight="bold")

    # -------------------------------------------------------------
    # Plot 2: Compression Ratio (CR) vs Accuracy Degradation Threshold
    # -------------------------------------------------------------
    scatter = axes[1].scatter(
        df["compression_ratio_cr"],
        df["accuracy_image"],
        c=df["visual_budget_setting"],
        cmap="viridis",
        s=120,
        edgecolor="black",
        zorder=3
    )
    axes[1].axvline(x=4.5, color="red", linestyle="--", linewidth=1.5, label="Degradation Cliff (CR ≈ 4.5x)")
    axes[1].set_xlabel("Compression Ratio (CR = m+q / k+q)", fontweight="bold")
    axes[1].set_ylabel("Retrieval Accuracy", fontweight="bold")
    axes[1].set_title("2. Degradation Threshold Discovery", fontweight="bold")
    axes[1].set_ylim(-0.1, 1.15)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left")

    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Visual Budget", rotation=270, labelpad=15)

    # -------------------------------------------------------------
    # Plot 3: Token Savings % across Context Lengths
    # -------------------------------------------------------------
    avg_savings = df.groupby("context_length")["token_savings_pct"].mean()
    bars = axes[2].bar([f"{k} tok" for k in avg_savings.index], avg_savings.values, color="#3b82f6", width=0.5, edgecolor="black")
    axes[2].set_xlabel("Document Context Length", fontweight="bold")
    axes[2].set_ylabel("Average Token Savings (%)", fontweight="bold")
    axes[2].set_title("3. Efficiency Gains (% Token Reduction)", fontweight="bold")
    axes[2].set_ylim(0, 100)
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    for bar in bars:
        h = bar.get_height()
        axes[2].annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    out_img = os.path.join(output_dir, "full_grid_analysis_dashboard.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Full grid analysis dashboard saved to: {out_img}")

if __name__ == "__main__":
    plot_full_grid_analysis()
