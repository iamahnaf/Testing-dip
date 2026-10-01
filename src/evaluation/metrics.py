"""
Evaluation Metrics & Logging Engine
Computes:
- Task Accuracy (Exact Match / Substring)
- Compression Ratio: CR = (m + q) / (k + q)
- Token Reduction percentage
- Latency (prefill, generation, total)
"""

from typing import Dict, Any

def compute_accuracy(prediction: str, ground_truth: str) -> float:
    """Computes whether ground truth needle is successfully extracted."""
    if not prediction or not ground_truth:
        return 0.0
    return 1.0 if ground_truth.lower() in prediction.lower() else 0.0

def compute_metrics(
    pred_text: str,
    pred_image: str,
    ground_truth: str,
    tokens_text: int,
    tokens_image: int,
    latency_text: float,
    latency_image: float
) -> Dict[str, Any]:
    """
    Computes comparative metrics between Condition A (Text) and Condition B (Image).
    """
    acc_text = compute_accuracy(pred_text, ground_truth)
    acc_image = compute_accuracy(pred_image, ground_truth)
    
    cr = (tokens_text / tokens_image) if tokens_image and tokens_image > 0 else 0.0
    token_savings_pct = (1.0 - (1.0 / cr)) * 100 if cr > 1.0 else 0.0
    latency_speedup = (latency_text / latency_image) if latency_image and latency_image > 0 else 0.0

    return {
        "accuracy_text": acc_text,
        "accuracy_image": acc_image,
        "tokens_text_m_q": tokens_text,
        "tokens_image_k_q": tokens_image,
        "compression_ratio_cr": round(cr, 4),
        "token_savings_pct": round(token_savings_pct, 2),
        "latency_text_sec": round(latency_text, 3),
        "latency_image_sec": round(latency_image, 3),
        "latency_speedup_x": round(latency_speedup, 2)
    }
