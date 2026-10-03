"""Py mirror of the 02 Qwen3.5 loader + metrics (for git diff only; Kaggle runs the .ipynb).

Download responsibility: `load_qwen35()` below (`from_pretrained` pulls weights).
T4 rules: fp16 (Turing has no bf16), sdpa (no flash-attn-2), 4-bit, batch=1.
"""
import re
import string

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers import TorchAoConfig

MODEL_ID = "Qwen/Qwen3.5-9B"  # only model for this arm — no fallback


def load_qwen35(model_id: str = MODEL_ID, max_pixels: int = 1024 * 28 * 28):
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, min_pixels=256 * 28 * 28, max_pixels=max_pixels
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        quantization_config=TorchAoConfig("int4_weight_only", group_size=128),
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().translate(str.maketrans("", "", string.punctuation))).strip()


def tok_f1(pred: str, gt: str) -> float:
    p, g = set(norm(pred).split()), set(norm(gt).split())
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec, rec = tp / len(p), tp / len(g)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
