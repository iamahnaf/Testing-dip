"""
RULER Needle-in-a-Haystack Benchmark Generator
Generates synthetic long-context retrieval challenges scaled across
500, 1000, 2000, 4000, and 5000 context token lengths with randomized needle placement.
"""

import random
from typing import Dict, Any

DISTRACTOR_FACTS = [
    "The thermal radiator on the secondary boom radiates excess heat into cold space at roughly 4 kelvins.",
    "Subsystem bus voltage fluctuates within an allowable tolerance of positive or negative 0.15 volts direct current.",
    "Star tracking sensor arrays refresh coordinate quaternions at an operational cadence of thirty hertz.",
    "Liquid nitrogen boil-off lines are purged automatically during solar occultation intervals.",
    "Reaction wheels one and three reached peak angular momentum before momentum dumping thrusters fired.",
    "Micrometeoroid shielding panels withstood seventy-four microscopic impact events during the outer flyby.",
    "Optical payload mirrors are coated with ultra-thin protected gold films optimized for near-infrared reflection.",
    "The radioisotope thermoelectric generator generates approximately one hundred and fifty-seven watts of thermal output.",
    "Telemetry packets transmitted across the S-band downlink utilize Reed-Solomon forward error correction codes.",
    "Autonomous attitude determination algorithms rely on gyroscopic ring laser drift sensors during occultation."
]

def generate_ruler_sample(
    target_tokens: int = 1000,
    needle_depth_ratio: float = 0.5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generates a RULER needle-in-a-haystack sample with a buried fact.
    
    Args:
        target_tokens: Desired context token length (500 to 5000).
        needle_depth_ratio: Float between 0.0 (top) and 1.0 (bottom) representing needle position.
        seed: Random seed for deterministic reproducibility.
    """
    rng = random.Random(seed)
    
    # Unique needle fact
    secret_code = f"CIPHER-{rng.randint(1000, 9999)}-{rng.choice(['OMEGA', 'KAPPA', 'DELTA', 'SIGMA'])}"
    needle = f"Special authorization key {secret_code} was recorded in emergency protocol ledger section 9."
    question = "What special authorization key was recorded in emergency protocol ledger section 9?"
    ground_truth = secret_code

    # Build haystack to reach target length (approx 1.3 words per token)
    target_words = int(target_tokens * 0.75)
    
    paragraphs = []
    current_words = 0
    
    while current_words < target_words:
        # Construct a paragraph of 4-6 distractor sentences
        num_sentences = rng.randint(4, 6)
        para_sentences = [rng.choice(DISTRACTOR_FACTS) for _ in range(num_sentences)]
        para = " ".join(para_sentences)
        paragraphs.append(para)
        current_words += len(para.split())
        
    # Insert needle at specified depth ratio
    insert_idx = int(len(paragraphs) * needle_depth_ratio)
    insert_idx = max(0, min(insert_idx, len(paragraphs) - 1))
    
    # Place needle inside that paragraph
    p = paragraphs[insert_idx]
    words = p.split()
    mid = len(words) // 2
    words.insert(mid, needle)
    paragraphs[insert_idx] = " ".join(words)
    
    full_context = "\n\n".join(paragraphs)
    
    return {
        "target_tokens": target_tokens,
        "actual_word_count": len(full_context.split()),
        "needle_depth_ratio": needle_depth_ratio,
        "context": full_context,
        "question": question,
        "ground_truth": ground_truth,
        "needle": needle
    }
