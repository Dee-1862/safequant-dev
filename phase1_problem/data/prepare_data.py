"""
SafeQuant Phase 1: Prepare datasets.

Uses HarmBench behaviors as harmful prompts and Alpaca-Cleaned as harmless.
Splits into train/val for direction extraction and keeps eval set for ASR.

Run: python phase1_problem/data/prepare_data.py
"""
import sys
import os
import csv
from datasets import load_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
from config import N_TRAIN, N_VAL, DATA_DIR

random.seed(42)


def load_harmbench_prompts():
    """
    Load harmful prompts from HarmBench.
    NOTE: In future if you see any errors with importing HarmBench then fall back to JailbreakBench or a manual list.
    """
    prompts = []
    
    # Try HarmBench first
    harmbench_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "repos", "HarmBench",
        "data", "behavior_datasets", "harmbench_behaviors_text_all.csv"
    )
    
    if not os.path.exists(harmbench_path):
        raise FileNotFoundError(
            f"HarmBench dataset not found at {harmbench_path}"
        )
    else:
        print(f"Loading harmful prompts from HarmBench: {harmbench_path}")
        with open(harmbench_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior = row.get("Behavior", row.get("behavior", ""))
                if behavior.strip():
                    prompts.append(behavior.strip())
        print(f"    Loaded {len(prompts)} harmful behaviors from HarmBench")
    
    return prompts


def load_alpaca_prompts():
    """Load harmless prompts from Alpaca-Cleaned dataset."""

    print("Loading Alpaca-Cleaned harmless prompts")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    # Filter to instruction-only (no input field) for cleaner prompts
    prompts = []
    for row in ds:
        if not row.get("input", "").strip():
            prompts.append(row["instruction"].strip())
        if len(prompts) >= 1000:  # don't need more than 1000
            break
    print(f"Loaded {len(prompts)} harmless prompts from Alpaca-Cleaned")
    return prompts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. Harmful prompts
    harmful = load_harmbench_prompts()
    random.shuffle(harmful)
    
    # N_TRAIN for training, N_VAL for validation, rest for eval
    assert len(harmful) >= N_TRAIN + N_VAL + 50, f"Need at least {N_TRAIN + N_VAL + 50} harmful prompts, got {len(harmful)}"
    
    harmful_train = harmful[:N_TRAIN]
    harmful_val = harmful[N_TRAIN:N_TRAIN + N_VAL]
    eval_harmful = harmful[N_TRAIN + N_VAL:N_TRAIN + N_VAL + 100]
    
    # 2. Harmless prompts
    harmless = load_alpaca_prompts()
    random.shuffle(harmless)
    
    assert len(harmless) >= N_TRAIN + N_VAL, f"Need at least {N_TRAIN + N_VAL} harmless prompts, got {len(harmless)}"
    
    harmless_train = harmless[:N_TRAIN]
    harmless_val = harmless[N_TRAIN:N_TRAIN + N_VAL]
    
    # 3. Save all splits
    splits = {
        "harmful_train.json": harmful_train,
        "harmful_val.json": harmful_val,
        "harmless_train.json": harmless_train,
        "harmless_val.json": harmless_val,
        "eval_harmful.json": eval_harmful,
    }
    
    for filename, data in splits.items():
        path = os.path.join(DATA_DIR, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved {filename}: {len(data)} prompts")
    
    # 4. Print sample prompts
    print("\nSample harmful (train)")
    for p in harmful_train[:3]:
        print(f"  - {p[:80]}...")
    
    print("\n=== Sample harmless (train) ===")
    for p in harmless_train[:3]:
        print(f"  - {p[:80]}...")
    
    print("\n=== Sample eval (harmful) ===")
    for p in eval_harmful[:3]:
        print(f"  - {p[:80]}...")
    
    print(f"\n=== Data preparation complete. Files saved to {DATA_DIR} ===")


if __name__ == "__main__":
    main()
