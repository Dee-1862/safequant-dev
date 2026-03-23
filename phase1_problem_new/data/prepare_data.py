"""
SafeQuant Phase 1 — Prepare datasets.

Uses HarmBench behaviors as harmful prompts and Alpaca-Cleaned as harmless.
Splits into train/val for direction extraction and keeps eval set for ASR.

Run: python phase1_problem/data/prepare_data.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import random
from config import N_TRAIN, N_VAL, DATA_DIR

random.seed(42)


def load_harmbench_prompts():
    """
    Load harmful prompts from HarmBench.
    Falls back to JailbreakBench or a manual list if HarmBench isn't cloned.
    """
    prompts = []
    
    # Try HarmBench first
    harmbench_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "repos", "HarmBench",
        "data", "behavior_datasets", "harmbench_behaviors_text_all.csv"
    )
    
    if os.path.exists(harmbench_path):
        print(f"Loading from HarmBench: {harmbench_path}")
        import csv
        with open(harmbench_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                behavior = row.get("Behavior", row.get("behavior", ""))
                if behavior.strip():
                    prompts.append(behavior.strip())
        print(f"  Loaded {len(prompts)} harmful behaviors from HarmBench")
    
    # Try alternative HarmBench path
    if not prompts:
        alt_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "repos", "HarmBench",
            "data", "behavior_datasets", "harmbench_behaviors_text_val.csv"
        )
        if os.path.exists(alt_path):
            print(f"Loading from HarmBench (val): {alt_path}")
            import csv
            with open(alt_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    behavior = row.get("Behavior", row.get("behavior", ""))
                    if behavior.strip():
                        prompts.append(behavior.strip())
            print(f"  Loaded {len(prompts)} harmful behaviors")
    
    # Fallback: try loading from datasets library (AdvBench)
    if not prompts:
        print("HarmBench CSV not found. Trying to load AdvBench via datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("walledai/AdvBench", split="train")
            prompts = [row["prompt"] for row in ds]
            print(f"  Loaded {len(prompts)} prompts from AdvBench")
        except Exception as e:
            print(f"  Could not load AdvBench: {e}")
    
    # Last fallback: JailbreakBench
    if not prompts:
        print("Trying JailbreakBench...")
        try:
            import jailbreakbench as jbb
            dataset = jbb.read_dataset()
            prompts = [item.goal for item in dataset.goals]
            print(f"  Loaded {len(prompts)} prompts from JailbreakBench")
        except Exception as e:
            print(f"  Could not load JailbreakBench: {e}")
    
    if not prompts:
        raise RuntimeError(
            "Could not load harmful prompts from any source.\n"
            "Please run clone_repos.sh first, or pip install jailbreakbench."
        )
    
    return prompts


def load_alpaca_prompts():
    """Load harmless prompts from Alpaca-Cleaned dataset."""
    print("Loading Alpaca-Cleaned harmless prompts...")
    try:
        from datasets import load_dataset
        ds = load_dataset("yahma/alpaca-cleaned", split="train")
        # Filter to instruction-only (no input field) for cleaner prompts
        prompts = []
        for row in ds:
            if not row.get("input", "").strip():
                prompts.append(row["instruction"].strip())
            if len(prompts) >= 1000:  # don't need more than 1000
                break
        print(f"  Loaded {len(prompts)} harmless prompts from Alpaca-Cleaned")
        return prompts
    except Exception as e:
        print(f"  Error loading Alpaca: {e}")
        print("  Falling back to simple harmless prompts...")
        # Minimal fallback
        return [
            "What is the capital of France?",
            "Explain how photosynthesis works.",
            "Write a short poem about autumn.",
            "What are the benefits of exercise?",
            "How does a computer work?",
            "Describe the water cycle.",
            "What is machine learning?",
            "Tell me about the solar system.",
            "How do you make pasta?",
            "What is the theory of relativity?",
        ] * 20  # repeat to get enough


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # === Harmful prompts ===
    harmful = load_harmbench_prompts()
    random.shuffle(harmful)
    
    # We need: N_TRAIN for training, N_VAL for validation, rest for eval
    assert len(harmful) >= N_TRAIN + N_VAL + 50, \
        f"Need at least {N_TRAIN + N_VAL + 50} harmful prompts, got {len(harmful)}"
    
    harmful_train = harmful[:N_TRAIN]
    harmful_val = harmful[N_TRAIN:N_TRAIN + N_VAL]
    eval_harmful = harmful[N_TRAIN + N_VAL:N_TRAIN + N_VAL + 100]
    
    # === Harmless prompts ===
    harmless = load_alpaca_prompts()
    random.shuffle(harmless)
    
    assert len(harmless) >= N_TRAIN + N_VAL, \
        f"Need at least {N_TRAIN + N_VAL} harmless prompts, got {len(harmless)}"
    
    harmless_train = harmless[:N_TRAIN]
    harmless_val = harmless[N_TRAIN:N_TRAIN + N_VAL]
    
    # === Save all splits ===
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
    
    # === Print sample prompts ===
    print("\n=== Sample harmful (train) ===")
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
