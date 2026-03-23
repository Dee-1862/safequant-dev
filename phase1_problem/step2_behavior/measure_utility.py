"""
SafeQuant Phase 1, Step 2 — Measure general utility (HellaSwag / BoolQ).

Quick accuracy check to show quantization preserves general capabilities
even if safety degrades. Uses a simple multiple-choice evaluation.

Run:
  python phase1_problem/step2_behavior/measure_utility.py --model qwen1.5b --precision fp16
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG


def load_model(model_name, precision):
    """Load model in specified precision on CPU."""
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map = "cpu",
            trust_remote_code = True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = DTYPE,
            device_map = "cpu",
            trust_remote_code = True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def eval_boolq(model, tokenizer, n_samples=200):
    """
    Evaluate on BoolQ (yes/no questions).
    Simple approach: generate answer, check if it matches.
    """
    print("  Loading BoolQ dataset...")
    ds = load_dataset("google/boolq", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))
    
    correct = 0
    total = 0
    
    for item in tqdm(ds, desc="  BoolQ"):
        question = item["question"]
        passage = item["passage"][:500]  # truncate passage for speed
        answer = "yes" if item["answer"] else "no"
        
        prompt = f"Read the passage and answer with just 'yes' or 'no'.\n\nPassage: {passage}\n\nQuestion: {question}\n\nAnswer:"
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()
        
        # Check if response starts with correct answer
        predicted = "yes" if response.startswith("yes") else "no" if response.startswith("no") else "unknown"
        if predicted == answer:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    return {"benchmark": "BoolQ", "correct": correct, "total": total, "accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, choices = list(MODELS.keys()))
    parser.add_argument("--precision", required = True, choices = ["fp16", "nf4"])
    parser.add_argument("--n_samples", type = int, default = 200)
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n=== Measuring utility: {args.model} ({args.precision}) ===\n")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    
    # Run BoolQ
    boolq_results = eval_boolq(model, tokenizer, n_samples = args.n_samples)
    
    print(f"\n  BoolQ Accuracy: {boolq_results['accuracy']:.1f}% "
          f"({boolq_results['correct']}/{boolq_results['total']})")
    
    # Save
    results = {
        "model": args.model,
        "precision": args.precision,
        "benchmarks": [boolq_results],
    }
    
    results_path = os.path.join(output_dir, f"utility_{args.precision}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent = 2)
    
    print(f"\n  Results saved to {results_path}")
    
    del model
    print("=== Utility measurement complete ===")


if __name__ == "__main__":
    main()
