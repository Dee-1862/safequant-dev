"""
SafeQuant Phase 1, Step 2c — Measure Perplexity (Standard Quality Metric).

Shows that standard quantization quality metrics (perplexity on WikiText-2)
barely change after NF4 quantization, even though safety behavior degrades
significantly. This is the core motivation for the entire paper:

    STANDARD METRICS SAY THE MODEL IS FINE.
    SAFETY METRICS SAY IT'S BROKEN.

Perplexity measures how well the model predicts the next token in a held-out
text corpus. It's the standard metric used by the quantization community
(GPTQ, AWQ, QuIP papers all report it) to evaluate quantization quality.

Run:
  python phase1_problem/step2_behavior/measure_perplexity.py --model qwen1.5b --precision fp16
  python phase1_problem/step2_behavior/measure_perplexity.py --model qwen1.5b --precision nf4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG


def load_model(model_name, precision):
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPE,
            device_map="auto", trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def compute_perplexity(model, tokenizer, text, max_length=2048, stride=512):
    """
    Compute perplexity using sliding window approach.
    Standard method from Hugging Face perplexity documentation.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, stride), desc="  Computing perplexity"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop

        input_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_slice.clone()
        target_ids[:, :-trg_len] = -100  # mask non-target tokens

        with torch.no_grad():
            outputs = model(input_slice, labels=target_ids)
            nll = outputs.loss.float().item()

        nlls.append(nll)
        prev_end_loc = end_loc

        if end_loc >= seq_len:
            break

    ppl = math.exp(sum(nlls) / len(nlls))
    return ppl


def load_wikitext():
    """Load WikiText-2 test set — the standard benchmark for perplexity."""
    print("  Loading WikiText-2 test set...")
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join([t for t in dataset["text"] if t.strip()])
        print(f"  Loaded {len(text)} characters")
        return text
    except Exception as e:
        print(f"  Error loading WikiText-2: {e}")
        print("  Falling back to a small text sample...")
        return ("The quick brown fox jumps over the lazy dog. " * 500 +
                "Machine learning is a subfield of artificial intelligence. " * 500)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Measuring Perplexity: {args.model} ({args.precision}) ===\n")

    # Load model
    model, tokenizer = load_model(model_name, args.precision)

    # Load text
    text = load_wikitext()

    # Compute perplexity
    ppl = compute_perplexity(model, tokenizer, text,
                             max_length=args.max_length,
                             stride=args.stride)

    print(f"\n  Perplexity ({args.precision}): {ppl:.2f}")

    # Save
    results = {
        "model": args.model,
        "precision": args.precision,
        "perplexity": ppl,
        "dataset": "wikitext-2-raw-v1",
        "max_length": args.max_length,
        "stride": args.stride,
    }

    save_path = os.path.join(output_dir, f"perplexity_{args.precision}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved to {save_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("=== Perplexity measurement complete ===")


if __name__ == "__main__":
    main()
