#!/usr/bin/env python
"""Quick model-quality check for a quantized checkpoint.

Purpose: discriminate the two hypotheses for why our GPTQ refusal-direction
cosine reads ~0.5 instead of the literature's >0.97:
  - GARBAGE generations  -> the load kernel / quantization corrupts the forward
    pass. The model is broken; switching to a different toolchain (auto-gptq)
    or loader is the fix.
  - COHERENT generations -> the model is fine, so the ~0.5 is a MEASUREMENT bug
    in the direction-drift residual extraction, not the quantization.

It loads through core.model_loader.load_quantized -- the SAME path the
experiments use (shim, kernel, v1->v2 conversion and all) -- so it tests the
real stack, not an idealized one. Greedy decoding for determinism.

Usage:
  python -m scripts.check_gptq_quality --model llama7b --quantizer gptq4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from core.model_loader import load_quantized  # noqa: E402

PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is, in two sentences.",
    "Write a haiku about the ocean.",
    "List three prime numbers.",
]


def main():
    ap = argparse.ArgumentParser(description="Coherence check for a quantized model.")
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="gptq4")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    args = ap.parse_args()

    print(f"Loading {args.model} / {args.quantizer} ...", flush=True)
    model, tok = load_quantized(args.model, args.quantizer)
    model.eval()
    dev = next(model.parameters()).device

    for p in PROMPTS:
        msgs = [{"role": "user", "content": p}]
        input_ids = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(dev)
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        text = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
        print("\n" + "=" * 64, flush=True)
        print(f"PROMPT: {p}")
        print(f"OUTPUT: {text.strip()}")

    print("\n" + "=" * 64)
    print("VERDICT GUIDE:")
    print("  COHERENT answers -> model is FINE -> the ~0.5 is a MEASUREMENT bug")
    print("                      in direction_drift, NOT the quantization.")
    print("  GARBAGE answers  -> forward pass is CORRUPTED -> kernel/toolchain")
    print("                      problem -> auto-gptq (classic kernels) is the fix.")


if __name__ == "__main__":
    main()
