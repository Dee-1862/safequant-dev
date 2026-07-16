#!/usr/bin/env python
"""Kernel test: load OUR gptq4 checkpoint via auto-gptq's exLlama kernel instead
of the gptqmodel-Triton fallback, and compare its residual stream to FP32.

Background: diagnose_gptq_residuals showed the hooks are faithful (TEST 1) but
the gptqmodel-Triton-loaded model's residuals diverge from FP32 to ~0.72 by
layer 4 (TEST 2) -- a broken forward pass, because the gptqmodel CUDA/exLlama
extension can't build under torch 2.9 and it falls back to a numerically-wrong
Triton kernel. This reloads the SAME checkpoint through auto-gptq's exLlama
kernel and re-measures.

  ~0.99 across layers -> exLlama is faithful; the fix is to load GPTQ via
                         auto-gptq, not gptqmodel. (Then wire that into _load_gptq.)
  still ~0.72         -> the problem is deeper than the kernel choice.
  load error          -> auto-gptq can't read the gptqmodel-format checkpoint;
                         pivot to reputable auto-gptq-format checkpoints.

Usage:  python -m scripts.diag_autogptq_kernel --model llama7b --n 24
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from config import QUANTIZED_CHECKPOINTS  # noqa: E402
from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts  # noqa: E402
from core.model_loader import load_fp32  # noqa: E402
from core.residuals import default_token_pos, prompt_text  # noqa: E402


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


def hs_residuals(model, tok, prompts, layers, device, token_pos):
    """Per-layer token_pos residuals via output_hidden_states (kernel-agnostic)."""
    caps = {l: [] for l in layers}
    for p in prompts:
        text = prompt_text(p, tok)
        inputs = tok(text, return_tensors="pt", truncation=True,
                     max_length=512).to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # len n_layers+1; hs[l+1] == layer l output
        for l in layers:
            caps[l].append(hs[l + 1][:, token_pos, :].detach().cpu().float().squeeze(0))
    return caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    device = "cuda"
    tp = default_token_pos(args.model)
    ckpt = QUANTIZED_CHECKPOINTS[(args.model, "gptq4")]
    print(f"checkpoint: {ckpt}\ntoken_pos:  {tp}")

    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM

    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

    print("\nLoading via auto-gptq (exLlamaV2 kernel) ...", flush=True)
    try:
        ag = AutoGPTQForCausalLM.from_quantized(
            ckpt,
            device="cuda:0",
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
            disable_exllamav2=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"\n[LOAD FAILED] auto-gptq could not load the checkpoint:\n  {type(e).__name__}: {e}")
        print("=> auto-gptq can't read the gptqmodel-format checkpoint; "
              "pivot to reputable auto-gptq-format checkpoints (hugging-quants).")
        return

    hf = ag.model  # underlying HF LlamaForCausalLM with auto-gptq QuantLinear
    hf.eval()

    # Report which quant-linear kernel is actually in use
    qtype = "?"
    for m in hf.modules():
        n = type(m).__name__
        if "QuantLinear" in n or "Exllama" in n:
            qtype = n
            break
    print(f"  loaded. quant-linear module: {qtype}")

    n_layers = hf.config.num_hidden_layers
    layers = list(range(n_layers))

    harmful = [
        (it.get("behavior") or it.get("prompt") or "") if isinstance(it, dict) else str(it)
        for it in load_wildjailbreak_prompts(n_samples=args.n, config="train")
    ]
    harmless = load_harmless_prompts(n_samples=args.n)
    prompts = harmful + harmless

    ag_caps = hs_residuals(hf, tok, prompts, layers, device, tp)
    del ag, hf
    torch.cuda.empty_cache()

    fp, _ = load_fp32(args.model)
    fp_caps = hs_residuals(fp, tok, prompts, layers, device, tp)

    print("\n=== auto-gptq(exLlama) gptq4 vs FP32 raw residuals ===")
    print("  ~0.99 => exLlama kernel is faithful -> switch _load_gptq to auto-gptq")
    print("  (compare to gptqmodel-Triton which read ~0.72 from layer 4 on)\n")
    worst = 1.0
    for l in layers:
        cs = float(np.mean([_cos(a, b) for a, b in zip(fp_caps[l], ag_caps[l])]))
        worst = min(worst, cs)
        if l % 4 == 0 or l == n_layers - 1:
            print(f"   layer {l:2d}: mean cos(fp32, autogptq-gptq) = {cs:.4f}")
    print(f"\n  WORST layer: {worst:.4f}   "
          f"({'FAITHFUL -> kernel was the bug' if worst > 0.95 else 'still degraded'})")


if __name__ == "__main__":
    main()
