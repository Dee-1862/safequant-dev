#!/usr/bin/env python
"""Localize the direction_drift measurement bug for GPTQ.

The gptq4 model generates coherently (see check_gptq_quality), so the ~0.5
refusal-direction cosine is a MEASUREMENT artifact, not a bad quantization.
This script pinpoints where the residual capture diverges.

TEST 1 -- hook vs output_hidden_states (gptq only):
  For each prompt+layer, capture the residual BOTH via the forward hook (exactly
  as extract_last_token_residuals does) AND via the model's own
  output_hidden_states, in the SAME forward pass. If mean cos(hook, hs) < 1.0,
  the forward-hook capture is unfaithful for gptqmodel-loaded models -> the fix
  is to extract residuals from output_hidden_states instead of hooks.

TEST 2 -- FP32 vs GPTQ raw residuals (output_hidden_states, apples-to-apples):
  Compare per-prompt residuals FP32 vs GPTQ per layer. If these are ~0.99, the
  models agree and any low direction-cosine is a downstream (diff-in-means)
  artifact; if they diverge, the residual streams genuinely differ.

Usage:  python -m scripts.diagnose_gptq_residuals --model llama7b --quantizer gptq4 --n 24
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts  # noqa: E402
from core.model_loader import load_fp32, load_quantized  # noqa: E402
from core.residuals import default_token_pos, get_inner, prompt_text  # noqa: E402


def _cos(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-9))


def capture_both(model, tok, prompts, layers, device, token_pos):
    """Capture each layer's token_pos residual via forward-hook AND via
    output_hidden_states, in one forward per prompt. Returns two dicts
    {layer -> [per-prompt vectors]}."""
    inner = get_inner(model)
    buf = {}

    def mk(l):
        def h(_m, _i, o):
            r = o[0] if isinstance(o, (tuple, list)) else o
            if r.dim() == 3:
                buf[l] = r[:, token_pos, :].detach().cpu().float().squeeze(0)
        return h

    handles = [inner.layers[l].register_forward_hook(mk(l)) for l in layers]
    hook_caps = {l: [] for l in layers}
    hs_caps = {l: [] for l in layers}
    try:
        for p in prompts:
            text = prompt_text(p, tok)
            inputs = tok(text, return_tensors="pt", truncation=True,
                         max_length=512).to(device)
            buf.clear()
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states  # tuple length n_layers+1; hs[l+1] == layer l output
            for l in layers:
                hook_caps[l].append(buf.get(l))
                hs_caps[l].append(
                    hs[l + 1][:, token_pos, :].detach().cpu().float().squeeze(0)
                )
    finally:
        for h in handles:
            h.remove()
    return hook_caps, hs_caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="gptq4")
    ap.add_argument("--n", type=int, default=24, help="prompts per class")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tp = default_token_pos(args.model)
    print(f"token_pos = {tp}")

    harmful = [
        (it.get("behavior") or it.get("prompt") or "") if isinstance(it, dict) else str(it)
        for it in load_wildjailbreak_prompts(n_samples=args.n, config="train")
    ]
    harmless = load_harmless_prompts(n_samples=args.n)
    prompts = harmful + harmless

    # --- gptq: hook vs hidden_states ---
    qmodel, tok = load_quantized(args.model, args.quantizer)
    n_layers = qmodel.config.num_hidden_layers
    layers = list(range(n_layers))
    q_hook, q_hs = capture_both(qmodel, tok, prompts, layers, device, tp)

    print("\n=== TEST 1: hook vs output_hidden_states (gptq) ===")
    print("  cos < 1.0 => the forward-HOOK capture is the bug\n")
    worst, worst_l = 1.0, -1
    for l in layers:
        pairs = [(a, b) for a, b in zip(q_hook[l], q_hs[l]) if a is not None]
        cs = float(np.mean([_cos(a, b) for a, b in pairs])) if pairs else float("nan")
        if l % 8 == 0 or cs < 0.999:
            print(f"   layer {l:2d}: mean cos(hook, hidden_states) = {cs:.4f}")
        if cs < worst:
            worst, worst_l = cs, l
    print(f"  --> WORST layer agreement: {worst:.4f} (layer {worst_l})")

    # --- fp32 vs gptq raw residuals (via hidden_states) ---
    del qmodel
    torch.cuda.empty_cache()
    fp, _ = load_fp32(args.model)
    _, f_hs = capture_both(fp, tok, prompts, layers, device, tp)

    print("\n=== TEST 2: FP32 vs GPTQ raw residuals (output_hidden_states) ===")
    print("  cos ~0.99 => residual streams agree; low direction-cosine is downstream\n")
    for l in layers:
        cs = float(np.mean([_cos(a, b) for a, b in zip(f_hs[l], q_hs[l])]))
        if l % 4 == 0:
            print(f"   layer {l:2d}: mean cos(fp32, gptq) raw residual = {cs:.4f}")


if __name__ == "__main__":
    main()
