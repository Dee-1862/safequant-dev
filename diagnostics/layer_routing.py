import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.seeds import set_global_seed, DEFAULT_SEED
    SEED = set_global_seed(DEFAULT_SEED)
except ModuleNotFoundError:
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts


def _get_inner(model):
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def format_prompt(p, tok):
    text = p.get("prompt") or p.get("vanilla") or p.get("behavior") or "" if isinstance(p, dict) else str(p)
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        return text


def extract_last_token_residuals(model, tokenizer, prompts, layers, device):
    """Returns {layer: list of (hidden,) CPU tensors, one per prompt}."""
    inner = _get_inner(model)
    buf = {}

    def make_hook(l):
        def hook(module, inputs, output):
            r = output[0] if isinstance(output, (tuple, list)) else output
            if r.dim() == 3:
                buf[l] = r[:, -1, :].detach().cpu().float().squeeze(0)
        return hook

    handles = [inner.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    out = {l: [] for l in layers}
    try:
        for p in prompts:
            text = format_prompt(p, tokenizer)
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512).to(device)
            buf.clear()
            with torch.no_grad():
                _ = model(**inputs)
            for l, r in buf.items():
                out[l].append(r)
    finally:
        for h in handles:
            h.remove()
    return out


def compute_r_hat(harmful_lasts, harmless_lasts):
    h = torch.stack(harmful_lasts).mean(0)
    s = torch.stack(harmless_lasts).mean(0)
    r = h - s
    n = r.norm()
    return r / n if n > 1e-6 else r


def per_layer_separation(harmful_lasts, harmless_lasts, r_hat):
    h_scores = np.array([float(t @ r_hat) for t in harmful_lasts], dtype=np.float64)
    s_scores = np.array([float(t @ r_hat) for t in harmless_lasts], dtype=np.float64)
    pooled = np.sqrt((h_scores.var() + s_scores.var()) / 2 + 1e-12)
    return float((h_scores.mean() - s_scores.mean()) / pooled)


def per_prompt_projections(residuals_by_layer, r_hat):
    """
    Returns {prompt_idx: [layer_projections]}.
    residuals_by_layer is dict {L: [tensor_per_prompt]}.
    """
    layers = sorted(residuals_by_layer.keys())
    n_prompts = len(residuals_by_layer[layers[0]])
    out = np.zeros((n_prompts, len(layers)), dtype=np.float64)
    for li, L in enumerate(layers):
        for pi, t in enumerate(residuals_by_layer[L]):
            out[pi, li] = float(t @ r_hat)
    return out, layers


def operative_layer_per_prompt(projections, layers):
    """Returns list of layer indices, one per prompt, where |projection| is max."""
    abs_proj = np.abs(projections)
    argmax_indices = abs_proj.argmax(axis=1)
    return [layers[i] for i in argmax_indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="awq")
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = f"outputs/layer_routing_{args.model}_{args.quantizer}.json"

    print(f"=== Layer routing analysis | {args.model} / {args.quantizer} | n_calib={args.n_calib} | seed={SEED}")

    # Calibration prompts
    print("Loading calibration prompts...")
    harmful_full = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless_full = load_harmless_prompts(n_samples=args.n_calib)
    harmful = harmful_full
    harmless = harmless_full
    print(f"  n harmful: {len(harmful)}  n harmless: {len(harmless)}")

    # FP32 pass
    print(f"\n[FP32] loading {args.model} FP32 ...")
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    n_layers = len(_get_inner(fp32_model).layers)
    layers = list(range(n_layers))
    print(f"  n_layers = {n_layers}")

    print("[FP32] extracting per-layer residuals ...")
    fp_h_resid = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s_resid = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)

    # Per-layer r_hat and sigma on FP32
    fp_r_hats = {L: compute_r_hat(fp_h_resid[L], fp_s_resid[L]) for L in layers}
    fp_seps = {L: per_layer_separation(fp_h_resid[L], fp_s_resid[L], fp_r_hats[L]) for L in layers}
    fp32_best_layer = max(layers, key=lambda L: fp_seps[L])
    r_hat_FP = fp_r_hats[fp32_best_layer]
    print(f"  FP32 best-sigma layer = {fp32_best_layer} (sigma={fp_seps[fp32_best_layer]:.3f})")

    # Free FP32 model
    del fp32_model
    import gc; gc.collect(); torch.cuda.empty_cache()

    # Quant pass
    print(f"\n[{args.quantizer}] loading quantized model ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device

    print(f"[{args.quantizer}] extracting per-layer residuals ...")
    q_h_resid = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s_resid = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)

    q_r_hats = {L: compute_r_hat(q_h_resid[L], q_s_resid[L]) for L in layers}
    q_seps = {L: per_layer_separation(q_h_resid[L], q_s_resid[L], q_r_hats[L]) for L in layers}
    quant_best_layer = max(layers, key=lambda L: q_seps[L])
    r_hat_q = q_r_hats[quant_best_layer]
    print(f"  quant best-sigma layer = {quant_best_layer} (sigma={q_seps[quant_best_layer]:.3f})")

    del q_model
    gc.collect(); torch.cuda.empty_cache()

    # === Routing analysis ===
    # In the QUANT model's residual stream, where does the activation
    # have peak projection onto r_hat_FP and r_hat_q respectively?
    print(f"\n=== Computing per-prompt operative layers ===")
    q_proj_FP_harmful,  _ = per_prompt_projections(q_h_resid, r_hat_FP)
    q_proj_FP_harmless, _ = per_prompt_projections(q_s_resid, r_hat_FP)
    q_proj_q_harmful,   _ = per_prompt_projections(q_h_resid, r_hat_q)
    q_proj_q_harmless,  _ = per_prompt_projections(q_s_resid, r_hat_q)

    op_FP_harmful  = operative_layer_per_prompt(q_proj_FP_harmful,  layers)
    op_FP_harmless = operative_layer_per_prompt(q_proj_FP_harmless, layers)
    op_q_harmful   = operative_layer_per_prompt(q_proj_q_harmful,   layers)
    op_q_harmless  = operative_layer_per_prompt(q_proj_q_harmless,  layers)

    def summary(op_list):
        c = Counter(op_list)
        mode_layer, mode_count = c.most_common(1)[0]
        return {
            "mode_layer": int(mode_layer),
            "mode_count": int(mode_count),
            "mode_fraction": float(mode_count / len(op_list)),
            "mean": float(np.mean(op_list)),
            "std": float(np.std(op_list)),
            "n_distinct_layers": len(c),
            "concentration_top3_fraction": float(sum(n for _, n in c.most_common(3)) / len(op_list)),
        }

    routing_summary = {
        "operative_layer_FP_rhat_harmful": summary(op_FP_harmful),
        "operative_layer_FP_rhat_harmless": summary(op_FP_harmless),
        "operative_layer_q_rhat_harmful":  summary(op_q_harmful),
        "operative_layer_q_rhat_harmless": summary(op_q_harmless),
    }

    print(f"\n  Operative layer (peak |projection|) summary in quant model:")
    print(f"    r_hat_FP on harmful prompts:  mode={routing_summary['operative_layer_FP_rhat_harmful']['mode_layer']} "
          f"({routing_summary['operative_layer_FP_rhat_harmful']['mode_fraction']:.1%}), "
          f"mean={routing_summary['operative_layer_FP_rhat_harmful']['mean']:.1f}, "
          f"std={routing_summary['operative_layer_FP_rhat_harmful']['std']:.2f}")
    print(f"    r_hat_FP on harmless prompts: mode={routing_summary['operative_layer_FP_rhat_harmless']['mode_layer']} "
          f"({routing_summary['operative_layer_FP_rhat_harmless']['mode_fraction']:.1%})")
    print(f"    r_hat_q  on harmful prompts:  mode={routing_summary['operative_layer_q_rhat_harmful']['mode_layer']} "
          f"({routing_summary['operative_layer_q_rhat_harmful']['mode_fraction']:.1%}), "
          f"mean={routing_summary['operative_layer_q_rhat_harmful']['mean']:.1f}, "
          f"std={routing_summary['operative_layer_q_rhat_harmful']['std']:.2f}")
    print(f"    r_hat_q  on harmless prompts: mode={routing_summary['operative_layer_q_rhat_harmless']['mode_layer']} "
          f"({routing_summary['operative_layer_q_rhat_harmless']['mode_fraction']:.1%})")
    print(f"\n  Script-chosen layers: fp32_best_layer={fp32_best_layer}, quant_best_layer={quant_best_layer}")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_calib": args.n_calib,
        "n_layers": n_layers,
        "seed": SEED,
        "fp32_best_layer": fp32_best_layer,
        "quant_best_layer": quant_best_layer,
        "fp32_separations": fp_seps,
        "quant_separations": q_seps,
        "routing_summary": routing_summary,
        "operative_layers_FP_rhat_harmful": op_FP_harmful,
        "operative_layers_FP_rhat_harmless": op_FP_harmless,
        "operative_layers_q_rhat_harmful": op_q_harmful,
        "operative_layers_q_rhat_harmless": op_q_harmless,
        "mean_abs_proj_per_layer_q_resid_onto_r_hat_FP_harmful":  np.abs(q_proj_FP_harmful).mean(0).tolist(),
        "mean_abs_proj_per_layer_q_resid_onto_r_hat_FP_harmless": np.abs(q_proj_FP_harmless).mean(0).tolist(),
        "mean_abs_proj_per_layer_q_resid_onto_r_hat_q_harmful":   np.abs(q_proj_q_harmful).mean(0).tolist(),
        "mean_abs_proj_per_layer_q_resid_onto_r_hat_q_harmless":  np.abs(q_proj_q_harmless).mean(0).tolist(),
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
