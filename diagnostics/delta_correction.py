"""
Delta Correction: Safety-Preserving Quantization via Lightweight Adapters.

Instead of needing FP32 weights at deploy time, we:
1. Compute correction delta from a cheaper reference (INT8, NF4)
2. Store delta as a lightweight adapter
3. Apply at load time: W_corrected = W_2bit + delta

Compares correction sources: FP32 vs INT8 vs NF4
Tests α values: 0.25, 0.50, 0.75, 1.0
Reports: ASR, ARC, delta size, compression ratio

Usage:
    python diagnostics/delta_correction.py --model llama7b --quantizer aqlm2
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import gc
import os
import numpy as np
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import load_fp32, load_quantized, get_weight
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior, _logprob_score

ALL_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
             "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def _get_submodule(layer, proj_path):
    parts = proj_path.split(".")
    m = layer
    for p in parts:
        m = getattr(m, p)
    return m


def _set_submodule(layer, proj_path, new_module):
    parts = proj_path.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _get_inner_model(model):
    inner = model.model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def extract_all_weights(model, proj_paths, device):
    """Extract weights from all layers for all projections."""
    inner = _get_inner_model(model)
    n_layers = model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for l in range(n_layers):
            layer = inner.layers[l]
            weights[l] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[l][proj] = get_weight(m, as_dtype=torch.float32).cpu().clone()
    return weights


def compute_delta(ref_weights, quant_weights, proj_paths, n_layers):
    """Compute per-dimension error-weighted delta between reference and quantized."""
    hidden_dim = ref_weights[0][proj_paths[0]].shape[1]

    # First compute per-dim quantization error for error-weighting
    error_per_dim = torch.zeros(hidden_dim)
    for l in range(n_layers):
        for proj in proj_paths:
            ref_w = ref_weights[l][proj]
            q_w = quant_weights[l][proj]
            in_f, out_f = ref_w.shape[1], ref_w.shape[0]
            if in_f == hidden_dim:
                error_per_dim += (ref_w - q_w).norm(dim=0).cpu()
            elif out_f == hidden_dim:
                error_per_dim += (ref_w - q_w).norm(dim=1).cpu()

    # Normalize to [0, 1]
    error_importance = error_per_dim / error_per_dim.max()

    # Compute delta per layer per projection
    deltas = {}
    total_params = 0
    total_nonzero = 0
    for l in range(n_layers):
        deltas[l] = {}
        for proj in proj_paths:
            ref_w = ref_weights[l][proj]
            q_w = quant_weights[l][proj]
            raw_delta = ref_w - q_w

            # Error-weighted: apply per-dim importance
            in_f, out_f = raw_delta.shape[1], raw_delta.shape[0]
            if in_f == hidden_dim:
                weighted_delta = raw_delta * error_importance.unsqueeze(0)
            elif out_f == hidden_dim:
                weighted_delta = raw_delta * error_importance.unsqueeze(1)
            else:
                weighted_delta = raw_delta

            deltas[l][proj] = weighted_delta
            total_params += weighted_delta.numel()
            total_nonzero += (weighted_delta.abs() > 1e-6).sum().item()

    return deltas, error_importance, total_params, total_nonzero


def apply_delta(quant_model, deltas, proj_paths, alpha, device):
    """Apply pre-computed delta to quantized model."""
    inner = _get_inner_model(quant_model)
    act_dtype = inner.norm.weight.dtype
    n_layers = quant_model.config.num_hidden_layers

    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, 'in_features', None) or getattr(orig_module, 'infeatures', None)
            out_f = getattr(orig_module, 'out_features', None) or getattr(orig_module, 'outfeatures', None)
            has_bias = orig_module.bias is not None

            q_w = get_weight(orig_module, as_dtype=torch.float32).to(device)
            delta = deltas[l][proj].to(device)
            corrected = q_w + alpha * delta

            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(corrected.to(act_dtype), requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(),
                                               requires_grad=False)
            _set_submodule(layer, proj, new_linear)


def measure_arc(model, tokenizer, arc_items):
    device = next(model.parameters()).device
    correct = 0
    for item in arc_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)
        if pred == item["answer_idx"]:
            correct += 1
    return 100.0 * correct / len(arc_items)


def compute_delta_size_mb(deltas, proj_paths, n_layers):
    """Compute size of delta in MB (FP16 storage)."""
    total_elements = 0
    for l in range(n_layers):
        for proj in proj_paths:
            total_elements += deltas[l][proj].numel()
    # FP16 = 2 bytes per element
    return total_elements * 2 / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama7b")
    parser.add_argument("--quantizer", default="aqlm2")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_reasoning", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output is None:
        args.output = f"outputs/delta_correction_{args.model}_{args.quantizer}.json"

    print("=" * 70)
    print(f"  Delta Correction: Safety-Preserving Quantization")
    print(f"  Model: {args.model} | Quantizer: {args.quantizer}")
    print(f"  Correction sources: FP32, INT8, NF4")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    all_responses = {}
    all_arc = {}

    # ================================================================
    # Step 1: FP32 baseline
    # ================================================================
    print("\n[Step 1] FP32 baseline...")
    fp32_model, tokenizer = load_fp32(args.model)
    n_layers = fp32_model.config.num_hidden_layers
    hidden_dim = fp32_model.config.hidden_size

    print("  Extracting FP32 weights...")
    fp32_weights = extract_all_weights(fp32_model, ALL_PROJS, device)

    fp32_responses = generate_only(fp32_model, tokenizer, eval_items, desc="FP32")
    fp32_arc = measure_arc(fp32_model, tokenizer, arc_items)
    all_responses["fp32"] = fp32_responses
    all_arc["fp32"] = fp32_arc
    print(f"  FP32 ARC: {fp32_arc:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Quantized baseline (generate only, don't store weights)
    # ================================================================
    print(f"\n[Step 2] {args.quantizer} baseline...")
    q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
    q_responses = generate_only(q_model, tokenizer, eval_items, desc=args.quantizer)
    q_arc = measure_arc(q_model, tokenizer, arc_items)
    all_responses[args.quantizer] = q_responses
    all_arc[args.quantizer] = q_arc
    print(f"  {args.quantizer} ARC: {q_arc:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 3-5: Process each reference source ONE AT A TIME
    # Peak memory: fp32_weights(~16GB) + quant_weights(~16GB) + deltas(~16GB) = ~48GB
    # After delta computed: fp32_weights(~16GB) + deltas(~16GB) = ~32GB
    # ================================================================
    alphas = [0.25, 0.50, 0.75, 1.0]
    conditions = []

    for src_name, src_quantizer in [("fp32", None), ("int8", "int8"), ("nf4", "nf4")]:
        print(f"\n{'='*70}")
        print(f"  Processing reference: {src_name}")
        print(f"{'='*70}")

        # Free fp32_weights before loading other refs to reduce peak memory
        if src_name != "fp32" and fp32_weights is not None:
            del fp32_weights; fp32_weights = None; gc.collect()

        # Get reference weights
        if src_name == "fp32":
            ref_weights = fp32_weights
        else:
            print(f"  Loading {src_name} model...")
            ref_model, _ = load_quantized(args.model, quantizer=src_quantizer)
            ref_weights = extract_all_weights(ref_model, ALL_PROJS, device)
            del ref_model; gc.collect(); torch.cuda.empty_cache()

        # Load quant weights fresh for delta computation
        print(f"  Loading {args.quantizer} for delta computation...")
        q_tmp, _ = load_quantized(args.model, quantizer=args.quantizer)
        quant_weights = extract_all_weights(q_tmp, ALL_PROJS, device)
        del q_tmp; gc.collect(); torch.cuda.empty_cache()

        # Compute delta — then immediately free ref and quant weights
        print(f"  Computing {src_name} delta...")
        deltas, error_imp, total_p, nonzero_p = compute_delta(
            ref_weights, quant_weights, ALL_PROJS, n_layers)
        delta_mb = compute_delta_size_mb(deltas, ALL_PROJS, n_layers)
        print(f"  {src_name} delta: {delta_mb:.0f} MB (FP16), "
              f"{total_p/1e6:.0f}M params, sparsity: {1-nonzero_p/total_p:.1%}")

        # Free BOTH ref and quant weights immediately — only keep deltas (~16GB)
        del quant_weights, ref_weights; gc.collect()

        # Run correction experiments (only deltas + GPU model in memory)
        for alpha in alphas:
            tag = f"{src_name}_a{int(alpha*100)}"
            print(f"\n  [{tag}] {src_name} reference, α={alpha}...")

            q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
            apply_delta(q_model, deltas, ALL_PROJS, alpha, device)

            all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
            all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
            print(f"  ARC: {all_arc[tag]:.0f}%")
            conditions.append((tag, src_name, alpha))
            del q_model; gc.collect(); torch.cuda.empty_cache()

        # Free deltas
        del deltas; gc.collect()

    if fp32_weights is not None:
        del fp32_weights; gc.collect()

    # ================================================================
    # Step 6: Classify all responses
    # ================================================================
    print("\n" + "=" * 70)
    print("  Classifying all responses...")
    print("=" * 70)

    behaviors = [_extract_behavior(item) for item in eval_items]
    results = {}
    for tag, responses in all_responses.items():
        judgments = classify_batch(behaviors, responses)
        asr = 100.0 * sum(judgments) / len(eval_items)
        results[tag] = {"asr": asr, "arc": all_arc.get(tag)}

    # ================================================================
    # Step 7: Print results
    # ================================================================
    fp32_asr = results["fp32"]["asr"]
    q_asr = results[args.quantizer]["asr"]
    damage = q_asr - fp32_asr

    print("\n" + "=" * 70)
    print(f"  Delta Correction: {args.model} / {args.quantizer}")
    print(f"  Damage: {damage:+.1f}pp (FP32={fp32_asr:.0f}%, {args.quantizer}={q_asr:.0f}%)")
    print("=" * 70)

    # Header
    print(f"\n{'Condition':<20s} {'ASR':>6s} {'ΔASR':>8s} {'Recov':>7s} {'ARC':>5s} {'Source':>8s} {'α':>5s}")
    print("-" * 65)
    print(f"{'fp32':<20s} {fp32_asr:>5.1f}% {'---':>8s} {'':>7s} {fp32_arc:>4.0f}%  {'---':>8s} {'---':>5s}")
    print(f"{args.quantizer:<20s} {q_asr:>5.1f}% {damage:>+7.1f}pp {'0%':>7s} {q_arc:>4.0f}%  {'---':>8s} {'---':>5s}")

    for tag, src_name, alpha in conditions:
        asr = results[tag]["asr"]
        delta_from_fp32 = asr - fp32_asr
        if abs(damage) > 0.5:
            recovery = f"{-(asr - q_asr)/damage*100:.0f}%"
        else:
            recovery = "N/A"
        arc = all_arc.get(tag, 0)
        print(f"{tag:<20s} {asr:>5.1f}% {delta_from_fp32:>+7.1f}pp {recovery:>7s} {arc:>4.0f}%  {src_name:>8s} {alpha:>5.2f}")

    # Comparison table: best per source
    print(f"\n{'='*65}")
    print(f"  Best result per correction source:")
    print(f"{'='*65}")
    for src_name in ["fp32", "int8", "nf4"]:
        best_tag = None
        best_dist = float('inf')
        for tag, sn, alpha in conditions:
            if sn == src_name:
                dist = abs(results[tag]["asr"] - fp32_asr)
                if dist < best_dist:
                    best_dist = dist
                    best_tag = (tag, alpha)
        if best_tag:
            tag, alpha = best_tag
            asr = results[tag]["asr"]
            arc = all_arc[tag]
            # Delta size is the same for all sources (same number of params)
            # Approximate: 7 projections × n_layers × hidden_dim² × 2 bytes (FP16)
            approx_delta_mb = 7 * n_layers * hidden_dim * hidden_dim * 2 / (1024 * 1024)
            print(f"  {src_name:>5s} ref: {tag:<18s} ASR={asr:>5.1f}% ARC={arc:.0f}% "
                  f"(delta: ~{approx_delta_mb:.0f}MB, α={alpha})")

    # Save
    out_data = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_eval": args.n_eval,
        "fp32_asr": fp32_asr, "quantized_asr": q_asr, "damage_pp": damage,
        "fp32_arc": fp32_arc, "quantized_arc": q_arc,
        "conditions": {tag: {"asr": results[tag]["asr"], "arc": all_arc[tag],
                              "source": src, "alpha": a}
                       for tag, src, a in conditions},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
