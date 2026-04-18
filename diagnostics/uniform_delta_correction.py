"""
Uniform Delta Correction: Simple interpolation using INT8/NF4 references.

No error-weighting — just plain: W_corrected = W_quant + α × (W_ref - W_quant)

This is the simplest possible correction method. If it works, it's the most
practical approach: no refusal direction, no activation collection, no error
computation. Just blend two quantized models.

Usage:
    python diagnostics/uniform_delta_correction.py --model llama7b --quantizer aqlm2
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import gc
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


def apply_uniform_blend(quant_model, ref_weights, quant_weights, proj_paths, alpha, device):
    """Simple uniform blend: W = (1-α) × W_quant + α × W_ref"""
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

            q_w = quant_weights[l][proj].to(device)
            ref_w = ref_weights[l][proj].to(device)

            # Simple uniform interpolation
            blended = (1 - alpha) * q_w + alpha * ref_w

            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(blended.to(act_dtype), requires_grad=False)
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
        args.output = f"outputs/uniform_delta_{args.model}_{args.quantizer}.json"

    print("=" * 70)
    print(f"  Uniform Delta Correction (no error weighting)")
    print(f"  Model: {args.model} | Quantizer: {args.quantizer}")
    print(f"  W_corrected = (1-α) × W_quant + α × W_ref")
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

    fp32_responses = generate_only(fp32_model, tokenizer, eval_items, desc="FP32")
    fp32_arc = measure_arc(fp32_model, tokenizer, arc_items)
    all_responses["fp32"] = fp32_responses
    all_arc["fp32"] = fp32_arc
    print(f"  FP32 ARC: {fp32_arc:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Quantized baseline
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
    # Step 3: Process each reference ONE AT A TIME
    # ================================================================
    alphas = [0.25, 0.50, 0.75, 1.0]
    conditions = []

    for src_name, src_quantizer in [("int8", "int8"), ("nf4", "nf4")]:
        print(f"\n{'='*70}")
        print(f"  Reference: {src_name}")
        print(f"{'='*70}")

        # Load reference weights
        print(f"  Loading {src_name} model...")
        ref_model, _ = load_quantized(args.model, quantizer=src_quantizer)
        ref_weights = extract_all_weights(ref_model, ALL_PROJS, device)
        del ref_model; gc.collect(); torch.cuda.empty_cache()

        # Load quant weights
        print(f"  Loading {args.quantizer} weights...")
        q_tmp, _ = load_quantized(args.model, quantizer=args.quantizer)
        quant_weights = extract_all_weights(q_tmp, ALL_PROJS, device)
        del q_tmp; gc.collect(); torch.cuda.empty_cache()

        # Run experiments
        for alpha in alphas:
            tag = f"uniform_{src_name}_a{int(alpha*100)}"
            print(f"\n  [{tag}] {src_name} ref, α={alpha}...")

            q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
            apply_uniform_blend(q_model, ref_weights, quant_weights, ALL_PROJS,
                                alpha, device)

            all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
            all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
            print(f"  ARC: {all_arc[tag]:.0f}%")
            conditions.append((tag, src_name, alpha))
            del q_model; gc.collect(); torch.cuda.empty_cache()

        # Free weights
        del ref_weights, quant_weights; gc.collect()

    # ================================================================
    # Classify
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
    # Results
    # ================================================================
    fp32_asr = results["fp32"]["asr"]
    q_asr = results[args.quantizer]["asr"]
    damage = q_asr - fp32_asr

    print("\n" + "=" * 70)
    print(f"  Uniform Delta: {args.model} / {args.quantizer}")
    print(f"  Damage: {damage:+.1f}pp (FP32={fp32_asr:.0f}%, {args.quantizer}={q_asr:.0f}%)")
    print("=" * 70)

    print(f"\n{'Condition':<28s} {'ASR':>6s} {'ΔASR':>8s} {'Recov':>7s} {'ARC':>5s} {'Src':>5s} {'α':>5s}")
    print("-" * 70)
    print(f"{'fp32':<28s} {fp32_asr:>5.1f}% {'---':>8s} {'':>7s} {fp32_arc:>4.0f}%")
    print(f"{args.quantizer:<28s} {q_asr:>5.1f}% {damage:>+7.1f}pp {'0%':>7s} {q_arc:>4.0f}%")

    for tag, src, alpha in conditions:
        asr = results[tag]["asr"]
        delta = asr - fp32_asr
        recov = f"{-(asr-q_asr)/damage*100:.0f}%" if abs(damage) > 0.5 else "N/A"
        arc = all_arc[tag]
        print(f"{tag:<28s} {asr:>5.1f}% {delta:>+7.1f}pp {recov:>7s} {arc:>4.0f}% {src:>5s} {alpha:>5.2f}")

    # Compare with error-weighted results
    print(f"\n{'='*70}")
    print(f"  Comparison: Uniform vs Error-Weighted (from previous run)")
    print(f"{'='*70}")
    print(f"  (Load error-weighted results from outputs/delta_correction_{args.model}_{args.quantizer}.json)")

    # Save
    out_data = {
        "model": args.model, "quantizer": args.quantizer,
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
