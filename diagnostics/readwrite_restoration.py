"""
Read-Side vs Write-Side Restoration — generalized for any quantizer.

Tests whether quantization damages read-side (detection) or write-side
(refusal) matrices more, using FP32 weight restoration experiments.

Conditions tested:
  1. FP32 baseline
  2. Quantized baseline (nf4, gptq4, gptq3, etc.)
  3. Quantized + write matrices restored (o_proj, down_proj)
  4. Quantized + read matrices restored (q/k/v/gate/up_proj)
  5. Quantized + both restored

Usage:
    python diagnostics/readwrite_restoration.py --model llama7b --quantizer gptq4 --n_eval 100
    python diagnostics/readwrite_restoration.py --model mistral7b --quantizer nf4 --n_eval 100
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import gc
import random
import numpy as np
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import load_fp32, load_quantized, get_weight
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior, _logprob_score

# Matrix categorization
WRITE_PROJS = ["self_attn.o_proj", "mlp.down_proj"]
READ_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.gate_proj", "mlp.up_proj"]


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


def extract_fp32_weights(fp32_model, proj_paths):
    n_layers = fp32_model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for l in range(n_layers):
            layer = fp32_model.model.layers[l]
            weights[l] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[l][proj] = get_weight(m, as_dtype=torch.float32).cpu().clone()
    return weights


def _get_inner_model(model):
    """Unwrap to the inner model that has .layers and .norm (e.g. LlamaModel)."""
    inner = model.model
    # AutoGPTQForCausalLM adds an extra wrapper: model.model is LlamaForCausalLM
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def _is_awq_module(module):
    """Check if a module is an AWQ WQLinear layer."""
    cls_name = type(module).__name__
    return "WQLinear" in cls_name


def _transform_fp32_to_awq_space(fp32_weight, awq_module):
    """
    Transform FP32 weight into AWQ's scaled weight space.

    AWQ stores: W_quantized ≈ (W_fp32 * scale), then quantizes.
    During inference: Y = (X / scale) @ W_quantized^T
    So to put FP32 weights into AWQ space, we multiply by scale.

    Args:
        fp32_weight: [out_features, in_features] FP32 weight
        awq_module: AWQ WQLinear module with .scales and .group_size

    Returns:
        Transformed weight in AWQ space [out_features, in_features]
    """
    scales = awq_module.scales.float()   # [in_features // group_size, out_features]
    group_size = awq_module.group_size
    in_features = fp32_weight.shape[1]

    # Expand scales to match weight dimensions
    # scales: [n_groups, out_features] → [in_features, out_features]
    scales_expanded = scales.repeat_interleave(group_size, dim=0)  # [in_features, out_features]

    # Transform: W_awq_space = W_fp32 * scale (per input group, per output channel)
    # fp32_weight is [out, in], scales_expanded is [in, out]
    w_transformed = fp32_weight.float() * scales_expanded.T  # [out, in] * [out, in]

    return w_transformed


def restore_projections(model, fp32_weights, proj_paths):
    inner = _get_inner_model(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, 'in_features', None) or orig_module.infeatures
            out_f = getattr(orig_module, 'out_features', None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[l][proj]

            # If AWQ module, transform FP32 weights into AWQ's scaled space
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)

            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(w.to(device).to(act_dtype), requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(), requires_grad=False)
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
    parser.add_argument("--model", required=True, help="Model alias (llama7b, mistral7b, gemma9b)")
    parser.add_argument("--quantizer", default="nf4", help="Quantizer: nf4, gptq4, gptq3, int8")
    parser.add_argument("--n_eval", type=int, default=100, help="WildJailbreak prompts for ASR")
    parser.add_argument("--n_reasoning", type=int, default=50, help="ARC-Challenge questions")
    parser.add_argument("--output", default=None, help="Output path (auto-generated if not set)")
    args = parser.parse_args()

    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-generate output path
    if args.output is None:
        args.output = f"outputs/readwrite_{args.model}_{qtz}.json"

    print("=" * 70)
    print(f"  Read-Side vs Write-Side Restoration")
    print(f"  Model: {args.model} | Quantizer: {qtz}")
    print(f"  ASR prompts: {args.n_eval} | ARC questions: {args.n_reasoning}")
    print("=" * 70)

    # Load datasets once
    print("\nLoading datasets...")
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(item) for item in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    print(f"  WildJailbreak: {len(items)} adversarial prompts")
    print(f"  ARC-Challenge: {len(arc_items)} reasoning questions")

    all_responses = {}
    all_arc = {}

    # ========================================================================
    # Phase A: Generate responses + measure ARC for all 5 conditions
    # ========================================================================
    print(f"\n{'='*70}")
    print("  Phase A: Generating responses for all conditions")
    print("=" * 70)

    # ── Condition 1: FP32 baseline ────────────────────────────────────────
    print(f"\n[1/5] FP32 baseline...")
    fp32_model, tokenizer = load_fp32(args.model)

    print("  Extracting FP32 weights...")
    write_weights = extract_fp32_weights(fp32_model, WRITE_PROJS)
    read_weights = extract_fp32_weights(fp32_model, READ_PROJS)
    print("  Done extracting weights")

    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['fp32']:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # ── Condition 2: Quantized baseline ───────────────────────────────────
    print(f"\n[2/5] {qtz} baseline...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc[qtz]:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ── Condition 3: Write restored ───────────────────────────────────────
    print(f"\n[3/5] {qtz} + write restored (o_proj, down_proj)...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    restore_projections(q_model, write_weights, WRITE_PROJS)
    all_responses["write_restored"] = generate_only(q_model, tokenizer, items, desc="Write-Restored")
    all_arc["write_restored"] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['write_restored']:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ── Condition 4: Read restored ────────────────────────────────────────
    print(f"\n[4/5] {qtz} + read restored (q/k/v/gate/up)...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    restore_projections(q_model, read_weights, READ_PROJS)
    all_responses["read_restored"] = generate_only(q_model, tokenizer, items, desc="Read-Restored")
    all_arc["read_restored"] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['read_restored']:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ── Condition 5: Both restored ────────────────────────────────────────
    print(f"\n[5/5] {qtz} + both restored...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    restore_projections(q_model, write_weights, WRITE_PROJS)
    restore_projections(q_model, read_weights, READ_PROJS)
    all_responses["both_restored"] = generate_only(q_model, tokenizer, items, desc="Both-Restored")
    all_arc["both_restored"] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['both_restored']:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    del write_weights, read_weights

    # ========================================================================
    # Phase B: Classify all responses in one batch
    # ========================================================================
    print(f"\n{'='*70}")
    print("  Phase B: Classifying all responses (single classifier load)")
    print(f"{'='*70}\n")

    conditions = list(all_responses.keys())
    n = args.n_eval
    flat_behaviors = []
    flat_responses = []
    for cond in conditions:
        flat_behaviors.extend(behaviors)
        flat_responses.extend(all_responses[cond])

    flat_flags = classify_batch(flat_behaviors, flat_responses, device=device)

    asr_results = {}
    for i, cond in enumerate(conditions):
        start = i * n
        end = start + n
        flags = flat_flags[start:end]
        n_complied = sum(flags)
        asr = 100.0 * n_complied / n
        asr_results[cond] = asr

    # ========================================================================
    # Summary
    # ========================================================================
    fp32_asr = asr_results["fp32"]
    q_asr = asr_results[qtz]
    damage = q_asr - fp32_asr

    print(f"\n{'='*70}")
    print(f"  Read-Side vs Write-Side — {args.model} / {qtz}")
    print(f"  {args.n_eval} WildJailbreak prompts | {args.n_reasoning} ARC questions")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'ASR':>7} {'D ASR':>8} {'Recovery':>10} {'ARC':>6}")
    print("-" * 55)

    phase2 = ["fp32", qtz, "write_restored", "read_restored", "both_restored"]
    for cond in phase2:
        asr = asr_results[cond]
        delta = asr - fp32_asr
        if cond == "fp32":
            recovery = ""
        elif damage != 0:
            rec = (q_asr - asr) / damage * 100
            recovery = f"{rec:.0f}%"
        else:
            recovery = "N/A"
        arc = f"{all_arc[cond]:.0f}%"
        print(f"{cond:<20} {asr:>6.1f}% {delta:>+7.1f}pp {recovery:>9} {arc:>5}")

    write_recovery = q_asr - asr_results["write_restored"]
    read_recovery = q_asr - asr_results["read_restored"]
    print(f"\nTotal damage: {damage:+.1f}pp")
    print(f"Write restoration: {write_recovery:+.1f}pp ({100*write_recovery/damage if damage else 0:.0f}% of damage)")
    print(f"Read  restoration: {read_recovery:+.1f}pp ({100*read_recovery/damage if damage else 0:.0f}% of damage)")
    if abs(read_recovery) > abs(write_recovery):
        print("=> Read-Side Collapse CONFIRMED")
    elif damage == 0:
        print("=> No quantization damage detected")
    else:
        print("=> Write-side appears more important")

    # ── Save ──────────────────────────────────────────────────────────────
    out_data = {
        "model": args.model,
        "quantizer": qtz,
        "n_eval": args.n_eval,
        "n_reasoning": args.n_reasoning,
        "fp32_asr": fp32_asr,
        "quantized_asr": q_asr,
        "damage_pp": damage,
        "write_restored_asr": asr_results["write_restored"],
        "read_restored_asr": asr_results["read_restored"],
        "both_restored_asr": asr_results["both_restored"],
        "write_recovery_pp": write_recovery,
        "read_recovery_pp": read_recovery,
        "write_recovery_pct": 100 * write_recovery / damage if damage else 0,
        "read_recovery_pct": 100 * read_recovery / damage if damage else 0,
        "arc": all_arc,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n> Results saved to {out_path}")


if __name__ == "__main__":
    main()
