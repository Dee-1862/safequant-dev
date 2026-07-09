import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.seeds import set_global_seed, DEFAULT_SEED
    SEED = set_global_seed(DEFAULT_SEED)
except ModuleNotFoundError:
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

# Reuse helpers from existing F3a
from diagnostics.readwrite_restoration import (
    extract_fp32_weights,
    measure_arc,
    _get_inner_model,
    _get_submodule,
    _set_submodule,
    _is_awq_module,
    _transform_fp32_to_awq_space,
    WRITE_PROJS,
    READ_PROJS,
)
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior


ALL_PROJS = READ_PROJS + WRITE_PROJS


def restore_projections_in_range(model, fp32_weights, proj_paths, layer_range):
    """Same as restore_projections but only restores layers in [layer_range[0], layer_range[1])."""
    inner = _get_inner_model(model)
    act_dtype = inner.norm.weight.dtype
    device = next(model.parameters()).device
    lo, hi = layer_range
    for l in range(lo, hi):
        layer = inner.layers[l]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, 'in_features', None) or orig_module.infeatures
            out_f = getattr(orig_module, 'out_features', None) or orig_module.outfeatures
            has_bias = orig_module.bias is not None
            w = fp32_weights[l][proj]
            if _is_awq_module(orig_module):
                w = _transform_fp32_to_awq_space(w, orig_module)
            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(w.to(device).to(act_dtype), requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(), requires_grad=False)
            _set_submodule(layer, proj, new_linear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_reasoning", type=int, default=50)
    # Defaults assume 32-layer model (Llama, Mistral). Override for Qwen (28).
    ap.add_argument("--n_layers", type=int, default=None,
                    help="Auto-detected from FP32 model if not set")
    ap.add_argument("--early_end", type=int, default=None,
                    help="Layer range early=[0, early_end). Default: n_layers//3")
    ap.add_argument("--mid_end", type=int, default=None,
                    help="Layer range mid=[early_end, mid_end). Default: 2*n_layers//3")
    ap.add_argument("--proj_group", choices=["read", "write", "all"], default="all",
                    help="which projections to restore per band: read-side, write-side, or all")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    PROJ_SET = {"read": READ_PROJS, "write": WRITE_PROJS, "all": ALL_PROJS}[args.proj_group]

    if args.output is None:
        suffix = "" if args.proj_group == "all" else f"_{args.proj_group}"
        args.output = f"outputs/readwrite_per_layer_range{suffix}_{args.model}_{qtz}.json"

    print("=" * 70)
    print(f"  F3a Decomposition -- Per-layer-range restoration")
    print(f"  Model: {args.model} | Quantizer: {qtz}")
    print(f"  n_eval={args.n_eval}  n_arc={args.n_reasoning}  seed={SEED}")
    print("=" * 70)

    print("\nLoading datasets...")
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    print(f"  WJ: {len(items)}, ARC: {len(arc_items)}")

    all_responses = {}
    all_arc = {}

    # -- FP32 baseline + extract weights for ALL projections ----------------
    print(f"\n[1/5] FP32 baseline ...")
    fp32_model, tokenizer = load_fp32(args.model)
    n_layers = args.n_layers or fp32_model.config.num_hidden_layers
    early_end = args.early_end or (n_layers // 3)
    mid_end   = args.mid_end   or (2 * n_layers // 3)
    ranges = {
        "early":  (0, early_end),
        "middle": (early_end, mid_end),
        "late":   (mid_end, n_layers),
    }
    print(f"  n_layers={n_layers}  ranges:")
    for name, (lo, hi) in ranges.items():
        print(f"    {name}:  [{lo}, {hi})  ({hi-lo} layers)")

    print("  Extracting FP32 weights for all read + write projections ...")
    fp32_w = extract_fp32_weights(fp32_model, PROJ_SET)
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['fp32']:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # -- Quantized baseline --------------------------------------------------
    print(f"\n[2/5] {qtz} baseline ...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc[qtz]:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # -- 3 layer-range restoration variants ----------------------------------
    for i, (name, lo_hi) in enumerate(ranges.items(), start=3):
        cond = f"layers_{name}"
        print(f"\n[{i}/5] {qtz} + layers {lo_hi[0]}-{lo_hi[1]-1} restored (all projections) ...")
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        restore_projections_in_range(q_model, fp32_w, PROJ_SET, lo_hi)
        all_responses[cond] = generate_only(q_model, tokenizer, items, desc=cond)
        all_arc[cond] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[cond]:.0f}%")
        del q_model; gc.collect(); torch.cuda.empty_cache()

    # -- Batch classification ------------------------------------------------
    print(f"\nClassifying {len(all_responses)*args.n_eval} responses ...")
    conditions = list(all_responses.keys())
    flat_behaviors, flat_responses = [], []
    for cond in conditions:
        flat_behaviors.extend(behaviors)
        flat_responses.extend(all_responses[cond])
    flat_flags = classify_batch(flat_behaviors, flat_responses, device=device)
    n = args.n_eval
    asr_results = {}
    for i, cond in enumerate(conditions):
        flags = flat_flags[i*n:(i+1)*n]
        asr_results[cond] = 100.0 * sum(flags) / n

    # -- Report --------------------------------------------------------------
    fp32_asr = asr_results["fp32"]
    q_asr = asr_results[qtz]
    damage = q_asr - fp32_asr
    print()
    print("=" * 86)
    print(f"  Per-layer-range restoration -- {args.model} / {qtz}")
    print(f"  Damage to recover: {damage:+.1f}pp (FP32={fp32_asr:.1f}% -> quant={q_asr:.1f}%)")
    print("=" * 86)
    print(f"  {'Condition':<22} {'Layers':>11} {'ASR':>7} {'DeltaASR':>8} {'Recovery_pp':>12} {'% of dmg':>9} {'ARC':>6}")
    print("-" * 86)
    for name in ["fp32", qtz, "layers_early", "layers_middle", "layers_late"]:
        if name in ("fp32", qtz):
            lo_hi = ""
        else:
            range_key = name.replace("layers_", "")
            lo, hi = ranges[range_key]
            lo_hi = f"[{lo},{hi})"
        asr = asr_results[name]
        d = asr - fp32_asr
        rec_pp = q_asr - asr
        pct = (rec_pp / damage * 100) if damage else 0
        print(f"  {name:<22} {lo_hi:>11} {asr:>6.1f}% {d:>+7.1f}pp {rec_pp:>+10.1f}pp {pct:>8.0f}% {all_arc[name]:>5.0f}%")

    out = {
        "model": args.model, "quantizer": qtz,
        "n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED,
        "n_layers": n_layers,
        "ranges": {name: list(lo_hi) for name, lo_hi in ranges.items()},
        "fp32_asr": fp32_asr, "quantized_asr": q_asr, "damage_pp": damage,
        "asr_per_condition": asr_results,
        "arc_per_condition": all_arc,
        "recovery_pp_per_range": {
            name: q_asr - asr_results[f"layers_{name}"] for name in ranges
        },
        "recovery_pct_per_range": {
            name: (q_asr - asr_results[f"layers_{name}"]) / damage * 100 if damage else 0
            for name in ranges
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
