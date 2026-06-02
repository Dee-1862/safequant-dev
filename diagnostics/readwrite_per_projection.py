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

# Reuse helpers from readwrite_restoration so we don't duplicate AWQ-space transform / etc.
from diagnostics.readwrite_restoration import (
    extract_fp32_weights,
    restore_projections,
    measure_arc,
)
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior


# All 7 projection types in a single decoder layer
PROJ_TYPES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_reasoning", type=int, default=50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output is None:
        args.output = f"outputs/readwrite_per_projection_{args.model}_{qtz}.json"

    print("=" * 70)
    print(f"  F3a Decomposition -- Per-projection-type restoration")
    print(f"  Model: {args.model} | Quantizer: {qtz}")
    print(f"  Projections tested individually: {len(PROJ_TYPES)}")
    print(f"  n_eval={args.n_eval}  n_arc={args.n_reasoning}  seed={SEED}")
    print("=" * 70)

    print("\nLoading datasets...")
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    print(f"  WJ: {len(items)}, ARC: {len(arc_items)}")

    all_responses = {}
    all_arc = {}

    # -- FP32 baseline + extract all projection weights ----------------------
    print(f"\n[1/{len(PROJ_TYPES)+2}] FP32 baseline ...")
    fp32_model, tokenizer = load_fp32(args.model)
    print("  Extracting FP32 weights for all 7 projection types ...")
    fp32_weights_by_proj = {p: extract_fp32_weights(fp32_model, [p]) for p in PROJ_TYPES}
    all_responses["fp32"] = generate_only(fp32_model, tokenizer, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc['fp32']:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # -- Quantized baseline --------------------------------------------------
    print(f"\n[2/{len(PROJ_TYPES)+2}] {qtz} baseline ...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tokenizer, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tokenizer, arc_items)
    print(f"  ARC: {all_arc[qtz]:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # -- 7 per-projection restoration variants -------------------------------
    for i, proj in enumerate(PROJ_TYPES, start=3):
        cond = f"only_{proj.replace('.', '_')}"
        print(f"\n[{i}/{len(PROJ_TYPES)+2}] {qtz} + only {proj} restored ...")
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        restore_projections(q_model, fp32_weights_by_proj[proj], [proj])
        all_responses[cond] = generate_only(q_model, tokenizer, items, desc=cond)
        all_arc[cond] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[cond]:.0f}%")
        del q_model; gc.collect(); torch.cuda.empty_cache()

    # -- Batch classification ------------------------------------------------
    print(f"\nClassifying {(len(PROJ_TYPES)+2)*args.n_eval} responses in one batch ...")
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
    print(f"  Per-projection-type restoration -- {args.model} / {qtz}")
    print(f"  Damage to recover: {damage:+.1f}pp (FP32={fp32_asr:.1f}% -> quant={q_asr:.1f}%)")
    print("=" * 86)
    print(f"  {'Condition':<32} {'ASR':>7} {'DeltaASR':>8} {'Recovery_pp':>12} {'% of dmg':>9} {'ARC':>6}")
    print("-" * 86)
    for cond in conditions:
        asr = asr_results[cond]
        d = asr - fp32_asr
        rec_pp = q_asr - asr
        pct = (rec_pp / damage * 100) if damage != 0 else 0
        print(f"  {cond:<32} {asr:>6.1f}% {d:>+7.1f}pp {rec_pp:>+10.1f}pp {pct:>8.0f}% {all_arc[cond]:>5.0f}%")

    # Sorted by recovery
    print()
    print("  Sorted by recovery (best -> worst):")
    ranked = sorted(PROJ_TYPES, key=lambda p: q_asr - asr_results[f"only_{p.replace('.','_')}"], reverse=True)
    for p in ranked:
        cond = f"only_{p.replace('.','_')}"
        rec_pp = q_asr - asr_results[cond]
        pct = (rec_pp / damage * 100) if damage != 0 else 0
        print(f"    {p:<20} -> {rec_pp:>+6.1f}pp ({pct:>4.0f}% of damage)")

    out = {
        "model": args.model, "quantizer": qtz,
        "n_eval": args.n_eval, "n_reasoning": args.n_reasoning, "seed": SEED,
        "fp32_asr": fp32_asr, "quantized_asr": q_asr, "damage_pp": damage,
        "asr_per_condition": asr_results,
        "arc_per_condition": all_arc,
        "recovery_pp_per_projection": {
            p: q_asr - asr_results[f"only_{p.replace('.','_')}"] for p in PROJ_TYPES
        },
        "recovery_pct_per_projection": {
            p: (q_asr - asr_results[f"only_{p.replace('.','_')}"]) / damage * 100 if damage else 0
            for p in PROJ_TYPES
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
