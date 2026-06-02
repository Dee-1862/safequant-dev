import argparse, gc, json, sys
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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

from diagnostics.readwrite_restoration import extract_fp32_weights, measure_arc
from diagnostics.readwrite_per_layer_range import restore_projections_in_range
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior


PROJ_TYPES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--layer_lo", type=int, default=11)
    ap.add_argument("--layer_hi", type=int, default=22)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_reasoning", type=int, default=50)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    qtz = args.quantizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.output is None:
        args.output = f"outputs/per_projection_middle_{args.model}_{qtz}_L{args.layer_lo}-{args.layer_hi}.json"

    print("=" * 70)
    print(f"  Per-projection x middle-layer restoration")
    print(f"  {args.model} / {qtz}  | layers [{args.layer_lo}, {args.layer_hi})")
    print(f"  n_eval={args.n_eval}  n_arc={args.n_reasoning}  seed={SEED}")
    print("=" * 70)

    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    all_responses, all_arc = {}, {}

    # FP32 baseline + extract FP32 weights for each projection
    print(f"\n[1/{len(PROJ_TYPES)+2}] FP32 baseline ...")
    fp32_model, tok = load_fp32(args.model)
    fp32_weights_by_proj = {p: extract_fp32_weights(fp32_model, [p]) for p in PROJ_TYPES}
    all_responses["fp32"] = generate_only(fp32_model, tok, items, desc="FP32")
    all_arc["fp32"] = measure_arc(fp32_model, tok, arc_items)
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # Quantized baseline
    print(f"\n[2/{len(PROJ_TYPES)+2}] {qtz} baseline ...")
    q_model, _ = load_quantized(args.model, quantizer=qtz)
    all_responses[qtz] = generate_only(q_model, tok, items, desc=qtz)
    all_arc[qtz] = measure_arc(q_model, tok, arc_items)
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # 7 per-projection restoration variants, restricted to middle layers
    for i, proj in enumerate(PROJ_TYPES, start=3):
        cond = f"only_{proj.replace('.', '_')}_middle"
        print(f"\n[{i}/{len(PROJ_TYPES)+2}] {qtz} + only {proj} restored in layers [{args.layer_lo},{args.layer_hi}) ...")
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        restore_projections_in_range(q_model, fp32_weights_by_proj[proj], [proj], (args.layer_lo, args.layer_hi))
        all_responses[cond] = generate_only(q_model, tok, items, desc=cond)
        all_arc[cond] = measure_arc(q_model, tok, arc_items)
        del q_model; gc.collect(); torch.cuda.empty_cache()

    # Batch classify
    print(f"\n  classifying {len(all_responses)*args.n_eval} responses ...")
    conditions = list(all_responses.keys())
    flat_b, flat_r = [], []
    for k in conditions:
        flat_b.extend(behaviors); flat_r.extend(all_responses[k])
    flags = classify_batch(flat_b, flat_r, device=device)
    n = args.n_eval
    asr = {k: 100.0 * sum(flags[i*n:(i+1)*n]) / n for i, k in enumerate(conditions)}

    fp = asr["fp32"]; q = asr[qtz]; damage = q - fp
    print()
    print("="*86)
    print(f"  Per-projection x middle-layer restoration -- {args.model} / {qtz}")
    print(f"  Damage to recover: {damage:+.1f}pp (FP32={fp:.1f}% -> quant={q:.1f}%)")
    print(f"  Restoration limited to layers [{args.layer_lo},{args.layer_hi})")
    print("="*86)
    print(f"  {'Condition':<36} {'ASR':>7} {'DeltaASR':>8} {'Recov_pp':>9} {'% of dmg':>9} {'ARC':>5}")
    print("-"*86)
    for cond in conditions:
        a = asr[cond]
        d = a - fp
        rec = q - a
        pct = (rec/damage*100) if damage else 0
        print(f"  {cond:<36} {a:>6.1f}% {d:>+7.1f}pp {rec:>+7.1f}pp {pct:>7.0f}% {all_arc[cond]:>4.0f}%")

    print()
    print("  Sorted by recovery (best -> worst within middle layers):")
    ranked = sorted(PROJ_TYPES, key=lambda p: q - asr[f"only_{p.replace('.','_')}_middle"], reverse=True)
    for p in ranked:
        cond = f"only_{p.replace('.','_')}_middle"
        rec = q - asr[cond]
        pct = (rec / damage * 100) if damage else 0
        print(f"    {p:<20} -> {rec:>+6.1f}pp ({pct:>4.0f}% of damage)")

    out = {
        "model": args.model, "quantizer": qtz, "seed": SEED,
        "layer_lo": args.layer_lo, "layer_hi": args.layer_hi,
        "n_eval": args.n_eval, "n_reasoning": args.n_reasoning,
        "fp32_asr": fp, "quantized_asr": q, "damage_pp": damage,
        "asr_per_condition": asr, "arc_per_condition": all_arc,
        "recovery_pp_per_projection": {
            p: q - asr[f"only_{p.replace('.','_')}_middle"] for p in PROJ_TYPES
        },
        "recovery_pct_per_projection": {
            p: (q - asr[f"only_{p.replace('.','_')}_middle"]) / damage * 100 if damage else 0
            for p in PROJ_TYPES
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
