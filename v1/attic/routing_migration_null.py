"""Condition-4 (corrected): refusal-specific operative-layer MIGRATION vs a migration null.

The original layer-routing screen compared the refusal direction's operative layer to
random directions' AVERAGE POSITION, using a raw dot-product projection. That projection
is t . r_hat = |t| * cos(angle); since |t| grows with depth, the operative-layer argmax
is dominated by activation magnitude, not by alignment with r_hat. Under that artifact the
random-control SD collapsed to ~1 and the z inflated (the paper's "z>9" is that artifact).
It also measured POSITION, while condition 4's wording is "the geometric shift must be
refusal-specific" -- i.e. a SHIFT, tested against chance.

This corrects BOTH, by a definition fixed BEFORE looking at any cell's pass/fail:
  * magnitude-normalized (cosine) projection -> operative layer reflects alignment;
  * a FIXED refusal direction r_hat_FP is projected into the FP residual stream and the
    quant residual stream; the migration is delta = |op(r_hat_FP in quant) - op(r_hat_FP in FP)|.
    Using one fixed direction avoids confounding the layer shift with the direction rotation
    (which the separate probe-rotation screen already measures);
  * NULL: K random unit directions migrated through the SAME quantization; delta_rand(k) =
    |op(r_k in quant) - op(r_k in FP)|.

PASS (refusal-specific reorganization): z = (delta_obs - mean delta_rand)/std >= 3 on harmful
(equivalently empirical p <= ~0.003). harmless reported alongside, never hidden.
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.seeds import set_global_seed, DEFAULT_SEED  # noqa: E402
SEED = set_global_seed(DEFAULT_SEED)
from core.model_loader import load_fp32, load_quantized  # noqa: E402
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts  # noqa: E402
from diagnostics.layer_routing import (  # noqa: E402
    _get_inner, extract_last_token_residuals, compute_r_hat, per_layer_separation,
    per_prompt_projections, operative_layer_per_prompt,
)
from diagnostics.routing_random_control import random_unit_vector  # noqa: E402


def _token_pos(model, override):
    if override is not None:
        return override
    m = model.lower()
    if "llama" in m:
        return -5
    if "phi" in m:
        return -3
    return -1


def _op_median(residuals_by_layer, direction, layers):
    """Median per-prompt operative layer of `direction`, cosine (normalized) projection."""
    proj, _ = per_prompt_projections(residuals_by_layer, direction, normalize=True)
    ops = operative_layer_per_prompt(proj, layers)
    return float(np.median(ops))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--rand_seed", type=int, default=12345)
    ap.add_argument("--token_pos", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/routing_migration_null_{args.model}_{args.quantizer}.json"
    tp = _token_pos(args.model, args.token_pos)
    print(f"=== migration-null | {args.model}/{args.quantizer} | token_pos={tp} "
          f"n_random={args.n_random} seed={SEED} (PASS: z>=3 on harmful)")

    harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)
    print(f"  n harmful={len(harmful)} n harmless={len(harmless)}")

    # ---- FP pass: residuals + refusal direction r_hat_FP at sigma-best layer ----
    print(f"[FP32] loading {args.model} ...")
    fp_model, fp_tok = load_fp32(args.model)
    device = next(fp_model.parameters()).device
    layers = list(range(len(_get_inner(fp_model).layers)))
    fp_h = extract_last_token_residuals(fp_model, fp_tok, harmful, layers, device, token_pos=tp)
    fp_s = extract_last_token_residuals(fp_model, fp_tok, harmless, layers, device, token_pos=tp)
    fp_rhats = {L: compute_r_hat(fp_h[L], fp_s[L]) for L in layers}
    fp_seps = {L: per_layer_separation(fp_h[L], fp_s[L], fp_rhats[L]) for L in layers}
    fp_best = max(layers, key=lambda L: fp_seps[L])
    r_hat_FP = fp_rhats[fp_best]
    hidden = int(fp_model.config.hidden_size)
    print(f"  FP sigma-best layer = {fp_best}")
    del fp_model
    gc.collect(); torch.cuda.empty_cache()

    # ---- Quant pass: residuals in the SAME prompt order ----
    print(f"[{args.quantizer}] loading quantized model ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    qdev = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, qdev, token_pos=tp)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, qdev, token_pos=tp)
    del q_model
    gc.collect(); torch.cuda.empty_cache()

    # residuals are CPU tensors; move r_hat_FP to CPU to match
    r_hat_FP = r_hat_FP.detach().cpu().float()

    rng = np.random.default_rng(args.rand_seed)
    rand_dirs = [random_unit_vector(hidden, rng).cpu().float() for _ in range(args.n_random)]

    result = {
        "model": args.model, "quantizer": args.quantizer, "token_pos": tp,
        "seed": SEED, "rand_seed": args.rand_seed, "n_random": args.n_random,
        "n_calib": args.n_calib, "fp_best_layer": int(fp_best), "normalize": True,
        "direction": "fixed r_hat_FP in both FP and quant residuals",
        "statistic": "delta = |op(r_hat_FP in quant) - op(r_hat_FP in FP)|; null = same with random dirs",
        "pass_rule": "z >= 3 on harmful",
    }

    print(f"\n  {'cond':<9}{'opFP':>6}{'opQ':>6}{'delta':>7}{'randDelta(mean+/-sd)':>22}{'z':>8}{'p':>8}  verdict")
    for cond, fpr, qr in [("harmful", fp_h, q_h), ("harmless", fp_s, q_s)]:
        op_fp = _op_median(fpr, r_hat_FP, layers)
        op_q = _op_median(qr, r_hat_FP, layers)
        delta_obs = abs(op_q - op_fp)
        deltas = np.array([abs(_op_median(qr, r, layers) - _op_median(fpr, r, layers))
                           for r in rand_dirs], dtype=np.float64)
        z = float((delta_obs - deltas.mean()) / (deltas.std() + 1e-9))
        p = float((deltas >= delta_obs).mean())
        verdict = "PASS" if abs(z) >= 3 else "FAIL"
        result[cond] = {
            "op_FP_in_FP": op_fp, "op_FP_in_quant": op_q, "delta_obs": delta_obs,
            "rand_delta_mean": float(deltas.mean()), "rand_delta_std": float(deltas.std()),
            "z": z, "empirical_p": p, "pass_z3": bool(abs(z) >= 3),
        }
        print(f"  {cond:<9}{op_fp:>6.1f}{op_q:>6.1f}{delta_obs:>7.1f}"
              f"{f'{deltas.mean():.1f}+/-{deltas.std():.2f}':>22}{z:>8.2f}{p:>8.3f}  {verdict}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
