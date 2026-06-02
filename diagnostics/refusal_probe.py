import argparse
import gc
import json
import os
import sys
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

# Reuse residual-extraction from layer_routing
from diagnostics.layer_routing import (
    extract_last_token_residuals,
    _get_inner,
)
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold


def _make_probe(probe_type):
    """Return a fresh sklearn estimator for the given probe type."""
    if probe_type == "linear":
        return LogisticRegression(C=1.0, max_iter=2000, n_jobs=1, random_state=SEED)
    if probe_type == "mlp":
        # Small 1-hidden-layer MLP. Standardize first via Pipeline-equivalent
        # (we'll wrap externally).
        return MLPClassifier(
            hidden_layer_sizes=(256,),
            activation="relu",
            solver="adam",
            alpha=1e-3,         # L2 regularization
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=SEED,
        )
    raise ValueError(f"Unknown probe_type: {probe_type}")


def fit_probe_per_layer(harmful_by_layer, harmless_by_layer, n_folds=5, probe_type="linear"):
    """
    For each layer, fit a 5-fold CV probe (linear or MLP) on
    [harmful=1, harmless=0]. Return per-layer mean test accuracy +
    per-layer std across folds.

    MLP probes get standardized features (zero-mean, unit-var per fold's
    train set) for stable convergence.
    """
    layers = sorted(harmful_by_layer.keys())
    results = {}
    for L in layers:
        h = np.stack([t.cpu().numpy() for t in harmful_by_layer[L]])
        s = np.stack([t.cpu().numpy() for t in harmless_by_layer[L]])
        X = np.concatenate([h, s], axis=0).astype(np.float64)
        y = np.concatenate([np.ones(len(h)), np.zeros(len(s))], axis=0)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
        fold_accs = []
        fold_h_accs = []
        fold_s_accs = []
        for tr_idx, te_idx in skf.split(X, y):
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xte, yte = X[te_idx], y[te_idx]
            if probe_type == "mlp":
                # Standardize within fold to avoid train/test leakage
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(Xtr)
                Xte = scaler.transform(Xte)
            clf = _make_probe(probe_type)
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            fold_accs.append((pred == yte).mean())
            mask_h = (yte == 1); mask_s = (yte == 0)
            if mask_h.sum() > 0:
                fold_h_accs.append((pred[mask_h] == yte[mask_h]).mean())
            if mask_s.sum() > 0:
                fold_s_accs.append((pred[mask_s] == yte[mask_s]).mean())
        results[L] = {
            "mean_acc": float(np.mean(fold_accs)),
            "std_acc":  float(np.std(fold_accs)),
            "harmful_acc": float(np.mean(fold_h_accs)) if fold_h_accs else None,
            "harmless_acc": float(np.mean(fold_s_accs)) if fold_s_accs else None,
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--probe_type", default="linear", choices=["linear", "mlp"],
                    help="linear=logistic regression; mlp=1-hidden-layer MLP (256 units)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        tag = "" if args.probe_type == "linear" else f"_{args.probe_type}"
        args.output = f"outputs/refusal_probe{tag}_{args.model}_{args.quantizer}.json"

    print(f"=== Refusal probe | {args.probe_type} | {args.model} / {args.quantizer} | n_calib={args.n_calib} | seed={SEED}")

    print("Loading calibration prompts...")
    harmful  = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)
    print(f"  n harmful: {len(harmful)}  n harmless: {len(harmless)}")

    # --------------------------- FP32 ---------------------------
    print(f"\n[FP32] loading {args.model} ...")
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    n_layers = len(_get_inner(fp32_model).layers)
    layers = list(range(n_layers))
    print(f"  n_layers = {n_layers}")

    print("[FP32] extracting per-layer residuals (harmful + harmless) ...")
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    print("[FP32] fitting per-layer probes ...")
    fp_probes = fit_probe_per_layer(fp_h, fp_s, n_folds=args.n_folds, probe_type=args.probe_type)
    del fp_h, fp_s; gc.collect()

    # --------------------------- QUANT ---------------------------
    print(f"\n[{args.quantizer}] loading quantized model ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device

    print(f"[{args.quantizer}] extracting per-layer residuals ...")
    q_h = extract_last_token_residuals(q_model, q_tok, harmful,  layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    del q_model; gc.collect(); torch.cuda.empty_cache()

    print(f"[{args.quantizer}] fitting per-layer probes ...")
    q_probes = fit_probe_per_layer(q_h, q_s, n_folds=args.n_folds, probe_type=args.probe_type)
    del q_h, q_s; gc.collect()

    # --------------------------- ANALYSIS ---------------------------
    print("\n" + "=" * 96)
    print(f"  Per-layer refusal probe accuracy: {args.model} (FP32 vs {args.quantizer})")
    print(f"  Train: 80% of {2*args.n_calib} prompts; Test: 5-fold CV; LR with L2, C=1, seed={SEED}")
    print("=" * 96)
    print(f"  {'L':>3} | {'FP_acc':>8} {'FP_std':>7} | {'Q_acc':>8} {'Q_std':>7} | {'DeltaAcc':>8}  bar")
    print("-" * 96)
    fp_accs = []
    q_accs = []
    for L in layers:
        fa = fp_probes[L]["mean_acc"]
        fs = fp_probes[L]["std_acc"]
        qa = q_probes[L]["mean_acc"]
        qs = q_probes[L]["std_acc"]
        d = qa - fa
        fp_accs.append(fa); q_accs.append(qa)
        bar = "" * int(round(qa * 30))
        marker = "  DROP" if d < -0.05 else ("  rise" if d > 0.05 else "")
        print(f"  {L:>3} | {fa:>7.1%} {fs:>6.1%} | {qa:>7.1%} {qs:>6.1%} | {d:>+7.1%}  {bar}{marker}")

    fp_mean = float(np.mean(fp_accs))
    q_mean  = float(np.mean(q_accs))
    fp_min  = float(np.min(fp_accs))
    q_min   = float(np.min(q_accs))
    fp_argmax = int(np.argmax(fp_accs))
    q_argmax  = int(np.argmax(q_accs))
    layers_drop = [L for L in layers if (q_probes[L]["mean_acc"] - fp_probes[L]["mean_acc"]) < -0.05]
    flat_degraded = (q_mean < fp_mean - 0.02) and (len(layers_drop) > 0.5 * n_layers)
    localized_drop = len(layers_drop) > 0 and len(layers_drop) <= 0.3 * n_layers
    preserved = (q_mean >= fp_mean - 0.02) and (len(layers_drop) == 0)

    print()
    print(f"  FP32 mean acc: {fp_mean:.1%}  min: {fp_min:.1%}  argmax L={fp_argmax}")
    print(f"  Quant mean acc: {q_mean:.1%}  min: {q_min:.1%}  argmax L={q_argmax}")
    print(f"  Layers with >5pp drop ({len(layers_drop)}/{n_layers}): {layers_drop}")
    print()
    if preserved:
        print("  -> INTERPRETATION: probe accuracy PRESERVED across all layers in quant.")
        print("    Read circuit (detection) intact. Damage is downstream -- write-side / action.")
        print("    Supports two-stage story (Story 2).")
    elif localized_drop:
        print(f"  -> INTERPRETATION: LOCALIZED probe degradation at layers {layers_drop}.")
        print("    Read circuit broken at specific points. Supports a layer-specific mechanism.")
    elif flat_degraded:
        print("  -> INTERPRETATION: DISTRIBUTED probe degradation across many layers.")
        print("    Read damage is genuinely scattered. Supports the 'safety is scattered' framing.")
    else:
        print("  -> INTERPRETATION: mixed / inconclusive pattern. Needs more conditions to disambiguate.")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "probe_type": args.probe_type,
        "n_calib": args.n_calib,
        "n_folds": args.n_folds,
        "n_layers": n_layers,
        "seed": SEED,
        "fp32_probe_per_layer": {str(L): fp_probes[L] for L in layers},
        "quant_probe_per_layer": {str(L): q_probes[L] for L in layers},
        "summary": {
            "fp32_mean_acc": fp_mean,
            "quant_mean_acc": q_mean,
            "fp32_min_acc": fp_min,
            "quant_min_acc": q_min,
            "fp32_argmax_layer": fp_argmax,
            "quant_argmax_layer": q_argmax,
            "layers_with_drop_gt_5pp": layers_drop,
            "n_layers_with_drop": len(layers_drop),
            "interpretation": (
                "preserved" if preserved else
                "localized" if localized_drop else
                "distributed" if flat_degraded else
                "mixed"
            ),
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
