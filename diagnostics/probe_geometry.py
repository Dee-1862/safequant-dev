import argparse, gc, json, sys
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

from diagnostics.layer_routing import extract_last_token_residuals, _get_inner
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def fit_probe_get_weights(harmful_by_layer, harmless_by_layer):
    """For each layer, fit one logistic regression on ALL data; return weight vector + bias + train acc."""
    layers = sorted(harmful_by_layer.keys())
    out = {}
    for L in layers:
        h = np.stack([t.cpu().numpy() for t in harmful_by_layer[L]])
        s = np.stack([t.cpu().numpy() for t in harmless_by_layer[L]])
        X = np.concatenate([h, s], axis=0).astype(np.float64)
        y = np.concatenate([np.ones(len(h)), np.zeros(len(s))], axis=0)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
        clf.fit(X, y)
        w = clf.coef_[0]
        out[L] = {"weight": w.tolist(), "bias": float(clf.intercept_[0]), "train_acc": float(clf.score(X, y))}
    return out


def cos(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12: return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/probe_geometry_{args.model}_{args.quantizer}.json"

    print(f"=== Probe-geometry comparison | {args.model} / {args.quantizer} | n_calib={args.n_calib}, seed={SEED}")

    harmful  = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    print(f"[FP32] loading + extracting residuals ...")
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    n_layers = len(_get_inner(fp32_model).layers)
    layers = list(range(n_layers))
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
    fp_probes = fit_probe_get_weights(fp_h, fp_s)
    del fp32_model, fp_h, fp_s; gc.collect(); torch.cuda.empty_cache()

    print(f"[{args.quantizer}] loading + extracting residuals ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    q_probes = fit_probe_get_weights(q_h, q_s)
    del q_model, q_h, q_s; gc.collect(); torch.cuda.empty_cache()

    print(f"\n  {'L':>3} | {'FP_train_acc':>12} {'Q_train_acc':>11} {'cos(W_FP, W_q)':>16}  signal_status")
    print("  " + "-"*70)
    cos_per_layer = {}
    for L in layers:
        c = cos(fp_probes[L]["weight"], q_probes[L]["weight"])
        cos_per_layer[L] = c
        status = "IN-PLACE" if c > 0.9 else ("MOVED" if c < 0.5 else "PARTIAL-ROTATION")
        print(f"  {L:>3} | {fp_probes[L]['train_acc']:>11.1%} {q_probes[L]['train_acc']:>10.1%} {c:>16.4f}  {status}")

    vals = list(cos_per_layer.values())
    print(f"\n  cos_sim across layers: min={min(vals):.3f}  median={np.median(vals):.3f}  max={max(vals):.3f}")
    in_place = sum(1 for v in vals if v > 0.9)
    moved    = sum(1 for v in vals if v < 0.5)
    partial  = len(vals) - in_place - moved
    print(f"  layers IN-PLACE (cos>0.9): {in_place}/{n_layers}")
    print(f"  layers PARTIAL-ROTATION (0.5<cos<=0.9): {partial}/{n_layers}")
    print(f"  layers MOVED (cos<=0.5): {moved}/{n_layers}")

    interp = ("preserved-in-place" if in_place >= 0.7*n_layers else
              "preserved-but-moved" if moved >= 0.5*n_layers else
              "partially-rotated")
    print(f"\n  INTERPRETATION: {interp.upper()}")

    out = {
        "model": args.model, "quantizer": args.quantizer, "n_calib": args.n_calib, "seed": SEED,
        "n_layers": n_layers,
        "fp32_probe_weights": {str(L): fp_probes[L]["weight"] for L in layers},
        "quant_probe_weights": {str(L): q_probes[L]["weight"] for L in layers},
        "cos_sim_per_layer": {str(L): cos_per_layer[L] for L in layers},
        "summary": {
            "cos_min": float(min(vals)),
            "cos_median": float(np.median(vals)),
            "cos_max": float(max(vals)),
            "n_in_place": in_place,
            "n_moved": moved,
            "n_partial": partial,
            "interpretation": interp,
        },
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
