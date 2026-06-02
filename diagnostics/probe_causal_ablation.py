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
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

from diagnostics.layer_routing import extract_last_token_residuals, _get_inner
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts
from core.evaluation import generate_only, classify_batch, _extract_behavior
from sklearn.linear_model import LogisticRegression


def fit_probe_get_directions(harmful_by_layer, harmless_by_layer):
    """Per layer: normalized weight vector of a logistic-regression probe."""
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
        out[L] = w / (np.linalg.norm(w) + 1e-12)
    return out


def install_ablation_hook(model, layer_idx, direction_t):
    """Install a forward hook that projects `direction_t` out of the residual
    stream output at the given layer."""
    inner = _get_inner(model)
    def hook(module, inputs, output):
        r = output[0] if isinstance(output, (tuple, list)) else output
        if r.dim() == 3:
            d = direction_t.to(r.device).to(r.dtype)
            proj = (r * d).sum(dim=-1, keepdim=True)  # (B, T, 1)
            r_new = r - proj * d
            if isinstance(output, tuple):
                return (r_new,) + output[1:]
            return r_new
        return output
    return inner.layers[layer_idx].register_forward_hook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=100, help="ASR prompts; default 100 to keep this affordable across all 32 layers")
    ap.add_argument("--layers_to_ablate", default=None, help="Comma-sep list, e.g. '4,11,16,22,29'. Default: every 4th layer")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/probe_causal_ablation_{args.model}_{args.quantizer}.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Probe causal ablation | {args.model} / {args.quantizer} | seed={SEED}")

    print("Loading prompts ...")
    calib_harmful  = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    calib_harmless = load_harmless_prompts(n_samples=args.n_calib)
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    eval_behaviors = [_extract_behavior(it) for it in eval_items]

    # Fit probes on quant model (per-layer directions)
    print(f"\n[{args.quantizer}] loading + extracting calibration residuals ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device
    n_layers = len(_get_inner(q_model).layers)
    layers = list(range(n_layers))

    h_resid = extract_last_token_residuals(q_model, q_tok, calib_harmful,  layers, q_device)
    s_resid = extract_last_token_residuals(q_model, q_tok, calib_harmless, layers, q_device)
    directions = fit_probe_get_directions(h_resid, s_resid)
    del h_resid, s_resid; gc.collect()

    # Determine which layers to ablate
    if args.layers_to_ablate:
        ablate_layers = [int(x) for x in args.layers_to_ablate.split(",")]
    else:
        ablate_layers = list(range(0, n_layers, 4))   # every 4th: 0,4,8,12,...
    print(f"  ablating layers: {ablate_layers}")

    # Baseline ASR (no ablation)
    print(f"\n  measuring baseline ASR (no ablation, n_eval={args.n_eval}) ...")
    baseline_responses = generate_only(q_model, q_tok, eval_items, desc=f"{args.quantizer} baseline")

    # Per-layer ablation
    all_responses = {"baseline": baseline_responses}
    for L in ablate_layers:
        direction_t = torch.tensor(directions[L], dtype=torch.float32)
        print(f"\n  measuring ASR with probe-direction ablation at L={L} ...")
        h = install_ablation_hook(q_model, L, direction_t)
        try:
            all_responses[f"ablate_L{L}"] = generate_only(q_model, q_tok, eval_items, desc=f"abl L{L}")
        finally:
            h.remove()

    del q_model; gc.collect(); torch.cuda.empty_cache()

    # Batch classify
    print(f"\n  classifying all responses ({len(all_responses)} x {args.n_eval}) ...")
    flat_behaviors, flat_responses = [], []
    keys = list(all_responses.keys())
    for k in keys:
        flat_behaviors.extend(eval_behaviors)
        flat_responses.extend(all_responses[k])
    flags = classify_batch(flat_behaviors, flat_responses, device=device)
    asr_results = {}
    n = args.n_eval
    for i, k in enumerate(keys):
        asr_results[k] = 100.0 * sum(flags[i*n:(i+1)*n]) / n

    # Report
    base = asr_results["baseline"]
    print()
    print("="*70)
    print(f"  Causal probe ablation -- {args.model} / {args.quantizer}")
    print(f"  baseline ASR (no ablation): {base:.1f}%")
    print("="*70)
    print(f"  {'Layer':>6}  {'ASR':>7}  {'Delta from base':>13}")
    print("  " + "-"*40)
    for L in ablate_layers:
        a = asr_results[f"ablate_L{L}"]
        d = a - base
        print(f"  L={L:<4} {a:>6.1f}% {d:>+11.1f}pp")

    out = {
        "model": args.model, "quantizer": args.quantizer,
        "n_calib": args.n_calib, "n_eval": args.n_eval, "seed": SEED,
        "ablated_layers": ablate_layers,
        "baseline_asr": base,
        "asr_per_layer": {f"L{L}": asr_results[f"ablate_L{L}"] for L in ablate_layers},
        "delta_per_layer": {f"L{L}": asr_results[f"ablate_L{L}"] - base for L in ablate_layers},
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
