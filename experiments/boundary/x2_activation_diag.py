# X2 activation diagnostic: FP32 vs a quantized model on the X2 probe-training prompts
# (WildJailbreak harmful + harmless, n_calib each, last token). Per layer and per model:
#   - probe cross-validated accuracy on that training data (core.probes.fit_probe_per_layer,
#     5-fold, same probe as X2/X3/Table 1)
#   - activation stats: mean L2 norm, element std, share of exact zeros, non-finite count
# and, per layer, the cosine between FP32 and quantized residuals on the SAME prompt
# (mean / min over prompts): a direct check that the quantized kernel computes sane activations.
# Written for the Mistral AQLM-2 (2x8 Triton kernel on aarch64) X2 check.
#   python -m experiments.boundary.x2_activation_diag --model mistral7b-v02 --quantizer aqlm2 --tag delta
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.manifest import RunConfig, assert_new_result, save_result
from core.model_loader import load_fp32, load_quantized
from core.probes import fit_probe_per_layer
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed
from experiments.boundary.probe_leakage_control import _as_text

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _stats(h, s):
    """
    Activation statistics over the harmful + harmless residuals of one layer.

    Inputs:
        - h (list[Tensor]): Harmful last-token residuals
        - s (list[Tensor]): Harmless last-token residuals

    Outputs:
        - stats (dict): norm_mean, std, zero_share, nonfinite
    """
    X = np.concatenate([np.stack([t.float().numpy() for t in h]),
                        np.stack([t.float().numpy() for t in s])]).astype(np.float64)
    fin = np.isfinite(X)
    Xf = np.where(fin, X, 0.0)
    return {
        "norm_mean": float(np.linalg.norm(Xf, axis = 1).mean()),
        "std": float(Xf.std()),
        "zero_share": float((X == 0).mean()),
        "nonfinite": int((~fin).sum()),
    }


def _cv_layer(h, s):
    """
    5-fold probe accuracy for one layer (same helper as X3 / Table 1).

    Inputs:
        - h (list[Tensor]): Harmful residuals of the layer
        - s (list[Tensor]): Harmless residuals of the layer

    Outputs:
        - acc (float): Cross-validated mean accuracy
    """
    r = fit_probe_per_layer({0: h}, {0: s})[0]
    return float(r["mean_acc"])


def _per_model(h, s, layers, n_jobs):
    """
    Probe accuracy and activation statistics for every layer of one model.

    Inputs:
        - h (dict): layer -> harmful residuals
        - s (dict): layer -> harmless residuals
        - layers (list[int]): Layers to evaluate
        - n_jobs (int): Parallel probe fits

    Outputs:
        - per_layer (dict): str(layer) -> {cv_acc, norm_mean, std, zero_share, nonfinite}
    """
    accs = Parallel(n_jobs = n_jobs)(delayed(_cv_layer)(h[L], s[L]) for L in layers)
    return {str(L): {"cv_acc": a, **_stats(h[L], s[L])} for L, a in zip(layers, accs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required = True)
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--tag", default = None)
    args = ap.parse_args()

    cfg = RunConfig(model = args.model, quantizer = args.quantizer,
                    variant = args.tag or "x2diag",
                    extra = {"n_calib": args.n_calib, "seed": SEED})
    assert_new_result(_SCRIPT, cfg)
    n_jobs = int(os.environ.get("SQ_FIT_JOBS") or os.environ.get("OMP_NUM_THREADS") or os.cpu_count())

    harmful = [_as_text(x) for x in load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")]
    harmless = [_as_text(x) for x in load_harmless_prompts(n_samples = args.n_calib)]

    res = {}
    for name in ("fp32", args.quantizer):
        model, tok = load_fp32(args.model) if name == "fp32" else load_quantized(args.model, quantizer = name)
        device = next(model.parameters()).device
        layers = list(range(len(get_inner(model).layers)))
        h = extract_last_token_residuals(model, tok, harmful, layers, device)
        s = extract_last_token_residuals(model, tok, harmless, layers, device)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        res[name] = (h, s)
        print(f"[{name}] extracted {len(harmful)}+{len(harmless)} prompts x {len(layers)} layers", flush = True)

    (fh, fs), (qh, qs) = res["fp32"], res[args.quantizer]
    per_layer = {"fp32": _per_model(fh, fs, layers, n_jobs),
                 args.quantizer: _per_model(qh, qs, layers, n_jobs)}

    # FP32 vs quantized residual cosine on the same prompt
    cross = {}
    for L in layers:
        F = np.stack([t.float().numpy() for t in fh[L] + fs[L]]).astype(np.float64)
        Q = np.stack([t.float().numpy() for t in qh[L] + qs[L]]).astype(np.float64)
        c = (F * Q).sum(1) / (np.linalg.norm(F, axis = 1) * np.linalg.norm(Q, axis = 1) + 1e-12)
        cross[str(L)] = {"cos_mean": float(np.nanmean(c)), "cos_min": float(np.nanmin(c)),
                         "norm_ratio_mean": float(np.mean(np.linalg.norm(Q, axis = 1) /
                                                          (np.linalg.norm(F, axis = 1) + 1e-12)))}

    print(f"{'L':>3} | {'acc fp':>6} {'acc q':>6} | {'norm fp':>8} {'norm q':>8} | {'std fp':>7} {'std q':>7} | "
          f"{'zero% fp':>8} {'zero% q':>8} | {'nonfin q':>8} | {'cos fp-q':>8}")
    for L in layers:
        a, b, c = per_layer["fp32"][str(L)], per_layer[args.quantizer][str(L)], cross[str(L)]
        print(f"{L:>3} | {a['cv_acc']:6.3f} {b['cv_acc']:6.3f} | {a['norm_mean']:8.2f} {b['norm_mean']:8.2f} | "
              f"{a['std']:7.3f} {b['std']:7.3f} | {100 * a['zero_share']:8.3f} {100 * b['zero_share']:8.3f} | "
              f"{b['nonfinite']:8d} | {c['cos_mean']:8.3f}")

    path = save_result(_SCRIPT, cfg, {"per_layer": per_layer, "fp32_vs_quant": cross,
                                      "n_layers": len(layers)})
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
