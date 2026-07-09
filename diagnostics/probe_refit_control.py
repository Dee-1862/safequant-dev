"""Noise-floor control for the probe-geometry (rotation) screen.

probe_geometry.py reports cos(W_FP, W_q) per layer and calls AQLM-2 "drifted" at
cos_med ~ 0.79. The audit question (post-routing-artifact): is 0.79 real rotation, or
just probe refit noise? This fits TWO probes on the SAME FP model -- on disjoint random
halves of the calibration data -- and reports cos(W_FP_a, W_FP_b) per layer. That is the
noise floor: the cosine you get with NO geometry change at all, only finite-sample refit
variation. If the floor is ~0.97 while FP-vs-quant is 0.79, the drift is real; if the
floor is itself ~0.80, "drift" is an artifact. Same fit as probe_geometry: LogReg C=1.0.
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
from core.model_loader import load_fp32  # noqa: E402
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts  # noqa: E402
from diagnostics.layer_routing import _get_inner, extract_last_token_residuals  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


def _cos(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def _fit_w(h_list, s_list):
    X = np.stack([t.numpy() for t in h_list] + [t.numpy() for t in s_list])
    y = np.array([1] * len(h_list) + [0] * len(s_list))
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)
    clf.fit(X, y)
    return clf.coef_[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--token_pos", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/probe_refit_control_{args.model}.json"
    # match probe_geometry's extraction position (it uses the default -1 for all models)
    tp = args.token_pos if args.token_pos is not None else -1
    print(f"=== probe refit noise-floor | {args.model} | n_calib={args.n_calib} token_pos={tp} seed={SEED}")

    harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    fp_model, fp_tok = load_fp32(args.model)
    device = next(fp_model.parameters()).device
    layers = list(range(len(_get_inner(fp_model).layers)))
    h = extract_last_token_residuals(fp_model, fp_tok, harmful, layers, device, token_pos=tp)
    s = extract_last_token_residuals(fp_model, fp_tok, harmless, layers, device, token_pos=tp)
    del fp_model; gc.collect(); torch.cuda.empty_cache()

    # disjoint random halves of the calibration data (same split across layers)
    rng = np.random.default_rng(SEED)
    nh, ns = len(h[layers[0]]), len(s[layers[0]])
    hp = rng.permutation(nh); sp = rng.permutation(ns)
    hA, hB = hp[:nh // 2], hp[nh // 2:]
    sA, sB = sp[:ns // 2], sp[ns // 2:]

    cos_floor = {}
    for L in layers:
        wA = _fit_w([h[L][i] for i in hA], [s[L][i] for i in sA])
        wB = _fit_w([h[L][i] for i in hB], [s[L][i] for i in sB])
        cos_floor[L] = _cos(wA, wB)
    vals = list(cos_floor.values())
    summary = {"cos_floor_min": float(min(vals)), "cos_floor_median": float(np.median(vals)),
               "cos_floor_max": float(max(vals))}
    print(f"\n  FP-vs-FP refit cos (noise floor): min={summary['cos_floor_min']:.3f} "
          f"median={summary['cos_floor_median']:.3f} max={summary['cos_floor_max']:.3f}")
    print(f"  (FP-vs-quant cos_med for this model's AQLM-2 was ~0.79 Llama / ~0.70 Phi-3)")
    print(f"  -> if floor >> 0.79, the drift is REAL rotation, not refit noise")

    out = {"model": args.model, "n_calib": args.n_calib, "token_pos": tp, "seed": SEED,
           "summary": summary, "cos_floor_per_layer": {str(k): v for k, v in cos_floor.items()}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
