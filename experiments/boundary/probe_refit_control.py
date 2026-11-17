# Probe refit noise-floor control for probe-rotation screen.
import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.manifest import RunConfig, assert_new_result, save_result
from core.model_loader import load_fp32
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def _cos(a, b):
    """
    Cosine similarity between two vectors.

    Inputs:
        - a (array-like): First vector
        - b (array-like): Second vector

    Outputs:
        - cos (float): Cosine similarity, or 0.0 if either vector is near-zero
    """
    # Cast to float64 for stable norms
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)

    # Guard against near-zero norms
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na < 1e-12 or nb < 1e-12 else float(np.dot(a, b) / (na * nb))


def _fit_w(h_list, s_list):
    """
    Fit a linear probe and return its weight vector.

    Inputs:
        - h_list (list[Tensor]): Harmful residual vectors
        - s_list (list[Tensor]): Harmless residual vectors

    Outputs:
        - weight (ndarray): LogisticRegression coef_[0]
    """
    # Stack features and binary labels (harmful = 1)
    X = np.stack([t.numpy() for t in h_list] + [t.numpy() for t in s_list])
    y = np.array([1] * len(h_list) + [0] * len(s_list))

    # Fit L2-regularized logistic regression
    clf = LogisticRegression(C = 1.0, max_iter = 2000, random_state = SEED)
    clf.fit(X, y)
    return clf.coef_[0]


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--token_pos", type = int, default = -1)
    # X4: number of independent random half-splits. Default 1 = the original
    # single-split control (identical numbers); e.g. --n_splits 100 for the 95th-percentile floor.
    ap.add_argument("--n_splits", type = int, default = 1)
    # Opt-in output suffix (e.g. smoke runs: --tag smoke -> ..._splits100_smoke.json)
    ap.add_argument("--tag", default = None)
    args = ap.parse_args()

    # Multi-split runs get their own filename and never overwrite the single-split result
    variant = f"splits{args.n_splits}" if args.n_splits != 1 else None
    if args.tag:
        variant = f"{variant}_{args.tag}" if variant else args.tag
    if variant:
        assert_new_result(
            _SCRIPT, RunConfig(model = args.model, omit_quantizer = True, variant = variant)
        )

    # Load calibration prompts
    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # Extract last-token residuals from FP32
    fp_model, fp_tok = load_fp32(args.model)
    device = next(fp_model.parameters()).device
    layers = list(range(len(get_inner(fp_model).layers)))
    h = extract_last_token_residuals(fp_model, fp_tok, harmful, layers, device, token_pos = args.token_pos)
    s = extract_last_token_residuals(fp_model, fp_tok, harmless, layers, device, token_pos = args.token_pos)
    del fp_model
    gc.collect()
    torch.cuda.empty_cache()

    # Random half-splits of harmful and harmless indices. Split 0 draws exactly what the
    # original single-split code drew (same rng, same order), so n_splits=1 is unchanged.
    rng = np.random.default_rng(SEED)
    nh, ns = len(h[layers[0]]), len(s[layers[0]])
    splits = []
    for _ in range(args.n_splits):
        hp, sp = rng.permutation(nh), rng.permutation(ns)
        splits.append((hp[: nh // 2], hp[nh // 2 :], sp[: ns // 2], sp[ns // 2 :]))

    # Per-layer cosine between independent probe fits, for every split
    cos_by_split = {layer_idx: [] for layer_idx in layers}
    for hA, hB, sA, sB in splits:
        for layer_idx in layers:
            wA = _fit_w([h[layer_idx][i] for i in hA], [s[layer_idx][i] for i in sA])
            wB = _fit_w([h[layer_idx][i] for i in hB], [s[layer_idx][i] for i in sB])
            cos_by_split[layer_idx].append(_cos(wA, wB))

    # First split reproduces the original control's per-layer values
    cos_floor = {layer_idx: cos_by_split[layer_idx][0] for layer_idx in layers}
    vals = list(cos_floor.values())

    # Persist noise-floor summary
    extra = {"n_calib": args.n_calib, "token_pos": args.token_pos, "seed": SEED}
    if args.n_splits != 1:
        extra["n_splits"] = args.n_splits
    cfg = RunConfig(
        model = args.model,
        omit_quantizer = True,
        variant = variant,
        extra = extra,
    )
    data = {
        "summary": {
            "cos_floor_min": float(min(vals)),
            "cos_floor_median": float(np.median(vals)),
            "cos_floor_max": float(max(vals)),
        },
        "cos_floor_per_layer": {str(k): v for k, v in cos_floor.items()},
    }
    if args.n_splits != 1:
        # Refit distance = 1 - cos(wA, wB). The floor is the 95th percentile of the
        # FP-vs-FP refit distance (pooled over layers x splits, plus per layer); the
        # matching cosine floor is 1 - that distance (= 5th percentile of cos).
        all_cos = np.array([c for L in layers for c in cos_by_split[L]])
        dist_p95 = float(np.percentile(1.0 - all_cos, 95))
        data["summary_splits"] = {
            "n_splits": args.n_splits,
            "refit_distance_p95_pooled": dist_p95,
            "cos_floor_p5_pooled": 1.0 - dist_p95,
            "cos_median_pooled": float(np.median(all_cos)),
            "refit_distance_p95_per_layer_median": float(
                np.median([np.percentile(1.0 - np.array(cos_by_split[L]), 95) for L in layers])
            ),
        }
        data["refit_distance_p95_per_layer"] = {
            str(L): float(np.percentile(1.0 - np.array(cos_by_split[L]), 95)) for L in layers
        }
        data["cos_by_split_per_layer"] = {str(L): cos_by_split[L] for L in layers}
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
