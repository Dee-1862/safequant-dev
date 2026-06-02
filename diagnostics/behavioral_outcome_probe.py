import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from core.seeds import set_global_seed, DEFAULT_SEED
SEED = set_global_seed(DEFAULT_SEED)

from core.model_loader import load_quantized
from core.datasets import load_wildjailbreak_prompts
from core.evaluation import generate_only, classify_batch, _extract_behavior

from layer_routing import (
    extract_last_token_residuals, _get_inner,
)
from directional_ablation import install_ablation, remove_handles

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


def per_layer_probe_accuracy(residuals_by_layer, labels):
    """5-fold CV accuracy per layer for a logistic regression probe."""
    layers = sorted(residuals_by_layer.keys())
    accs = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = np.asarray(labels, dtype=np.int64)

    for L in layers:
        X = np.stack([t.numpy() for t in residuals_by_layer[L]]).astype(np.float64)
        # If degenerate label (only one class present), skip
        if len(np.unique(y)) < 2:
            accs[L] = float("nan")
            continue
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1,
                                     random_state=SEED)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
        accs[L] = float(np.mean(fold_accs))
    return accs


def behavioral_direction(residuals_at_layer, labels):
    """Mean(label=1) - mean(label=0), normalized."""
    y = np.asarray(labels, dtype=np.int64)
    X = torch.stack(residuals_at_layer).float()    # [N, D]
    mu1 = X[y == 1].mean(dim=0)
    mu0 = X[y == 0].mean(dim=0)
    r = (mu1 - mu0).float()
    n = r.norm()
    return r / n if n > 1e-6 else r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_train_prompts", type=int, default=200,
                    help="prompts used to train per-layer probes (response-labeled)")
    ap.add_argument("--n_eval_ablation", type=int, default=100,
                    help="held-out prompts for the ASR ablation eval")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = (f"outputs/behavioral_outcome_probe_"
                       f"{args.model}_{args.quantizer}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  Behavioral-outcome probe")
    print(f"  {args.model} / {args.quantizer}")
    print(f"  n_train_prompts={args.n_train_prompts}  n_eval_ablation={args.n_eval_ablation}")
    print(f"  seed={SEED}")
    print("=" * 70)

    # Use disjoint slices of WildJailbreak: train on first N from `train`,
    # ablation eval on first M from `eval`.
    print("\nLoading prompts ...")
    train_items = load_wildjailbreak_prompts(n_samples=args.n_train_prompts,
                                              config="train")
    train_behaviors = [_extract_behavior(it) for it in train_items]
    ablation_items = load_wildjailbreak_prompts(n_samples=args.n_eval_ablation,
                                                 config="eval")
    ablation_behaviors = [_extract_behavior(it) for it in ablation_items]
    print(f"  n_train_prompts: {len(train_items)}")
    print(f"  n_eval_ablation: {len(ablation_items)}")

    # -- Quant model: gather residuals + behavioral labels ------------------
    print(f"\nLoading {args.quantizer} model ...")
    q_model, tokenizer = load_quantized(args.model, quantizer=args.quantizer)
    inner = _get_inner(q_model)
    n_layers = len(inner.layers)
    layers = list(range(n_layers))
    q_device = next(q_model.parameters()).device
    print(f"  n_layers = {n_layers}")

    # Generate responses for the probe-training prompts
    print(f"\n[probe-train] generating responses on {len(train_items)} prompts ...")
    train_responses = generate_only(q_model, tokenizer, train_items,
                                     desc="probe-train")

    # Extract per-layer residuals on the same training prompts
    print(f"[probe-train] extracting per-layer residuals ...")
    train_residuals = extract_last_token_residuals(
        q_model, tokenizer, train_items, layers, q_device,
    )

    # Generate baseline responses on the held-out ablation set
    print(f"\n[ablation] generating baseline responses on {len(ablation_items)} prompts ...")
    baseline_responses = generate_only(q_model, tokenizer, ablation_items,
                                        desc="baseline")

    # -- Phase B: HarmBench classify train + baseline + (deferred) ablation
    # We classify train + baseline now. Ablation responses are classified later in the same batch to keep classifier loads minimal.
    # -- We do all classification in one HarmBench load at the end ----------

    # Generate ablation responses with the probe-direction hook installed.
    # This needs the probe direction first, so we have to:
    #   (a) classify the train responses now to get labels,
    #   (b) train probes, pick best layer & direction,
    #   (c) install ablation, regenerate ablation responses.
    # Since classifier load is the heavy step, we do (a) and the final (baseline + ablation) classification in two HarmBench loads total.

    # (a) Classify TRAINING responses to get behavioral labels
    print(f"\n[classifier] classifying {len(train_responses)} training responses ...")
    train_flags = classify_batch(train_behaviors, train_responses, device=device)
    train_labels = [int(f) for f in train_flags]
    n_jb = sum(train_labels)
    n_rf = len(train_labels) - n_jb
    print(f"  jailbroken: {n_jb}   refused: {n_rf}   "
          f"jailbreak rate: {100.0*n_jb/len(train_labels):.1f}%")

    if n_jb < 10 or n_rf < 10:
        print(f"  WARNING: degenerate label distribution; probe will be unstable.")

    # -- Per-layer probe accuracy -------------------------------------------
    print(f"\n[probe] training per-layer logistic regression (5-fold CV) ...")
    accs = per_layer_probe_accuracy(train_residuals, train_labels)
    best_layer = max(accs, key=lambda L: accs[L] if not np.isnan(accs[L]) else -1)
    print(f"  per-layer accuracy:")
    for L in sorted(accs):
        marker = " <-- best" if L == best_layer else ""
        print(f"    L{L:>2}: {accs[L]:.3f}{marker}")
    print(f"  best layer = {best_layer}  acc = {accs[best_layer]:.3f}")

    # -- Behavioral direction at best layer ---------------------------------
    r_behav = behavioral_direction(train_residuals[best_layer], train_labels)
    r_behav_cpu = r_behav.cpu().clone()
    print(f"  ||r_behav|| = {r_behav.norm():.4f}  (should be 1.0)")

    # -- Free training residuals --------------------------------------------
    del train_residuals
    gc.collect()
    torch.cuda.empty_cache()

    # -- Ablation: install hook with r_behav at every layer, regenerate -----
    print(f"\n[ablation] generating responses with r_behav (best layer={best_layer}) ablated ...")
    handles = install_ablation(q_model, r_behav_cpu, layers)
    try:
        ablation_responses = generate_only(q_model, tokenizer, ablation_items,
                                            desc="behav-ablation")
    finally:
        remove_handles(handles)

    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # (b) Classify baseline + ablation responses
    print(f"\n[classifier] classifying {2*len(ablation_items)} ablation-eval responses ...")
    flat_b = ablation_behaviors + ablation_behaviors
    flat_r = baseline_responses + ablation_responses
    flat_flags = classify_batch(flat_b, flat_r, device=device)
    n = len(ablation_items)
    base_flags = flat_flags[:n]
    abl_flags = flat_flags[n:]
    base_asr = 100.0 * sum(base_flags) / n
    abl_asr = 100.0 * sum(abl_flags) / n
    delta_asr = abl_asr - base_asr

    print()
    print("=" * 70)
    print(f"  Behavioral-outcome probe summary -- {args.model} / {args.quantizer}")
    print("=" * 70)
    print(f"  Probe-train jailbreak rate: {100.0*n_jb/len(train_labels):.1f}%   ({n_jb}/{len(train_labels)})")
    print(f"  Best probe layer / acc:     L{best_layer}  ({accs[best_layer]:.3f})")
    print(f"  Held-out baseline ASR:      {base_asr:.1f}%")
    print(f"  Held-out ablation  ASR:     {abl_asr:.1f}%")
    print(f"  DeltaASR (behav-direction abl): {delta_asr:+.1f}pp")
    if abs(delta_asr) >= 5:
        print(f"  => OPERATIVE: behavioral-outcome direction moves ASR >= 5pp")
    else:
        print(f"  => INOPERATIVE: behavioral-outcome direction does not move ASR")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_train_prompts": args.n_train_prompts,
        "n_eval_ablation": args.n_eval_ablation,
        "n_layers": n_layers,
        "seed": SEED,
        "n_jailbroken_train": int(n_jb),
        "n_refused_train": int(n_rf),
        "train_jailbreak_rate": float(n_jb / len(train_labels)),
        "per_layer_probe_accuracy": {int(L): float(a) for L, a in accs.items()},
        "best_layer": int(best_layer),
        "best_layer_accuracy": float(accs[best_layer]),
        "baseline_asr": float(base_asr),
        "behavioral_ablation_asr": float(abl_asr),
        "delta_asr": float(delta_asr),
        "operative": bool(abs(delta_asr) >= 5),
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
