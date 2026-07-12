"""Probe causal ablation null (decodability vs operativity)."""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.evaluation import _extract_behavior, classify_batch, generate_only
from core.hooks import install_ablation, remove_handles
from core.manifest import RunConfig, save_result
from core.model_loader import load_quantized
from core.probes import per_layer_probe_accuracy
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def fit_probe_get_directions(harmful_by_layer, harmless_by_layer):
    """
    Fit a probe to get directions for each layer.

    Inputs:
        harmful_by_layer: A dictionary of harmful activations by layer.
        harmless_by_layer: A dictionary of harmless activations by layer.

    Outputs:
        A dictionary of directions by layer.
    """
    layers = sorted(harmful_by_layer.keys())
    out = {}

    # Fit a probe to get directions for each layer
    for layer_idx in layers:
        # Stack the harmful and harmless activations into a single array
        h = np.stack([t.cpu().numpy() for t in harmful_by_layer[layer_idx]])
        s = np.stack([t.cpu().numpy() for t in harmless_by_layer[layer_idx]])
        X = np.concatenate([h, s], axis=0).astype(np.float64)

        # Create the labels for the logistic regression
        y = np.concatenate([np.ones(len(h)), np.zeros(len(s))], axis = 0)
        clf = LogisticRegression(C = 1.0, max_iter = 2000, random_state = SEED)
        clf.fit(X, y)
        w = clf.coef_[0]
        out[layer_idx] = w / (np.linalg.norm(w) + 1e-12)
    return out


def main():
    # Command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required = True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load prompts
    calib_harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    calib_harmless = load_harmless_prompts(n_samples=args.n_calib)
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    eval_behaviors = [_extract_behavior(it) for it in eval_items]

    # Load quantized model and get directions
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device
    
    layers = list(range(len(get_inner(q_model).layers)))

    # Extract last token residuals
    h_resid = extract_last_token_residuals(model = q_model, tok = q_tok, prompts = calib_harmful, layers = layers, device = q_device)
    s_resid = extract_last_token_residuals(model = q_model, tok = q_tok, prompts = calib_harmless, layers = layers, device = q_device)
    
    # Fit a probe to get directions for each layer
    directions = fit_probe_get_directions(h_resid, s_resid)
    resid_by_layer = {L: h_resid[L] + s_resid[L] for L in layers}
    
    # Create the labels for the logistic regression and calculate the accuracy for each layer
    labels = [1] * len(calib_harmful) + [0] * len(calib_harmless)
    accs = per_layer_probe_accuracy(resid_by_layer, labels)

    # Get the best layer
    best_L = max(accs, key=lambda L: accs[L] if not np.isnan(accs[L]) else -1)

    # Generate baseline responses and install ablation handles
    baseline_responses = generate_only(
        model = q_model,
        tok = q_tok,
        prompts = eval_items,
        desc = "baseline",
        max_new_tokens = args.max_new_tokens
    )
    best_dir = torch.tensor(directions[best_L], dtype = torch.float32)
    handles = install_ablation(model = q_model, direction = best_dir, layers = layers)

    # Generate multilayer ablation responses
    try:
        ablation_responses = generate_only(
            model = q_model,
            tok = q_tok,
            prompts = eval_items,
            desc = "multilayer-abl",
            max_new_tokens = args.max_new_tokens
        )
    finally:
        remove_handles(handles)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Classify responses and calculate the ASR for each key
    all_responses = {"baseline": baseline_responses, "multilayer_ablation": ablation_responses}
    flat_behaviors, flat_responses = [], []
    for k in all_responses:
        flat_behaviors.extend(eval_behaviors)
        flat_responses.extend(all_responses[k])
    flags = classify_batch(flat_behaviors, flat_responses, device=device)
    n = args.n_eval
    asr_results = {
        k: 100.0 * sum(flags[i * n : (i + 1) * n]) / n for i, k in enumerate(all_responses)
    }

    # Save results
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        extra = {"n_calib": args.n_calib, "n_eval": args.n_eval, "seed": SEED},
    )
    data = {
        "best_probe_layer": best_L,
        "best_probe_cv_acc": accs[best_L],
        "baseline_asr": asr_results["baseline"],
        "multilayer_ablation_asr": asr_results["multilayer_ablation"],
        "delta_asr": asr_results["multilayer_ablation"] - asr_results["baseline"],
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
