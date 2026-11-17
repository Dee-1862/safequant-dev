# Table 1: per-layer probe accuracy (a_min).
import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.manifest import RunConfig, assert_new_result, save_result
from core.model_loader import load_fp32, load_quantized
from core.probes import fit_probe_per_layer, permute_labels_by_layer
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--n_folds", type = int, default = 5)
    ap.add_argument("--probe_type", default = "linear", choices = ["linear", "mlp"])
    # X3 control: permute harmful/harmless labels (seed 42) before fitting. Expect ~50%.
    ap.add_argument("--shuffle_labels", action = "store_true")
    args = ap.parse_args()

    # Shuffled-label runs get their own filename (never overwrite the real probe JSON)
    variant = None if args.probe_type == "linear" else args.probe_type
    if args.shuffle_labels:
        variant = "shuffled" if variant is None else f"{variant}_shuffled"
        assert_new_result(
            _SCRIPT, RunConfig(model = args.model, quantizer = args.quantizer, variant = variant)
        )

    # Load calibration harmful and harmless prompts
    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # Fit FP32 cross-validated probes
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    layers = list(range(len(get_inner(fp32_model).layers)))
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
    if args.shuffle_labels:
        fp_h, fp_s = permute_labels_by_layer(fp_h, fp_s, seed = SEED)
    fp_probes = fit_probe_per_layer(fp_h, fp_s, n_folds = args.n_folds, probe_type = args.probe_type)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Fit quantized cross-validated probes
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    q_device = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    if args.shuffle_labels:
        q_h, q_s = permute_labels_by_layer(q_h, q_s, seed = SEED)
    q_probes = fit_probe_per_layer(q_h, q_s, n_folds = args.n_folds, probe_type = args.probe_type)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summarize accuracy deltas and large drops
    fp_accs = [fp_probes[L]["mean_acc"] for L in layers]
    q_accs = [q_probes[L]["mean_acc"] for L in layers]
    layers_drop = [L for L in layers if (q_probes[L]["mean_acc"] - fp_probes[L]["mean_acc"]) < -0.05]

    # Persist per-layer accuracies
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = variant,
        extra = {
            "n_calib": args.n_calib,
            "n_folds": args.n_folds,
            "probe_type": args.probe_type,
            "seed": SEED,
            **({"shuffle_labels": True} if args.shuffle_labels else {}),
        },
    )
    data = {
        "n_layers": len(layers),
        "fp32_probe_per_layer": {str(L): fp_probes[L] for L in layers},
        "quant_probe_per_layer": {str(L): q_probes[L] for L in layers},
        "summary": {
            "fp32_mean_acc": float(np.mean(fp_accs)),
            "quant_mean_acc": float(np.mean(q_accs)),
            "fp32_min_acc": float(np.min(fp_accs)),
            "quant_min_acc": float(np.min(q_accs)),
            "fp32_argmax_layer": int(np.argmax(fp_accs)),
            "quant_argmax_layer": int(np.argmax(q_accs)),
            "layers_with_drop_gt_5pp": layers_drop,
            "n_layers_with_drop": len(layers_drop),
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
