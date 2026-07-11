"""Table 1: probe-axis cosine (probe rotation)."""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.directions import cos_sim
from core.manifest import RunConfig, save_result
from core.model_loader import load_fp32, load_quantized
from core.probes import fit_probe_get_weights
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    args = ap.parse_args()

    harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    layers = list(range(len(get_inner(fp32_model).layers)))
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
    fp_probes = fit_probe_get_weights(fp_h, fp_s)
    del fp32_model
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    q_device = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    q_probes = fit_probe_get_weights(q_h, q_s)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    cos_per_layer = {}
    for layer_idx in layers:
        fp_w = np.asarray(fp_probes[layer_idx]["weight"], dtype=np.float64)
        q_w = np.asarray(q_probes[layer_idx]["weight"], dtype=np.float64)
        cos_per_layer[str(layer_idx)] = cos_sim(
            torch.from_numpy(fp_w), torch.from_numpy(q_w)
        )

    cfg = RunConfig(
        model=args.model,
        quantizer=args.quantizer,
        extra={"n_calib": args.n_calib, "seed": SEED},
    )
    data = {
        "n_layers": len(layers),
        "cos_probe_weight_per_layer": cos_per_layer,
        "summary": {
            "cos_min": float(min(cos_per_layer.values())),
            "cos_median": float(np.median(list(cos_per_layer.values()))),
            "cos_max": float(max(cos_per_layer.values())),
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
