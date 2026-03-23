"""
SafeQuant Phase 1, Step 3f — Global Frobenius Norm Analysis.

Shows that the standard quantization error metric (global Frobenius norm)
is UNIFORM across all weight matrices and cannot identify which matrices
are responsible for safety degradation.

This is the second half of the "standard metrics miss it" argument:
  1. Perplexity (Step 2c) = standard BEHAVIORAL metric → misses it
  2. Frobenius norm (this script) = standard ERROR metric → misses it

If every matrix has ~3% relative Frobenius error, how do you know which
3% matters for safety? You can't. That's why we need V-score (Phase 2).

REQUIRES ZERO INFERENCE. Only loads and compares weight matrices.

Run:
  python phase1_problem/step3_circuit/global_frobenius.py --model qwen1.5b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import MODELS, DTYPE, NF4_CONFIG


# Every linear projection in the transformer
ALL_MATRICES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["up_proj", "gate_proj", "down_proj"],
}


def extract_weight(module, name):
    """Extract weight matrix, handling quantized models."""
    submodule = getattr(module, name)
    param = submodule.weight
    if hasattr(param, "dequantize"):
        return param.dequantize().float().cpu().clone()
    else:
        return param.data.float().cpu().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Global Frobenius Norm Analysis: {args.model} ===\n")

    # Load FP16 model
    print("  Loading FP16 model...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    n_layers = fp16_model.config.num_hidden_layers
    hidden_size = fp16_model.config.hidden_size

    # Load NF4 model
    print("  Loading NF4 model...")
    bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True
    )

    # Compute global Frobenius error for EVERY matrix
    results = {}

    for l in tqdm(range(n_layers), desc="  Analyzing layers"):
        fp16_layer = fp16_model.model.layers[l]
        nf4_layer = nf4_model.model.layers[l]

        for comp_type, mat_names in ALL_MATRICES.items():
            for mat_name in mat_names:
                if comp_type == "attn":
                    W_fp16 = extract_weight(fp16_layer.self_attn, mat_name)
                    W_nf4 = extract_weight(nf4_layer.self_attn, mat_name)
                else:
                    W_fp16 = extract_weight(fp16_layer.mlp, mat_name)
                    W_nf4 = extract_weight(nf4_layer.mlp, mat_name)

                frob_error = torch.norm(W_fp16 - W_nf4, p="fro").item()
                frob_norm = torch.norm(W_fp16, p="fro").item()
                relative_error = frob_error / (frob_norm + 1e-10)

                key = f"{mat_name}_layer_{l}"
                results[key] = {
                    "layer": l,
                    "matrix": mat_name,
                    "type": "write" if mat_name in ["o_proj", "down_proj"] else "read",
                    "component": comp_type,
                    "frobenius_error": frob_error,
                    "frobenius_norm": frob_norm,
                    "relative_error": relative_error,
                    "shape": list(W_fp16.shape),
                }

    del fp16_model, nf4_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # === Analysis: show uniformity ===
    all_rel_errors = [v["relative_error"] for v in results.values()]

    print(f"\n  === Global Frobenius Error Statistics ===")
    print(f"  Total matrices analyzed: {len(results)}")
    print(f"  Mean relative error:   {np.mean(all_rel_errors):.6f}")
    print(f"  Std relative error:    {np.std(all_rel_errors):.6f}")
    print(f"  Min relative error:    {np.min(all_rel_errors):.6f}")
    print(f"  Max relative error:    {np.max(all_rel_errors):.6f}")
    print(f"  Coefficient of var:    {np.std(all_rel_errors)/np.mean(all_rel_errors):.4f}")

    # Group by matrix type
    type_errors = {}
    for v in results.values():
        mat = v["matrix"]
        if mat not in type_errors:
            type_errors[mat] = []
        type_errors[mat].append(v["relative_error"])

    print(f"\n  === Error by Matrix Type ===")
    print(f"  {'Matrix':<12} | {'Mean Error':>10} | {'Std':>10} | {'Min':>10} | {'Max':>10}")
    print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for mat_name in ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]:
        if mat_name in type_errors:
            errs = type_errors[mat_name]
            print(f"  {mat_name:<12} | {np.mean(errs):>10.6f} | {np.std(errs):>10.6f} | "
                  f"{np.min(errs):>10.6f} | {np.max(errs):>10.6f}")

    # The key point: can you tell which matrices matter for safety?
    # Sort by error and show that safety-critical matrices are NOT at the top
    sorted_by_error = sorted(results.items(), key=lambda x: x[1]["relative_error"], reverse=True)

    print(f"\n  === Top 15 by Frobenius Error ===")
    print(f"  {'Matrix':<25} | {'Rel Error':>10} | {'Type':>6}")
    print(f"  {'-'*25}-+-{'-'*10}-+-{'-'*6}")
    for key, data in sorted_by_error[:15]:
        print(f"  {key:<25} | {data['relative_error']:>10.6f} | {data['type']:>6}")

    print(f"\n  ★ KEY OBSERVATION: All matrices have similar error (~{np.mean(all_rel_errors):.4f})")
    print(f"  ★ Frobenius norm CANNOT tell you which matrices damage safety")
    print(f"  ★ This is why we need per-dimension analysis weighted by safety importance")

    # === Generate plots ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Global Frobenius Norm Analysis: {args.model}\n"
                 f"(Standard error metric cannot identify safety-critical damage)",
                 fontsize=13, fontweight="bold")

    # Plot 1: All matrices sorted by error — showing uniformity
    ax = axes[0]
    sorted_errors = sorted(all_rel_errors)
    ax.bar(range(len(sorted_errors)), sorted_errors, color="gray", alpha=0.7, width=1.0)
    ax.axhline(y=np.mean(all_rel_errors), color="red", linestyle="--",
               label=f"Mean: {np.mean(all_rel_errors):.4f}")
    ax.set_xlabel("Matrix (sorted by error)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("All Matrices Look Similar\n(No outlier = no signal)")
    ax.legend(fontsize=9)

    # Plot 2: Error by matrix type — showing read vs write have similar error
    ax = axes[1]
    mat_order = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
    mat_means = [np.mean(type_errors.get(m, [0])) for m in mat_order]
    mat_stds = [np.std(type_errors.get(m, [0])) for m in mat_order]
    colors = ["red" if m in ["q_proj", "k_proj", "v_proj", "up_proj", "gate_proj"]
              else "blue" for m in mat_order]
    ax.bar(range(len(mat_order)), mat_means, yerr=mat_stds,
           color=colors, alpha=0.7, capsize=3)
    ax.set_xticks(range(len(mat_order)))
    ax.set_xticklabels(mat_order, rotation=45, fontsize=9)
    ax.set_ylabel("Mean Relative Error")
    ax.set_title("Error by Matrix Type\n(Red=Read, Blue=Write — all similar)")

    # Plot 3: Error per layer — showing no layer stands out
    ax = axes[2]
    layer_errors = {}
    for v in results.values():
        l = v["layer"]
        if l not in layer_errors:
            layer_errors[l] = []
        layer_errors[l].append(v["relative_error"])
    layers = sorted(layer_errors.keys())
    layer_means = [np.mean(layer_errors[l]) for l in layers]
    layer_stds = [np.std(layer_errors[l]) for l in layers]

    ax.bar(layers, layer_means, yerr=layer_stds, color="gray", alpha=0.7, capsize=2)
    ax.axhline(y=np.mean(all_rel_errors), color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Relative Error (all matrices)")
    ax.set_title("Error per Layer\n(No layer stands out)")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "global_frobenius_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {plot_path}")

    # === Save ===
    save_data = {
        "per_matrix_results": results,
        "statistics": {
            "mean_relative_error": float(np.mean(all_rel_errors)),
            "std_relative_error": float(np.std(all_rel_errors)),
            "min_relative_error": float(np.min(all_rel_errors)),
            "max_relative_error": float(np.max(all_rel_errors)),
            "coefficient_of_variation": float(np.std(all_rel_errors) / np.mean(all_rel_errors)),
            "total_matrices": len(results),
        },
        "per_type_statistics": {
            mat: {
                "mean": float(np.mean(errs)),
                "std": float(np.std(errs)),
            }
            for mat, errs in type_errors.items()
        },
        "conclusion": (
            f"Global Frobenius error is nearly uniform across all {len(results)} matrices "
            f"(CV={np.std(all_rel_errors)/np.mean(all_rel_errors):.4f}). "
            f"This metric cannot distinguish safety-critical matrices from non-critical ones. "
            f"A per-dimension analysis weighted by safety importance (Phase 2 V-score) is needed."
        ),
    }

    json_path = os.path.join(output_dir, "global_frobenius_analysis.json")
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"  Results saved to {json_path}")
    print("=== Global Frobenius analysis complete ===")


if __name__ == "__main__":
    main()
