"""
SafeQuant Phase 1, Step 1 — Compare FP16 vs NF4 refusal directions.

Computes cosine similarity between the best directions extracted from
FP16 and NF4 models. Also computes per-layer comparison across all
candidate directions.

This produces Finding 1: The direction is preserved after quantization.

Run:
  python phase1_problem/step1_direction/compare_directions.py --model qwen1.5b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import MODELS


def cosine_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    
    print(f"\n=== Comparing FP16 vs NF4 directions: {args.model} ===\n")
    
    # Load best directions
    fp16_data = torch.load(os.path.join(output_dir, "best_direction_fp16.pt"), map_location = "cpu")
    nf4_data = torch.load(os.path.join(output_dir, "best_direction_nf4.pt"), map_location = "cpu")
    
    fp16_dir = fp16_data["direction_normalized"]
    nf4_dir = nf4_data["direction_normalized"]
    
    # Main result: cosine similarity of best directions
    cos_sim = cosine_sim(fp16_dir, nf4_dir)
    
    print(f"=== FINDING 1: Direction Preservation ===")
    print(f"  FP16 best: layer {fp16_data['layer']}, pos {fp16_data['position_offset']}")
    print(f"  NF4 best:  layer {nf4_data['layer']}, pos {nf4_data['position_offset']}")
    print(f"  Cosine similarity (best vs best): {cos_sim:.4f}")
    print(f"  Absolute cosine similarity:       {abs(cos_sim):.4f}")
    
    # Per-layer comparison of ALL candidate directions
    fp16_cands = torch.load(os.path.join(output_dir, "candidates_fp16.pt"), map_location = "cpu")["candidates"]
    nf4_cands = torch.load(os.path.join(output_dir, "candidates_nf4.pt"), map_location = "cpu")["candidates"]
    
    # Find common keys
    common_keys = set(fp16_cands.keys()) & set(nf4_cands.keys())
    
    # Group by layer
    layer_cosines = {}
    for key in common_keys:
        layer = key[0]
        cs = cosine_sim(fp16_cands[key], nf4_cands[key])
        if layer not in layer_cosines:
            layer_cosines[layer] = []
        layer_cosines[layer].append(cs)
    
    # Average cosine per layer
    layers = sorted(layer_cosines.keys())
    avg_cosines = []
    abs_avg_cosines = []
    for l in layers:
        avg_cosines.append(np.mean(layer_cosines[l]))
        abs_avg_cosines.append(np.mean([abs(c) for c in layer_cosines[l]]))
    
    print(f"\n  Per-layer cosine similarity (FP16 vs NF4):")
    print(f"  {'Layer':>5} | {'Mean Cosine':>12} | {'Abs Mean':>10}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*10}")
    for l, c, ac in zip(layers, avg_cosines, abs_avg_cosines):
        if ac == min(abs_avg_cosines):
            marker = " <-- most divergent"
        else:
            marker = ""
        print(f"  {l:5d} | {c:12.4f} | {ac:10.4f}{marker}")
    
    overall_mean = np.mean(avg_cosines)
    overall_abs_mean = np.mean(abs_avg_cosines)
    print(f"\n  Overall mean cosine: {overall_mean:.4f}")
    print(f"  Overall abs mean cosine: {overall_abs_mean:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Per-layer cosine similarity
    axes[0].bar(layers, abs_avg_cosines, color="steelblue", alpha=0.8)
    axes[0].axhline(y = overall_abs_mean, color = "red", linestyle = "--",
                     label = f"Mean: {overall_abs_mean:.3f}")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Absolute Cosine Similarity")
    axes[0].set_title(f"Direction Preservation per Layer\n{args.model}: FP16 vs NF4")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    
    # Plot 2: Raw cosine (for sign flips)
    colors = []
    for c in avg_cosines:
        if c > 0:
            colors.append("green")
        else:
            colors.append("red")
    axes[1].bar(layers, avg_cosines, color = colors, alpha = 0.8)
    axes[1].axhline(y = 0, color = "black", linewidth = 0.5)
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Cosine Similarity (with sign)")
    axes[1].set_title(f"Direction Alignment per Layer\n(Red = sign-flipped)")
    axes[1].set_ylim(-1.05, 1.05)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "direction_comparison.png")
    plt.savefig(plot_path, dpi = 150, bbox_inches = "tight")
    print(f"\n  Plot saved to {plot_path}")
    
    # Dictionary of layer vs absolute cosine similarity (per_layer_abs_cosine)
    dc ={}
    for l, c in zip(layers, abs_avg_cosines):
        dc[l] = c

    # Saving results
    results = {
        "best_cosine_similarity": cos_sim,
        "best_abs_cosine": abs(cos_sim),
        "fp16_best_layer": fp16_data["layer"],
        "fp16_best_position": fp16_data["position_offset"],
        "nf4_best_layer": nf4_data["layer"],
        "nf4_best_position": nf4_data["position_offset"],
        "overall_mean_cosine": float(overall_mean),
        "overall_abs_mean_cosine": float(overall_abs_mean),
        "per_layer_abs_cosine": dc,
    }
    
    results_path = os.path.join(output_dir, "direction_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to {results_path}")
    
    print("\n=== Direction comparison complete ===")


if __name__ == "__main__":
    main()
