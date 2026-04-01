"""
SafeQuant Phase 1, Step 3 - Compare FP16 vs NF4 circuits and visualize.

Shows which components lost the most refusal signal after quantization.
This is Finding 3: the mechanistic explanation for safety degradation.

Run:
  python phase1_problem/step3_circuit/compare_and_visualize.py --model qwen1.5b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from config import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, choices = list(MODELS.keys()))
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    
    print(f"\n=== Comparing circuits: {args.model} ===\n")
    
    # Load DFA scores
    with open(os.path.join(output_dir, "dfa_scores_fp16.json")) as f:
        fp16_data = json.load(f)
    with open(os.path.join(output_dir, "dfa_scores_nf4.json")) as f:
        nf4_data = json.load(f)
    
    fp16_scores = fp16_data["dfa_scores"]
    nf4_scores = nf4_data["dfa_scores"]
    
    # Compute per-component changes
    components = sorted(fp16_scores.keys())
    changes = {}
    for comp in components:
        fp16_val = fp16_scores[comp]
        nf4_val = nf4_scores.get(comp, 0)
        changes[comp] = {
            "fp16_dfa": fp16_val,
            "nf4_dfa": nf4_val,
            "absolute_change": nf4_val - fp16_val,
            "relative_change": (nf4_val - fp16_val) / (abs(fp16_val) + 1e-8),
        }
    
    # Signal retention
    total_fp16 = sum(abs(fp16_scores[c]) for c in components)
    total_nf4 = sum(abs(nf4_scores.get(c, 0)) for c in components)
    signal_retention = total_nf4 / total_fp16 * 100 if total_fp16 > 0 else 0
    
    # Net signal
    net_fp16 = sum(fp16_scores[c] for c in components)
    net_nf4 = sum(nf4_scores.get(c, 0) for c in components)
    
    print(f"=== FINDING 3: Circuit-Level Analysis ===")
    print(f"  Total |DFA| signal FP16: {total_fp16:.4f}")
    print(f"  Total |DFA| signal NF4:  {total_nf4:.4f}")
    print(f"  Signal retention: {signal_retention:.1f}%")
    print(f"  Net signal FP16: {net_fp16:.4f}")
    print(f"  Net signal NF4:  {net_nf4:.4f}")
    
    # Top degraded components
    sorted_by_loss = sorted(
        changes.items(),
        key=lambda x: -abs(x[1]["absolute_change"])
    )
    
    print(f"\n  Top 10 most affected components:")
    print(f"  {'Component':<20} | {'FP16 DFA':>10} | {'NF4 DFA':>10} | {'Change':>10} | {'Rel Change':>10}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for comp, vals in sorted_by_loss[:10]:
        print(f"  {comp:<20} | {vals['fp16_dfa']:>10.4f} | {vals['nf4_dfa']:>10.4f} | "
              f"{vals['absolute_change']:>10.4f} | {vals['relative_change']:>10.2f}")
    
    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    
    # Parse layer numbers for plotting
    attn_layers = sorted([c for c in components if c.startswith("attn_")],
                         key=lambda x: int(x.split("_")[-1]))
    mlp_layers = sorted([c for c in components if c.startswith("mlp_")],
                        key=lambda x: int(x.split("_")[-1]))
    
    layer_nums_attn = [int(c.split("_")[-1]) for c in attn_layers]
    layer_nums_mlp = [int(c.split("_")[-1]) for c in mlp_layers]
    
    fig, axes = plt.subplots(2, 2, figsize = (16, 12))
    fig.suptitle(f"SafeQuant Circuit Analysis: {args.model}", fontsize = 14, fontweight = "bold")
    
    # --- Plot 1: FP16 DFA scores (the refusal circuit) ---
    ax = axes[0, 0]
    attn_fp16 = [fp16_scores[c] for c in attn_layers]
    mlp_fp16 = [fp16_scores[c] for c in mlp_layers]
    
    x = np.arange(len(layer_nums_attn))
    width = 0.35
    ax.bar(x - width/2, attn_fp16, width, label = "Attention", color = "steelblue", alpha = 0.8)
    ax.bar(x + width/2, mlp_fp16, width, label = "MLP", color = "coral", alpha = 0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("DFA Score")
    ax.set_title("FP16 Refusal Circuit (DFA per component)")
    ax.legend()
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([str(l) for l in layer_nums_attn[::max(1, len(x)//10)]])
    ax.axhline(y = 0, color = "black", linewidth = 0.5)

    # --- Plot 2: FP16 vs NF4 comparison ---
    ax = axes[0, 1]
    attn_nf4 = [nf4_scores.get(c, 0) for c in attn_layers]
    mlp_nf4 = [nf4_scores.get(c, 0) for c in mlp_layers]
    
    all_fp16 = attn_fp16 + mlp_fp16
    all_nf4 = attn_nf4 + mlp_nf4
    all_labels = [f"A{l}" for l in layer_nums_attn] + [f"M{l}" for l in layer_nums_mlp]
    
    x2 = np.arange(len(all_fp16))
    ax.bar(x2 - width/2, all_fp16, width, label = "FP16", color = "steelblue", alpha = 0.7)
    ax.bar(x2 + width/2, all_nf4, width, label = "NF4", color = "orange", alpha = 0.7)
    ax.set_xlabel("Component (A=Attn, M=MLP)")
    ax.set_ylabel("DFA Score")
    ax.set_title(f"DFA Comparison: FP16 vs NF4\nSignal Retention: {signal_retention:.1f}%")
    ax.legend()
    step = max(1, len(x2) // 15)
    ax.set_xticks(x2[::step])
    ax.set_xticklabels([all_labels[i] for i in range(0, len(all_labels), step)], rotation = 45, fontsize = 7)
    ax.axhline(y = 0, color = "black", linewidth = 0.5)

    # --- Plot 3: Absolute change per component ---
    ax = axes[1, 0]
    abs_changes_attn = [changes[c]["absolute_change"] for c in attn_layers]
    abs_changes_mlp = [changes[c]["absolute_change"] for c in mlp_layers]
    
    ax.bar(x - width/2, abs_changes_attn, width, label = "Attention", color = "steelblue", alpha = 0.8)
    ax.bar(x + width/2, abs_changes_mlp, width, label = "MLP", color = "coral", alpha = 0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("DFA Change (NF4 - FP16)")
    ax.set_title("Per-Component Signal Change After Quantization")
    ax.legend()
    ax.set_xticks(x[::max(1, len(x)//10)])
    ax.set_xticklabels([str(l) for l in layer_nums_attn[::max(1, len(x)//10)]])
    ax.axhline(y = 0, color = "black", linewidth = 0.5)

    # --- Plot 4: Top affected components ---
    ax = axes[1, 1]
    top_n = 15
    top_comps = sorted_by_loss[:top_n]
    names = [c[0] for c in top_comps]
    abs_vals = [abs(c[1]["absolute_change"]) for c in top_comps]
    colors = ["red" if c[1]["absolute_change"] < 0 else "green" for c in top_comps]
    
    ax.barh(range(top_n), abs_vals, color = colors, alpha = 0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize = 8)
    ax.set_xlabel("|DFA Change|")
    ax.set_title(f"Top {top_n} Most Affected Components\n(Red=lost signal, Green=gained)")
    ax.invert_yaxis()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "circuit_analysis.png")
    plt.savefig(plot_path, dpi = 150, bbox_inches = "tight")
    print(f"\n  Circuit analysis plot saved to {plot_path}")
    
    # Save comparison results
    comparison = {
        "signal_retention_percent": signal_retention,
        "total_abs_dfa_fp16": total_fp16,
        "total_abs_dfa_nf4": total_nf4,
        "net_signal_fp16": net_fp16,
        "net_signal_nf4": net_nf4,
        "top_degraded": [
            {"component": comp, **vals}
            for comp, vals in sorted_by_loss[:20]
        ],
    }
    
    comp_path = os.path.join(output_dir, "circuit_comparison.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent = 2)
    
    print(f"  Comparison saved to {comp_path}")
    print("\n=== Circuit comparison complete ===")


if __name__ == "__main__":
    main()
