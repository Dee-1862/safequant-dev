"""
SafeQuant Phase 1, Step 3 - Compute DFA (Direct Feature Attribution) scores.

For each component, computes how much its output projects onto the
refusal direction. High DFA = important for refusal behavior.

DFA(component) = mean(component_output · r_hat)

Run:
  python phase1_problem/step3_circuit/compute_dfa.py --model qwen1.5b --precision fp16
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from config import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, choices = list(MODELS.keys()))
    parser.add_argument("--precision", required = True, choices = ["fp16", "nf4"])
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "step1_direction", "outputs", args.model
    )
    
    print(f"\nComputing DFA: {args.model} ({args.precision})\n")
    
    # Load the best refusal direction (use FP16 direction as reference for both)
    # This is intentional: we want to measure how much each component
    # projects onto the SAME direction in both FP16 and NF4
    dir_data = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"),
        map_location = "cpu"
    )
    r_hat = dir_data["direction_normalized"]  # unit vector
    
    print(f"Using FP16 refusal direction from layer {dir_data['layer']}")
    
    # Load component outputs
    comp_data = torch.load(
        os.path.join(output_dir, f"component_outputs_{args.precision}.pt"),
        map_location = "cpu"
    )
    attn_means = comp_data["attn_means"]
    mlp_means = comp_data["mlp_means"]
    n_layers = comp_data["n_layers"]
    
    # Compute DFA for each component
    dfa_scores = {}
    
    for l in range(n_layers):
        # Attention DFA at this layer
        attn_dfa = torch.dot(attn_means[l], r_hat).item()
        dfa_scores[f"attn_layer_{l}"] = attn_dfa
        
        # MLP DFA at this layer
        mlp_dfa = torch.dot(mlp_means[l], r_hat).item()
        dfa_scores[f"mlp_layer_{l}"] = mlp_dfa
    
    # Rank by absolute DFA
    ranked = sorted(dfa_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Compute total signal
    total_positive = sum(v for v in dfa_scores.values() if v > 0)
    total_negative = sum(v for v in dfa_scores.values() if v < 0)
    total_signal = sum(dfa_scores.values())
    
    print(f"DFA scores ({args.precision}):")
    print(f"  Total components: {len(dfa_scores)}")
    print(f"  Total positive signal: {total_positive:.4f}")
    print(f"  Total negative signal: {total_negative:.4f}")
    print(f"  Net signal: {total_signal:.4f}")
    
    print(f"\n  Top 10 components by |DFA|:")
    for name, score in ranked[:10]:
        pct = abs(score) / sum(abs(v) for v in dfa_scores.values()) * 100
        print(f"    {name:<20}  DFA={score:>10.4f}  |DFA|={abs(score):>10.4f}  ({pct:.1f}%)")
    
    # Save
    results = {
        "precision": args.precision,
        "reference_direction": "fp16",
        "dfa_scores": dfa_scores,
        "ranked_by_abs": [(name, score) for name, score in ranked],
        "total_positive_signal": total_positive,
        "total_negative_signal": total_negative,
        "net_signal": total_signal,
    }
    
    save_path = os.path.join(output_dir, f"dfa_scores_{args.precision}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent = 2)
    
    print(f"\nSaved to {save_path}")
    print("DFA computation complete")


if __name__ == "__main__":
    main()
