"""
SafeQuant Phase 1, Step 1 — Compute candidate refusal directions.

For each (layer, position) pair, computes:
    r(l,i) = mean_harmful(l,i) - mean_harmless(l,i)

This is the Difference-in-Means (DIM) method from Arditi et al. (2024).

Run:
  python phase1_problem/step1_direction/compute_directions.py --model qwen1.5b --precision fp16
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import torch
from config import MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    
    # Load cached activations
    act_path = os.path.join(output_dir, f"activations_{args.precision}.pt")
    print(f"\n=== Computing directions: {args.model} ({args.precision}) ===")
    print(f"Loading activations from {act_path}")
    
    data = torch.load(act_path, map_location="cpu")
    harmful_means = data["harmful_means"]
    harmless_means = data["harmless_means"]
    n_layers = data["n_layers"]
    hidden_size = data["hidden_size"]
    
    # Compute difference-in-means for each (layer, position) pair
    candidates = {}
    for key in harmful_means:
        if key in harmless_means:
            direction = harmful_means[key] - harmless_means[key]
            candidates[key] = direction
    
    print(f"Computed {len(candidates)} candidate directions")
    
    # Print some stats
    norms = {k: torch.norm(v).item() for k, v in candidates.items()}
    layers_seen = set(k[0] for k in candidates)
    positions_seen = set(k[1] for k in candidates)
    
    print(f"  Layers covered: {min(layers_seen)}-{max(layers_seen)}")
    print(f"  Position offsets covered: {min(positions_seen)}-{max(positions_seen)}")
    print(f"  Direction norms: min={min(norms.values()):.4f}, "
          f"max={max(norms.values()):.4f}, mean={sum(norms.values())/len(norms):.4f}")
    
    # Save
    save_path = os.path.join(output_dir, f"candidates_{args.precision}.pt")
    torch.save({
        "candidates": candidates,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
    }, save_path)
    
    print(f"\nSaved candidates to {save_path}")
    print("=== Direction computation complete ===")


if __name__ == "__main__":
    main()
