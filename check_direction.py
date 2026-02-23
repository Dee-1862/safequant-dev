"""
Quick script to check the direction file for Issue 1 (NaN values).
"""
import torch
import os

# Get the script directory and resolve relative path
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "phase1_problem", "step1_direction", "outputs", "qwen1.5b", "best_direction_fp16.pt")

if not os.path.exists(path):
    print(f"ERROR: File not found: {path}")
    print("You need to run Phase 1 step 1 direction selection first.")
    exit(1)

d = torch.load(path, map_location="cpu")

print("=== Direction File Analysis ===")
print(f"Keys: {list(d.keys())}")
print(f"Layer: {d.get('layer', 'NOT FOUND')}")
print(f"Position: {d.get('position', 'NOT FOUND')}")

if "direction_normalized" in d:
    dir_norm = d["direction_normalized"]
    print(f"Direction shape: {dir_norm.shape}")
    print(f"Direction norm: {torch.norm(dir_norm).item():.6f}")
    print(f"Any NaN: {torch.isnan(dir_norm).any().item()}")
    print(f"Any Inf: {torch.isinf(dir_norm).any().item()}")
    print(f"Min value: {dir_norm.min().item():.6f}")
    print(f"Max value: {dir_norm.max().item():.6f}")
    
    if d.get("layer", -1) == 0:
        print("\nWARNING: Layer is 0! This is likely wrong.")
        print("   Layer 0 has almost no refusal signal.")
        print("   Expected layer: 18-23 based on Phase 1 circuit analysis.")
    
    if torch.isnan(dir_norm).any().item():
        print("\nWARNING: Direction contains NaN values!")
        print("   This will cause all DFA scores to be NaN.")
else:
    print("ERROR: 'direction_normalized' key not found in file!")

