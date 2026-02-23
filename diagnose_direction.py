"""
Diagnostic script to check direction file and candidates.
"""
import torch
import os

# Get the script directory and resolve relative path
script_dir = os.path.dirname(os.path.abspath(__file__))

# Check what's in the direction file
dir_path = os.path.join(script_dir, "phase1_problem", "step1_direction", "outputs", "qwen1.5b", "best_direction_fp16.pt")
print("=== Direction File Analysis ===")
print(f"File exists: {os.path.exists(dir_path)}")

if os.path.exists(dir_path):
    d = torch.load(dir_path, map_location="cpu")
    print("Keys:", list(d.keys()))
    print("Layer:", d.get("layer"))
    print("Direction shape:", d["direction_normalized"].shape)
    print("Direction norm:", torch.norm(d["direction_normalized"]).item())
    print("NaN count:", torch.isnan(d["direction_normalized"]).sum().item())
    print("First 10 values:", d["direction_normalized"][:10])
else:
    print("Direction file not found!")

# Also check if candidates exist
cand_path = os.path.join(script_dir, "phase1_problem", "step1_direction", "outputs", "qwen1.5b", "candidates_fp16.pt")
print("\n=== Candidates File Analysis ===")
print(f"Candidates file exists: {os.path.exists(cand_path)}")

if os.path.exists(cand_path):
    c = torch.load(cand_path, map_location="cpu")
    print("Candidates keys:", list(c.keys()))
    if "candidates" in c:
        print("Num candidates:", len(c["candidates"]))
        # Check if any candidate is valid
        print("\nFirst 5 candidates:")
        for i, (key, vec) in enumerate(list(c["candidates"].items())[:5]):
            print(f"  Candidate {key}: norm={torch.norm(vec).item():.4f}, nan={torch.isnan(vec).any().item()}")
    else:
        print("No 'candidates' key found in file")
        print("Available keys:", list(c.keys()))
else:
    print("No candidates file found")

