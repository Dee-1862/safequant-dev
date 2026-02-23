"""
Fix the NF4 direction file by selecting a valid candidate from the candidates file.
"""
import torch
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "phase1_problem", "step1_direction", "outputs", "qwen1.5b")

# Load candidates
cand_path = os.path.join(output_dir, "candidates_nf4.pt")
if not os.path.exists(cand_path):
    print(f"NF4 candidates file not found: {cand_path}")
    exit(1)

c = torch.load(cand_path, map_location="cpu")
candidates = c["candidates"]

print("=== Finding best valid NF4 candidate ===")
print(f"Total candidates: {len(candidates)}")

# Filter to late layers (18-23) and find best by norm
best_candidate = None
best_norm = 0
best_key = None

for key, vec in candidates.items():
    layer, pos = key
    # Skip layer 0 and focus on late layers
    if layer >= 18 and layer <= 23:
        norm = torch.norm(vec).item()
        has_nan = torch.isnan(vec).any().item()
        if not has_nan and norm > best_norm:
            best_norm = norm
            best_candidate = vec
            best_key = key

if best_candidate is None:
    # Fallback: find best from any layer except 0
    print("No candidate found in layers 18-23, searching all layers...")
    for key, vec in candidates.items():
        layer, pos = key
        if layer > 0:  # Skip layer 0
            norm = torch.norm(vec).item()
            has_nan = torch.isnan(vec).any().item()
            if not has_nan and norm > best_norm:
                best_norm = norm
                best_candidate = vec
                best_key = key

if best_candidate is None:
    print("ERROR: No valid candidate found!")
    exit(1)

layer, pos = best_key
print(f"\nSelected candidate:")
print(f"  Layer: {layer}")
print(f"  Position: {pos}")
print(f"  Norm: {best_norm:.4f}")
print(f"  Has NaN: {torch.isnan(best_candidate).any().item()}")

# Load existing direction file to preserve other metadata
dir_path = os.path.join(output_dir, "best_direction_nf4.pt")
if not os.path.exists(dir_path):
    print(f"NF4 direction file not found: {dir_path}")
    exit(1)

d = torch.load(dir_path, map_location="cpu")

# Update with valid candidate
d["direction"] = best_candidate
d["direction_normalized"] = best_candidate / torch.norm(best_candidate)
d["layer"] = layer
d["position"] = pos
d["position_idx"] = pos

# Save fixed direction file
torch.save(d, dir_path)
print(f"\nFixed NF4 direction file saved to: {dir_path}")

