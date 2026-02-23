"""
SafeQuant Phase 2 v2, Step 2 — Per-Row Quantization Error on Write Matrices.

Computes per-row L2 error for WRITE matrices only (o_proj, down_proj).
Row i of a write matrix controls dimension i of the residual stream output,
so per-row error maps directly to per-dimension impact.

We do NOT compute per-row error for read matrices (q_proj, k_proj, v_proj,
up_proj, gate_proj) because for those, it's COLUMNS that map to residual
stream dimensions. Mixing this up would disconnect the error metric from
the DFA metric.

Also computes global (block-level) Frobenius error for comparison,
to demonstrate that global norms hide localized damage.

Run:
  python phase2_solution/diagnosis/per_row_quant_error.py --model qwen1.5b
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
from config import MODELS, DTYPE, NF4_CONFIG


def extract_write_weights(model, n_layers):
    """Extract o_proj and down_proj weights (the write matrices)."""
    weights = {}
    for l in range(n_layers):
        layer = model.model.layers[l]

        # Attention write matrix
        param = layer.self_attn.o_proj.weight
        if hasattr(param, "dequantize"):
            w = param.dequantize().float().cpu().clone()
        else:
            w = param.data.float().cpu().clone()
        weights[f"attn_layer_{l}"] = w

        # MLP write matrix
        param = layer.mlp.down_proj.weight
        if hasattr(param, "dequantize"):
            w = param.dequantize().float().cpu().clone()
        else:
            w = param.data.float().cpu().clone()
        weights[f"mlp_layer_{l}"] = w

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== Per-Row Quantization Error: {args.model} ===\n")

    # Load FP16 weights
    print("  Loading FP16 model...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    n_layers = fp16_model.config.num_hidden_layers
    hidden_size = fp16_model.config.hidden_size
    fp16_weights = extract_write_weights(fp16_model, n_layers)
    del fp16_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load NF4 weights
    print("  Loading NF4 model...")
    bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="auto", trust_remote_code=True
    )
    nf4_weights = extract_write_weights(nf4_model, n_layers)
    del nf4_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute per-row errors
    per_row_errors = {}  # {comp_name: tensor of shape (hidden_size,)}
    global_errors = {}   # {comp_name: scalar relative Frobenius error}

    for comp in tqdm(sorted(fp16_weights.keys()), desc="Computing per-row errors"):
        w_fp16 = fp16_weights[comp]
        w_nf4 = nf4_weights[comp]

        # Handle shape mismatch from BitsAndBytes block padding
        if w_fp16.shape != w_nf4.shape:
            # Truncate to the smaller shape (padding is zeros)
            min_rows = min(w_fp16.shape[0], w_nf4.shape[0])
            min_cols = min(w_fp16.shape[1], w_nf4.shape[1])
            w_fp16 = w_fp16[:min_rows, :min_cols]
            w_nf4 = w_nf4[:min_rows, :min_cols]

        # Global Frobenius error (the "standard" way)
        global_abs = torch.norm(w_fp16 - w_nf4, p="fro").item()
        global_fp16 = torch.norm(w_fp16, p="fro").item()
        global_errors[comp] = {
            "relative_error": global_abs / (global_fp16 + 1e-10),
            "absolute_error": global_abs,
        }

        # Per-row L2 error (identity-matrix normalized approach)
        # Row i of write matrix → dimension i of residual stream
        # Normalize by global matrix scale to get errors in 0.001-0.100 range
        n_rows = w_fp16.shape[0]  # should be hidden_size
        n_cols = w_fp16.shape[1]
        row_errors = torch.zeros(n_rows)

        # Identity-matrix approach: normalize by global Frobenius norm per element
        # This gives per-element scale comparable to identity matrix transformation
        global_fp16_norm = torch.norm(w_fp16, p="fro").item()
        per_element_scale = global_fp16_norm / np.sqrt(n_rows * n_cols) if (n_rows * n_cols) > 0 else 1.0

        for i in range(n_rows):
            row_diff = torch.norm(w_fp16[i, :] - w_nf4[i, :], p=2).item()
            # Normalize by per-element scale (identity-matrix approach)
            # This gives error relative to typical element magnitude, in 0.001-0.100 range
            row_errors[i] = row_diff / (per_element_scale * np.sqrt(n_cols) + 1e-10)

        per_row_errors[comp] = row_errors

    # Save per-row errors as tensors
    save_path = os.path.join(output_dir, "per_row_errors.pt")
    torch.save({
        "per_row_errors": per_row_errors,
        "global_errors": global_errors,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
    }, save_path)

    # === Demonstrate that global norm hides local damage ===
    print(f"\n  === Global vs Per-Row Error Analysis ===")
    print(f"  {'Component':<20} | {'Global Rel':>10} | {'Max Row Rel':>11} | {'Mean Row':>10} | {'Ratio':>7}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}")

    for comp in sorted(per_row_errors.keys()):
        g_err = global_errors[comp]["relative_error"]
        row_errs = per_row_errors[comp]
        max_row = row_errs.max().item()
        mean_row = row_errs.mean().item()
        ratio = max_row / (g_err + 1e-10)

        print(f"  {comp:<20} | {g_err:>10.6f} | {max_row:>11.6f} | {mean_row:>10.6f} | {ratio:>7.1f}x")

    # Show worst rows globally
    all_rows = []
    for comp, errs in per_row_errors.items():
        for i in range(errs.shape[0]):
            all_rows.append((comp, i, errs[i].item()))
    all_rows.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 10 most damaged individual rows (dimensions):")
    print(f"  {'Component':<20} | {'Row':>5} | {'Rel Error':>10}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*10}")
    for comp, row, err in all_rows[:10]:
        print(f"  {comp:<20} | {row:>5} | {err:>10.6f}")

    print(f"\n  Saved to {save_path}")
    print("=== Per-row error profiling complete ===")


if __name__ == "__main__":
    main()
