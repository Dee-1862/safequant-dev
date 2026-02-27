"""
Step 2: Per-Row Quantization Error on Write Matrices.

PURPOSE:
--------
This script measures how much quantization damages each individual row of the write matrices
(o_proj, down_proj). Since row i of a write matrix controls dimension i of the residual stream
output, per-row error maps directly to per-dimension impact on model behavior.

This complements Step 1 (per-dimension DFA) by identifying which dimensions are most damaged
by quantization, regardless of their importance to refusal behavior.

MATHEMATICAL FORMULA:
--------------------
For write matrices (o_proj, down_proj), row i controls dimension i of the residual stream output.
We compute per-row L2 error using identity-matrix normalization:

    ε_i(c) = ||W_fp16[i,:] - W_nf4[i,:]||_2 / (scale_factor)

Where:
  - ε_i(c): Quantization error for row i in component c
  - W_fp16[i,:]: The i-th row of the FP16 weight matrix
  - W_nf4[i,:]: The i-th row of the NF4 quantized weight matrix (dequantized)
  - scale_factor: Normalization factor based on global matrix scale (identity-matrix approach)

The identity-matrix normalization ensures errors are in a meaningful range (0.001-0.100),
making them comparable across different components and suitable for multiplication with DFA scores.

WHY PER-ROW (NOT PER-COLUMN):
------------------------------
- We ONLY compute per-row error for write matrices (o_proj, down_proj) because:
  * Row i of a write matrix directly maps to dimension i of the residual stream output.
  * This matches the per-dimension DFA analysis from Step 1.
  
- We do NOT compute per-row error for read matrices (q_proj, k_proj, v_proj, up_proj, gate_proj):
  * For read matrices, it's COLUMNS (not rows) that map to residual stream dimensions.
  * Mixing row/column analysis would disconnect the error metric from the DFA metric.
  * The vulnerability score (Step 3) requires DFA and error to be aligned on the same dimensions.

OUTPUT:
-------
Saves per-row quantization errors to:
  - phase2_solution/diagnosis/outputs/{model}/per_row_errors.pt

The saved file contains:
  - per_row_errors: Dictionary mapping component names to tensors of shape (hidden_size,)
  - global_errors: Dictionary with global Frobenius error metrics for comparison
  - n_layers: Number of layers in the model
  - hidden_size: Hidden dimension size

USAGE:
------
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
    """
    Extract o_proj and down_proj weights. This is the only way to get the write weights for the model.
    
    INPUT:
    - model: The model to extract write weights from
    - n_layers: Number of layers in the model

    OUTPUT:
    - weights: Dictionary mapping component names to weight tensors
      Format: {"attn_layer_{l}": tensor, "mlp_layer_{l}": tensor, ...}
      Each tensor has shape (out_features, in_features) on CPU in FP32.
    """
    weights = {}
    
    for l in range(n_layers):
        layer = model.model.layers[l]

        # Extract attention write matrix (o_proj)
        # This matrix maps attention output to residual stream dimensions
        param = layer.self_attn.o_proj.weight

        # Handle quantized weights (BitsAndBytes) vs standard weights
        # If quantized, dequantize to FP32; otherwise convert directly to FP32
        if hasattr(param, "dequantize"):
            w = param.dequantize().float().cpu().clone()
        else:
            w = param.data.float().cpu().clone()
        weights[f"attn_layer_{l}"] = w

        # Extract MLP write matrix (down_proj)
        # This matrix maps MLP output to residual stream dimensions
        param = layer.mlp.down_proj.weight

        # Handle quantized weights (BitsAndBytes) vs standard weights
        if hasattr(param, "dequantize"):
            w = param.dequantize().float().cpu().clone()
        else:
            w = param.data.float().cpu().clone()
        weights[f"mlp_layer_{l}"] = w

    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        required = True, 
        choices = list(MODELS.keys()))
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok = True)

    print(f"\n=== Per-Row Quantization Error: {args.model} ===\n")

    # Load FP16 weights
    print("  Loading FP16 model...")
    fp16_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype = DTYPE, 
        device_map = "cpu", 
        trust_remote_code = True
    )

    # Get model configuration for later use
    n_layers = fp16_model.config.num_hidden_layers
    hidden_size = fp16_model.config.hidden_size
    
    # Extract write matrix weights from FP16 model
    fp16_weights = extract_write_weights(fp16_model, n_layers)

    # Delete the FP16 model to free up memory
    del fp16_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Step 2: Load NF4 quantized model (damaged weights)
    # Why NF4?: This is the quantized version we want to measure damage for
    print("  Loading NF4 quantized model (damaged weights)...")
    bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
    nf4_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config = bnb_config,
        device_map = "auto", 
        trust_remote_code = True
    )

    # Extract write matrix weights from NF4 model (will be dequantized to FP32)
    nf4_weights = extract_write_weights(nf4_model, n_layers)

    # Delete the NF4 model to free up memory
    del nf4_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute per-row errors
    per_row_errors = {}  # {comp_name: tensor of shape (hidden_size,)}
    global_errors = {}   # {comp_name: scalar relative Frobenius error}

    print("  Computing per-row errors...")
    for comp in tqdm(sorted(fp16_weights.keys()), desc="  Processing components"):
        w_fp16 = fp16_weights[comp]
        w_nf4 = nf4_weights[comp]

        # Handle shape mismatch from BitsAndBytes block padding
        if w_fp16.shape != w_nf4.shape:
            # Truncate to the smaller shape (padding is zeros)
            min_rows = min(w_fp16.shape[0], w_nf4.shape[0])
            min_cols = min(w_fp16.shape[1], w_nf4.shape[1])
            w_fp16 = w_fp16[:min_rows, :min_cols]
            w_nf4 = w_nf4[:min_rows, :min_cols]

        # Compute global Frobenius error (standard matrix-level metric)
        # This is the "traditional" way to measure quantization damage
        # Used for comparison to show that global metrics hide local damage
        global_abs = torch.norm(w_fp16 - w_nf4, p="fro").item()
        global_fp16 = torch.norm(w_fp16, p="fro").item()
        global_errors[comp] = {
            "relative_error": global_abs / (global_fp16 + 1e-10),
            "absolute_error": global_abs,
        }

        # Compute per-row L2 error (identity-matrix normalized approach)
        # Why per-row?: Row i of write matrix -> dimension i of residual stream
        # Why normalize?: To get errors in meaningful range (0.001-0.100) for multiplication with DFA
        n_rows = w_fp16.shape[0]  # Should be hidden_size
        n_cols = w_fp16.shape[1]
        row_errors = torch.zeros(n_rows)

        # Identity-matrix normalization: normalize by global Frobenius norm per element
        # This gives per-element scale comparable to identity matrix transformation
        # Formula: scale = ||W||_F / sqrt(n_rows * n_cols)
        global_fp16_norm = torch.norm(w_fp16, p="fro").item()
        per_element_scale = global_fp16_norm / np.sqrt(n_rows * n_cols) if (n_rows * n_cols) > 0 else 1.0

        # Compute L2 error for each row
        for i in range(n_rows):
            # L2 norm of the difference between FP16 and NF4 rows
            row_diff = torch.norm(w_fp16[i, :] - w_nf4[i, :], p=2).item()
            
            # Normalize by per-element scale (identity-matrix approach)
            # This gives error relative to typical element magnitude, in 0.001-0.100 range
            # Formula: ε_i = ||row_diff||_2 / (scale * sqrt(n_cols))
            row_errors[i] = row_diff / (per_element_scale * np.sqrt(n_cols) + 1e-10)

        per_row_errors[comp] = row_errors

    # Step 4: Save per-row errors for use in Step 3 (vulnerability scores)
    save_path = os.path.join(output_dir, "per_row_errors.pt")
    torch.save({
        "per_row_errors": per_row_errors,
        "global_errors": global_errors,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
    }, save_path)

    # Step 5: Display analysis comparing global vs per-row errors
    # Purpose: Demonstrate that global Frobenius norm hides local damage
    # This validates why per-row analysis is necessary for identifying critical dimensions
    print(f"\n  === Global vs Per-Row Error Analysis ===")
    print(f"  {'Component':<20} | {'Global Rel':>10} | {'Max Row Rel':>11} | {'Mean Row':>10} | {'Ratio':>7}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*11}-+-{'-'*10}-+-{'-'*7}")

    for comp in sorted(per_row_errors.keys()):
        g_err = global_errors[comp]["relative_error"]
        row_errs = per_row_errors[comp]
        max_row = row_errs.max().item()
        mean_row = row_errs.mean().item()
        
        # Ratio shows how much the worst row exceeds the global average
        # High ratios (5-20x) indicate that global metrics hide local damage
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
