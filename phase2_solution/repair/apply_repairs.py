"""
SafeQuant Phase 2 v2, Step 4 — Sparse Delta-W Repair.

Instead of "swapping rows back," we add a sparse FP16 error-compensation
matrix ΔW on top of the frozen quantized model:

    W'_c = W^{NF4}_c + ΔW_c

Where ΔW_c is non-zero only at high-vulnerability rows:

    (ΔW_c)_{i,:} = (W^{FP16}_c)_{i,:} - (W^{NF4}_c)_{i,:}   if V_i(c) > τ
                  = 0                                            otherwise

This is mathematically identical to row restoration but deployable as a
sparse LoRA-style adapter on any standard quantized inference engine.

Strategies:
  1. TARGETED (Ours): top-k dimensions globally by V_i(c)
  2. CHHABRA: AIRD on the most-changed layer (their method)
  3. RANDOM: k random dimensions (control)
  4. DFA-ONLY: top-k dimensions by |DFA_i| alone (ablation)
  5. ERROR-ONLY: top-k dimensions by ε_i alone (ablation)

Run:
  python phase2_solution/repair/apply_repairs.py --model qwen1.5b --top_k 100
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import random
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DTYPE, NF4_CONFIG


random.seed(42)
np.random.seed(42)


def load_fp16_write_weights(model_name, n_layers):
    """Load FP16 write matrix weights."""
    print("  Loading FP16 weights for repair...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    weights = {}
    for l in range(n_layers):
        layer = model.model.layers[l]
        weights[f"attn_layer_{l}"] = layer.self_attn.o_proj.weight.data.float().cpu().clone()
        weights[f"mlp_layer_{l}"] = layer.mlp.down_proj.weight.data.float().cpu().clone()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return weights


def load_nf4_model(model_name):
    bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config,
        device_map="cpu", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def compute_delta_w(fp16_weights, model, targets):
    """
    Compute sparse ΔW matrices and apply them.
    targets: list of (component_name, dimension_index) tuples
    """
    # Group targets by component
    comp_dims = {}
    for comp, dim in targets:
        if comp not in comp_dims:
            comp_dims[comp] = []
        comp_dims[comp].append(dim)

    n_rows_repaired = 0

    for comp, dims in comp_dims.items():
        parts = comp.split("_")
        comp_type = parts[0]
        layer_idx = int(parts[-1])

        layer = model.model.layers[layer_idx]
        if comp_type == "attn":
            module = layer.self_attn.o_proj
        else:
            module = layer.mlp.down_proj

        # Get current quantized weight via identity-matrix trick.
        # .dequantize() on CPU returns wrong shape (numel, 1); the forward
        # pass through an identity input always gives correct (out, in) shape.
        in_features = module.in_features
        identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            W_nf4 = module(identity).float().T.contiguous()  # (out, in)

        W_fp16 = fp16_weights[comp].to(W_nf4.device)

        # Trim to smallest common shape (no-op when shapes already match)
        min_rows = min(W_fp16.shape[0], W_nf4.shape[0])
        min_cols = min(W_fp16.shape[1], W_nf4.shape[1])
        W_fp16 = W_fp16[:min_rows, :min_cols]
        W_nf4 = W_nf4[:min_rows, :min_cols]

        # Compute ΔW (sparse — only at specified rows)
        W_new = W_nf4.clone()
        for dim in dims:
            W_new[dim, :] = W_fp16[dim, :]
            n_rows_repaired += 1

        # Replace weight
        module.weight = torch.nn.Parameter(W_new.to(DTYPE), requires_grad=False)

    return n_rows_repaired


def apply_aird_layer(model, layer_idx, r_fp16, r_nf4, alpha=1.0):
    """Chhabra's AIRD at layer level for baseline comparison."""
    layer = model.model.layers[layer_idx]

    r_fp16 = r_fp16.flatten()
    r_nf4 = r_nf4.flatten()

    if torch.isnan(r_fp16).any() or torch.isnan(r_nf4).any():
        raise ValueError(
            "Direction vectors contain NaN values. "
            "Rerun Phase 1 step 1 direction selection to regenerate valid directions."
        )

    r_fp16_norm = torch.norm(r_fp16)
    r_nf4_norm = torch.norm(r_nf4)
    if r_fp16_norm == 0 or r_nf4_norm == 0:
        raise ValueError("Direction vectors have zero norm")

    r_fp16_hat = r_fp16 / r_fp16_norm
    r_nf4_hat = r_nf4 / r_nf4_norm

    proj_nf4 = torch.outer(r_nf4_hat, r_nf4_hat)
    proj_fp16 = torch.outer(r_fp16_hat, r_fp16_hat)

    for module in [layer.self_attn.o_proj, layer.mlp.down_proj]:
        # Identity-matrix trick: always returns correct (out, in) shape on CPU.
        # Avoids .dequantize() which returns (numel, 1) for BitsAndBytes on CPU.
        in_features = module.in_features
        identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            W = module(identity).float().T.contiguous()  # (out_features, in_features)

        # AIRD projection: remove NF4 direction, add FP16 direction
        W_new = W - proj_nf4 @ W + alpha * proj_fp16 @ W
        module.weight = torch.nn.Parameter(W_new.to(DTYPE), requires_grad=False)


def get_targets_by_vscore(vuln_path, top_k):
    """Our method: top-k dimensions globally by vulnerability score."""
    with open(vuln_path) as f:
        data = json.load(f)
    targets = [(item["component"], item["dimension"])
               for item in data["top_200_vulnerable"][:top_k]]
    return targets


def get_targets_by_dfa(vuln_path, top_k):
    """Ablation: top-k by DFA importance alone."""
    with open(vuln_path) as f:
        data = json.load(f)
    ranked = sorted(data["top_200_vulnerable"],
                    key=lambda x: x["dfa_importance"], reverse=True)
    return [(item["component"], item["dimension"]) for item in ranked[:top_k]]


def get_targets_by_error(vuln_path, top_k):
    """Ablation: top-k by quantization error alone."""
    with open(vuln_path) as f:
        data = json.load(f)
    ranked = sorted(data["top_200_vulnerable"],
                    key=lambda x: x["quant_error"], reverse=True)
    return [(item["component"], item["dimension"]) for item in ranked[:top_k]]


def get_targets_random(vuln_path, top_k):
    """Control: k random dimensions from the vulnerability list."""
    with open(vuln_path) as f:
        data = json.load(f)
    all_dims = [(item["component"], item["dimension"])
                for item in data["top_200_vulnerable"]]
    return random.sample(all_dims, min(top_k, len(all_dims)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--top_k", type=int, default=100,
                        help="Number of dimensions to repair")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    diag_dir = os.path.join(
        os.path.dirname(__file__), "..", "diagnosis", "outputs", args.model
    )
    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step1_direction", "outputs", args.model
    )
    circuit_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step3_circuit", "outputs", args.model
    )

    vuln_path = os.path.join(diag_dir, "vulnerability_scores.json")

    # Load refusal directions for AIRD baseline
    r_fp16 = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"), map_location="cpu"
    )["direction_normalized"]
    r_nf4 = torch.load(
        os.path.join(direction_dir, "best_direction_nf4.pt"), map_location="cpu"
    )["direction_normalized"]
    
    # Ensure directions are 1D vectors
    r_fp16 = r_fp16.flatten()
    r_nf4 = r_nf4.flatten()
    
    # Validate shapes
    if r_fp16.shape != r_nf4.shape:
        raise ValueError(
            f"Direction shape mismatch: r_fp16 {r_fp16.shape} vs r_nf4 {r_nf4.shape}"
        )
    
    if len(r_fp16.shape) != 1:
        raise ValueError(
            f"Direction should be 1D but has shape {r_fp16.shape}"
        )
    
    # Check for NaN
    if torch.isnan(r_fp16).any() or torch.isnan(r_nf4).any():
        raise ValueError(
            "Direction vectors contain NaN values. "
            "You must rerun Phase 1 step 1 direction selection to regenerate valid directions."
        )

    print(f"\n=== Applying Repairs: {args.model}, top_k={args.top_k} ===\n")

    # Pre-load FP16 weights for delta-W strategies
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    n_layers = temp_model.config.num_hidden_layers
    hidden_size = temp_model.config.hidden_size
    del temp_model
    fp16_weights = load_fp16_write_weights(model_name, n_layers)

    # Find worst layer for Chhabra baseline
    with open(os.path.join(circuit_dir, "circuit_comparison.json")) as f:
        circ_data = json.load(f)
    layer_damage = {}
    for item in circ_data["top_degraded"]:
        l = int(item["component"].split("_")[-1])
        layer_damage[l] = layer_damage.get(l, 0) + abs(item["absolute_change"])
    worst_layer = max(layer_damage, key=layer_damage.get)

    # Define strategies
    strategies = {
        "targeted": get_targets_by_vscore(vuln_path, args.top_k),
        "random": get_targets_random(vuln_path, args.top_k),
        "dfa_only": get_targets_by_dfa(vuln_path, args.top_k),
        "error_only": get_targets_by_error(vuln_path, args.top_k),
    }

    repair_info = {}

    # Delta-W strategies
    for strategy_name, targets in strategies.items():
        print(f"\n  --- Strategy: {strategy_name} ({len(targets)} dimensions) ---")

        model, tokenizer = load_nf4_model(model_name)
        n_repaired = compute_delta_w(fp16_weights, model, targets)

        # Compute overhead
        total_params = hidden_size * n_layers * 2  # rough count of write matrix rows
        overhead_pct = n_repaired / total_params * 100

        print(f"      Rows repaired: {n_repaired}")
        print(f"      FP16 overhead: {overhead_pct:.2f}% of write matrices")

        # Save repaired model
        save_dir = os.path.join(output_dir, f"repaired_{strategy_name}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
        torch.save({
            "strategy": strategy_name,
            "targets": targets,
            "n_repaired": n_repaired,
            "overhead_pct": overhead_pct,
        }, os.path.join(save_dir, "repair_info.pt"))

        repair_info[strategy_name] = {
            "n_dimensions": len(targets),
            "n_rows_repaired": n_repaired,
            "overhead_pct": overhead_pct,
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Chhabra AIRD baseline (layer-level, not dimension-level)
    print(f"\n  --- Strategy: chhabra (AIRD on layer {worst_layer}) ---")
    model, tokenizer = load_nf4_model(model_name)
    apply_aird_layer(model, worst_layer, r_fp16, r_nf4, alpha=args.alpha)

    save_dir = os.path.join(output_dir, "repaired_chhabra")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_state.pt"))
    torch.save({
        "strategy": "chhabra",
        "layer": worst_layer,
    }, os.path.join(save_dir, "repair_info.pt"))
    repair_info["chhabra"] = {"layer": worst_layer, "method": "AIRD"}

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save summary
    with open(os.path.join(output_dir, "repair_strategies.json"), "w") as f:
        json.dump(repair_info, f, indent=2)

    # Compute and report delta-W sparsity
    print(f"\n  === Repair Summary ===")
    print(f"  {'Strategy':<15} | {'Dims':>6} | {'Rows':>6} | {'Overhead':>8}")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}")
    for name, info in repair_info.items():
        dims = info.get("n_dimensions", "-")
        rows = info.get("n_rows_repaired", "-")
        ovh = f"{info.get('overhead_pct', 0):.2f}%" if "overhead_pct" in info else "full layer"
        print(f"  {name:<15} | {str(dims):>6} | {str(rows):>6} | {ovh:>8}")

    print(f"\n  Framing for paper: 'We freeze the quantized model and add a")
    print(f"  highly sparse FP16 error-compensation matrix ΔW, where only")
    print(f"  {args.top_k} high-vulnerability rows are non-zero.'")
    print("=== All repairs applied ===")


if __name__ == "__main__":
    main()
