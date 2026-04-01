# Applies sparse delta-W repair to a quantized model.
#
# Instead of swapping out entire rows, we add a sparse correction matrix on top
# of the frozen NF4 model:
#
#   W_repaired = W_nf4 + delta_W
#
# where delta_W is non-zero only at the rows selected by whichever strategy we use:
#
#   delta_W[i, :] = W_fp16[i, :] - W_nf4[i, :]   if dimension i is targeted
#   delta_W[i, :] = 0                              otherwise
#
# This is mathematically equivalent to full row replacement, but framed as a
# sparse adapter that could sit on top of any standard quantized inference engine.
#
# Strategies tested:
#   targeted  - top-k dimensions by V-score (our method)
#   dfa_only  - top-k by |DFA_i| alone (ablation)
#   error_only - top-k by err_i alone (ablation)
#   random    - k random dimensions (control)
#   chhabra   - AIRD on the most-changed layer (baseline from their paper)
#
# Run:
#   python phase2_solution/repair/apply_repairs.py --model qwen1.5b --top_k 100

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
    """Apply sparse row corrections to the write matrices of the NF4 model.

    targets is a list of (component_name, row_index) pairs telling us which
    rows to replace with their FP16 counterparts.
    """
    # group by component so we only do one pass per layer module
    comp_dims = {}
    for comp, dim in targets:
        comp_dims.setdefault(comp, []).append(dim)

    n_rows_repaired = 0

    for comp, dims in comp_dims.items():
        parts = comp.split("_")
        layer_idx = int(parts[-1])
        layer = model.model.layers[layer_idx]

        if parts[0] == "attn":
            module = layer.self_attn.o_proj
        else:
            module = layer.mlp.down_proj

        # identity-matrix trick to get the current quantized weight in the right shape
        # calling .dequantize() on CPU gives back (numel, 1) instead of (out, in)
        in_features = module.in_features
        identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            W_nf4 = module(identity).float().T.contiguous()

        W_fp16 = fp16_weights[comp].to(W_nf4.device)

        # trim to matching shape in case of minor size mismatches
        min_rows = min(W_fp16.shape[0], W_nf4.shape[0])
        min_cols = min(W_fp16.shape[1], W_nf4.shape[1])
        W_fp16 = W_fp16[:min_rows, :min_cols]
        W_nf4 = W_nf4[:min_rows, :min_cols]

        W_new = W_nf4.clone()
        for dim in dims:
            W_new[dim, :] = W_fp16[dim, :]
            n_rows_repaired += 1

        module.weight = torch.nn.Parameter(W_new.to(DTYPE), requires_grad=False)

    return n_rows_repaired


def apply_aird_layer(model, layer_idx, r_fp16, r_nf4, alpha=1.0):
    """Chhabra's AIRD method applied at a single layer (used as baseline)."""
    layer = model.model.layers[layer_idx]

    r_fp16 = r_fp16.flatten()
    r_nf4 = r_nf4.flatten()

    if torch.isnan(r_fp16).any() or torch.isnan(r_nf4).any():
        raise ValueError(
            "Direction vectors contain NaN. "
            "Rerun phase1 step1 direction selection to get valid directions."
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
        in_features = module.in_features
        identity = torch.eye(in_features, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            W = module(identity).float().T.contiguous()

        # remove the NF4 direction component, replace with FP16 direction
        W_new = W - proj_nf4 @ W + alpha * proj_fp16 @ W
        module.weight = torch.nn.Parameter(W_new.to(DTYPE), requires_grad=False)


def get_targets_by_vscore(vuln_path, top_k):
    with open(vuln_path) as f:
        data = json.load(f)
    return [(item["component"], item["dimension"])
            for item in data["top_200_vulnerable"][:top_k]]


def get_targets_by_dfa(vuln_path, top_k):
    # ablation: ignore quantization error, use DFA importance only
    with open(vuln_path) as f:
        data = json.load(f)
    ranked = sorted(data["top_200_vulnerable"], key=lambda x: x["dfa_importance"], reverse=True)
    return [(item["component"], item["dimension"]) for item in ranked[:top_k]]


def get_targets_by_error(vuln_path, top_k):
    # ablation: ignore DFA, use quantization error only
    with open(vuln_path) as f:
        data = json.load(f)
    ranked = sorted(data["top_200_vulnerable"], key=lambda x: x["quant_error"], reverse=True)
    return [(item["component"], item["dimension"]) for item in ranked[:top_k]]


def get_targets_random(vuln_path, top_k):
    # control: pick random dimensions from the pool
    with open(vuln_path) as f:
        data = json.load(f)
    all_dims = [(item["component"], item["dimension"])
                for item in data["top_200_vulnerable"]]
    return random.sample(all_dims, min(top_k, len(all_dims)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--top_k", type=int, default=100,
                        help="how many dimensions to repair")
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    diag_dir = os.path.join(os.path.dirname(__file__), "..", "diagnosis", "outputs", args.model)
    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step1_direction", "outputs", args.model
    )
    circuit_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step3_circuit", "outputs", args.model
    )

    vuln_path = os.path.join(diag_dir, "vulnerability_scores.json")

    # load refusal directions for the AIRD baseline
    r_fp16 = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"), map_location="cpu"
    )["direction_normalized"].flatten()
    r_nf4 = torch.load(
        os.path.join(direction_dir, "best_direction_nf4.pt"), map_location="cpu"
    )["direction_normalized"].flatten()

    if r_fp16.shape != r_nf4.shape:
        raise ValueError(f"Direction shape mismatch: {r_fp16.shape} vs {r_nf4.shape}")
    if len(r_fp16.shape) != 1:
        raise ValueError(f"Expected 1D direction, got shape {r_fp16.shape}")
    if torch.isnan(r_fp16).any() or torch.isnan(r_nf4).any():
        raise ValueError(
            "Direction vectors contain NaN. "
            "Rerun phase1 step1 direction selection."
        )

    print(f"\nApplying Repairs: {args.model}, top_k={args.top_k}\n")

    # get model dimensions by loading a temp FP16 model
    temp_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="cpu", trust_remote_code=True
    )
    n_layers = temp_model.config.num_hidden_layers
    hidden_size = temp_model.config.hidden_size
    del temp_model

    fp16_weights = load_fp16_write_weights(model_name, n_layers)

    # find the most-damaged layer for the Chhabra baseline
    with open(os.path.join(circuit_dir, "circuit_comparison.json")) as f:
        circ_data = json.load(f)
    layer_damage = {}
    for item in circ_data["top_degraded"]:
        l = int(item["component"].split("_")[-1])
        layer_damage[l] = layer_damage.get(l, 0) + abs(item["absolute_change"])
    worst_layer = max(layer_damage, key=layer_damage.get)

    strategies = {
        "targeted": get_targets_by_vscore(vuln_path, args.top_k),
        "random": get_targets_random(vuln_path, args.top_k),
        "dfa_only": get_targets_by_dfa(vuln_path, args.top_k),
        "error_only": get_targets_by_error(vuln_path, args.top_k),
    }

    repair_info = {}

    # run all delta-W strategies
    for strategy_name, targets in strategies.items():
        print(f"\n  Strategy: {strategy_name} ({len(targets)} dimensions)")

        model, tokenizer = load_nf4_model(model_name)
        n_repaired = compute_delta_w(fp16_weights, model, targets)

        total_write_rows = hidden_size * n_layers * 2
        overhead_pct = n_repaired / total_write_rows * 100

        print(f"    Rows repaired: {n_repaired}")
        print(f"    FP16 overhead: {overhead_pct:.2f}% of write matrices")

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

    # Chhabra AIRD baseline
    print(f"\n  Strategy: chhabra (AIRD on layer {worst_layer})")
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

    with open(os.path.join(output_dir, "repair_strategies.json"), "w") as f:
        json.dump(repair_info, f, indent=2)

    print(f"\nRepair summary:")
    for name, info in repair_info.items():
        dims = info.get("n_dimensions", "-")
        rows = info.get("n_rows_repaired", "-")
        ovh = f"{info.get('overhead_pct', 0):.2f}%" if "overhead_pct" in info else "full layer"
        print(f"  {name:<15}  dims={dims}  rows={rows}  overhead={ovh}")

    print("\nAll repairs applied.")


if __name__ == "__main__":
    main()
