"""
SafeQuant Phase 2 v2, Step 1 — Per-Dimension DFA Decomposition.

Decomposes the block-level DFA from Phase 1 into per-dimension contributions.
For write matrices (o_proj, down_proj), row i controls dimension i of the
residual stream output. So:

    DFA_i(c) = r_hat[i] × mean(output_c[i] across harmful prompts)

This tells us: how much does dimension i of component c contribute to the
refusal direction?

IMPORTANT: We ONLY decompose write matrices (o_proj, down_proj) because
these directly map rows → residual stream dimensions. Read matrices
(q_proj, k_proj, v_proj, up_proj, gate_proj) have the inverse mapping
(columns → residual stream) and are excluded from per-dimension analysis.

Run:
  python phase2_solution/diagnosis/per_dim_dfa.py --model qwen1.5b --precision fp16
  python phase2_solution/diagnosis/per_dim_dfa.py --model qwen1.5b --precision nf4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR


def load_model(model_name, precision):
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=DTYPE,
            device_map="auto", trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def extract_per_dim_outputs(model, tokenizer, prompts, n_layers, hidden_size):
    """
    For each prompt, extract the per-dimension output of each write component
    (self_attn as a whole, and mlp as a whole) at the last token position.

    Returns mean per-dimension output vectors across all prompts.
    Shape per component: (hidden_size,)
    """
    # Accumulators
    attn_sums = {l: torch.zeros(hidden_size) for l in range(n_layers)}
    mlp_sums = {l: torch.zeros(hidden_size) for l in range(n_layers)}
    count = 0

    for prompt in tqdm(prompts, desc="  Extracting per-dim outputs"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        attn_outputs = {}
        mlp_outputs = {}

        def make_attn_hook(layer_idx):
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                # Last token position, all dimensions
                attn_outputs[layer_idx] = out[0, -1, :].detach().cpu().float()
            return hook_fn

        def make_mlp_hook(layer_idx):
            def hook_fn(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                mlp_outputs[layer_idx] = out[0, -1, :].detach().cpu().float()
            return hook_fn

        hooks = []
        for l in range(n_layers):
            layer = model.model.layers[l]
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for l in range(n_layers):
            if l in attn_outputs:
                attn_sums[l] += attn_outputs[l]
            if l in mlp_outputs:
                mlp_sums[l] += mlp_outputs[l]
        count += 1

        if count % 20 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    attn_means = {l: attn_sums[l] / count for l in range(n_layers)}
    mlp_means = {l: mlp_sums[l] / count for l in range(n_layers)}
    return attn_means, mlp_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step1_direction", "outputs", args.model
    )

    print(f"\n=== Per-Dimension DFA: {args.model} ({args.precision}) ===\n")

    # Load FP16 refusal direction (always use FP16 as reference)
    dir_data = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"), map_location="cpu"
    )
    r_hat = dir_data["direction_normalized"]  # (hidden_size,)
    print(f"  Refusal direction from layer {dir_data['layer']}")

    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    print(f"  {n_layers} layers, hidden_size={hidden_size}")

    # Load harmful prompts
    with open(os.path.join(DATA_DIR, "harmful_train.json"), encoding="utf-8") as f:
        harmful_prompts = json.load(f)
    print(f"  {len(harmful_prompts)} harmful prompts")

    # Extract per-dimension outputs
    attn_means, mlp_means = extract_per_dim_outputs(
        model, tokenizer, harmful_prompts, n_layers, hidden_size
    )

    # Compute per-dimension DFA
    # DFA_i(c) = r_hat[i] * mean_output_i(c)
    per_dim_dfa = {}

    for l in range(n_layers):
        # Attention: element-wise r_hat * mean_attn_output
        attn_dfa_vec = r_hat * attn_means[l]  # (hidden_size,)
        per_dim_dfa[f"attn_layer_{l}"] = attn_dfa_vec

        # MLP: same
        mlp_dfa_vec = r_hat * mlp_means[l]  # (hidden_size,)
        per_dim_dfa[f"mlp_layer_{l}"] = mlp_dfa_vec

    # Save
    save_path = os.path.join(output_dir, f"per_dim_dfa_{args.precision}.pt")
    torch.save({
        "per_dim_dfa": per_dim_dfa,
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "refusal_direction": r_hat,
    }, save_path)

    # Print summary
    print(f"\n  Per-dimension DFA computed for {len(per_dim_dfa)} components")
    print(f"  Each component has {hidden_size} dimension-level DFA scores")

    # Show which dimensions matter most globally
    all_dfa_abs = []
    for comp, vec in per_dim_dfa.items():
        for i in range(hidden_size):
            all_dfa_abs.append((comp, i, abs(vec[i].item())))
    all_dfa_abs.sort(key=lambda x: x[2], reverse=True)

    print(f"\n  Top 10 most important (component, dimension) pairs:")
    print(f"  {'Component':<20} | {'Dim':>5} | {'|DFA_i|':>10}")
    print(f"  {'-'*20}-+-{'-'*5}-+-{'-'*10}")
    for comp, dim, val in all_dfa_abs[:10]:
        print(f"  {comp:<20} | {dim:>5} | {val:>10.6f}")

    print(f"\n  Saved to {save_path}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("=== Per-dimension DFA complete ===")


if __name__ == "__main__":
    main()
