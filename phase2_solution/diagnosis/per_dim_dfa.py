"""
Step 1: Per-Dimension DFA (Direct Feature Attribution) Decomposition.

PURPOSE:
--------
This script decomposes the block-level DFA from Phase 1 into per-dimension contributions.
It identifies which individual dimensions (out of thousands) in each component's output contribute most to the refusal behavior.

MATHEMATICAL FORMULA:
---------------------
For write matrices (o_proj, down_proj), row i controls dimension i of the residual stream output:

    DFA_i(c) = r_hat[i] x mean(output_c[i] across harmful prompts)

Where:
  - DFA_i(c): Contribution of dimension i in component c to the refusal direction
  - r_hat[i]: The i-th component of the normalized refusal direction vector
  - output_c[i]: The i-th dimension of component c's output (captured via hooks)

This tells us: "How much does dimension i of component c contribute to the refusal direction?"

HOW IT WORKS:
-------------
1. Uses PyTorch forward hooks to intercept and capture activations from each layer's attention (o_proj) and MLP (down_proj) outputs during the forward pass. 
2. For each harmful prompt, extracts the last token's activation vector from each component.
3. Computes the mean activation across all harmful prompts for each dimension.
4. Multiplies by the refusal direction vector (r_hat) to get per-dimension DFA scores.
5. Saves the results for use in Step 2 (quantization error) and Step 3 (vulnerability scores).


IMPORTANT CONSTRAINTS:
----------------------
- We ONLY decompose write matrices (o_proj, down_proj) because these directly map rows -> residual stream dimensions.
- Read matrices (q_proj, k_proj, v_proj, up_proj, gate_proj) have the inverse mapping (columns -> residual stream) and are excluded from per-dimension analysis.

OUTPUT:
-------
Saves per-dimension DFA scores to:
  - phase2_solution/diagnosis/outputs/{model}/per_dim_dfa_fp16.pt
  - phase2_solution/diagnosis/outputs/{model}/per_dim_dfa_nf4.pt

USAGE:
------
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
    """Load model in specified precision."""

    # NF4
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config = bnb_config,
            device_map = "auto", 
            trust_remote_code = True,
        )

    # elif precision == "nf2":
    #     bnb_config = BitsAndBytesConfig(**NF2_CONFIG)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_name, 
    #         quantization_config = bnb_config,
    #         device_map = "auto", 
    #         trust_remote_code = True,
    #     )

    else:
        # FP16
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype = DTYPE,
            device_map = "auto", 
            trust_remote_code = True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)

    # Set pad token for tokenizer
    # Why?: Because the old tokenizer (like GPT2 style) needs a pad token to pad the input tokens to the same length. If the pad token is not set then it will raise an error.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()

    # Return model and tokenizer
    return model, tokenizer


def extract_per_dim_outputs(model, tokenizer, prompts, n_layers, hidden_size):
    """
    PURPOSE:
    For each prompt, extract the per-dimension output of each write component at the last token position.

    INPUT:
    - model: the model to extract the per-dim outputs from
    - tokenizer: the tokenizer to use to tokenize the prompts
    - prompts: the prompts to extract the per-dim outputs from
    - n_layers: the number of layers in the model
    - hidden_size: the hidden size of the model

    OUTPUT:
    - attn_sums: (n_layers, hidden_size) -> mean(output_c[i] across harmful prompts)
    - mlp_sums: (n_layers, hidden_size) -> mean(output_c[i] across harmful prompts)
    """
    # Accumulators: A dictionary of layer indices for running sums of output vectors.
    attn_sums = {}
    mlp_sums = {}
    for l in range(n_layers):
        attn_sums[l] = torch.zeros(hidden_size)
        mlp_sums[l] = torch.zeros(hidden_size)

    # Count the number of prompts processed
    count = 0

    # Processing each prompt
    for prompt in tqdm(prompts, desc="Extracting per-dim outputs"):
        messages = [{"role": "user", "content": prompt}]

        # Format the prompt for the model
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize = False, 
            add_generation_prompt = True
        )

        # Tokenize the prompt and move to the model device
        inputs = tokenizer(text, return_tensors = "pt").to(model.device)

        # Accumulators for the attention and MLP outputs
        attn_outputs = {}
        mlp_outputs = {}

        def make_output_hook(layer_idx, output_dict):
            """Create a hook that captures the last token's output for a given layer."""

            # NOTE: This is the hook function structure required for the register_forward_hook method in PyTorch so dont change it.
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    out = output[0]
                else:
                    out = output

                # Last token position, all dimensions
                output_dict[layer_idx] = out[0, -1, :].detach().cpu().float()

            return hook_fn

        hooks = []
        for l in range(n_layers):
            layer = model.model.layers[l]
            hooks.append(layer.self_attn.register_forward_hook(make_output_hook(l, attn_outputs)))
            hooks.append(layer.mlp.register_forward_hook(make_output_hook(l, mlp_outputs)))

        with torch.no_grad():
            model(**inputs)

        # Remove hooks after forward pass
        for h in hooks:
            h.remove()

        # Accumulate the outputs
        for l in range(n_layers):
            if l in attn_outputs:
                attn_sums[l] += attn_outputs[l]
            if l in mlp_outputs:
                mlp_sums[l] += mlp_outputs[l]
        count += 1

        # Empty cache every 20 prompts due to my limited GPU memory
        # So you can increase the number of prompts if you have more memory.
        if count % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                None


    # Computing the means across all prompts
    attn_means = {}
    mlp_means = {}
    for l in range(n_layers):
        attn_means[l] = attn_sums[l] / count
        mlp_means[l] = mlp_sums[l] / count

    return attn_means, mlp_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, choices = list(MODELS.keys()))
    parser.add_argument("--precision", required = True, choices = ["fp16", "nf4"])
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
