"""
SafeQuant Phase 1, Step 3 — Extract per-component outputs for circuit analysis.

For each layer, separately captures:
  - Each attention head's contribution to the residual stream
  - The MLP's contribution to the residual stream

These are the building blocks for DFA (Direct Feature Attribution).

Run:
  python phase1_problem/step3_circuit/extract_component_outputs.py --model qwen1.5b --precision fp16
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


def extract_component_contributions(model, tokenizer, prompts, n_layers, n_heads, hidden_size):
    """
    For each prompt, extract:
    - Attention output per layer (combined heads, from the attention block)
    - MLP output per layer (from the MLP block)
    
    We hook into the attention output projection and MLP output to get
    each block's contribution to the residual stream.
    
    Returns mean contribution vectors across all prompts.
    """
    # Accumulators: {layer_idx: running sum of output vectors}
    attn_sums = {l: torch.zeros(hidden_size) for l in range(n_layers)}
    mlp_sums = {l: torch.zeros(hidden_size) for l in range(n_layers)}
    count = 0
    
    for prompt in tqdm(prompts, desc="Extracting components"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        attn_outputs = {}
        mlp_outputs = {}
        
        def make_attn_hook(layer_idx):
            def hook_fn(module, input, output):
                # output of self_attn is typically (attn_output, attn_weights, ...)
                if isinstance(output, tuple):
                    attn_out = output[0]
                else:
                    attn_out = output
                # Take the last token position (where generation happens)
                attn_outputs[layer_idx] = attn_out[0, -1, :].detach().cpu().float()
            return hook_fn
        
        def make_mlp_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    mlp_out = output[0]
                else:
                    mlp_out = output
                mlp_outputs[layer_idx] = mlp_out[0, -1, :].detach().cpu().float()
            return hook_fn
        
        # Register hooks
        hooks = []
        for l in range(n_layers):
            layer = model.model.layers[l]
            # Hook attention output (after o_proj)
            h1 = layer.self_attn.register_forward_hook(make_attn_hook(l))
            # Hook MLP output
            h2 = layer.mlp.register_forward_hook(make_mlp_hook(l))
            hooks.extend([h1, h2])
        
        # Forward pass
        with torch.no_grad():
            model(**inputs)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Accumulate
        for l in range(n_layers):
            if l in attn_outputs:
                attn_sums[l] += attn_outputs[l]
            if l in mlp_outputs:
                mlp_sums[l] += mlp_outputs[l]
        count += 1
        
        if count % 20 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute means
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
    
    print(f"\n=== Extracting component outputs: {args.model} ({args.precision}) ===\n")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    
    print(f"Architecture: {n_layers} layers, {n_heads} heads, {hidden_size} hidden")
    
    # Load harmful prompts (we compute DFA on harmful prompts)
    with open(os.path.join(DATA_DIR, "harmful_train.json")) as f:
        harmful_prompts = json.load(f)
    
    print(f"Processing {len(harmful_prompts)} harmful prompts...")
    
    # Extract
    attn_means, mlp_means = extract_component_contributions(
        model, tokenizer, harmful_prompts, n_layers, n_heads, hidden_size
    )
    
    # Save
    save_path = os.path.join(output_dir, f"component_outputs_{args.precision}.pt")
    torch.save({
        "attn_means": attn_means,
        "mlp_means": mlp_means,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "hidden_size": hidden_size,
    }, save_path)
    
    print(f"\nSaved to {save_path}")
    print(f"  Attention components: {n_layers}")
    print(f"  MLP components: {n_layers}")
    
    # Quick sanity check: print norms
    print(f"\n  Component output norms:")
    for l in range(0, n_layers, max(1, n_layers // 8)):
        attn_norm = torch.norm(attn_means[l]).item()
        mlp_norm = torch.norm(mlp_means[l]).item()
        print(f"    Layer {l:2d}: attn={attn_norm:.4f}, mlp={mlp_norm:.4f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n=== Component extraction complete ===")


if __name__ == "__main__":
    main()
