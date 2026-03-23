"""
SafeQuant Phase 1, Step 1 — Extract activations from residual stream.

For a given model and precision (FP16 or NF4), runs all train prompts
and caches the residual stream activations at every layer and
post-instruction token position.

Adapted from Arditi et al. (2024) generate_directions pipeline.

Run:
  python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision fp16
  python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision nf4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR


def load_model(model_name, precision):
    """Load model in specified precision."""
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


def find_post_instruction_positions(tokenizer, prompt):
    """
    Find the token positions AFTER the user's message ends.
    These are the positions where refusal information concentrates
    (per Arditi et al.).
    
    For Qwen2.5 chat template, the structure is:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    
    We want positions from <|im_end|> after user message onwards.
    """
    messages = [{"role": "user", "content": prompt}]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    
    # Find the user message tokens
    user_only_messages = [{"role": "user", "content": prompt}]
    user_text = tokenizer.apply_chat_template(
        user_only_messages, tokenize=False, add_generation_prompt=False
    )
    user_ids = tokenizer.encode(user_text, add_special_tokens=False)
    
    # Post-instruction positions: everything after user message ends
    # We take the last few tokens (typically the assistant header tokens)
    # A safe heuristic: last 3-5 tokens of the full prompt
    n_total = len(full_ids)
    n_user = len(user_ids)
    
    # Post-instruction = tokens from where user text ends to end of prompt
    # Typically this includes: <|im_end|>\n<|im_start|>assistant\n
    post_instr_start = max(n_user - 1, n_total - 6)
    post_instruction_positions = list(range(post_instr_start, n_total))
    
    return full_ids, post_instruction_positions


def extract_activations_for_prompts(model, tokenizer, prompts, n_layers):
    """
    Run prompts through model and collect residual stream activations
    at every layer and post-instruction position.
    
    Returns:
        activations: dict of {(layer, position_offset): list of activation tensors}
        position_counts: how many post-instruction positions per prompt
    """
    # We'll accumulate sum of activations for computing means
    # Key: (layer_idx, pos_offset_from_end) -> running sum tensor
    activation_sums = {}
    activation_counts = {}
    
    for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
        full_ids, post_positions = find_post_instruction_positions(tokenizer, prompt)
        
        input_ids = torch.tensor([full_ids]).to(model.device)
        
        # Storage for this prompt's hidden states
        hidden_states_per_layer = {}
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is typically a tuple; first element is hidden states
                if isinstance(output, tuple):
                    hs = output[0]
                else:
                    hs = output
                hidden_states_per_layer[layer_idx] = hs.detach().cpu()
            return hook_fn
        
        # Register hooks on every layer's output
        hooks = []
        for layer_idx in range(n_layers):
            layer = model.model.layers[layer_idx]
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        
        # Forward pass
        with torch.no_grad():
            model(input_ids)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Collect activations at post-instruction positions
        seq_len = input_ids.shape[1]
        for layer_idx in range(n_layers):
            hs = hidden_states_per_layer[layer_idx]  # (1, seq_len, hidden_dim)
            
            for pos_offset, abs_pos in enumerate(post_positions):
                if abs_pos >= seq_len:
                    continue
                    
                key = (layer_idx, pos_offset)
                activation = hs[0, abs_pos, :].float()  # (hidden_dim,)
                
                if key not in activation_sums:
                    activation_sums[key] = torch.zeros_like(activation)
                    activation_counts[key] = 0
                
                activation_sums[key] += activation
                activation_counts[key] += 1
        
        # Free memory
        del hidden_states_per_layer
        if prompt_idx % 20 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Compute means
    activation_means = {}
    for key in activation_sums:
        activation_means[key] = activation_sums[key] / activation_counts[key]
    
    return activation_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    print(f"\n=== Extracting activations: {args.model} ({args.precision}) ===\n")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, hidden_size={model.config.hidden_size}")
    
    # Load prompts
    with open(os.path.join(DATA_DIR, "harmful_train.json")) as f:
        harmful_prompts = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_train.json")) as f:
        harmless_prompts = json.load(f)
    
    print(f"Harmful prompts: {len(harmful_prompts)}")
    print(f"Harmless prompts: {len(harmless_prompts)}")
    
    # Extract activations
    print("\n--- Processing harmful prompts ---")
    harmful_means = extract_activations_for_prompts(
        model, tokenizer, harmful_prompts, n_layers
    )
    
    print("\n--- Processing harmless prompts ---")
    harmless_means = extract_activations_for_prompts(
        model, tokenizer, harmless_prompts, n_layers
    )
    
    # Save
    output_dir = os.path.join(
        os.path.dirname(__file__), "outputs", args.model
    )
    os.makedirs(output_dir, exist_ok=True)
    
    save_path = os.path.join(output_dir, f"activations_{args.precision}.pt")
    torch.save({
        "harmful_means": harmful_means,
        "harmless_means": harmless_means,
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "model_name": model_name,
        "precision": args.precision,
    }, save_path)
    
    print(f"\nSaved activations to {save_path}")
    print(f"  Keys (layer, pos_offset): {len(harmful_means)} candidates")
    print(f"  Hidden size: {model.config.hidden_size}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("\n=== Activation extraction complete ===")


if __name__ == "__main__":
    main()
