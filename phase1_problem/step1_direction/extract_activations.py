"""
Step 1a: Extract activations from residual stream.

For a given model and precision (FP16 or NF4), runs all train prompts and
computes mean activations at end-of-instruction token positions, then
computes difference-in-means (DIM) between harmful and harmless. Uses
Arditi et al. (2024) methodology; runs on CPU only.

Run:
  python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision fp16
  python phase1_problem/step1_direction/extract_activations.py --model qwen1.5b --precision nf4
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Add refusal_direction repo to path for imports
REFUSAL_DIR_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "repos", "refusal_direction"
))
if REFUSAL_DIR_PATH not in sys.path:
    sys.path.insert(0, REFUSAL_DIR_PATH)
    print(f"Added refusal_direction repo to path: {REFUSAL_DIR_PATH}")

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR

# Arditi refusal_direction repo: tokenization and position extraction (CPU-safe)
from pipeline.model_utils.qwen_model import (  # type: ignore
    tokenize_instructions_qwen_chat,
    QWEN_CHAT_TEMPLATE,
    QWEN_REFUSAL_TOKS
)


def load_model(model_name, precision):
    """Load model in specified precision."""
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config = bnb_config,
            device_map = "cpu",
            trust_remote_code = True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = DTYPE,
            device_map = "cpu",
            trust_remote_code = True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_eoi_toks(tokenizer):
    """
    Get end-of-instruction tokens using the chat template.
    Replicated from Arditi's QwenModel._get_eoi_toks (pipeline/model_utils/qwen_model.py).
    This extracts tokens that appear after {instruction} in the chat template.
    """
    return tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])


def get_mean_activations_memory_efficient(model, tokenizer, instructions, tokenize_fn, block_modules, positions):
    """
    Compute mean activations at specified positions across all layers.

    Memory-efficient: process prompts one at a time, move activations to CPU
    immediately, and accumulate in float64 on CPU for numerical precision.
    
    Args:
        model: The model to extract activations from
        tokenizer: Tokenizer for the model
        instructions: List of instruction strings
        tokenize_fn: Function to tokenize instructions (from Arditi's code)
        block_modules: List of transformer layer modules
        positions: List of token positions (negative indices from end)
    
    Returns:
        mean_activations: Tensor of shape (n_positions, n_layers, d_model) on CPU
    """
    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size
    
    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype = torch.float64, device = "cpu")

    for prompt_idx, prompt in enumerate(tqdm(instructions, desc = "Extracting activations")):
        # Tokenize using Arditi's tokenization function
        inputs = tokenize_fn([prompt])
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # Storage for this prompt's hidden states per layer
        hidden_states_per_layer = {}
        
        def make_hook(layer_idx):
            """Hook to capture hidden states at layer input (resid_pre)."""
            def hook_fn(module, input):
                # Extract hidden states from input tuple
                if isinstance(input, tuple):
                    hs = input[0]
                else:
                    hs = input
                # Store on CPU immediately
                hidden_states_per_layer[layer_idx] = hs.detach().cpu().float()
            return hook_fn
        
        # Register hooks on every layer's input (resid_pre)
        hooks = []
        for layer_idx in range(n_layers):
            layer = block_modules[layer_idx]
            h = layer.register_forward_pre_hook(make_hook(layer_idx))
            hooks.append(h)
        
        # Forward pass
        with torch.no_grad():
            model(input_ids = input_ids, attention_mask = attention_mask)
        
        # Remove hooks
        for h in hooks:
            h.remove()
        
        # Extract activations at specified positions and accumulate
        seq_len = input_ids.shape[1]
        for pos_idx, pos in enumerate(positions):
            # Convert negative position to absolute index
            if pos < 0:
                abs_pos = seq_len + pos
            else:
                abs_pos = pos

            if abs_pos < 0 or abs_pos >= seq_len:
                continue

            for layer_idx in range(n_layers):
                if layer_idx in hidden_states_per_layer:
                    hs = hidden_states_per_layer[layer_idx]
                    activation = hs[0, abs_pos, :].double()
                    mean_activations[pos_idx, layer_idx] += (1.0 / n_samples) * activation
        
        # Periodic cleanup when running on CPU (no-op if no GPU)
        if prompt_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return mean_activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required = True, choices = list(MODELS.keys()))
    parser.add_argument("--precision", required = True, choices = ["fp16", "nf4"])
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    print(f"\nExtracting activations: {args.model} ({args.precision})\n")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers, hidden_size = {model.config.hidden_size}")
    
    # Get end-of-instruction tokens (using Arditi's approach)
    eoi_toks = get_eoi_toks(tokenizer)
    print(f"End-of-instruction tokens: {len(eoi_toks)} tokens")
    print(f"  Token IDs: {eoi_toks}")
    print(f"  Decoded: {tokenizer.decode(eoi_toks)}")
    
    # Get block modules (transformer layers)
    block_modules = model.model.layers
    
    # Create tokenize function (using Arditi's tokenization)
    tokenize_fn = lambda instructions: tokenize_instructions_qwen_chat(
        tokenizer = tokenizer,
        instructions = instructions,
        system = None,
        include_trailing_whitespace = True
    )
    
    # Load prompts
    with open(os.path.join(DATA_DIR, "harmful_train.json")) as f:
        harmful_prompts = json.load(f)
    print(f"Harmful prompts: {len(harmful_prompts)}")

    with open(os.path.join(DATA_DIR, "harmless_train.json")) as f:
        harmless_prompts = json.load(f)    
    print(f"Harmless prompts: {len(harmless_prompts)}")
    
    # Extract positions: last len(eoi_toks) positions (using Arditi's approach)
    positions = list(range(-len(eoi_toks), 0))
    print(f"Extracting from positions: {positions}")
    
    print("\nComputing mean activations and difference...")

    # Process harmful prompts
    print("  Processing harmful prompts...")
    mean_activations_harmful = get_mean_activations_memory_efficient(
        model, 
        tokenizer, 
        harmful_prompts, 
        tokenize_fn,
        block_modules, 
        positions
    )

    # Process harmless prompts
    print("  Processing harmless prompts...")
    mean_activations_harmless = get_mean_activations_memory_efficient(
        model, 
        tokenizer, 
        harmless_prompts, 
        tokenize_fn,
        block_modules, 
        positions
    )
    
    # Compute difference-in-means (following Arditi's approach)
    mean_diffs = mean_activations_harmful - mean_activations_harmless
    
    assert mean_diffs.shape == (len(eoi_toks), n_layers, model.config.hidden_size)
    assert not mean_diffs.isnan().any()
    
    # Save
    output_dir = os.path.join(
        os.path.dirname(__file__), "outputs", args.model
    )
    os.makedirs(output_dir, exist_ok = True)
    
    save_path = os.path.join(output_dir, f"activations_{args.precision}.pt")
    torch.save({
        "mean_diffs": mean_diffs,
        "eoi_toks": eoi_toks,
        "positions": positions,
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "model_name": model_name,
        "precision": args.precision,
    }, save_path)
    
    print(f"\nSaved activations to {save_path}")
    print(f"  Shape: {mean_diffs.shape} (n_positions = {len(eoi_toks)}, n_layers = {n_layers}, d_model = {model.config.hidden_size})")
    
    del model
    print("\nActivation extraction complete")


if __name__ == "__main__":
    main()