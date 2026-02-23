"""
SafeQuant Phase 1, Step 1 — Select the single best refusal direction.

For each candidate direction, tests on validation set:
  - Ablate it on harmful prompts -> measure refusal score (should decrease)
  - Add it to harmless prompts -> measure refusal score (should increase)
  - Compute KL divergence on harmless prompts (should be low)

Uses Arditi et al. (2024) select_direction methodology directly.

Run:
  python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision fp16
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Add refusal_direction repo to path for imports
# Note: Linter may show warnings for these imports, but they work at runtime
REFUSAL_DIR_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "repos", "refusal_direction"
))
if REFUSAL_DIR_PATH not in sys.path:
    sys.path.insert(0, REFUSAL_DIR_PATH)
    print(f"Added refusal_direction repo to path: {REFUSAL_DIR_PATH}")

import argparse
import functools
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR, MAX_NEW_TOKENS

# Import from Arditi's code (from repos/refusal_direction/pipeline/)
# We only import CPU-safe functions; memory-intensive functions are replicated below
from pipeline.submodules.select_direction import ( # type: ignore
    refusal_score,
    kl_div_fn,
    filter_fn
)
from pipeline.utils.hook_utils import ( # type: ignore
    add_hooks,
    get_direction_ablation_input_pre_hook,
    get_direction_ablation_output_hook,
    get_activation_addition_input_pre_hook
)
from pipeline.model_utils.qwen_model import ( # type: ignore
    tokenize_instructions_qwen_chat,
    QWEN_REFUSAL_TOKS
)


def get_refusal_scores_memory_efficient(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks = [], fwd_hooks = [], batch_size = 1):
    """
    Memory-efficient version of Arditi's get_refusal_scores.
    
    Replicated from pipeline/submodules/select_direction.py but modified to:
    - Process prompts one at a time (batch_size=1 internally) to minimize GPU memory
    - Move refusal scores to CPU immediately after computation
    - Accumulate on CPU to avoid GPU memory accumulation
    
    Original Arditi code processes in batches and accumulates on GPU, which causes OOM on limited GPU memory.
    This version avoids that by processing individually and accumulating on CPU.
    """
    refusal_score_fn = functools.partial(refusal_score, refusal_toks = refusal_toks)
    
    # Accumulate on CPU to avoid GPU memory issues
    refusal_scores = torch.zeros(len(instructions), dtype = torch.float32, device = "cpu")
    
    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions = instructions[i:i+batch_size])
        
        with add_hooks(module_forward_pre_hooks = fwd_pre_hooks, module_forward_hooks = fwd_hooks):
            with torch.no_grad():
                logits = model(
                    input_ids = tokenized_instructions.input_ids.to(model.device),
                    attention_mask = tokenized_instructions.attention_mask.to(model.device),
                ).logits
        
        # Compute refusal scores and move to CPU immediately
        batch_scores = refusal_score_fn(logits = logits).cpu()
        refusal_scores[i:i+batch_size] = batch_scores
        
        # Clear GPU cache periodically
        if (i // batch_size) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return refusal_scores


def get_last_position_logits_memory_efficient(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks = [], fwd_hooks = [], batch_size = 1):
    """
    Memory-efficient version of Arditi's get_last_position_logits.
    
    Replicated from pipeline/submodules/select_direction.py but modified to:
    - Process prompts one at a time (batch_size=1 internally) to minimize GPU memory
    - Move logits to CPU immediately after computation
    - Accumulate on CPU to avoid GPU memory accumulation
    
    Original Arditi code processes in batches and accumulates on GPU, which causes OOM on limited GPU memory.
    This version avoids that by processing individually and accumulating on CPU.
    """
    last_position_logits_list = []
    
    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions = instructions[i:i+batch_size])
        
        with add_hooks(module_forward_pre_hooks = fwd_pre_hooks, module_forward_hooks = fwd_hooks):
            with torch.no_grad():
                logits = model(
                    input_ids = tokenized_instructions.input_ids.to(model.device),
                    attention_mask = tokenized_instructions.attention_mask.to(model.device),
                ).logits
        
        # Extract last position logits and move to CPU immediately
        batch_logits = logits[:, -1, :].detach().cpu()
        last_position_logits_list.append(batch_logits)
        
        # Clear GPU cache periodically
        if (i // batch_size) % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate on CPU
    last_position_logits = torch.cat(last_position_logits_list, dim = 0)
    return last_position_logits


def load_model(model_name, precision):
    """Load model in specified precision."""
    if precision == "nf4":
        bnb_config = BitsAndBytesConfig(**NF4_CONFIG)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config = bnb_config,
            device_map = "auto", trust_remote_code = True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = DTYPE,
            device_map = "auto", trust_remote_code = True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded: {tokenizer.pad_token}")
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    parser.add_argument("--top_k", type = int, default = 20,help = "Evaluate top-k candidates (for speed)")
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    
    print(f"\nSelecting best direction: {args.model} ({args.precision})\n")
    
    # Load candidates
    cand_path = os.path.join(output_dir, f"activations_{args.precision}.pt")
    data = torch.load(cand_path, map_location = "cpu")
    mean_diffs = data["mean_diffs"]  # Shape: (n_pos, n_layers, d_model)
    eoi_toks = data["eoi_toks"]
        
    n_pos, n_layers, d_model = mean_diffs.shape
    print(f"Loaded {n_pos * n_layers} candidate directions")
    print(f"  Positions: {n_pos}, Layers: {n_layers}, Dim: {d_model}")
    
    # Load validation prompts
    with open(os.path.join(DATA_DIR, "harmful_val.json")) as f:
        harmful_val = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_val.json")) as f:
        harmless_val = json.load(f)
    
    print(f"Harmful validation: {len(harmful_val)}")
    print(f"Harmless validation: {len(harmless_val)}")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    refusal_toks = torch.tensor(QWEN_REFUSAL_TOKS, device = model.device)
    print(f"Refusal tokens: {refusal_toks.tolist()}")
    
    # Get model modules
    block_modules = model.model.layers
    # Note: Qwen2 models use 'self_attn', not 'attn' like older Qwen models
    # Arditi's code (pipeline/model_utils/qwen_model.py) uses 'attn' for Qwen1
    # We need 'self_attn' for Qwen2.5 models

    attn_lst = []
    mlp_lst = []
    for block in block_modules:
        attn_lst.append(block.self_attn)
        mlp_lst.append(block.mlp)

    attn_modules = torch.nn.ModuleList(attn_lst)
    mlp_modules = torch.nn.ModuleList(mlp_lst)
    
    tokenize_fn = lambda instructions: tokenize_instructions_qwen_chat(
        tokenizer = tokenizer,
        instructions = instructions,
        system = None,
        include_trailing_whitespace = True
    )
    
    # Compute baseline refusal scores using memory-efficient version
    print("\nComputing baseline refusal scores...")
    baseline_refusal_scores_harmful = get_refusal_scores_memory_efficient(
        model, harmful_val, tokenize_fn, refusal_toks, fwd_pre_hooks = [], fwd_hooks = [], batch_size = 1
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    baseline_refusal_scores_harmless = get_refusal_scores_memory_efficient(
        model, harmless_val, tokenize_fn, refusal_toks, fwd_pre_hooks = [], fwd_hooks = [], batch_size = 1
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"  Baseline harmful refusal score: {baseline_refusal_scores_harmful.mean().item():.4f}")
    print(f"  Baseline harmless refusal score: {baseline_refusal_scores_harmless.mean().item():.4f}")
    
    # Get baseline logits for KL computation using memory-efficient version
    baseline_harmless_logits = get_last_position_logits_memory_efficient(
        model, tokenizer, harmless_val, tokenize_fn,
        fwd_pre_hooks = [], fwd_hooks = [], batch_size = 1
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize score arrays
    ablation_kl_div_scores = torch.zeros((n_pos, n_layers), device = model.device, dtype = torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layers), device = model.device, dtype = torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layers), device = model.device, dtype = torch.float64)
    
    # For speed, only evaluate top_k candidates by norm
    all_candidates = []
    for pos_idx in range(n_pos):
        for layer_idx in range(n_layers):
            direction = mean_diffs[pos_idx, layer_idx]
            norm = torch.norm(direction).item()
            all_candidates.append((norm, pos_idx, layer_idx))
    
    all_candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = all_candidates[:args.top_k]
    print(f"\nEvaluating top {len(top_candidates)} candidates by norm...")
    
    # Evaluate each candidate using memory-efficient versions
    from tqdm import tqdm
    for norm, pos_idx, layer_idx in tqdm(top_candidates, desc="Evaluating candidates"):
        source_pos = pos_idx - n_pos  # Convert to negative index
        # Keep direction on CPU; hooks will move it to GPU when needed
        direction = mean_diffs[pos_idx, layer_idx]
        
        # 1. Compute KL divergence (ablating on harmless)
        fwd_pre_hooks = [
            (block_modules[layer], get_direction_ablation_input_pre_hook(direction = direction))
            for layer in range(n_layers)
        ]
        fwd_hooks = [
            (attn_modules[layer], get_direction_ablation_output_hook(direction = direction))
            for layer in range(n_layers)
        ]
        fwd_hooks += [
            (mlp_modules[layer], get_direction_ablation_output_hook(direction = direction))
            for layer in range(n_layers)
        ]
        
        intervention_logits = get_last_position_logits_memory_efficient(
            model, tokenizer, harmless_val, tokenize_fn,
            fwd_pre_hooks = fwd_pre_hooks, fwd_hooks = fwd_hooks, batch_size = 1
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Both baseline and intervention logits are on CPU, so kl_div_fn will work correctly
        kl_div = kl_div_fn(baseline_harmless_logits, intervention_logits).mean().item()
        ablation_kl_div_scores[pos_idx, layer_idx] = kl_div
        
        # Clear intervention_logits from memory
        del intervention_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 2. Compute refusal score when ablating on harmful
        refusal_scores = get_refusal_scores_memory_efficient(
            model, harmful_val, tokenize_fn, refusal_toks,
            fwd_pre_hooks = fwd_pre_hooks, fwd_hooks = fwd_hooks, batch_size = 1
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ablation_refusal_scores[pos_idx, layer_idx] = refusal_scores.mean().item()
        
        # Clear refusal_scores from memory
        del refusal_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 3. Compute refusal score when adding to harmless
        coeff = torch.tensor(1.0)
        fwd_pre_hooks_add = [(block_modules[layer_idx], get_activation_addition_input_pre_hook(vector = direction, coeff = coeff))]
        
        refusal_scores = get_refusal_scores_memory_efficient(
            model, harmless_val, tokenize_fn, refusal_toks,
            fwd_pre_hooks = fwd_pre_hooks_add, fwd_hooks = [], batch_size = 1
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        steering_refusal_scores[pos_idx, layer_idx] = refusal_scores.mean().item()
        
        # Clear all intermediate tensors and hooks
        del refusal_scores, fwd_pre_hooks, fwd_hooks, fwd_pre_hooks_add
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Filter and select best using Arditi's filter function
    filtered_scores = []
    kl_threshold = 0.1
    induce_refusal_threshold = 0.0
    prune_layer_percentage = 0.2
    
    for pos_idx in range(n_pos):
        for layer_idx in range(n_layers):
            refusal_score_val = ablation_refusal_scores[pos_idx, layer_idx].item()
            steering_score = steering_refusal_scores[pos_idx, layer_idx].item()
            kl_div_score = ablation_kl_div_scores[pos_idx, layer_idx].item()
            
            # Use Arditi's filter function
            should_filter = filter_fn(
                refusal_score = refusal_score_val,
                steering_score = steering_score,
                kl_div_score = kl_div_score,
                layer = layer_idx,
                n_layer = n_layers,
                kl_threshold = kl_threshold,
                induce_refusal_threshold = induce_refusal_threshold,
                prune_layer_percentage = prune_layer_percentage
            )
            
            if should_filter:
                continue
            
            # Sorting score: lower refusal score is better (multiply by -1)
            sorting_score = -refusal_score_val
            filtered_scores.append((sorting_score, pos_idx, layer_idx))
    
    if not filtered_scores:
        print("WARNING: All candidates filtered out! Using best by norm.")
        best_norm, best_pos, best_layer = all_candidates[0]
        best_pos_idx = best_pos
        best_layer_idx = best_layer
    else:
        filtered_scores.sort(key=lambda x: x[0], reverse=True)
        _, best_pos_idx, best_layer_idx = filtered_scores[0]
    
    best_direction = mean_diffs[best_pos_idx, best_layer_idx]
    best_direction_normalized = best_direction / torch.norm(best_direction)
    
    print(f"\n=== BEST DIRECTION ===")
    print(f"  Position: {best_pos_idx - n_pos} (index {best_pos_idx})")
    print(f"  Layer: {best_layer_idx}")
    print(f"  Ablation refusal score: {ablation_refusal_scores[best_pos_idx, best_layer_idx].item():.4f} (baseline: {baseline_refusal_scores_harmful.mean().item():.4f})")
    print(f"  Steering refusal score: {steering_refusal_scores[best_pos_idx, best_layer_idx].item():.4f} (baseline: {baseline_refusal_scores_harmless.mean().item():.4f})")
    print(f"  KL divergence: {ablation_kl_div_scores[best_pos_idx, best_layer_idx].item():.4f}")
    
    # Save
    save_path = os.path.join(output_dir, f"best_direction_{args.precision}.pt")
    torch.save({
        "direction": best_direction,
        "direction_normalized": best_direction_normalized,
        "position": best_pos_idx - n_pos,
        "position_idx": best_pos_idx,
        "layer": best_layer_idx,
        "ablation_refusal_score": ablation_refusal_scores[best_pos_idx, best_layer_idx].item(),
        "steering_refusal_score": steering_refusal_scores[best_pos_idx, best_layer_idx].item(),
        "kl_div_score": ablation_kl_div_scores[best_pos_idx, best_layer_idx].item(),
        "baseline_harmful_refusal": baseline_refusal_scores_harmful.mean().item(),
        "baseline_harmless_refusal": baseline_refusal_scores_harmless.mean().item(),
    }, save_path)
    
    print(f"\nSaved best direction to {save_path}")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDirection selection complete")


if __name__ == "__main__":
    main()
