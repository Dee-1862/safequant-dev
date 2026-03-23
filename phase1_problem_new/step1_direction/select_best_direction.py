"""
SafeQuant Phase 1, Step 1 — Select the single best refusal direction.

For each candidate direction, tests on validation set:
  - Ablate it on harmful prompts -> does model stop refusing?
  - Add it to harmless prompts -> does model start refusing?
Score = ablation_bypass_rate + addition_refusal_rate

Following Arditi et al. (2024) select_direction methodology.

Run:
  python phase1_problem/step1_direction/select_best_direction.py --model qwen1.5b --precision fp16
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG, DATA_DIR, REFUSAL_PREFIXES, MAX_NEW_TOKENS


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


def is_refusal(text):
    """Check if generated text contains refusal indicators."""
    text_lower = text.strip().lower()
    for prefix in REFUSAL_PREFIXES:
        if text_lower.startswith(prefix.lower()):
            return True
    return False


def generate_with_hook(model, tokenizer, prompt, hook_fn, hook_layer):
    """Generate a completion with a forward hook applied at a specific layer."""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Register hook
    layer = model.model.layers[hook_layer]
    handle = layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,  # short for speed during selection
            do_sample=False,
        )
    
    handle.remove()
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def make_ablation_hook(direction, pos_offset=-1):
    """
    Creates a hook that REMOVES the refusal direction from activations.
    x' = x - (r_hat @ r_hat^T) @ x  (project out the direction)
    """
    r_hat = direction / torch.norm(direction)
    r_hat = r_hat.to(DTYPE)
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        
        r = r_hat.to(hs.device)
        # Ablate at all positions (following Arditi)
        proj = torch.einsum("...d,d->...", hs, r).unsqueeze(-1) * r
        hs_modified = hs - proj
        
        if isinstance(output, tuple):
            return (hs_modified,) + output[1:]
        return hs_modified
    
    return hook_fn


def make_addition_hook(direction, scale=1.0):
    """
    Creates a hook that ADDS the refusal direction to activations.
    x' = x + scale * r
    """
    r = direction.to(DTYPE)
    
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        
        r_dev = r.to(hs.device)
        # Add at all positions
        hs_modified = hs + scale * r_dev
        
        if isinstance(output, tuple):
            return (hs_modified,) + output[1:]
        return hs_modified
    
    return hook_fn


def evaluate_candidate(model, tokenizer, direction, layer_idx,
                       harmful_prompts, harmless_prompts):
    """
    Score a candidate direction by:
    1. Ablating it on harmful prompts -> count how many bypass refusal
    2. Adding it to harmless prompts -> count how many induce refusal
    """
    # Test ablation on harmful prompts (should bypass refusal)
    ablation_hook = make_ablation_hook(direction)
    bypass_count = 0
    for prompt in harmful_prompts:
        response = generate_with_hook(model, tokenizer, prompt, ablation_hook, layer_idx)
        if not is_refusal(response):
            bypass_count += 1
    bypass_rate = bypass_count / len(harmful_prompts)
    
    # Test addition on harmless prompts (should induce refusal)
    # Scale the direction to have similar magnitude as typical activations
    direction_norm = torch.norm(direction).item()
    addition_hook = make_addition_hook(direction, scale=3.0)
    induce_count = 0
    for prompt in harmless_prompts:
        response = generate_with_hook(model, tokenizer, prompt, addition_hook, layer_idx)
        if is_refusal(response):
            induce_count += 1
    induce_rate = induce_count / len(harmless_prompts)
    
    score = bypass_rate + induce_rate
    return score, bypass_rate, induce_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--precision", required=True, choices=["fp16", "nf4"])
    parser.add_argument("--top_k", type=int, default=10,
                        help="Evaluate top-k candidates by norm (for speed)")
    args = parser.parse_args()
    
    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    
    print(f"\n=== Selecting best direction: {args.model} ({args.precision}) ===\n")
    
    # Load candidates
    cand_path = os.path.join(output_dir, f"candidates_{args.precision}.pt")
    data = torch.load(cand_path, map_location="cpu")
    candidates = data["candidates"]
    
    # Load validation prompts
    with open(os.path.join(DATA_DIR, "harmful_val.json")) as f:
        harmful_val = json.load(f)
    with open(os.path.join(DATA_DIR, "harmless_val.json")) as f:
        harmless_val = json.load(f)
    
    # For speed, use subset of validation prompts
    harmful_subset = harmful_val[:8]
    harmless_subset = harmless_val[:8]
    
    # Pre-filter: take top-k candidates by direction norm
    # (stronger directions are more likely to be effective)
    norms = {k: torch.norm(v).item() for k, v in candidates.items()}
    sorted_keys = sorted(norms, key=norms.get, reverse=True)
    top_keys = sorted_keys[:args.top_k]
    
    print(f"Evaluating top {len(top_keys)} candidates (by norm) out of {len(candidates)}")
    
    # Load model
    model, tokenizer = load_model(model_name, args.precision)
    
    # Evaluate each candidate
    results = []
    for key in tqdm(top_keys, desc="Evaluating candidates"):
        layer_idx, pos_offset = key
        direction = candidates[key]
        
        score, bypass_rate, induce_rate = evaluate_candidate(
            model, tokenizer, direction, layer_idx,
            harmful_subset, harmless_subset
        )
        
        results.append({
            "key": key,
            "layer": layer_idx,
            "position_offset": pos_offset,
            "score": score,
            "bypass_rate": bypass_rate,
            "induce_rate": induce_rate,
            "norm": norms[key],
        })
        
        print(f"  Layer {layer_idx}, Pos {pos_offset}: "
              f"score={score:.3f} (bypass={bypass_rate:.2f}, induce={induce_rate:.2f})")
    
    # Select best
    results.sort(key=lambda x: x["score"], reverse=True)
    best = results[0]
    best_key = best["key"]
    best_direction = candidates[best_key]
    best_direction_normalized = best_direction / torch.norm(best_direction)
    
    print(f"\n=== BEST DIRECTION ===")
    print(f"  Layer: {best['layer']}")
    print(f"  Position offset: {best['position_offset']}")
    print(f"  Score: {best['score']:.3f}")
    print(f"  Bypass rate: {best['bypass_rate']:.2f}")
    print(f"  Induce rate: {best['induce_rate']:.2f}")
    print(f"  Norm: {best['norm']:.4f}")
    
    # Save
    save_path = os.path.join(output_dir, f"best_direction_{args.precision}.pt")
    torch.save({
        "direction": best_direction,
        "direction_normalized": best_direction_normalized,
        "layer": best["layer"],
        "position_offset": best["position_offset"],
        "score": best["score"],
        "bypass_rate": best["bypass_rate"],
        "induce_rate": best["induce_rate"],
        "all_results": results,
    }, save_path)
    
    print(f"\nSaved best direction to {save_path}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("=== Direction selection complete ===")


if __name__ == "__main__":
    main()
