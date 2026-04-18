"""
Safety-Weighted Interpolation Restoration.

Instead of restoring a few dimensions fully (binary swap), this applies
a smooth correction to ALL dimensions proportional to their safety importance.

    W_corrected = W_quantized + α_j × (W_fp32 - W_quantized)

Three methods:
  1. Uniform α:         Same α for all dimensions
  2. Safety-weighted α:  α_j ∝ |refusal_dir[j]| × quant_error[j] × activation[j]
  3. Error-only α:       α_j ∝ quant_error[j]  (ablation: no safety targeting)

Key insight: Instead of mixing 2-bit and 16-bit columns (Frankenstein),
every column gets a small nudge toward FP32. This preserves weight matrix
coherence while prioritizing safety-critical corrections.

Usage:
    python diagnostics/interpolation_restore.py --model llama7b --quantizer aqlm2
"""

import argparse
import json
import sys
import torch
import torch.nn as nn
import gc
import random
import numpy as np
from pathlib import Path

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import load_fp32, load_quantized, get_weight
from core.datasets import load_wildjailbreak_prompts, load_arc_challenge
from core.evaluation import generate_only, classify_batch, _extract_behavior, _logprob_score

READ_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.gate_proj", "mlp.up_proj"]
ALL_PROJS = READ_PROJS + ["self_attn.o_proj", "mlp.down_proj"]


def _get_submodule(layer, proj_path):
    parts = proj_path.split(".")
    m = layer
    for p in parts:
        m = getattr(m, p)
    return m


def _set_submodule(layer, proj_path, new_module):
    parts = proj_path.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _get_inner_model(model):
    inner = model.model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def collect_activations(model, tokenizer, prompts, device):
    """Collect mean activation magnitude per hidden dim on harmful prompts."""
    inner = _get_inner_model(model)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    act_sum = torch.zeros(n_layers, hidden_dim, device=device)
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0]
            act_sum[layer_idx] += x.abs().sum(dim=[0, 1])
        return hook_fn

    for l in range(n_layers):
        h = inner.layers[l].mlp.up_proj.register_forward_hook(make_hook(l))
        hooks.append(h)

    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for i, item in enumerate(prompts):
            prompt = item["prompt"] if isinstance(item, dict) else item
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            total_tokens += inputs["input_ids"].shape[1]
            model(**inputs)
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(prompts)} prompts processed")

    for h in hooks:
        h.remove()

    mean_act = (act_sum / total_tokens).mean(dim=0)
    print(f"  Activation stats: mean={mean_act.mean():.4f}, max={mean_act.max():.4f}")
    return mean_act


def compute_refusal_direction(model, tokenizer, harmful_prompts, harmless_prompts, device):
    """Compute refusal direction via difference-in-means on last layer activations."""
    inner = _get_inner_model(model)
    harmful_acts, harmless_acts = [], []

    def collect_hook(storage):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            storage.append(output.mean(dim=1).detach())
        return hook_fn

    hook = inner.norm.register_forward_hook(collect_hook(harmful_acts))
    model.eval()
    with torch.no_grad():
        for item in harmful_prompts:
            prompt = item["prompt"] if isinstance(item, dict) else item
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            model(**inputs)
    hook.remove()

    hook = inner.norm.register_forward_hook(collect_hook(harmless_acts))
    with torch.no_grad():
        for item in harmless_prompts:
            prompt = item if isinstance(item, str) else item.get("prompt", "")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            model(**inputs)
    hook.remove()

    r_hat = torch.cat(harmful_acts, 0).mean(0) - torch.cat(harmless_acts, 0).mean(0)
    r_hat = r_hat / r_hat.norm()
    print(f"  Refusal direction: dim={r_hat.shape[0]}")
    return r_hat


def extract_fp32_weights(model, proj_paths):
    n_layers = model.config.num_hidden_layers
    weights = {}
    with torch.no_grad():
        for l in range(n_layers):
            layer = model.model.layers[l]
            weights[l] = {}
            for proj in proj_paths:
                m = _get_submodule(layer, proj)
                weights[l][proj] = get_weight(m, as_dtype=torch.float32).cpu().clone()
    return weights


def interpolation_restore(quant_model, fp32_weights, proj_paths, alpha_per_dim,
                          hidden_dim, device):
    """
    Apply weighted interpolation: W = W_q + α_j × (W_fp32 - W_q) for each dim j.

    Args:
        alpha_per_dim: tensor of shape (hidden_dim,) with per-dim interpolation weights [0,1]
    """
    inner = _get_inner_model(quant_model)
    act_dtype = inner.norm.weight.dtype
    n_layers = quant_model.config.num_hidden_layers
    total_correction = 0.0

    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, 'in_features', None) or getattr(orig_module, 'infeatures', None)
            out_f = getattr(orig_module, 'out_features', None) or getattr(orig_module, 'outfeatures', None)
            has_bias = orig_module.bias is not None

            q_w = get_weight(orig_module, as_dtype=torch.float32).to(device)
            fp32_w = fp32_weights[l][proj].to(device)
            delta = fp32_w - q_w  # correction needed

            # Apply per-dimension interpolation
            if in_f == hidden_dim:
                # Read matrix: columns = hidden dims
                # alpha_per_dim shape: (hidden_dim,) → broadcast over rows
                corrected = q_w + alpha_per_dim.unsqueeze(0) * delta
            elif out_f == hidden_dim:
                # Write matrix: rows = hidden dims
                corrected = q_w + alpha_per_dim.unsqueeze(1) * delta
            else:
                corrected = q_w  # skip if dims don't match

            total_correction += (corrected - q_w).abs().sum().item()

            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(corrected.to(act_dtype), requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(),
                                               requires_grad=False)
            _set_submodule(layer, proj, new_linear)

    return total_correction


def measure_arc(model, tokenizer, arc_items):
    device = next(model.parameters()).device
    correct = 0
    for item in arc_items:
        prompt = f"Question: {item['question']}\nAnswer:"
        pred = _logprob_score(model, tokenizer, prompt, item["choices"], device)
        if pred == item["answer_idx"]:
            correct += 1
    return 100.0 * correct / len(arc_items)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama7b")
    parser.add_argument("--quantizer", default="aqlm2")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_reasoning", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output is None:
        args.output = f"outputs/interpolation_{args.model}_{args.quantizer}.json"

    print("=" * 70)
    print(f"  Safety-Weighted Interpolation Restoration")
    print(f"  Model: {args.model} | Quantizer: {args.quantizer}")
    print(f"  ASR prompts: {args.n_eval} | ARC: {args.n_reasoning}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    from core.datasets import load_harmless_prompts
    harmless = load_harmless_prompts(n_samples=100)

    # ================================================================
    # Step 1: FP32 baseline + extract everything we need
    # ================================================================
    print("\n[Step 1] Loading FP32 model...")
    fp32_model, tokenizer = load_fp32(args.model)
    hidden_dim = fp32_model.config.hidden_size

    print("  Collecting activations...")
    mean_activation = collect_activations(fp32_model, tokenizer, eval_items, device)

    print("  Computing refusal direction...")
    r_hat = compute_refusal_direction(fp32_model, tokenizer,
                                       eval_items[:100], harmless[:100], device)

    print("  Extracting FP32 weights...")
    fp32_weights = extract_fp32_weights(fp32_model, ALL_PROJS)

    print("  Generating FP32 baseline...")
    fp32_responses = generate_only(fp32_model, tokenizer, eval_items, desc="FP32")
    fp32_arc = measure_arc(fp32_model, tokenizer, arc_items)
    print(f"  FP32 ARC: {fp32_arc:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Quantized baseline + compute error
    # ================================================================
    print(f"\n[Step 2] Loading {args.quantizer} model...")
    q_model, _ = load_quantized(args.model, quantizer=args.quantizer)

    # Compute per-dim quantization error
    inner_q = _get_inner_model(q_model)
    n_layers = q_model.config.num_hidden_layers
    quant_error = torch.zeros(hidden_dim, device=device)
    for l in range(n_layers):
        for proj in ALL_PROJS:
            fp32_w = fp32_weights[l][proj].to(device)
            q_w = get_weight(_get_submodule(inner_q.layers[l], proj),
                             as_dtype=torch.float32).to(device)
            in_f, out_f = fp32_w.shape[1], fp32_w.shape[0]
            if in_f == hidden_dim:
                quant_error += (fp32_w - q_w).norm(dim=0)
            elif out_f == hidden_dim:
                quant_error += (fp32_w - q_w).norm(dim=1)

    print(f"  Quant error: mean={quant_error.mean():.2f}, max={quant_error.max():.2f}")

    q_responses = generate_only(q_model, tokenizer, eval_items, desc=args.quantizer)
    q_arc = measure_arc(q_model, tokenizer, arc_items)
    print(f"  {args.quantizer} ARC: {q_arc:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 3: Compute importance scores for weighted interpolation
    # ================================================================
    print("\n[Step 3] Computing importance scores...")
    r_hat_abs = r_hat.abs().to(device)
    mean_act = mean_activation.to(device)

    # Safety importance: refusal × error × activation
    safety_importance = r_hat_abs * quant_error * mean_act
    # Normalize to [0, 1]
    safety_importance = safety_importance / safety_importance.max()

    # Error-only importance (ablation)
    error_importance = quant_error / quant_error.max()

    print(f"  Safety importance: mean={safety_importance.mean():.4f}, "
          f"nonzero={(safety_importance > 0.01).sum().item()}/{hidden_dim}")

    # ================================================================
    # Step 4: Run interpolation experiments
    # ================================================================
    print("\n[Step 4] Running interpolation experiments...")

    # Global alpha values to test
    global_alphas = [0.1, 0.25, 0.5, 0.75, 1.0]

    all_responses = {"fp32": fp32_responses, args.quantizer: q_responses}
    all_arc = {"fp32": fp32_arc, args.quantizer: q_arc}
    conditions = []

    for alpha in global_alphas:
        # --- Method 1: Uniform α (same correction everywhere) ---
        tag = f"uniform_a{int(alpha*100)}"
        print(f"\n  [{tag}] Uniform α={alpha} on ALL projections...")
        q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
        alpha_vec = torch.full((hidden_dim,), alpha, device=device)
        correction = interpolation_restore(q_model, fp32_weights, ALL_PROJS,
                                           alpha_vec, hidden_dim, device)
        print(f"  Total correction magnitude: {correction:.0f}")
        all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
        all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[tag]:.0f}%")
        conditions.append((tag, "uniform"))
        del q_model; gc.collect(); torch.cuda.empty_cache()

        # --- Method 2: Safety-weighted α ---
        tag = f"safety_a{int(alpha*100)}"
        print(f"\n  [{tag}] Safety-weighted α (max={alpha}) on ALL projections...")
        q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
        alpha_vec = safety_importance * alpha  # scale by global alpha
        correction = interpolation_restore(q_model, fp32_weights, ALL_PROJS,
                                           alpha_vec, hidden_dim, device)
        print(f"  Total correction magnitude: {correction:.0f}")
        print(f"  Effective mean α: {alpha_vec.mean():.4f}, max α: {alpha_vec.max():.4f}")
        all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
        all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[tag]:.0f}%")
        conditions.append((tag, "safety-weighted"))
        del q_model; gc.collect(); torch.cuda.empty_cache()

        # --- Method 3: Error-only α (ablation) ---
        tag = f"error_a{int(alpha*100)}"
        print(f"\n  [{tag}] Error-only α (max={alpha}) on ALL projections...")
        q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
        alpha_vec = error_importance * alpha
        correction = interpolation_restore(q_model, fp32_weights, ALL_PROJS,
                                           alpha_vec, hidden_dim, device)
        print(f"  Total correction magnitude: {correction:.0f}")
        all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
        all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[tag]:.0f}%")
        conditions.append((tag, "error-only"))
        del q_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 5: Classify all responses
    # ================================================================
    print("\n" + "=" * 70)
    print("  Classifying all responses...")
    print("=" * 70)

    behaviors = [_extract_behavior(item) for item in eval_items]
    results = {}
    for tag, responses in all_responses.items():
        judgments = classify_batch(behaviors, responses)
        asr = 100.0 * sum(judgments) / len(eval_items)
        results[tag] = {"asr": asr, "arc": all_arc.get(tag)}

    # ================================================================
    # Step 6: Print results
    # ================================================================
    fp32_asr = results["fp32"]["asr"]
    q_asr = results[args.quantizer]["asr"]
    damage = q_asr - fp32_asr

    print("\n" + "=" * 70)
    print(f"  Interpolation Restoration: {args.model} / {args.quantizer}")
    print(f"  Damage: {damage:+.1f}pp (FP32={fp32_asr:.0f}%, {args.quantizer}={q_asr:.0f}%)")
    print("=" * 70)
    print(f"{'Condition':<25s} {'ASR':>6s} {'ΔASR':>8s} {'Recov':>7s} {'ARC':>5s} {'Method'}")
    print("-" * 70)
    print(f"{'fp32':<25s} {fp32_asr:>5.1f}% {'---':>8s} {'':>7s} {fp32_arc:>4.0f}%  baseline")
    print(f"{args.quantizer:<25s} {q_asr:>5.1f}% {damage:>+7.1f}pp {'0%':>7s} {q_arc:>4.0f}%  quantized")

    for tag, method_name in conditions:
        asr = results[tag]["asr"]
        delta = asr - q_asr
        if abs(damage) > 0.5:
            recovery = f"{-delta/damage*100:.0f}%"
        else:
            recovery = "N/A"
        arc = all_arc.get(tag, 0)
        print(f"{tag:<25s} {asr:>5.1f}% {asr-fp32_asr:>+7.1f}pp {recovery:>7s} {arc:>4.0f}%  {method_name}")

    # Best result
    best_tag = min(conditions, key=lambda x: abs(results[x[0]]["asr"] - fp32_asr))
    best_asr = results[best_tag[0]]["asr"]
    best_arc = all_arc[best_tag[0]]
    print(f"\n  Best: {best_tag[0]} — ASR={best_asr:.0f}% (Δ={best_asr-fp32_asr:+.1f}pp), ARC={best_arc:.0f}%")

    # Save
    out_data = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_eval": args.n_eval,
        "fp32_asr": fp32_asr, "quantized_asr": q_asr, "damage_pp": damage,
        "fp32_arc": fp32_arc, "quantized_arc": q_arc,
        "conditions": {tag: {"asr": results[tag]["asr"], "arc": all_arc[tag],
                              "method": method} for tag, method in conditions},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
