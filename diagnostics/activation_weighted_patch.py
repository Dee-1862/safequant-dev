"""
Activation-Weighted Selective Restoration for Llama-3.1-8B GPTQ-4.

Fixes the V-score targeting by weighting with actual activation magnitudes
from FP32 model on harmful prompts. Then runs sparse restoration experiment
comparing: activation-weighted V-score vs plain V-score vs random.

Pipeline:
  1. Load FP32 model, collect activations on harmful prompts (~2 min)
  2. Compute refusal direction r-hat
  3. Load GPTQ-4 model, compute weight error per dimension
  4. Compute three targeting metrics:
     - V2: |r_hat[j]| × weight_error[j]  (old, broken)
     - V3: |r_hat[j]| × weight_error[j] × mean_activation[j]  (new)
     - Random: random selection (control)
  5. Run restoration at 5%, 15%, 25%, 50% for all three metrics
  6. Measure ASR + ARC for each condition
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

# Reproducibility
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

# Read-side matrices only (what we're patching)
READ_PROJS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "mlp.gate_proj", "mlp.up_proj"]


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
    """
    Collect mean activation magnitude per hidden dimension across harmful prompts.
    Hooks into all read-side matrix INPUTS to measure what the model actually uses.

    Returns: tensor of shape (hidden_dim,) with mean |activation[j]| across
             all layers, prompts, and sequence positions.
    """
    inner = _get_inner_model(model)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size

    # Accumulator: sum of |activation| per dimension, per layer
    act_sum = torch.zeros(n_layers, hidden_dim, device=device)
    act_count = 0

    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            nonlocal act_count
            x = input[0]  # (batch, seq_len, hidden_dim)
            # Sum absolute values across batch and sequence
            act_sum[layer_idx] += x.abs().sum(dim=[0, 1])
        return hook_fn

    # Register hooks on up_proj (representative read-side matrix)
    for l in range(n_layers):
        layer = inner.layers[l]
        h = layer.mlp.up_proj.register_forward_hook(make_hook(l))
        hooks.append(h)

    # Forward pass on each prompt (no generation, just encode)
    print(f"  Collecting activations on {len(prompts)} prompts...")
    model.eval()
    total_tokens = 0
    with torch.no_grad():
        for i, item in enumerate(prompts):
            prompt = item["prompt"] if isinstance(item, dict) else item
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            seq_len = inputs["input_ids"].shape[1]
            total_tokens += seq_len
            model(**inputs)
            if (i + 1) % 20 == 0:
                print(f"    {i+1}/{len(prompts)} prompts processed")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average: divide by total tokens (sum over batch*seq was accumulated)
    mean_act = act_sum / total_tokens  # (n_layers, hidden_dim)

    # Aggregate across layers (mean across all layers)
    mean_act_per_dim = mean_act.mean(dim=0)  # (hidden_dim,)

    print(f"  Activation stats: mean={mean_act_per_dim.mean():.4f}, "
          f"max={mean_act_per_dim.max():.4f}, "
          f"min={mean_act_per_dim.min():.4f}")

    return mean_act_per_dim


def compute_quant_error_per_dim(fp32_model, quant_model, device):
    """Compute per-dimension quantization error across all read-side matrices."""
    inner_fp32 = _get_inner_model(fp32_model)
    inner_q = _get_inner_model(quant_model)
    n_layers = fp32_model.config.num_hidden_layers
    hidden_dim = fp32_model.config.hidden_size

    error_sum = torch.zeros(hidden_dim, device=device)

    for l in range(n_layers):
        for proj in READ_PROJS:
            fp32_w = get_weight(_get_submodule(inner_fp32.layers[l], proj),
                                as_dtype=torch.float32).to(device)
            q_w = get_weight(_get_submodule(inner_q.layers[l], proj),
                             as_dtype=torch.float32).to(device)

            in_f = fp32_w.shape[1]
            out_f = fp32_w.shape[0]

            if in_f == hidden_dim:
                # Read matrix: columns correspond to hidden dims
                col_error = (fp32_w - q_w).norm(dim=0)  # (hidden_dim,)
                error_sum += col_error
            elif out_f == hidden_dim:
                # Write matrix: rows correspond to hidden dims
                row_error = (fp32_w - q_w).norm(dim=1)  # (hidden_dim,)
                error_sum += row_error

    return error_sum


def compute_refusal_direction(model, tokenizer, harmful_prompts, harmless_prompts, device):
    """Compute refusal direction via difference-in-means on last layer activations."""
    inner = _get_inner_model(model)

    harmful_acts = []
    harmless_acts = []

    def collect_hook(storage):
        def hook_fn(module, input, output):
            # output is the hidden state after the final norm
            if isinstance(output, tuple):
                output = output[0]
            # Take mean across sequence length
            storage.append(output.mean(dim=1).detach())
        return hook_fn

    # Hook on the final layer norm
    hook_handle = inner.norm.register_forward_hook(collect_hook(harmful_acts))
    print(f"  Computing refusal direction (harmful={len(harmful_prompts)}, "
          f"harmless={len(harmless_prompts)})...")

    model.eval()
    with torch.no_grad():
        for item in harmful_prompts:
            prompt = item["prompt"] if isinstance(item, dict) else item
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            model(**inputs)

    hook_handle.remove()

    hook_handle = inner.norm.register_forward_hook(collect_hook(harmless_acts))
    with torch.no_grad():
        for item in harmless_prompts:
            prompt = item if isinstance(item, str) else item.get("prompt", "")
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(device)
            model(**inputs)

    hook_handle.remove()

    harmful_mean = torch.cat(harmful_acts, dim=0).mean(dim=0)
    harmless_mean = torch.cat(harmless_acts, dim=0).mean(dim=0)
    r_hat = harmful_mean - harmless_mean
    r_hat = r_hat / r_hat.norm()

    print(f"  Refusal direction: dim={r_hat.shape[0]}")
    return r_hat


def extract_fp32_weights(model, proj_paths):
    """Extract FP32 weights for all layers and projections."""
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


def restore_with_targeting(quant_model, fp32_weights, proj_paths, target_dims,
                           hidden_dim, device):
    """Restore specific dimensions of read-side matrices using target_dims indices."""
    inner = _get_inner_model(quant_model)
    act_dtype = inner.norm.weight.dtype
    n_layers = quant_model.config.num_hidden_layers
    n_restored = 0

    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            orig_module = _get_submodule(layer, proj)
            in_f = getattr(orig_module, 'in_features', None) or getattr(orig_module, 'infeatures', None)
            out_f = getattr(orig_module, 'out_features', None) or getattr(orig_module, 'outfeatures', None)
            has_bias = orig_module.bias is not None

            q_w = get_weight(orig_module, as_dtype=torch.float32).to(device)
            fp32_w = fp32_weights[l][proj].to(device)

            patched_w = q_w.clone()

            if in_f == hidden_dim:
                for dim in target_dims:
                    if dim < in_f:
                        patched_w[:, dim] = fp32_w[:, dim]
                        n_restored += out_f
            elif out_f == hidden_dim:
                for dim in target_dims:
                    if dim < out_f:
                        patched_w[dim, :] = fp32_w[dim, :]
                        n_restored += in_f

            new_linear = nn.Linear(in_f, out_f, bias=has_bias).to(device).to(act_dtype)
            new_linear.weight = nn.Parameter(patched_w.to(act_dtype), requires_grad=False)
            if has_bias:
                new_linear.bias = nn.Parameter(orig_module.bias.data.clone(),
                                               requires_grad=False)
            _set_submodule(layer, proj, new_linear)

    return n_restored


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
    parser.add_argument("--quantizer", default="gptq4")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--n_reasoning", type=int, default=50)
    parser.add_argument("--n_activation_prompts", type=int, default=100,
                        help="Harmful prompts for activation collection")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.output is None:
        args.output = f"outputs/activation_patch_{args.model}_{args.quantizer}.json"

    print("=" * 70)
    print(f"  Activation-Weighted Selective Restoration")
    print(f"  Model: {args.model} | Quantizer: {args.quantizer}")
    print(f"  ASR prompts: {args.n_eval} | ARC: {args.n_reasoning}")
    print("=" * 70)

    # Load datasets
    print("\nLoading datasets...")
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    activation_items = load_wildjailbreak_prompts(
        n_samples=args.n_activation_prompts, config="eval")
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)

    # Load harmless prompts for refusal direction
    from core.datasets import load_harmless_prompts
    harmless = load_harmless_prompts(n_samples=100)

    print(f"  Eval: {len(eval_items)} WildJailbreak prompts")
    print(f"  Activation collection: {len(activation_items)} prompts")
    print(f"  ARC: {len(arc_items)} questions")

    # ================================================================
    # Step 1: Load FP32, collect activations + refusal direction
    # ================================================================
    print("\n[Step 1] Loading FP32 model...")
    fp32_model, tokenizer = load_fp32(args.model)

    print("\n  Collecting FP32 activations on harmful prompts...")
    mean_activation = collect_activations(fp32_model, tokenizer,
                                          activation_items, device)

    print("\n  Computing refusal direction...")
    r_hat = compute_refusal_direction(fp32_model, tokenizer,
                                       activation_items[:100], harmless[:100],
                                       device)

    print("\n  Extracting FP32 weights...")
    fp32_weights = extract_fp32_weights(fp32_model, READ_PROJS)
    hidden_dim = fp32_model.config.hidden_size

    # Generate FP32 baseline responses
    print("\n  Generating FP32 baseline responses...")
    fp32_responses = generate_only(fp32_model, tokenizer, eval_items, desc="FP32")
    fp32_arc = measure_arc(fp32_model, tokenizer, arc_items)
    print(f"  FP32 ARC: {fp32_arc:.0f}%")
    del fp32_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 2: Load quantized, compute weight error
    # ================================================================
    print(f"\n[Step 2] Loading {args.quantizer} model for weight error...")
    q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
    quant_error = compute_quant_error_per_dim(
        # Need FP32 model again for comparison -- reload lightweight
        # Actually we already have fp32_weights, compute from those
        q_model, q_model, device  # placeholder, we compute differently
    )
    # Actually recompute properly using stored FP32 weights
    inner_q = _get_inner_model(q_model)
    n_layers = q_model.config.num_hidden_layers
    quant_error = torch.zeros(hidden_dim, device=device)
    for l in range(n_layers):
        for proj in READ_PROJS:
            fp32_w = fp32_weights[l][proj].to(device)
            q_w = get_weight(_get_submodule(inner_q.layers[l], proj),
                             as_dtype=torch.float32).to(device)
            in_f = fp32_w.shape[1]
            out_f = fp32_w.shape[0]
            if in_f == hidden_dim:
                quant_error += (fp32_w - q_w).norm(dim=0)
            elif out_f == hidden_dim:
                quant_error += (fp32_w - q_w).norm(dim=1)

    print(f"  Quant error per dim: mean={quant_error.mean():.4f}, max={quant_error.max():.4f}")

    # Generate quantized baseline
    print(f"\n  Generating {args.quantizer} baseline responses...")
    q_responses = generate_only(q_model, tokenizer, eval_items, desc=args.quantizer)
    q_arc = measure_arc(q_model, tokenizer, arc_items)
    print(f"  {args.quantizer} ARC: {q_arc:.0f}%")
    del q_model; gc.collect(); torch.cuda.empty_cache()

    # ================================================================
    # Step 3: Compute three targeting metrics
    # ================================================================
    print("\n[Step 3] Computing targeting metrics...")

    r_hat_abs = r_hat.abs().to(device)
    mean_act = mean_activation.to(device)

    # V2: |r_hat| × error (old, broken)
    v2_scores = r_hat_abs * quant_error
    # V3: |r_hat| × error × activation (new)
    v3_scores = r_hat_abs * quant_error * mean_act

    print(f"  V2 top dims: {v2_scores.topk(5).indices.tolist()}")
    print(f"  V3 top dims: {v3_scores.topk(5).indices.tolist()}")
    overlap = set(v2_scores.topk(50).indices.tolist()) & set(v3_scores.topk(50).indices.tolist())
    print(f"  Top-50 overlap between V2 and V3: {len(overlap)}/50")

    # ================================================================
    # Step 4: Run restoration experiments
    # ================================================================
    top_pcts = [5, 15, 25, 50]
    all_responses = {"fp32": fp32_responses, args.quantizer: q_responses}
    all_arc = {"fp32": fp32_arc, args.quantizer: q_arc}

    for pct in top_pcts:
        k = max(1, int(hidden_dim * pct / 100))

        for method_name, scores in [("v2", v2_scores), ("v3_act", v3_scores)]:
            tag = f"{method_name}_{pct}pct"
            dims = scores.topk(k).indices.cpu().tolist()

            print(f"\n  [{tag}] Restoring top {pct}% ({k} dims)...")
            q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
            n_restored = restore_with_targeting(q_model, fp32_weights, READ_PROJS,
                                                dims, hidden_dim, device)
            overhead = 100.0 * n_restored / sum(p.numel() for p in q_model.parameters())
            print(f"  Restored {n_restored:,} params ({overhead:.1f}% overhead)")

            all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
            all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
            print(f"  ARC: {all_arc[tag]:.0f}%")
            del q_model; gc.collect(); torch.cuda.empty_cache()

        # Random control
        tag = f"random_{pct}pct"
        dims = random.sample(range(hidden_dim), k)

        print(f"\n  [{tag}] Restoring random {pct}% ({k} dims)...")
        q_model, _ = load_quantized(args.model, quantizer=args.quantizer)
        n_restored = restore_with_targeting(q_model, fp32_weights, READ_PROJS,
                                            dims, hidden_dim, device)
        all_responses[tag] = generate_only(q_model, tokenizer, eval_items, desc=tag)
        all_arc[tag] = measure_arc(q_model, tokenizer, arc_items)
        print(f"  ARC: {all_arc[tag]:.0f}%")
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
        n_complied = sum(judgments)
        asr = 100.0 * n_complied / len(eval_items)
        results[tag] = {"asr": asr, "arc": all_arc.get(tag, None),
                        "n_complied": n_complied}

    # ================================================================
    # Step 6: Print results
    # ================================================================
    print("\n" + "=" * 70)
    print(f"  Activation-Weighted Restoration: {args.model} / {args.quantizer}")
    print("=" * 70)

    fp32_asr = results["fp32"]["asr"]
    q_asr = results[args.quantizer]["asr"]

    print(f"{'Condition':<25s} {'ASR':>6s} {'ΔASR':>8s} {'ARC':>5s} {'Method'}")
    print("-" * 65)
    print(f"{'fp32':<25s} {fp32_asr:>5.1f}% {'---':>8s} {all_arc['fp32']:>4.0f}%  unquantized")
    print(f"{args.quantizer:<25s} {q_asr:>5.1f}% {q_asr-fp32_asr:>+7.1f}pp {all_arc[args.quantizer]:>4.0f}%  baseline")

    for pct in top_pcts:
        for method in ["v2", "v3_act", "random"]:
            tag = f"{method}_{pct}pct"
            if tag in results:
                asr = results[tag]["asr"]
                delta = asr - q_asr
                arc = all_arc.get(tag, 0)
                label = {"v2": "r̂×error (old)", "v3_act": "r̂×error×act (NEW)",
                         "random": "random (control)"}[method]
                print(f"{tag:<25s} {asr:>5.1f}% {delta:>+7.1f}pp {arc:>4.0f}%  {label}")

    # Key comparison
    print("\n  Key question: Does V3 (activation-weighted) beat V2 and random?")
    for pct in top_pcts:
        v2_asr = results.get(f"v2_{pct}pct", {}).get("asr", -1)
        v3_asr = results.get(f"v3_act_{pct}pct", {}).get("asr", -1)
        rand_asr = results.get(f"random_{pct}pct", {}).get("asr", -1)
        winner = "V3" if v3_asr < v2_asr and v3_asr < rand_asr else \
                 "random" if rand_asr < v2_asr and rand_asr < v3_asr else "V2"
        print(f"    {pct}%: V2={v2_asr:.0f}% V3={v3_asr:.0f}% Random={rand_asr:.0f}% → {winner} wins")

    # Save
    out_data = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_eval": args.n_eval,
        "n_activation_prompts": args.n_activation_prompts,
        "v2_top5_dims": v2_scores.topk(5).indices.tolist(),
        "v3_top5_dims": v3_scores.topk(5).indices.tolist(),
        "top50_overlap": len(overlap),
        "results": {k: {"asr": v["asr"], "arc": v.get("arc"),
                         "n_complied": v["n_complied"]}
                    for k, v in results.items()},
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n✓ Saved to {out_path}")


if __name__ == "__main__":
    main()
