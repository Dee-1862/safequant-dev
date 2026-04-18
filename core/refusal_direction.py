"""
Compute refusal direction using difference-in-means method.

The refusal direction r̂ is a vector in the residual stream that represents
"this is a dangerous/harmful request". It's computed by taking the difference
between mean activations on harmful vs harmless prompts.

Uses benchmark datasets:
- AdvBench (harmful): Zou et al., "Universal and Transferable Adversarial Attacks on Aligned Language Models", 2023
- Anthropic HH-RLHF (harmless): Bai et al., "Training a Helpful and Harmless Assistant with RLHF", 2022
"""

import torch
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.datasets import load_harmful_prompts, load_harmless_prompts


def load_prompts(n_samples=100):
    """
    Load harmful and harmless prompts for refusal direction computation.

    Uses benchmark datasets - NO FALLBACKS.

    Returns:
        (harmful_prompts, harmless_prompts)
    """
    harmful_items = load_harmful_prompts(n_samples=n_samples)
    harmful = [item["goal"] for item in harmful_items]
    harmless = load_harmless_prompts(n_samples=n_samples)
    return harmful, harmless


def extract_activations(model, tokenizer, prompts, layer_idx=-1):
    """
    Extract residual stream activations at a specific layer.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        layer_idx: Layer to extract from (-1 = last layer)

    Returns:
        Tensor of shape (n_prompts, hidden_size) - mean-pooled activations
    """
    device = next(model.parameters()).device
    hidden_size = model.config.hidden_size
    activations = []

    # Hook to capture activations
    captured = {}

    def hook_fn(module, input, output):
        # output is tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        captured['activation'] = hidden_states.detach()

    # Register hook on the target layer
    if layer_idx == -1:
        layer_idx = model.config.num_hidden_layers - 1

    # Get the layer (handle different model architectures)
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Qwen2, Llama style
        layer = model.model.layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2 style
        layer = model.transformer.h[layer_idx]
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")

    handle = layer.register_forward_hook(hook_fn)

    try:
        for prompt in tqdm(prompts, desc="Extracting activations"):
            # Format prompt
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Tokenize
            inputs = tokenizer(text, return_tensors="pt").to(device)

            # Forward pass
            with torch.no_grad():
                _ = model(**inputs)

            # Get captured activation and mean-pool over sequence
            act = captured['activation']  # (1, seq_len, hidden_size)
            act_pooled = act.mean(dim=1).squeeze(0)  # (hidden_size,)
            activations.append(act_pooled.cpu())

    finally:
        handle.remove()

    return torch.stack(activations)  # (n_prompts, hidden_size)


def compute(model, tokenizer, n_samples=100, layer_idx=-1, normalize=True):
    """
    Compute refusal direction r̂ using difference-in-means.

    r̂ = mean(harmful_activations) - mean(harmless_activations)

    Args:
        model: Language model
        tokenizer: Tokenizer
        n_samples: Number of samples per category
        layer_idx: Layer to extract from (-1 = last)
        normalize: Whether to normalize r̂ to unit length

    Returns:
        (r_hat, harmful_acts, harmless_acts)
        - r_hat: Refusal direction vector (hidden_size,)
        - harmful_acts: Activations on harmful prompts (n_samples, hidden_size)
        - harmless_acts: Activations on harmless prompts (n_samples, hidden_size)
    """
    print(f"\n{'='*60}")
    print("Computing Refusal Direction")
    print(f"{'='*60}")
    print(f"  Layer: {layer_idx if layer_idx >= 0 else 'last'}")
    print(f"  Samples: {n_samples} harmful + {n_samples} harmless")

    # Load prompts
    harmful, harmless = load_prompts(n_samples)
    print(f"  ✓ Loaded {len(harmful)} harmful and {len(harmless)} harmless prompts")

    # Extract activations
    print("\n[1/3] Extracting harmful activations...")
    harmful_acts = extract_activations(model, tokenizer, harmful, layer_idx)

    print("\n[2/3] Extracting harmless activations...")
    harmless_acts = extract_activations(model, tokenizer, harmless, layer_idx)

    # Compute difference-in-means
    print("\n[3/3] Computing difference-in-means...")
    mean_harmful = harmful_acts.mean(dim=0)
    mean_harmless = harmless_acts.mean(dim=0)
    r_hat = mean_harmful - mean_harmless

    if normalize:
        r_hat = r_hat / torch.norm(r_hat)
        print(f"  ✓ Normalized refusal direction (||r̂|| = 1.0)")
    else:
        print(f"  ✓ Unnormalized refusal direction (||r̂|| = {torch.norm(r_hat):.2f})")

    print(f"\n  Refusal direction computed:")
    print(f"    Shape: {r_hat.shape}")
    print(f"    Mean: {r_hat.mean():.4f}")
    print(f"    Std: {r_hat.std():.4f}")

    return r_hat, harmful_acts, harmless_acts


if __name__ == "__main__":
    import argparse
    from core.model_loader import load_fp32

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output", default="outputs/refusal_direction.pt")
    args = parser.parse_args()

    print(f"\nComputing refusal direction for {args.model}\n")

    model, tokenizer = load_fp32(args.model)
    r_hat, harmful_acts, harmless_acts = compute(model, tokenizer, args.n_samples, args.layer)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'r_hat': r_hat,
        'harmful_acts': harmful_acts,
        'harmless_acts': harmless_acts,
        'model': args.model,
        'layer': args.layer,
        'n_samples': args.n_samples
    }, output_path)

    print(f"\n✓ Saved to {output_path}")
