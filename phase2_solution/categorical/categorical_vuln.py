"""
SafeQuant Phase 2 v2 — Categorical Vulnerability Analysis (Secondary).

Computes category-specific refusal directions and shows that NF4
disproportionately damages dimensions for specific harm categories.

Categories from HarmBench:
  - Chemical/Biological weapons
  - Cybercrime
  - Harassment/Discrimination
  - Misinformation
  - etc.

Uses global direction as primary. Categorical decomposition reveals
finer structure without undermining the main pipeline.

Run:
  python phase2_solution/categorical/categorical_vuln.py --model qwen1.5b
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import json
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import MODELS, DEVICE, DTYPE, NF4_CONFIG


# HarmBench category keywords for classification
CATEGORY_KEYWORDS = {
    "chemical_biological": ["chemical", "biological", "virus", "pathogen", "toxin", "poison",
                            "nerve agent", "bioweapon", "anthrax", "ricin", "sarin"],
    "cybercrime": ["hack", "malware", "exploit", "phishing", "ransomware", "DDoS",
                   "vulnerability", "password", "SQL injection", "trojan", "keylogger"],
    "harassment": ["harass", "bully", "discriminat", "hate speech", "slur", "derogat",
                   "threaten", "stalk", "doxx", "intimidat"],
    "violence": ["kill", "murder", "weapon", "bomb", "gun", "shoot", "attack",
                 "assault", "violent", "explosion", "firearm"],
    "fraud": ["fraud", "scam", "counterfeit", "forge", "impersonat", "identity theft",
              "launder", "embezzl", "deceiv"],
}


def classify_prompt(prompt):
    """Assign a harm category based on keyword matching."""
    prompt_lower = prompt.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in prompt_lower)
        if score > 0:
            scores[cat] = score
    if scores:
        return max(scores, key=scores.get)
    return "other"


def extract_mean_activations(model, tokenizer, prompts, n_layers, hidden_size):
    """Get mean last-token activations across prompts at each layer."""
    sums = {l: torch.zeros(hidden_size) for l in range(n_layers)}
    count = 0

    for prompt in tqdm(prompts, desc="    Activations", leave=False):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        layer_outputs = {}

        def make_hook(l):
            def fn(module, inp, out):
                # Residual stream at layer output
                hidden = out[0] if isinstance(out, tuple) else out
                layer_outputs[l] = hidden[0, -1, :].detach().cpu().float()
            return fn

        hooks = []
        for l in range(n_layers):
            hooks.append(model.model.layers[l].register_forward_hook(make_hook(l)))

        with torch.no_grad():
            model(**inputs)

        for h in hooks:
            h.remove()

        for l in range(n_layers):
            if l in layer_outputs:
                sums[l] += layer_outputs[l]
        count += 1

    return {l: sums[l] / count for l in range(n_layers)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    args = parser.parse_args()

    model_name = MODELS[args.model]
    output_dir = os.path.join(os.path.dirname(__file__), "outputs", args.model)
    os.makedirs(output_dir, exist_ok=True)

    from config import DATA_DIR

    print(f"\n=== Categorical Vulnerability Analysis: {args.model} ===\n")

    # Load harmful prompts and classify
    with open(os.path.join(DATA_DIR, "harmful_train.json"), encoding="utf-8") as f:
        harmful_prompts = json.load(f)

    categories = {}
    for p in harmful_prompts:
        cat = classify_prompt(p)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(p)

    print("  Category distribution:")
    for cat, prompts in sorted(categories.items()):
        print(f"    {cat}: {len(prompts)} prompts")

    # Filter to categories with enough data
    min_prompts = 5
    valid_cats = {k: v for k, v in categories.items() if len(v) >= min_prompts and k != "other"}

    if len(valid_cats) < 2:
        print(f"\n  Not enough category diversity (need ≥2 with ≥{min_prompts} prompts)")
        print("  Skipping categorical analysis")
        return

    # Load harmless prompts for direction computation
    with open(os.path.join(DATA_DIR, "harmless_train.json"), encoding="utf-8") as f:
        harmless_prompts = json.load(f)

    # Load FP16 model
    print("\n  Loading FP16 model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    # Get harmless baseline activations
    print("  Computing harmless baseline...")
    harmless_acts = extract_mean_activations(
        model, tokenizer, harmless_prompts[:50], n_layers, hidden_size
    )

    # Compute category-specific directions
    cat_directions = {}
    for cat, prompts in valid_cats.items():
        print(f"  Computing direction for: {cat} ({len(prompts)} prompts)")
        cat_acts = extract_mean_activations(model, tokenizer, prompts, n_layers, hidden_size)

        # Direction per layer
        best_norm = 0
        best_dir = None
        best_layer = 0
        for l in range(n_layers):
            d = cat_acts[l] - harmless_acts[l]
            norm = torch.norm(d).item()
            if norm > best_norm:
                best_norm = norm
                best_dir = d / norm
                best_layer = l

        cat_directions[cat] = {
            "direction": best_dir,
            "layer": best_layer,
            "norm": best_norm,
        }

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Load global direction for comparison
    direction_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "phase1_problem",
        "step1_direction", "outputs", args.model
    )
    global_dir_data = torch.load(
        os.path.join(direction_dir, "best_direction_fp16.pt"), map_location="cpu"
    )
    r_global = global_dir_data["direction_normalized"]

    # Compare directions
    print(f"\n  === Direction Similarity ===")
    print(f"  {'Category A':<20} | {'Category B':<20} | {'Cosine Sim':>10}")
    print(f"  {'-'*20}-+-{'-'*20}-+-{'-'*10}")

    cat_names = sorted(cat_directions.keys())
    for cat in cat_names:
        sim = torch.dot(cat_directions[cat]["direction"], r_global).item()
        print(f"  {cat:<20} | {'global':<20} | {sim:>10.4f}")

    for i in range(len(cat_names)):
        for j in range(i+1, len(cat_names)):
            d1 = cat_directions[cat_names[i]]["direction"]
            d2 = cat_directions[cat_names[j]]["direction"]
            sim = torch.dot(d1, d2).item()
            print(f"  {cat_names[i]:<20} | {cat_names[j]:<20} | {sim:>10.4f}")

    # Load per-dimension vulnerability data
    diag_dir = os.path.join(
        os.path.dirname(__file__), "..", "diagnosis", "outputs", args.model
    )
    error_data = torch.load(os.path.join(diag_dir, "per_row_errors.pt"), map_location="cpu")
    per_row_errors = error_data["per_row_errors"]

    # Compute category-specific vulnerability profiles
    print(f"\n  === Category-Specific Vulnerability ===")
    cat_vulns = {}
    for cat in cat_names:
        r_cat = cat_directions[cat]["direction"]  # (hidden_size,)
        total_vuln = 0
        for comp, err_vec in per_row_errors.items():
            # Per-dim DFA using category direction
            dfa_cat = torch.abs(r_cat[:err_vec.shape[0]])
            v_cat = dfa_cat * err_vec
            total_vuln += v_cat.sum().item()
        cat_vulns[cat] = total_vuln

    max_vuln = max(cat_vulns.values())
    for cat in cat_names:
        bar_len = int(40 * cat_vulns[cat] / max_vuln)
        print(f"  {cat:<20}: {'█' * bar_len} {cat_vulns[cat]:.4f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Categorical Vulnerability: {args.model}", fontsize=13, fontweight="bold")

    # Heatmap of direction similarities
    ax = axes[0]
    all_dirs = cat_names + ["global"]
    all_vecs = [cat_directions[c]["direction"] for c in cat_names] + [r_global]
    n = len(all_dirs)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = torch.dot(all_vecs[i], all_vecs[j]).item()
    im = ax.imshow(sim_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(all_dirs, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_dirs, fontsize=8)
    ax.set_title("Refusal Direction Similarity")
    fig.colorbar(im, ax=ax, shrink=0.8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7)

    # Bar chart of total vulnerability per category
    ax = axes[1]
    cats_sorted = sorted(cat_vulns.keys(), key=lambda x: cat_vulns[x], reverse=True)
    vals = [cat_vulns[c] for c in cats_sorted]
    colors = plt.cm.Set2(np.linspace(0, 1, len(cats_sorted)))
    ax.barh(range(len(cats_sorted)), vals, color=colors)
    ax.set_yticks(range(len(cats_sorted)))
    ax.set_yticklabels(cats_sorted, fontsize=9)
    ax.set_xlabel("Total Vulnerability (Σ V_i across all components)")
    ax.set_title("Which Harm Categories Are Most\nVulnerable to Quantization?")
    ax.invert_yaxis()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "categorical_vulnerability.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved to {plot_path}")

    # Save
    results = {
        "categories": {cat: len(prompts) for cat, prompts in valid_cats.items()},
        "direction_layers": {cat: cat_directions[cat]["layer"] for cat in cat_names},
        "total_vulnerability_per_category": cat_vulns,
        "direction_similarities": {
            f"{cat}_vs_global": torch.dot(cat_directions[cat]["direction"], r_global).item()
            for cat in cat_names
        },
    }
    with open(os.path.join(output_dir, "categorical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved")
    print("=== Categorical analysis complete ===")


if __name__ == "__main__":
    main()
