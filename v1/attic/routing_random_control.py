import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.seeds import set_global_seed, DEFAULT_SEED
    SEED = set_global_seed(DEFAULT_SEED)
except ModuleNotFoundError:
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

from core.model_loader import load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts
from diagnostics.layer_routing import (
    _get_inner,
    extract_last_token_residuals,
    per_prompt_projections,
    operative_layer_per_prompt,
)


def random_unit_vector(dim, rng):
    v = rng.standard_normal(dim).astype(np.float64)
    return torch.from_numpy(v / (np.linalg.norm(v) + 1e-12)).float()


def summarize_operative(op_list):
    c = Counter(op_list)
    mode_layer, mode_count = c.most_common(1)[0]
    return {
        "mode_layer": int(mode_layer),
        "mode_count": int(mode_count),
        "mode_fraction": float(mode_count / len(op_list)),
        "mean": float(np.mean(op_list)),
        "std": float(np.std(op_list)),
        "n_distinct_layers": len(c),
        "concentration_top3_fraction": float(
            sum(n for _, n in c.most_common(3)) / len(op_list)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="awq")
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--token_pos", type=int, default=None,
                    help="residual token position; MUST match layer_routing "
                         "(llama=-5, phi3=-3) so the z compares like for like")
    ap.add_argument("--normalize", action="store_true", default=False,
                    help="cosine projection (must match layer_routing --normalize)")
    ap.add_argument("--n_random", type=int, default=20,
                    help="number of random unit vectors to sample")
    ap.add_argument("--rand_seed", type=int, default=12345,
                    help="seed for the random-direction sampler (kept separate "
                         "from the global SEED so we can vary controls without "
                         "perturbing the rest of the run)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.output is None:
        args.output = f"outputs/routing_random_control_{args.model}_{args.quantizer}.json"

    print(f"=== Random-direction routing control | {args.model} / {args.quantizer} "
          f"| n_calib={args.n_calib} n_random={args.n_random} | seed={SEED} "
          f"rand_seed={args.rand_seed}")

    print("Loading calibration prompts...")
    harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)
    print(f"  n harmful: {len(harmful)}  n harmless: {len(harmless)}")

    print(f"\n[{args.quantizer}] loading quantized model ...")
    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    device = next(q_model.parameters()).device
    inner = _get_inner(q_model)
    n_layers = len(inner.layers)
    layers = list(range(n_layers))
    # hidden_dim from config (QuantLinear modules don't expose .weight on
    # quantized backends, so reading it off a layer module is unreliable).
    hidden_dim = int(q_model.config.hidden_size)
    print(f"  n_layers = {n_layers}, hidden_dim = {hidden_dim}")

    if args.token_pos is not None:
        token_pos = args.token_pos
    elif "llama" in args.model.lower():
        token_pos = -5
    elif "phi" in args.model.lower():
        token_pos = -3
    else:
        token_pos = -1
    print(f"  token_pos = {token_pos} (must match layer_routing)")

    print(f"[{args.quantizer}] extracting per-layer residuals ...")
    q_h_resid = extract_last_token_residuals(q_model, q_tok, harmful, layers, device, token_pos=token_pos)
    q_s_resid = extract_last_token_residuals(q_model, q_tok, harmless, layers, device, token_pos=token_pos)
    # Defensive: confirm the extracted residual width matches the config.
    observed_dim = int(q_h_resid[layers[0]][0].shape[0])
    assert observed_dim == hidden_dim, \
        f"hidden_dim mismatch: config={hidden_dim} vs residual={observed_dim}"

    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Sample K random unit vectors
    rng = np.random.default_rng(args.rand_seed)
    print(f"\n=== Sampling {args.n_random} random unit vectors (dim={hidden_dim}) ===")

    per_trial_harmful = []   # list of summary dicts, one per random direction
    per_trial_harmless = []
    mode_layers_harmful = []
    mode_layers_harmless = []
    mean_layers_harmful = []
    mean_layers_harmless = []

    for k in range(args.n_random):
        r_rand = random_unit_vector(hidden_dim, rng)
        proj_harmful, _ = per_prompt_projections(q_h_resid, r_rand, normalize=args.normalize)
        proj_harmless, _ = per_prompt_projections(q_s_resid, r_rand, normalize=args.normalize)
        op_harmful = operative_layer_per_prompt(proj_harmful, layers)
        op_harmless = operative_layer_per_prompt(proj_harmless, layers)
        sh = summarize_operative(op_harmful)
        ss = summarize_operative(op_harmless)
        per_trial_harmful.append(sh)
        per_trial_harmless.append(ss)
        mode_layers_harmful.append(sh["mode_layer"])
        mode_layers_harmless.append(ss["mode_layer"])
        mean_layers_harmful.append(sh["mean"])
        mean_layers_harmless.append(ss["mean"])
        if k < 5 or k == args.n_random - 1:
            print(f"  trial {k+1:>2}/{args.n_random}: "
                  f"harmful mode={sh['mode_layer']:>2} (mean={sh['mean']:.1f}) | "
                  f"harmless mode={ss['mode_layer']:>2} (mean={ss['mean']:.1f})")

    # Aggregate across trials
    def agg(mode_list, mean_list):
        c = Counter(mode_list)
        most_common_mode, most_common_count = c.most_common(1)[0]
        return {
            "mode_layer_across_trials": int(most_common_mode),
            "mode_layer_across_trials_fraction": float(most_common_count / len(mode_list)),
            "mode_layer_per_trial": list(map(int, mode_list)),
            "mean_layer_mean": float(np.mean(mean_list)),
            "mean_layer_std": float(np.std(mean_list)),
            "mean_layer_min": float(np.min(mean_list)),
            "mean_layer_max": float(np.max(mean_list)),
        }

    summary_harmful = agg(mode_layers_harmful, mean_layers_harmful)
    summary_harmless = agg(mode_layers_harmless, mean_layers_harmless)

    print(f"\n=== Random-direction summary ({args.n_random} trials) ===")
    print(f"  harmful : mode-of-modes={summary_harmful['mode_layer_across_trials']} "
          f"({summary_harmful['mode_layer_across_trials_fraction']:.1%} of trials)  "
          f"mean-layer={summary_harmful['mean_layer_mean']:.1f} +/- "
          f"{summary_harmful['mean_layer_std']:.2f} "
          f"(min={summary_harmful['mean_layer_min']:.1f}, "
          f"max={summary_harmful['mean_layer_max']:.1f})")
    print(f"  harmless: mode-of-modes={summary_harmless['mode_layer_across_trials']} "
          f"({summary_harmless['mode_layer_across_trials_fraction']:.1%} of trials)  "
          f"mean-layer={summary_harmless['mean_layer_mean']:.1f} +/- "
          f"{summary_harmless['mean_layer_std']:.2f}")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_calib": args.n_calib,
        "n_random": args.n_random,
        "n_layers": n_layers,
        "hidden_dim": int(hidden_dim),
        "seed": SEED,
        "rand_seed": args.rand_seed,
        "random_direction_summary_harmful": summary_harmful,
        "random_direction_summary_harmless": summary_harmless,
        "per_trial_harmful": per_trial_harmful,
        "per_trial_harmless": per_trial_harmless,
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
