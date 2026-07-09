import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from core.seeds import set_global_seed, DEFAULT_SEED
SEED = set_global_seed(DEFAULT_SEED)

from core.model_loader import load_quantized
from core.datasets import load_wildjailbreak_prompts
from core.evaluation import generate_only, classify_batch, _extract_behavior

from directional_ablation import (
    install_ablation, remove_handles,
)
from layer_routing import _get_inner


def random_unit_vector(dim, rng):
    v = rng.standard_normal(dim).astype(np.float64)
    return torch.from_numpy(v / (np.linalg.norm(v) + 1e-12)).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_eval", type=int, default=100,
                    help="WildJailbreak prompts per ASR eval. 100 by default "
                         "to keep K=20 random trials tractable.")
    ap.add_argument("--n_random", type=int, default=20,
                    help="number of random unit vectors to sample")
    ap.add_argument("--rand_seed", type=int, default=12345,
                    help="separate sampler seed (matches routing_random_control.py)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = (f"outputs/random_ablation_dist_"
                       f"{args.model}_{args.quantizer}.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"  Random-direction ablation distribution")
    print(f"  {args.model} / {args.quantizer}")
    print(f"  n_eval={args.n_eval}  K={args.n_random}  seed={SEED}  rand_seed={args.rand_seed}")
    print("=" * 70)

    print("\nLoading WildJailbreak eval set ...")
    eval_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in eval_items]
    print(f"  n_eval = {len(eval_items)}")

    print(f"\nLoading {args.quantizer} model ...")
    q_model, tokenizer = load_quantized(args.model, quantizer=args.quantizer)
    inner = _get_inner(q_model)
    n_layers = len(inner.layers)
    hidden_dim = int(q_model.config.hidden_size)
    layers = list(range(n_layers))
    print(f"  n_layers = {n_layers}, hidden_dim = {hidden_dim}")

    # Phase A: collect ALL responses (baseline + K random vectors).
    # Then classify everything in one HarmBench load to keep VRAM saner.
    all_responses = {}

    print(f"\n--- Baseline (no ablation) ---")
    all_responses["baseline"] = generate_only(q_model, tokenizer, eval_items,
                                               desc="baseline")

    rng = np.random.default_rng(args.rand_seed)
    print(f"\n--- Sampling {args.n_random} random unit vectors (dim={hidden_dim}) ---")
    for k in range(args.n_random):
        r_rand = random_unit_vector(hidden_dim, rng)
        handles = install_ablation(q_model, r_rand, layers)
        try:
            tag = f"rand_{k:02d}"
            print(f"  trial {k+1:>2}/{args.n_random}  generating ...")
            all_responses[tag] = generate_only(q_model, tokenizer, eval_items,
                                                desc=tag)
        finally:
            remove_handles(handles)

    # Free the quant model before classifying
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Phase B: classify all responses in one batch
    print(f"\n--- Classifying {len(all_responses) * args.n_eval} responses ---")
    conditions = list(all_responses.keys())
    flat_b, flat_r = [], []
    for k in conditions:
        flat_b.extend(behaviors)
        flat_r.extend(all_responses[k])
    flags = classify_batch(flat_b, flat_r, device=device)
    n = args.n_eval
    asr = {k: 100.0 * sum(flags[i*n:(i+1)*n]) / n
           for i, k in enumerate(conditions)}

    base_asr = asr["baseline"]
    rand_asrs = np.array([asr[f"rand_{k:02d}"] for k in range(args.n_random)])
    rand_deltas = rand_asrs - base_asr

    print()
    print("=" * 70)
    print(f"  Random-direction ablation ASR distribution -- {args.model}/{args.quantizer}")
    print("=" * 70)
    print(f"  Baseline ASR: {base_asr:.1f}%")
    print(f"  Random trials (K={args.n_random}):")
    print(f"    DeltaASR mean   = {rand_deltas.mean():+.2f}pp")
    print(f"    DeltaASR std    = {rand_deltas.std():.2f}pp")
    print(f"    DeltaASR min    = {rand_deltas.min():+.2f}pp")
    print(f"    DeltaASR max    = {rand_deltas.max():+.2f}pp")
    print(f"    DeltaASR median = {np.median(rand_deltas):+.2f}pp")
    print(f"    |DeltaASR| mean = {np.abs(rand_deltas).mean():.2f}pp")
    print()
    print(f"  Per-trial ASRs: {[f'{a:.1f}' for a in rand_asrs]}")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "n_eval": args.n_eval,
        "n_random": args.n_random,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "seed": SEED,
        "rand_seed": args.rand_seed,
        "baseline_asr": float(base_asr),
        "random_asr_per_trial": rand_asrs.tolist(),
        "random_delta_asr_per_trial": rand_deltas.tolist(),
        "random_delta_asr_mean": float(rand_deltas.mean()),
        "random_delta_asr_std": float(rand_deltas.std()),
        "random_delta_asr_min": float(rand_deltas.min()),
        "random_delta_asr_max": float(rand_deltas.max()),
        "random_abs_delta_asr_mean": float(np.abs(rand_deltas).mean()),
    }
    Path("outputs").mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
