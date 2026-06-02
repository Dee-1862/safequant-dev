import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import load_fp32, load_quantized
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts


from core.seeds import set_global_seed, DEFAULT_SEED, fixed_rng, torch_gen
SEED = set_global_seed(DEFAULT_SEED)


def _get_inner(model):
    inner = model.model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def format_prompt(p, tok):
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        return p


def extract_last_token_residuals(model, tokenizer, prompts, layers, device):
    """Returns {layer: list of (hidden,) CPU tensors, one per prompt}."""
    inner = _get_inner(model)
    buf = {}

    def make_hook(l):
        def hook(module, inputs, output):
            r = output[0] if isinstance(output, (tuple, list)) else output
            if r.dim() == 3:
                buf[l] = r[:, -1, :].detach().cpu().float().squeeze(0)
        return hook

    handles = [inner.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    out = {l: [] for l in layers}
    try:
        for p in prompts:
            text = format_prompt(p, tokenizer)
            inputs = tokenizer(text, return_tensors="pt",
                               truncation=True, max_length=512).to(device)
            buf.clear()
            with torch.no_grad():
                _ = model(**inputs)
            for l, r in buf.items():
                out[l].append(r)
    finally:
        for h in handles:
            h.remove()
    return out


def compute_r_hat(harmful_lasts, harmless_lasts):
    h = torch.stack(harmful_lasts).mean(0)
    s = torch.stack(harmless_lasts).mean(0)
    r = h - s
    n = r.norm()
    return r / n if n > 1e-6 else r


def per_layer_separation(harmful_lasts, harmless_lasts, r_hat):
    h_scores = np.array([float(t @ r_hat) for t in harmful_lasts], dtype=np.float64)
    s_scores = np.array([float(t @ r_hat) for t in harmless_lasts], dtype=np.float64)
    pooled = np.sqrt((h_scores.var() + s_scores.var()) / 2 + 1e-12)
    sep = (h_scores.mean() - s_scores.mean()) / pooled
    return float(sep), float(h_scores.mean()), float(s_scores.mean())


def run_for_model(model, tokenizer, harmful, harmless, layers, device, label):
    print(f"\n[{label}] extracting last-token residuals across {len(layers)} layers")
    print(f"    forwarding harmful ({len(harmful)} prompts)...")
    h_caps = extract_last_token_residuals(model, tokenizer, harmful, layers, device)
    print(f"    forwarding harmless ({len(harmless)} prompts)...")
    s_caps = extract_last_token_residuals(model, tokenizer, harmless, layers, device)
    r_hats, seps = {}, {}
    for L in layers:
        r = compute_r_hat(h_caps[L], s_caps[L])
        r_hats[L] = r
        sep, hm, sm = per_layer_separation(h_caps[L], s_caps[L], r)
        seps[L] = {"separation_sigma": sep,
                   "harmful_mean": hm, "harmless_mean": sm}
    return r_hats, seps


def cos_sim(a, b):
    a = a.float(); b = b.float()
    na, nb = a.norm(), b.norm()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))


def analyze_and_save(quantizer, fp32_r_hats, fp32_seps, q_r_hats, q_seps,
                     layers, n_calib, model_name, output_path):
    cos_per_layer = {L: cos_sim(fp32_r_hats[L], q_r_hats[L]) for L in layers}

    print("\n" + "=" * 70)
    print(f"  Per-layer direction drift (FP32 vs {quantizer})")
    print("=" * 70)
    print(f"  layer   cos_sim   sep_FP32   sep_{quantizer}   Deltasep")
    for L in layers:
        ds = q_seps[L]["separation_sigma"] - fp32_seps[L]["separation_sigma"]
        print(f"  {L:5d}   {cos_per_layer[L]:7.4f}   "
              f"{fp32_seps[L]['separation_sigma']:8.3f}   "
              f"{q_seps[L]['separation_sigma']:8.3f}   {ds:+7.3f}")

    cos_vals = list(cos_per_layer.values())
    print(f"\n  cos_sim   min={min(cos_vals):.4f}  "
          f"median={np.median(cos_vals):.4f}  max={max(cos_vals):.4f}")
    weak = [L for L, c in cos_per_layer.items() if c < 0.95]
    very_weak = [L for L, c in cos_per_layer.items() if c < 0.80]
    print(f"  layers with cos_sim < 0.95: {weak}")
    print(f"  layers with cos_sim < 0.80: {very_weak}")

    fp32_seps_arr = np.array([fp32_seps[L]["separation_sigma"] for L in layers])
    cos_arr = np.array([cos_per_layer[L] for L in layers])
    if fp32_seps_arr.std() > 1e-9 and cos_arr.std() > 1e-9:
        pearson = float(np.corrcoef(fp32_seps_arr, cos_arr)[0, 1])
    else:
        pearson = 0.0
    print(f"\n  Pearson(FP32 separation sigma, cos_sim) = {pearson:+.3f}")
    print(f"    negative -> cos_sim drops at high-separation layers (hypothesis)")

    out = {
        "model": model_name,
        "quantizer": quantizer,
        "n_calib": n_calib,
        "n_layers": len(layers),
        "cos_sim_per_layer": {str(L): cos_per_layer[L] for L in layers},
        "fp32_separation_per_layer": {str(L): fp32_seps[L] for L in layers},
        f"{quantizer}_separation_per_layer": {str(L): q_seps[L] for L in layers},
        "summary": {
            "cos_sim_min": float(min(cos_vals)),
            "cos_sim_median": float(np.median(cos_vals)),
            "cos_sim_max": float(max(cos_vals)),
            "layers_below_0_95": weak,
            "layers_below_0_80": very_weak,
            "pearson_sep_vs_cos": pearson,
        }
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {output_path}")
    return cos_per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2",
                    help="single quantizer or comma-separated list "
                         "(e.g. 'gptq2,gptq3,gptq4')")
    ap.add_argument("--n_calib", type=int, default=50)
    ap.add_argument("--output", default=None,
                    help="output path (only valid for single quantizer)")
    args = ap.parse_args()

    quantizers = [q.strip() for q in args.quantizer.split(",") if q.strip()]
    if len(quantizers) > 1 and args.output is not None:
        raise ValueError("--output cannot be set when sweeping multiple quantizers; "
                         "outputs auto-named outputs/direction_drift_<model>_<q>.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print(f"  Direction drift: {args.model} FP32 vs {quantizers}")
    print(f"  n_calib={args.n_calib}")
    print("=" * 70)

    print("Loading calibration prompts...")
    harmful_items = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmful = []
    for it in harmful_items:
        if isinstance(it, dict):
            harmful.append(it.get("behavior") or it.get("prompt") or "")
        else:
            harmful.append(str(it))
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    fp32_model, tokenizer = load_fp32(args.model)
    n_layers = fp32_model.config.num_hidden_layers
    layers = list(range(n_layers))
    print(f"  num_hidden_layers: {n_layers}")

    fp32_r_hats, fp32_seps = run_for_model(
        fp32_model, tokenizer, harmful, harmless, layers, device, "FP32")

    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    cos_summary = {}
    for q in quantizers:
        print(f"\n{'#' * 70}")
        print(f"# Loading {q} model...")
        print(f"{'#' * 70}")
        q_model, _ = load_quantized(args.model, quantizer=q)
        q_r_hats, q_seps = run_for_model(
            q_model, tokenizer, harmful, harmless, layers, device, q)

        del q_model
        gc.collect()
        torch.cuda.empty_cache()

        out_path = (args.output if args.output is not None
                    else f"outputs/direction_drift_{args.model}_{q}.json")
        cos_per_layer = analyze_and_save(
            q, fp32_r_hats, fp32_seps, q_r_hats, q_seps,
            layers, args.n_calib, args.model, out_path)
        cos_summary[q] = cos_per_layer

    if len(quantizers) > 1:
        print("\n" + "=" * 70)
        print(f"  Cross-quantizer summary (cos_sim FP32 vs Q, per layer)")
        print("=" * 70)
        header = "  layer   " + "   ".join(f"{q:>8s}" for q in quantizers)
        print(header)
        for L in layers:
            row = "   ".join(f"{cos_summary[q][L]:8.4f}" for q in quantizers)
            print(f"  {L:5d}   {row}")
        print()
        for q in quantizers:
            vals = list(cos_summary[q].values())
            print(f"  {q:>8s}: min={min(vals):.4f}  "
                  f"median={np.median(vals):.4f}  max={max(vals):.4f}")


if __name__ == "__main__":
    main()
