"""Faithfulness check for the routing z: compute the operative layer of Eq. 7 TWO ways
and compare, against the same single-random-direction null the paper uses.

  (A) single-direction : op = argmax_l |<r_hat_q(L*), x(l)>|, one direction (sigma-best
      layer) projected across all layers. This is what my layer_routing.py computes and
      what reproduces the paper's Table 8 (Llama-AQLM-2 z=11.6).
  (B) per-layer (Eq. 7 literal): op = argmax_l |<r_hat_q(l), x(l)>|, each layer's activation
      projected onto THAT layer's own refusal direction r_hat_q(l).

If (A) and (B) agree, the per-layer notation in Eq. 7 is cosmetic and my z is faithful.
Reported raw (paper metric) and normalized (de-confounded) for both. z=(randMean-opq)/randSD.
"""
import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.seeds import set_global_seed, DEFAULT_SEED  # noqa: E402
SEED = set_global_seed(DEFAULT_SEED)
from core.model_loader import load_quantized  # noqa: E402
from core.datasets import load_wildjailbreak_prompts, load_harmless_prompts  # noqa: E402
from diagnostics.layer_routing import (  # noqa: E402
    _get_inner, extract_last_token_residuals, compute_r_hat, per_layer_separation,
)
from diagnostics.routing_random_control import random_unit_vector  # noqa: E402


def _tp(model, ov):
    if ov is not None:
        return ov
    m = model.lower()
    return -5 if "llama" in m else (-3 if "phi" in m else -1)


def _op_single(resid, direction, layers, normalize):
    ops = []
    for pi in range(len(resid[layers[0]])):
        best_l, best_v = layers[0], -1.0
        for L in layers:
            t = resid[L][pi]
            v = abs(float(t @ direction))
            if normalize:
                v /= (float(t.norm()) + 1e-8)
            if v > best_v:
                best_v, best_l = v, L
        ops.append(best_l)
    return ops


def _op_perlayer(resid, rhats, layers, normalize):
    ops = []
    for pi in range(len(resid[layers[0]])):
        best_l, best_v = layers[0], -1.0
        for L in layers:
            t = resid[L][pi]
            v = abs(float(t @ rhats[L]))
            if normalize:
                v /= (float(t.norm()) + 1e-8)
            if v > best_v:
                best_v, best_l = v, L
        ops.append(best_l)
    return ops


def _mode(xs):
    return Counter(xs).most_common(1)[0][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quantizer", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--rand_seed", type=int, default=12345)
    ap.add_argument("--token_pos", type=int, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/routing_eq7_{args.model}_{args.quantizer}.json"
    tp = _tp(args.model, args.token_pos)
    print(f"=== Eq.7 literal check | {args.model}/{args.quantizer} | token_pos={tp} seed={SEED}")

    harmful = load_wildjailbreak_prompts(n_samples=args.n_calib, config="train")
    harmless = load_harmless_prompts(n_samples=args.n_calib)

    q_model, q_tok = load_quantized(args.model, quantizer=args.quantizer)
    dev = next(q_model.parameters()).device
    layers = list(range(len(_get_inner(q_model).layers)))
    h = extract_last_token_residuals(q_model, q_tok, harmful, layers, dev, token_pos=tp)
    s = extract_last_token_residuals(q_model, q_tok, harmless, layers, dev, token_pos=tp)
    hidden = int(q_model.config.hidden_size)
    del q_model; gc.collect(); torch.cuda.empty_cache()

    rhats = {L: compute_r_hat(h[L], s[L]) for L in layers}
    seps = {L: per_layer_separation(h[L], s[L], rhats[L]) for L in layers}
    best = max(layers, key=lambda L: seps[L])
    r_single = rhats[best]
    print(f"  sigma-best layer = {best}")

    rng = np.random.default_rng(args.rand_seed)
    rand_dirs = [random_unit_vector(hidden, rng).cpu().float() for _ in range(args.n_random)]

    out = {"model": args.model, "quantizer": args.quantizer, "token_pos": tp, "seed": SEED,
           "sigma_best_layer": int(best), "n_random": args.n_random}
    print(f"\n  {'metric':<8}{'method':<12}{'opq(mode)':>10}{'randMean':>10}{'randSD':>8}{'z':>8}  verdict")
    for normalize in (False, True):
        tag = "norm" if normalize else "raw"
        rand_modes = [_mode(_op_single(h, r, layers, normalize)) for r in rand_dirs]
        rm, rs = float(np.mean(rand_modes)), float(np.std(rand_modes))
        for method, ops in [("single", _op_single(h, r_single, layers, normalize)),
                            ("perlayer", _op_perlayer(h, rhats, layers, normalize))]:
            om = _mode(ops)
            z = (rm - om) / (rs + 1e-9)
            out[f"{tag}_{method}"] = {"opq_mode": int(om), "rand_mean": rm, "rand_sd": rs, "z": float(z)}
            v = "PASS" if abs(z) >= 3 else "FAIL"
            print(f"  {tag:<8}{method:<12}{om:>10}{rm:>10.1f}{rs:>8.2f}{z:>8.2f}  {v}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
