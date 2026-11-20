# Table 1: per-layer probe accuracy (a_min).
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from core.datasets import load_harmless_prompts, load_wildjailbreak_prompts
from core.manifest import RunConfig, assert_new_result, result_path, save_result
from core.model_loader import load_fp32, load_quantized
from core.probes import fit_probe_per_layer, permute_labels_by_layer
from core.residuals import extract_last_token_residuals, get_inner
from core.seeds import DEFAULT_SEED, set_global_seed

SEED = set_global_seed(DEFAULT_SEED)
_SCRIPT = Path(__file__)


def main():
    # Parse CLI arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--n_calib", type = int, default = 200)
    ap.add_argument("--n_folds", type = int, default = 5)
    ap.add_argument("--probe_type", default = "linear", choices = ["linear", "mlp"])
    # X3 control: permute harmful/harmless labels (seed 42) before fitting. Expect ~50%.
    ap.add_argument("--shuffle_labels", action = "store_true")
    # Opt-in output suffix (e.g. smoke runs: --tag smoke -> ..._shuffled_smoke.json)
    ap.add_argument("--tag", default = None)
    # X3 matched pair: fit the probe twice on the SAME extracted activations, once with the
    # true labels and once with labels permuted (seed 42), identical layers / folds /
    # regularisation; selectivity = real - shuffled from that pair. -> ..._paired[_<tag>].json
    ap.add_argument("--paired", action = "store_true")
    # Paired mode only: reuse the FP32 pair (real + shuffled per-layer results) computed
    # once per model family. Same computation; skips reloading FP32 for every cell.
    ap.add_argument("--fp32_cache_dir", default = None)
    args = ap.parse_args()
    if args.paired and args.shuffle_labels:
        ap.error("--paired already includes the shuffled-label fit; drop --shuffle_labels")
    if args.paired:
        return main_paired(args)

    # Shuffled-label runs get their own filename (never overwrite the real probe JSON)
    variant = None if args.probe_type == "linear" else args.probe_type
    if args.shuffle_labels:
        variant = "shuffled" if variant is None else f"{variant}_shuffled"
    if args.tag:
        variant = f"{variant}_{args.tag}" if variant else args.tag
    if args.shuffle_labels or args.tag:
        assert_new_result(
            _SCRIPT, RunConfig(model = args.model, quantizer = args.quantizer, variant = variant)
        )

    # Load calibration harmful and harmless prompts
    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # Fit FP32 cross-validated probes
    fp32_model, fp32_tok = load_fp32(args.model)
    device = next(fp32_model.parameters()).device
    layers = list(range(len(get_inner(fp32_model).layers)))
    fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
    fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
    if args.shuffle_labels:
        fp_h, fp_s = permute_labels_by_layer(fp_h, fp_s, seed = SEED)
    fp_probes = fit_probe_per_layer(fp_h, fp_s, n_folds = args.n_folds, probe_type = args.probe_type)
    del fp32_model
    gc.collect()
    torch.cuda.empty_cache()

    # Fit quantized cross-validated probes
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    q_device = next(q_model.parameters()).device
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    if args.shuffle_labels:
        q_h, q_s = permute_labels_by_layer(q_h, q_s, seed = SEED)
    q_probes = fit_probe_per_layer(q_h, q_s, n_folds = args.n_folds, probe_type = args.probe_type)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()

    # Summarize accuracy deltas and large drops
    fp_accs = [fp_probes[L]["mean_acc"] for L in layers]
    q_accs = [q_probes[L]["mean_acc"] for L in layers]
    layers_drop = [L for L in layers if (q_probes[L]["mean_acc"] - fp_probes[L]["mean_acc"]) < -0.05]

    # Persist per-layer accuracies
    cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = variant,
        extra = {
            "n_calib": args.n_calib,
            "n_folds": args.n_folds,
            "probe_type": args.probe_type,
            "seed": SEED,
            **({"shuffle_labels": True} if args.shuffle_labels else {}),
        },
    )
    data = {
        "n_layers": len(layers),
        "fp32_probe_per_layer": {str(L): fp_probes[L] for L in layers},
        "quant_probe_per_layer": {str(L): q_probes[L] for L in layers},
        "summary": {
            "fp32_mean_acc": float(np.mean(fp_accs)),
            "quant_mean_acc": float(np.mean(q_accs)),
            "fp32_min_acc": float(np.min(fp_accs)),
            "quant_min_acc": float(np.min(q_accs)),
            "fp32_argmax_layer": int(np.argmax(fp_accs)),
            "quant_argmax_layer": int(np.argmax(q_accs)),
            "layers_with_drop_gt_5pp": layers_drop,
            "n_layers_with_drop": len(layers_drop),
        },
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


def _fit_pair(h, s, n_folds, probe_type):
    """
    Real-label and shuffled-label (seed 42) probes on the same activations.

    Inputs:
        - h (dict): layer -> harmful residuals
        - s (dict): layer -> harmless residuals
        - n_folds (int): Cross-validation folds
        - probe_type (str): Probe family

    Outputs:
        - real (dict): Per-layer fit with the true labels
        - shuf (dict): Per-layer fit with permuted labels
    """
    real = fit_probe_per_layer(h, s, n_folds = n_folds, probe_type = probe_type)
    hp, sp = permute_labels_by_layer(h, s, seed = SEED)
    shuf = fit_probe_per_layer(hp, sp, n_folds = n_folds, probe_type = probe_type)
    return real, shuf


def _pair_summary(real, shuf, layers):
    """
    Summary of a real / shuffled probe pair over layers.

    Inputs:
        - real (dict): Per-layer real-label fit
        - shuf (dict): Per-layer shuffled-label fit
        - layers (list[int]): Layers

    Outputs:
        - summary (dict): Mean / extreme accuracies with their layers and selectivity
    """
    r = np.array([real[L]["mean_acc"] for L in layers])
    s = np.array([shuf[L]["mean_acc"] for L in layers])
    sel = r - s
    return {
        "real_mean_acc": float(r.mean()), "real_min_acc": float(r.min()),
        "real_min_layer": int(layers[int(r.argmin())]),
        "shuffled_mean_acc": float(s.mean()), "shuffled_max_acc": float(s.max()),
        "shuffled_max_layer": int(layers[int(s.argmax())]),
        "selectivity_mean": float(r.mean() - s.mean()),
        "selectivity_min_layer": float(sel.min()),
        "selectivity_min_layer_idx": int(layers[int(sel.argmin())]),
    }


def _ints(d):
    """
    Convert JSON string layer keys back to ints.

    Inputs:
        - d (dict): str(layer) -> value

    Outputs:
        - out (dict): int(layer) -> value
    """
    return {int(k): v for k, v in d.items()}


def main_paired(args):
    """
    --paired run: real and shuffled-label probes on the same FP32 and quantized activations.

    Inputs:
        - args (Namespace): Parsed command-line arguments

    Outputs:
        - None (writes probe_accuracy_<model>_<quant>_paired[_<tag>].json)
    """
    variant = "paired" if args.probe_type == "linear" else f"{args.probe_type}_paired"
    if args.tag:
        variant = f"{variant}_{args.tag}"
    assert_new_result(_SCRIPT, RunConfig(model = args.model, quantizer = args.quantizer, variant = variant))

    harmful = load_wildjailbreak_prompts(n_samples = args.n_calib, config = "train")
    harmless = load_harmless_prompts(n_samples = args.n_calib)

    # FP32 pair, from the per-family cache when it holds this exact setting
    key = hashlib.sha256(json.dumps(
        {"model": args.model, "n_calib": args.n_calib, "n_folds": args.n_folds,
         "probe_type": args.probe_type, "seed": SEED, "harmful": harmful, "harmless": harmless},
        sort_keys = True, default = str).encode()).hexdigest()[:16]
    cache = (Path(args.fp32_cache_dir) / f"x3_fp32_pair_{args.model}_{key}.json"
             if args.fp32_cache_dir else None)
    if cache is not None and cache.exists():
        c = json.loads(cache.read_text())
        layers = list(range(c["n_layers"]))
        fp_real, fp_shuf = _ints(c["real_per_layer"]), _ints(c["shuffled_per_layer"])
        fp_source = {"from_cache": str(cache), "computed_by_job": c.get("job")}
        print(f"FP32 pair from cache {cache} (computed by job {c.get('job')})")
    else:
        fp32_model, fp32_tok = load_fp32(args.model)
        device = next(fp32_model.parameters()).device
        layers = list(range(len(get_inner(fp32_model).layers)))
        fp_h = extract_last_token_residuals(fp32_model, fp32_tok, harmful, layers, device)
        fp_s = extract_last_token_residuals(fp32_model, fp32_tok, harmless, layers, device)
        del fp32_model
        gc.collect()
        torch.cuda.empty_cache()
        fp_real, fp_shuf = _fit_pair(fp_h, fp_s, args.n_folds, args.probe_type)
        fp_source = {"computed_here": True, "cache": str(cache) if cache else None}
        if cache is not None:
            cache.parent.mkdir(parents = True, exist_ok = True)
            tmp = cache.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps({
                "model": args.model, "key": key, "n_layers": len(layers),
                "n_calib": args.n_calib, "n_folds": args.n_folds, "probe_type": args.probe_type,
                "seed": SEED, "job": os.environ.get("SLURM_JOB_ID"),
                "real_per_layer": {str(L): fp_real[L] for L in layers},
                "shuffled_per_layer": {str(L): fp_shuf[L] for L in layers}}))
            tmp.replace(cache)

    # Quantized pair, always computed for this cell
    q_model, q_tok = load_quantized(args.model, quantizer = args.quantizer)
    q_device = next(q_model.parameters()).device
    assert len(get_inner(q_model).layers) == len(layers), "layer count differs from FP32"
    q_h = extract_last_token_residuals(q_model, q_tok, harmful, layers, q_device)
    q_s = extract_last_token_residuals(q_model, q_tok, harmless, layers, q_device)
    del q_model
    gc.collect()
    torch.cuda.empty_cache()
    q_real, q_shuf = _fit_pair(q_h, q_s, args.n_folds, args.probe_type)

    fp_sum, q_sum = _pair_summary(fp_real, fp_shuf, layers), _pair_summary(q_real, q_shuf, layers)

    # Consistency check: this run's real-label a_min vs Table 1 (the original probe_accuracy run)
    t1_path = result_path(_SCRIPT, RunConfig(model = args.model, quantizer = args.quantizer))
    t1 = None
    if t1_path.exists():
        t1 = json.loads(t1_path.read_text()).get("summary", {}).get("quant_min_acc")
    print(f"[{args.model}/{args.quantizer}] real a_min {q_sum['real_min_acc']:.3f} (mean {q_sum['real_mean_acc']:.3f}) "
          f"| Table 1 a_min {('%.3f' % t1) if t1 is not None else 'n/a'}"
          + (f" (diff {100 * (q_sum['real_min_acc'] - t1):+.1f} pt)" if t1 is not None else ""))
    print(f"[{args.model}/{args.quantizer}] shuffled mean {q_sum['shuffled_mean_acc']:.3f} max {q_sum['shuffled_max_acc']:.3f} "
          f"| selectivity mean {q_sum['selectivity_mean']:.3f} min-layer {q_sum['selectivity_min_layer']:.3f}")

    cfg = RunConfig(
        model = args.model, quantizer = args.quantizer, variant = variant,
        extra = {"n_calib": args.n_calib, "n_folds": args.n_folds, "probe_type": args.probe_type,
                 "seed": SEED, "paired_shuffle_seed": SEED, "mode": "paired real + shuffled",
                 "fp32_source": fp_source},
    )
    data = {
        "n_layers": len(layers),
        "quant": {"real_per_layer": {str(L): q_real[L] for L in layers},
                  "shuffled_per_layer": {str(L): q_shuf[L] for L in layers}, "summary": q_sum},
        "fp32": {"real_per_layer": {str(L): fp_real[L] for L in layers},
                 "shuffled_per_layer": {str(L): fp_shuf[L] for L in layers}, "summary": fp_sum},
        "table1_check": {"table1_file": str(t1_path), "table1_a_min": t1,
                         "delta_real_a_min": (q_sum["real_min_acc"] - t1) if t1 is not None else None},
        "summary": {f"quant_{k}": v for k, v in q_sum.items()} | {f"fp32_{k}": v for k, v in fp_sum.items()},
    }
    path = save_result(_SCRIPT, cfg, data)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
