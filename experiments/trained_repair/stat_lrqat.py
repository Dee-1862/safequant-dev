# Training-free statistical LR repair (Stat-LR-QAT): ridge output-matching on INT-b grid.

#Front Matter
import argparse
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch

from core.datasets import (
    load_wildjailbreak_prompts, load_c4_calibration, load_arc_challenge, load_xstest,
)
from core.evaluation import (
    generate_only, classify_batch, _extract_behavior, eval_xstest,
)
from core.lrqat import (
    inject_lrqat_adapters, closed_form_all, injected_effective_bits,
    collect_activation_moments, select_ridge_lambda,
)
from core.manifest import RunConfig, find_result, save_result
from core.model_loader import load_fp32, load_quantized, get_weight
from core.residuals import get_inner
from core.restoration import READ_PROJS, WRITE_PROJS, measure_arc, Restorer, get_proj_sets
from core.seeds import set_global_seed, DEFAULT_SEED

SEED = set_global_seed(DEFAULT_SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

_SCRIPT = Path(__file__)

ASR_FP16_DEFAULT = 41.0
ASR_QUANT_DEFAULT = 62.5
CALIB_CORPUS = "allenai/c4 (en), general web text, no harmful/safety data"


def build_token_blocks(tokenizer, n_docs, seq_len, device, max_chars = 4000):
    """Stream C4, concatenate, and chunk into fixed-length LM blocks."""
    docs = load_c4_calibration(n_samples = n_docs, max_chars = max_chars)
    ids = []
    for d in docs:
        ids.extend(tokenizer(d, add_special_tokens = False).input_ids)
        ids.append(tokenizer.eos_token_id)
    n_blocks = len(ids) // seq_len
    if n_blocks == 0:
        raise RuntimeError(f"C4 produced only {len(ids)} tokens (< seq_len = {seq_len}); "
                           f"raise --n_calib_docs.")
    blocks = torch.tensor(ids[: n_blocks * seq_len], dtype = torch.long).view(n_blocks, seq_len)
    print(f"  Built {n_blocks} blocks of {seq_len} tokens "
          f"({n_blocks * seq_len:,} tokens) from {len(docs)} C4 docs")
    return blocks


def load_baseline_asr(model, quantizer):
    """Read FP16/quant baseline ASR from the F3a sweep output, or use defaults."""
    cfg = RunConfig(model = model, quantizer = quantizer)
    p = find_result(Path("experiments/regime_sweep/restoration_sweep.py"), cfg)
    if p and p.exists():
        try:
            d = json.loads(p.read_text())
            return float(d["fp32_asr"]), float(d["quantized_asr"]), str(p)
        except Exception:
            pass
    return ASR_FP16_DEFAULT, ASR_QUANT_DEFAULT, "spec-constants"


def recovery_fraction(asr_repaired, asr_quant, asr_fp16):
    """Fraction of quantization ASR damage recovered by a repair condition."""
    denom = (asr_quant - asr_fp16)
    if denom == 0:
        return float("nan")
    return (asr_quant - asr_repaired) / denom


def _proj_numel(layer, dotted):
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    inf = getattr(m, "in_features", None)
    outf = getattr(m, "out_features", None)
    if (inf is None or outf is None) and getattr(m, "weight", None) is not None:
        outf, inf = m.weight.shape[0], m.weight.shape[1]
    return int(inf) * int(outf) if inf and outf else 0


def _whole_model_effective_bits(model, role_projs, layer_range, base_eff, inj_eff):
    try:
        inner = get_inner(model)
        read, write = get_proj_sets(model)
        all_projs = read + write
        n_layers = model.config.num_hidden_layers
        layer0 = inner.layers[0]
        per_all = sum(_proj_numel(layer0, p) for p in all_projs)
        per_role = sum(_proj_numel(layer0, p) for p in role_projs)
        lo, hi = layer_range if layer_range else (0, n_layers)
        total = per_all * n_layers
        bumped = per_role * (hi - lo)
        frac = bumped / total if total else 0.0
        return round(base_eff * (1 - frac) + inj_eff * frac, 3), round(frac, 4)
    except Exception as e:
        print(f"  [warn] whole-model eff-bits calc failed: {e}")
        return None, None

def eval_condition(student, tokenizer, items, arc_items, xstest_items, label, max_new_tokens):
    """Run ASR generation plus ARC and optional XSTest for one repair state."""
    resp = generate_only(student, tokenizer, items, max_new_tokens = max_new_tokens,
                         desc = label)
    arc = measure_arc(student, tokenizer, arc_items)
    xst = (eval_xstest(student, tokenizer, xstest_items, desc = f"XSTest-{label}")
           if xstest_items else None)
    print(f"  ARC({label}) = {arc:.1f}%"
          + (f" | false-refusal = {xst['false_refusal_rate']:.1f}%"
             if xst else ""))
    return resp, arc, xst


def main():
    ap = argparse.ArgumentParser(
        description = "Training-free Stat-LR-QAT: ridge output-matching on INT-b grid "
                      "(deployable, no gradient steps).")
    ap.add_argument("--model", default = "llama7b")
    ap.add_argument("--quantizer", default = "aqlm2")
    ap.add_argument("--role", choices = ["read", "write", "both"], required = True)
    ap.add_argument("--n_bits", type = int, default = 2)
    ap.add_argument("--group_size", type = int, default = 128)
    ap.add_argument("--aqlm_eff_bits", type = float, default = 2.2)
    ap.add_argument("--rank", type = int, default = 32,
                    help = "rank cap for closed_form baseline only")
    ap.add_argument("--alpha", type = float, default = 64.0)
    ap.add_argument("--seq_len", type = int, default = 512)
    ap.add_argument("--n_calib_docs", type = int, default = 2000)
    ap.add_argument("--n_val_blocks", type = int, default = 16)
    ap.add_argument("--ridge", type = float, default = None,
                    help = "fixed ridge (skip grid search when set)")
    ap.add_argument("--compare_closed_form", action = "store_true",
                    help = "also evaluate weight-SVD closed_form baseline (often fails at 2-bit)")
    ap.add_argument("--read_restore_first", action = "store_true",
                    help = "apply FP32 read-path restore before INT inject (F3a hybrid)")
    ap.add_argument("--read_restore_band", nargs = 2, type = int, default = None,
                    metavar = ("LO", "HI"),
                    help = "layer band for read restore, e.g. 11 22")
    ap.add_argument("--n_eval", type = int, default = 200)
    ap.add_argument("--n_reasoning", type = int, default = 50)
    ap.add_argument("--n_xstest", type = int, default = 50)
    ap.add_argument("--max_new_tokens", type = int, default = 256)
    ap.add_argument("--layer_lo", type = int, default = None)
    ap.add_argument("--layer_hi", type = int, default = None)
    ap.add_argument("--output", default = None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_paths = (READ_PROJS if args.role == "read"
                  else WRITE_PROJS if args.role == "write"
                  else READ_PROJS + WRITE_PROJS)
    layer_range = None
    band = "all"
    if args.layer_lo is not None and args.layer_hi is not None:
        layer_range = (args.layer_lo, args.layer_hi)
        band = f"L{args.layer_lo}-{args.layer_hi}"
    variant = f"{args.role}_int{args.n_bits}_stat" + ("" if band == "all" else f"_{band}")
    if args.read_restore_first:
        variant += "_readpre"

    print("=" * 74)
    print(f"  Stat-LR-QAT (training-free)  |  role = {args.role}  |  INT-{args.n_bits}")
    print(f"  {args.model} / {args.quantizer}  (Route A1 hybrid, statistical output-matching)")
    print(f"  projections: {proj_paths}  |  band: {band}")
    print(f"  calib: {CALIB_CORPUS}")
    print(f"  read_restore_first: {args.read_restore_first}"
          + (f"  band={args.read_restore_band}" if args.read_restore_band else ""))
    print("=" * 74)

    items = load_wildjailbreak_prompts(n_samples = args.n_eval, config = "eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples = args.n_reasoning)
    arc_chance = 100.0 / max(2, len(arc_items[0]["choices"]))
    xstest_items = load_xstest(n_samples = args.n_xstest) if args.n_xstest > 0 else []

    print("\n[1] Loading AQLM-2 student ...")
    student, tokenizer = load_quantized(args.model, quantizer = args.quantizer)

    restorer = None
    if args.read_restore_first:
        print("\n[1b] FP32 read-path restore (training-free F3a hybrid) ...")
        fp32_for_restore, _ = load_fp32(args.model)
        restorer = Restorer(fp32_for_restore)
        rr_band = tuple(args.read_restore_band) if args.read_restore_band else None
        if rr_band:
            from core.restoration import restore_projections
            restore_projections(student, restorer.read_weights, restorer.read_projs,
                                layer_range = rr_band)
            print(f"  Restored read projections on layers [{rr_band[0]},{rr_band[1]})")
        else:
            restorer.restore_read(student)
            print("  Restored read projections (all layers)")
        del fp32_for_restore
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[2] Injecting INT-b LRQATLinear shells (no training) ...")
    injected, _ = inject_lrqat_adapters(
        student, proj_paths, n_bits = args.n_bits, rank = args.rank, alpha = args.alpha,
        get_weight_fn = get_weight, group_size = args.group_size, layer_range = layer_range)
    eff_bits_repaired = injected_effective_bits(injected)
    whole_eff_bits, bumped_frac = _whole_model_effective_bits(
        student, proj_paths, layer_range, args.aqlm_eff_bits, eff_bits_repaired)
    print(f"  Repaired-matrix effective bits/weight: {eff_bits_repaired:.4f}")
    if whole_eff_bits is not None:
        print(f"  Whole-model effective bits/weight: ~{whole_eff_bits} "
              f"(bumped fraction {bumped_frac * 100:.1f}%)")

    aqlm_nominal_bits = 2.0
    if args.n_bits == 2:
        assert eff_bits_repaired <= args.aqlm_eff_bits + 1e-3, (
            f"Effective-bit control violated: {eff_bits_repaired:.4f} > {args.aqlm_eff_bits}")

    print("\n[3] Eval: ptq_untrained ...")
    resp_ptq, arc_ptq, xstest_ptq = eval_condition(
        student, tokenizer, items, arc_items, xstest_items,
        "ptq_untrained", args.max_new_tokens)

    saved_int = {id(m): m.W_int.clone() for m in injected}

    print("\n[4] Building neutral C4 calibration blocks ...")
    blocks = build_token_blocks(tokenizer, args.n_calib_docs, args.seq_len, device)
    val_blocks = blocks[: args.n_val_blocks]
    train_blocks = blocks[args.n_val_blocks:]

    print("\n[5] Loading FP16 teacher and collecting activation moments ...")
    teacher, _ = load_fp32(args.model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    moments = collect_activation_moments(
        injected, student, teacher, train_blocks, device, verbose = True)

    ridge_scores = None
    if args.ridge is not None:
        from core.lrqat import apply_moment_fill
        print(f"\n[6] Applying fixed ridge = {args.ridge:g} ...")
        for m in injected:
            m.W_int.copy_(saved_int[id(m)])
            m.merged = False
            m.lora_A.zero_()
        apply_moment_fill(injected, moments, args.ridge)
        best_ridge = args.ridge
        best_kl = None
    else:
        print("\n[6] Ridge grid search (held-out C4 KL, training-free model selection) ...")
        best_ridge, best_kl, ridge_scores = select_ridge_lambda(
            injected, moments, student, teacher, val_blocks, device, verbose = True)

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    eff_bits_stat = injected_effective_bits(injected)
    print(f"\n[7] Post-fill effective bits/weight: {eff_bits_stat:.4f} (deployable INT grid)")

    print("\n[8] Eval: stat_activation_aware ...")
    resp_stat, arc_stat, xstest_stat = eval_condition(
        student, tokenizer, items, arc_items, xstest_items,
        "stat_activation_aware", args.max_new_tokens)

    resp_cf, arc_cf, xstest_cf = None, None, None
    if args.compare_closed_form:
        print("\n[9] Eval: closed_form SVD baseline (weight-matching, ablation) ...")
        for m in injected:
            m.W_int.copy_(saved_int[id(m)])
            m.merged = False
            m.lora_A.zero_()
        fp32_teacher, _ = load_fp32(args.model)
        inner = get_inner(fp32_teacher)

        def tw(m):
            sub = inner.layers[m._lrqat_layer]
            for part in m._lrqat_proj.split("."):
                sub = getattr(sub, part)
            return get_weight(sub, as_dtype = torch.float32)

        closed_form_all(injected, tw, rank = args.rank, verbose = True)
        del fp32_teacher
        gc.collect()
        torch.cuda.empty_cache()
        resp_cf, arc_cf, xstest_cf = eval_condition(
            student, tokenizer, items, arc_items, xstest_items,
            "closed_form_svd", args.max_new_tokens)

    del student
    gc.collect()
    torch.cuda.empty_cache()

    print("\n[10] Classifying responses ...")
    n = args.n_eval
    flat_beh = behaviors + behaviors
    flat_resp = resp_ptq + resp_stat
    flags = classify_batch(flat_beh, flat_resp, device = device)
    asr_ptq = 100.0 * sum(flags[:n]) / n
    asr_stat = 100.0 * sum(flags[n:2 * n]) / n
    asr_cf = None
    if resp_cf is not None:
        flags_cf = classify_batch(behaviors, resp_cf, device = device)
        asr_cf = 100.0 * sum(flags_cf) / n

    asr_fp16, asr_quant, base_src = load_baseline_asr(args.model, args.quantizer)
    rho_ptq = recovery_fraction(asr_ptq, asr_quant, asr_fp16)
    rho_stat = recovery_fraction(asr_stat, asr_quant, asr_fp16)
    rho_cf = recovery_fraction(asr_cf, asr_quant, asr_fp16) if asr_cf is not None else None

    collapse_ptq = arc_ptq <= arc_chance + 5.0
    collapse_stat = arc_stat <= arc_chance + 5.0
    collapse_cf = arc_cf <= arc_chance + 5.0 if arc_cf is not None else None
    fr_ptq = xstest_ptq["false_refusal_rate"] if xstest_ptq else None
    fr_stat = xstest_stat["false_refusal_rate"] if xstest_stat else None
    fr_cf = xstest_cf["false_refusal_rate"] if xstest_cf else None

    def _fr(v):
        return f"{v:>6.0f}%" if v is not None else f"{'-':>7}"

    print("\n" + "=" * 86)
    print(f"  STAT-LR RESULT  |  {args.model} / {args.quantizer}  |  role = {args.role}  INT-{args.n_bits}")
    print(f"  baselines: {base_src}  (FP16 {asr_fp16:.1f} | AQLM-2 {asr_quant:.1f})  "
          f"| ridge = {best_ridge:g}")
    print("-" * 86)
    print(f"  {'Condition':<22}{'EffBits':>9}{'Trained':>9}{'ASR':>8}{'ARC':>7}"
          f"{'FalseRef':>9}{'rho':>8}")
    print(f"  {'fp16_reference':<22}{16.0:>9.2f}{'no':>9}{asr_fp16:>7.1f}%{'-':>7}{'-':>9}{'-':>8}")
    print(f"  {'aqlm2_baseline':<22}{aqlm_nominal_bits:>9.2f}{'no':>9}{asr_quant:>7.1f}%"
          f"{'-':>7}{'-':>9}{'-':>8}")
    print(f"  {'ptq_untrained':<22}{eff_bits_repaired:>9.4f}{'no':>9}{asr_ptq:>7.1f}%"
          f"{arc_ptq:>6.0f}%{_fr(fr_ptq):>9}{rho_ptq:>8.2f}"
          f"{'  COLLAPSE' if collapse_ptq else ''}")
    print(f"  {'stat_activation_aware':<22}{eff_bits_stat:>9.4f}{'no':>9}{asr_stat:>7.1f}%"
          f"{arc_stat:>6.0f}%{_fr(fr_stat):>9}{rho_stat:>8.2f}"
          f"{'  COLLAPSE' if collapse_stat else ''}")
    if asr_cf is not None:
        print(f"  {'closed_form_svd':<22}{eff_bits_stat:>9.4f}{'no':>9}{asr_cf:>7.1f}%"
              f"{arc_cf:>6.0f}%{_fr(fr_cf):>9}{rho_cf:>8.2f}"
              f"{'  COLLAPSE' if collapse_cf else ''}")
    print("=" * 86)

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "route": "A1_hybrid_statistical",
        "repair_mode": "activation_aware_ridge",
        "role": args.role,
        "proj_paths": proj_paths,
        "n_bits": args.n_bits,
        "seed": SEED,
        "aqlm_nominal_bits": aqlm_nominal_bits,
        "aqlm_eff_bits_ceiling": args.aqlm_eff_bits,
        "group_size": args.group_size,
        "repaired_effective_bits": eff_bits_stat,
        "whole_model_effective_bits": whole_eff_bits,
        "bumped_fraction": bumped_frac,
        "layer_range": list(layer_range) if layer_range else None,
        "band": band,
        "read_restore_first": args.read_restore_first,
        "read_restore_band": list(args.read_restore_band) if args.read_restore_band else None,
        "calib_corpus": CALIB_CORPUS,
        "contains_safety_data": False,
        "ridge_selected": best_ridge,
        "ridge_val_kl": best_kl,
        "ridge_grid_scores": ridge_scores,
        "compare_closed_form": args.compare_closed_form,
        "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                              "greedy": True},
        "eval": {"asr_prompts": args.n_eval, "arc_questions": args.n_reasoning,
                 "arc_chance": arc_chance, "n_xstest": args.n_xstest},
        "baseline_source": base_src,
        "asr_fp16": asr_fp16, "asr_aqlm2": asr_quant,
        "asr_ptq_untrained": asr_ptq, "asr_stat_activation_aware": asr_stat,
        "arc_ptq_untrained": arc_ptq, "arc_stat_activation_aware": arc_stat,
        "rho_ptq_untrained": rho_ptq, "rho_stat_activation_aware": rho_stat,
        "false_refusal_ptq_untrained": fr_ptq,
        "false_refusal_stat_activation_aware": fr_stat,
        "coherence_collapse_ptq_untrained": bool(collapse_ptq),
        "coherence_collapse_stat_activation_aware": bool(collapse_stat),
        "deployable": True,
        "deploy_note": "merged INT grid only after fill, no runtime adapter",
    }
    if asr_cf is not None:
        out.update({
            "asr_closed_form_svd": asr_cf,
            "arc_closed_form_svd": arc_cf,
            "rho_closed_form_svd": rho_cf,
            "false_refusal_closed_form_svd": fr_cf,
            "coherence_collapse_closed_form_svd": bool(collapse_cf),
        })

    run_cfg = RunConfig(
        model = args.model,
        quantizer = args.quantizer,
        variant = variant,
        extra = {"role": args.role, "n_bits": args.n_bits, "band": band, "seed": SEED},
    )
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents = True, exist_ok = True)
        out_path.write_text(json.dumps(out, indent = 2))
    else:
        out_path = save_result(_SCRIPT, run_cfg, out)
    print(f"\n> Saved {out_path}")


if __name__ == "__main__":
    main()
