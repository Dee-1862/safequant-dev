"""
Headline-cell full evaluation: ASR + ARC + XSTest over the 5 restoration
conditions {FP, quant, +write, +read, +both}, with the statistical machinery a
causal recovery claim needs at AAAI -- paired McNemar on condition flips and
bootstrap CIs on the recovery percentages.

This is Tier 1 of the paper plan:
  1.1  XSTest false-refusal on every condition -- proves the recovered safety is
       genuine refusal, not the model refusing everything (the reviewer's first
       question about any safety-recovery result).
  1.2  ASR at n>=400 + McNemar(paired flips) + bootstrap CIs on recovery %.

Reuses the restoration helpers from readwrite_restoration (single source of truth
for WRITE/READ_PROJS and the weight-swap), so the conditions are identical to the
ASR/ARC run; only the metrics are richer. Raw responses are checkpointed after
each condition so a late failure never discards the expensive generation.

Usage:
  python -m diagnostics.headline_eval_full --model llama7b --quantizer aqlm2 \
      --n_eval 400 --n_reasoning 100 --output outputs/headline_full_llama7b_aqlm2.json
"""

import argparse
import json
import sys
import gc
import math
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.seeds import set_global_seed, DEFAULT_SEED  # noqa: E402
from core.model_loader import load_fp32, load_quantized  # noqa: E402
from core.datasets import (  # noqa: E402
    load_wildjailbreak_prompts, load_arc_challenge, load_xstest,
)
from core.evaluation import (  # noqa: E402
    generate_only, classify_batch, _extract_behavior, eval_xstest,
)
from diagnostics.readwrite_restoration import (  # noqa: E402
    WRITE_PROJS, READ_PROJS, get_proj_sets, extract_fp32_weights, restore_projections, measure_arc,
)


# --------------------------------------------------------------------------- #
# statistics (dependency-free)
# --------------------------------------------------------------------------- #
def mcnemar_exact(flags_a, flags_b):
    """Exact two-sided McNemar on paired binary flags (1=complied).
    b = A complied & B refused; c = A refused & B complied."""
    b = sum(1 for x, y in zip(flags_a, flags_b) if x and not y)
    c = sum(1 for x, y in zip(flags_a, flags_b) if (not x) and y)
    nbc = b + c
    if nbc == 0:
        p = 1.0
    else:
        k = min(b, c)
        p = 2.0 * sum(math.comb(nbc, i) for i in range(k + 1)) * (0.5 ** nbc)
        p = min(1.0, p)
    return {"b": b, "c": c, "n_discordant": nbc, "p_value": p}


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def bootstrap_recovery(flags_by_cond, qtz, n_boot=2000, seed=0):
    """Paired bootstrap (resample prompts, same indices across conditions) ->
    95% CIs on read/write/both recovery % and the read-minus-write margin."""
    rng = np.random.default_rng(seed)
    arrs = {c: np.asarray(v, dtype=float) for c, v in flags_by_cond.items()}
    n = len(next(iter(arrs.values())))
    rec = {"read": [], "write": [], "both": [], "read_minus_write": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        asr = {c: arrs[c][idx].mean() * 100 for c in arrs}
        dmg = asr[qtz] - asr["fp32"]
        if dmg == 0:
            continue
        rr = (asr[qtz] - asr["read_restored"]) / dmg * 100
        rw = (asr[qtz] - asr["write_restored"]) / dmg * 100
        rb = (asr[qtz] - asr["both_restored"]) / dmg * 100
        rec["read"].append(rr); rec["write"].append(rw); rec["both"].append(rb)
        rec["read_minus_write"].append(rr - rw)

    def ci(a):
        a = np.asarray(a)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    return {k: ci(v) for k, v in rec.items()}


# --------------------------------------------------------------------------- #
# per-condition evaluation
# --------------------------------------------------------------------------- #
def eval_condition(model, tokenizer, name, wj_items, arc_items, xstest_items,
                   asr_max_new_tokens):
    """Generate ASR responses, measure ARC, and run XSTest while the model is
    loaded. Returns (asr_responses, arc_pct, xstest_dict)."""
    resp = generate_only(model, tokenizer, wj_items,
                         max_new_tokens=asr_max_new_tokens, desc=f"{name}/ASR")
    arc = measure_arc(model, tokenizer, arc_items)
    xs = eval_xstest(model, tokenizer, xstest_items, desc=f"{name}/XSTest")
    print(f"  [{name}] ARC={arc:.0f}%  XSTest false-refusal={100*xs['false_refusal_rate']:.1f}%")
    return resp, arc, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--n_eval", type=int, default=400, help="WildJailbreak prompts for ASR (>=400)")
    ap.add_argument("--n_reasoning", type=int, default=100, help="ARC-Challenge questions")
    ap.add_argument("--asr_max_new_tokens", type=int, default=256)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    set_global_seed(DEFAULT_SEED)
    qtz = args.quantizer
    out = args.output or f"outputs/headline_full_{args.model}_{qtz}.json"
    raw_path = out.replace(".json", "_raw.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Headline full eval: {args.model} / {qtz}")
    print(f"  ASR n={args.n_eval} | ARC n={args.n_reasoning} | XSTest | boot={args.n_boot}")
    print("=" * 70)

    wj_items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in wj_items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    xstest_items = load_xstest()
    n_safe = sum(1 for it in xstest_items if it["type"] == "safe")
    print(f"  WJ={len(wj_items)}  ARC={len(arc_items)}  XSTest={len(xstest_items)} ({n_safe} safe)")

    all_resp, all_arc, all_xs = {}, {}, {}

    def checkpoint():
        with open(raw_path, "w") as f:
            json.dump({"responses": all_resp, "arc": all_arc,
                       "xstest": {k: {kk: vv for kk, vv in v.items() if kk != "per_prompt"}
                                  for k, v in all_xs.items()}}, f, indent=2)

    # ---- Condition 1: FP ----
    print("\n[1/5] FP baseline")
    fp_model, tok = load_fp32(args.model)
    # Use the model's actual projection sets (Phi-3 fuses qkv_proj / gate_up_proj),
    # else the read/write swap would target non-existent Llama-style names.
    READ_P, WRITE_P = get_proj_sets(fp_model)
    print(f"  proj sets: READ={READ_P}  WRITE={WRITE_P}")
    write_w = extract_fp32_weights(fp_model, WRITE_P)
    read_w = extract_fp32_weights(fp_model, READ_P)
    all_resp["fp32"], all_arc["fp32"], all_xs["fp32"] = eval_condition(
        fp_model, tok, "fp32", wj_items, arc_items, xstest_items, args.asr_max_new_tokens)
    del fp_model; gc.collect(); torch.cuda.empty_cache(); checkpoint()

    # ---- the four quantized conditions ----
    plan = [
        (qtz, [], []),
        ("write_restored", [(write_w, WRITE_P)], []),
        ("read_restored", [(read_w, READ_P)], []),
        ("both_restored", [(write_w, WRITE_P), (read_w, READ_P)], []),
    ]
    for i, (name, restores, _) in enumerate(plan, start=2):
        print(f"\n[{i}/5] {name}")
        q_model, _ = load_quantized(args.model, quantizer=qtz)
        for weights, projs in restores:
            restore_projections(q_model, weights, projs)
        all_resp[name], all_arc[name], all_xs[name] = eval_condition(
            q_model, tok, name, wj_items, arc_items, xstest_items, args.asr_max_new_tokens)
        del q_model; gc.collect(); torch.cuda.empty_cache(); checkpoint()

    del write_w, read_w; gc.collect()

    # ---- classify all ASR responses in one judge session ----
    print("\n[judge] classifying all ASR responses (HarmBench)")
    conditions = ["fp32", qtz, "write_restored", "read_restored", "both_restored"]
    flat_beh, flat_resp = [], []
    for c in conditions:
        flat_beh.extend(behaviors); flat_resp.extend(all_resp[c])
    flat_flags = classify_batch(flat_beh, flat_resp, device="cuda")

    n = args.n_eval
    flags_by_cond = {c: [bool(x) for x in flat_flags[i*n:(i+1)*n]]
                     for i, c in enumerate(conditions)}
    asr = {c: 100.0 * sum(flags_by_cond[c]) / n for c in conditions}

    fp_asr, q_asr = asr["fp32"], asr[qtz]
    damage = q_asr - fp_asr
    read_rec = q_asr - asr["read_restored"]
    write_rec = q_asr - asr["write_restored"]
    both_rec = q_asr - asr["both_restored"]

    # ---- stats ----
    mcnemar = {
        f"{qtz}_vs_read": mcnemar_exact(flags_by_cond[qtz], flags_by_cond["read_restored"]),
        f"{qtz}_vs_write": mcnemar_exact(flags_by_cond[qtz], flags_by_cond["write_restored"]),
        "read_vs_write": mcnemar_exact(flags_by_cond["read_restored"], flags_by_cond["write_restored"]),
        f"fp32_vs_read": mcnemar_exact(flags_by_cond["fp32"], flags_by_cond["read_restored"]),
        f"fp32_vs_both": mcnemar_exact(flags_by_cond["fp32"], flags_by_cond["both_restored"]),
    }
    boot = bootstrap_recovery(flags_by_cond, qtz, n_boot=args.n_boot)

    # XSTest false-refusal + Wilson CI per condition
    xstest_summary = {}
    for c in conditions:
        xs = all_xs[c]
        k = xs["n_refusal_on_safe"]; ns = xs["n_safe"]
        xstest_summary[c] = {
            "false_refusal_rate": xs["false_refusal_rate"],
            "false_refusal_ci": wilson_ci(k, ns),
            "n_refusal_on_safe": k, "n_safe": ns,
            "refusal_rate_on_unsafe": xs["refusal_rate_on_unsafe"],
        }

    # ---- report ----
    print("\n" + "=" * 78)
    print(f"  HEADLINE FULL EVAL -- {args.model}/{qtz}   (ASR n={n}, boot={args.n_boot})")
    print("=" * 78)
    print(f"{'condition':<16}{'ASR':>7}{'recov%':>9}{'ARC':>7}{'XSTest-FR':>11}{'  XSTest 95% CI'}")
    for c in conditions:
        rec = "" if c in ("fp32", qtz) else f"{100*(q_asr-asr[c])/damage:.0f}%" if damage else "n/a"
        fr = xstest_summary[c]["false_refusal_rate"]  # already a percentage
        lo, hi = xstest_summary[c]["false_refusal_ci"]
        print(f"{c:<16}{asr[c]:>6.1f}%{rec:>9}{all_arc[c]:>6.0f}%{fr:>10.1f}%   [{100*lo:.1f},{100*hi:.1f}]")
    print("-" * 78)
    print(f"damage={damage:+.1f}pp  read_recov={read_rec:+.1f}pp ({100*read_rec/damage if damage else 0:.0f}%)  "
          f"write_recov={write_rec:+.1f}pp ({100*write_rec/damage if damage else 0:.0f}%)")
    print(f"bootstrap 95% CI  read_recov%={boot['read']}  write_recov%={boot['write']}  "
          f"read-write margin={boot['read_minus_write']}")
    print("McNemar (b=left-complied/right-refused, c=opposite, p=exact two-sided):")
    for k, v in mcnemar.items():
        print(f"  {k:<22} b={v['b']:>3} c={v['c']:>3} p={v['p_value']:.2e}")
    print("\nXSTest read-back gate: read_restored false-refusal vs FP --",
          f"{xstest_summary['read_restored']['false_refusal_rate']:.1f}% vs "
          f"{xstest_summary['fp32']['false_refusal_rate']:.1f}% "
          "(near FP => genuine refusal recovery; >> FP => over-refusal)")

    result = {
        "model": args.model, "quantizer": qtz, "n_eval": n,
        "n_reasoning": args.n_reasoning, "asr": asr, "arc": all_arc,
        "damage_pp": damage,
        "recovery_pp": {"read": read_rec, "write": write_rec, "both": both_rec},
        "recovery_pct": {
            "read": 100 * read_rec / damage if damage else 0,
            "write": 100 * write_rec / damage if damage else 0,
            "both": 100 * both_rec / damage if damage else 0,
        },
        "recovery_ci_95": boot,
        "mcnemar": mcnemar,
        "xstest": xstest_summary,
        "flags_by_cond": flags_by_cond,
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
