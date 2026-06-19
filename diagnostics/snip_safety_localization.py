"""
SNIP safety-critical weight localization (Q-Resafe convergent-evidence pass).

This is an INDEPENDENT reimplementation of the gradient-saliency identification
that Q-Resafe's Algorithm 1 *describes* but its released code does NOT contain
(the repo ships vanilla alignment-handbook PEFT-DPO; no SNIP, no mask). Because
it is a clean-room reimplementation under a different method family, it is a
genuinely independent localization signal -- exactly what makes agreement with
the activation-swap localization "convergent evidence" rather than circular.

What it does (no training, gradient pass only):
  For the FP16/bf16 pre-quantization model, score every weight w by its SNIP
  connection sensitivity to a SAFETY loss on AdvBench harmful prompts,
        s(w) = | w * dL/dw |,   averaged over prompts,
  then ask WHERE the safety-critical weights concentrate: reader-side
  {q,k,v,gate,up} vs write-side {o,down}, and early/middle/late layers.

Two reported localization metrics:
  (1) saliency-MASS enrichment (threshold-free, primary):
        enrichment(group) = (saliency mass in group / total mass)
                          / (param count in group / total params)
      enrichment > 1 == safety-critical signal concentrated in the group
      BEYOND what its parameter share predicts. This is the param-count
      normalization that prevents "reader has 5 matrices vs 2" from being
      read as concentration -- the same confound flagged for the repair rows.
  (2) top-tau COUNT enrichment (matches the "where do the top-tau land" framing):
      global tau-quantile threshold over all weights (estimated from a large
      random subsample), then per-group counts above it, again normalized by
      param share.

Safety loss modes:
  --mode refusal      L = CE(refusal | harmful_prompt)              [default]
  --mode contrastive  L = CE(refusal | x) - CE(compliance | x)      [margin]
                      (needs harmful "target"/compliance strings; falls back
                       to refusal mode if unavailable)

Usage:
  python -m diagnostics.snip_safety_localization --model llama7b \
      --n_prompts 260 --taus 0.001 0.01 0.1 --output outputs/snip_llama7b.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import MODELS, HF_HOME  # noqa: E402
from diagnostics.readwrite_restoration import READ_PROJS, WRITE_PROJS  # noqa: E402
from core.datasets import load_wildjailbreak_prompts  # noqa: E402

ALL_PROJS = READ_PROJS + WRITE_PROJS  # 7 projection types, suffix form "self_attn.q_proj" etc.

# Canonical refusals -- averaging over a few reduces single-phrasing token bias.
DEFAULT_REFUSALS = [
    "I cannot help with that request.",
    "I'm sorry, but I can't help with that.",
    "I cannot assist with that.",
]

ADVBENCH_SPLIT = (
    Path(__file__).parent.parent
    / "refusal_direction/dataset/splits/harmful_train.json"
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_harmful(n_prompts, source="advbench", wj_config="eval"):
    """Harmful prompt texts for the refusal-CE saliency pass.

    source='wildjailbreak' must be used to match the displacement distribution
    (the contrastive pairs come from WildJailbreak); AdvBench floors out and
    localizes a gap that is not there.
    """
    if source == "wildjailbreak":
        items = load_wildjailbreak_prompts(n_samples=n_prompts, config=wj_config)
        return [it["prompt"] for it in items if it.get("prompt")]
    with open(ADVBENCH_SPLIT) as f:
        items = json.load(f)
    prompts = [it["instruction"] for it in items if it.get("instruction")]
    return prompts[:n_prompts] if n_prompts else prompts


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model_for_grad(model_name, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = MODELS.get(model_name, model_name)
    print(f"[load] {model_name} -> {model_path}  (dtype={dtype})")
    tok = AutoTokenizer.from_pretrained(model_path, cache_dir=HF_HOME)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map="cuda",
        cache_dir=HF_HOME, attn_implementation="eager",
    )
    model.eval()
    model.config.use_cache = False

    # Only the 7 target projection matrices need gradients; everything else off.
    model.requires_grad_(False)
    targets = {}  # name -> module
    for name, module in model.named_modules():
        if any(name.endswith(p) for p in ALL_PROJS) and hasattr(module, "weight"):
            module.weight.requires_grad_(True)
            targets[name] = module
    n_layers = model.config.num_hidden_layers
    print(f"[load] {len(targets)} target matrices over {n_layers} layers")
    return model, tok, targets, n_layers


def _proj_type(name):
    for p in ALL_PROJS:
        if name.endswith(p):
            return p
    return None


def _layer_idx(name):
    # ...model.layers.<i>.<proj>
    for tok in name.split("."):
        if tok.isdigit():
            return int(tok)
    return -1


# --------------------------------------------------------------------------- #
# SNIP accumulation
# --------------------------------------------------------------------------- #
def build_example(tok, instruction, response):
    """Return (input_ids, labels) with labels masked to the response tokens only."""
    msgs = [{"role": "user", "content": instruction}]
    prompt_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    )[0]
    resp_ids = tok(response, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, resp_ids])
    labels = torch.cat([torch.full_like(prompt_ids, -100), resp_ids])
    return input_ids, labels


def snip_scores(model, tok, targets, prompts, refusals, mode, sal_dtype, device="cuda"):
    """Accumulate mean |w * dL/dw| per target matrix. Returns {name: saliency_tensor}."""
    sal = {n: torch.zeros_like(m.weight, dtype=sal_dtype) for n, m in targets.items()}
    n_seen = 0
    for i, prompt in enumerate(prompts):
        # average over refusal phrasings to de-bias the target tokens
        for refusal in refusals:
            model.zero_grad(set_to_none=True)
            input_ids, labels = build_example(tok, prompt, refusal)
            input_ids = input_ids.unsqueeze(0).to(device)
            labels = labels.unsqueeze(0).to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss
            if mode == "contrastive":
                # margin: also pull DOWN the compliance loss (needs targets; here
                # we approximate compliance by an affirmative prefix continuation)
                comply = "Sure, here is how to do it: First,"
                c_ids, c_lab = build_example(tok, prompt, comply)
                out_c = model(input_ids=c_ids.unsqueeze(0).to(device),
                              labels=c_lab.unsqueeze(0).to(device))
                loss = loss - out_c.loss
            loss.backward()
            with torch.no_grad():
                for n, m in targets.items():
                    if m.weight.grad is not None:
                        sal[n] += (m.weight.detach() * m.weight.grad.detach()).abs().to(sal_dtype)
            n_seen += 1
        if (i + 1) % 25 == 0:
            print(f"  [snip] {i+1}/{len(prompts)} prompts")
    for n in sal:
        sal[n] /= max(n_seen, 1)
    model.zero_grad(set_to_none=True)
    return sal


def snip_scores_pairs(model, tok, targets, pairs, sal_dtype, device="cuda"):
    """Contrastive SNIP over REAL displacement pairs: score |w*(g_refuse - g_comply)|
    with yw = FP refusal, yl = quantized model's harmful completion (per prompt)."""
    sal = {n: torch.zeros_like(m.weight, dtype=sal_dtype) for n, m in targets.items()}
    n_seen = 0
    for i, pr in enumerate(pairs):
        model.zero_grad(set_to_none=True)
        w_ids, w_lab = build_example(tok, pr["instruction"], pr["yw"])
        l_ids, l_lab = build_example(tok, pr["instruction"], pr["yl"])
        out_w = model(input_ids=w_ids.unsqueeze(0).to(device), labels=w_lab.unsqueeze(0).to(device))
        out_l = model(input_ids=l_ids.unsqueeze(0).to(device), labels=l_lab.unsqueeze(0).to(device))
        loss = out_w.loss - out_l.loss  # margin; single backward => grad = g_refuse - g_comply
        loss.backward()
        with torch.no_grad():
            for n, m in targets.items():
                if m.weight.grad is not None:
                    sal[n] += (m.weight.detach() * m.weight.grad.detach()).abs().to(sal_dtype)
        n_seen += 1
        if (i + 1) % 25 == 0:
            print(f"  [snip-pairs] {i+1}/{len(pairs)} pairs")
    for n in sal:
        sal[n] /= max(n_seen, 1)
    model.zero_grad(set_to_none=True)
    return sal


# --------------------------------------------------------------------------- #
# aggregation / enrichment
# --------------------------------------------------------------------------- #
def bucketize_layer(i, n_layers):
    third = n_layers / 3.0
    if i < third:
        return "early"
    if i < 2 * third:
        return "middle"
    return "late"


def aggregate(sal, n_layers, taus, subsample=20_000_000):
    """Return a report dict with mass- and count-based enrichment by role/proj/layer."""
    # Per-matrix mass + params
    rows = []
    for name, s in sal.items():
        rows.append({
            "name": name,
            "proj": _proj_type(name),
            "layer": _layer_idx(name),
            "bucket": bucketize_layer(_layer_idx(name), n_layers),
            "role": "read" if _proj_type(name) in READ_PROJS else "write",
            "mass": float(s.sum().item()),
            "n_params": int(s.numel()),
        })
    total_mass = sum(r["mass"] for r in rows) or 1.0
    total_params = sum(r["n_params"] for r in rows) or 1

    def enrich(subset):
        m = sum(r["mass"] for r in subset)
        p = sum(r["n_params"] for r in subset)
        mass_frac = m / total_mass
        param_frac = p / total_params if total_params else 0
        return {
            "mass_frac": mass_frac,
            "param_frac": param_frac,
            "enrichment": (mass_frac / param_frac) if param_frac else 0.0,
            "n_params": p,
        }

    def group_by(key):
        keys = sorted({r[key] for r in rows}, key=lambda x: str(x))
        return {str(k): enrich([r for r in rows if r[key] == k]) for k in keys}

    # ---- top-tau count enrichment ----
    # Estimate global tau-quantile thresholds from a random subsample of all scores.
    flat_parts = []
    remaining = subsample
    n_mats = len(sal)
    per_mat = max(1, subsample // max(n_mats, 1))
    g = torch.Generator(device="cpu").manual_seed(0)
    for s in sal.values():
        flatv = s.detach().to("cpu", torch.float32).flatten()
        if flatv.numel() > per_mat:
            idx = torch.randint(0, flatv.numel(), (per_mat,), generator=g)
            flatv = flatv[idx]
        flat_parts.append(flatv)
        remaining -= flatv.numel()
    sample = torch.cat(flat_parts).numpy()
    # np.quantile has no 2**24 element cap that torch.quantile does; still subsample for speed.
    if sample.shape[0] > 8_000_000:
        rng = np.random.default_rng(0)
        sample = sample[rng.integers(0, sample.shape[0], 8_000_000)]
    thresholds = {tau: float(np.quantile(sample, 1.0 - tau)) for tau in taus}

    # exact per-matrix counts above each threshold (cheap: one comparison per matrix)
    count_rows = []
    for name, s in sal.items():
        sc = s.detach()
        rec = {"name": name, "proj": _proj_type(name),
               "role": "read" if _proj_type(name) in READ_PROJS else "write",
               "bucket": bucketize_layer(_layer_idx(name), n_layers),
               "n_params": int(s.numel())}
        for tau, thr in thresholds.items():
            rec[f"top_{tau}"] = int((sc >= thr).sum().item())
        count_rows.append(rec)

    def count_enrich(subset, tau):
        c = sum(r[f"top_{tau}"] for r in subset)
        p = sum(r["n_params"] for r in subset)
        total_top = sum(r[f"top_{tau}"] for r in count_rows) or 1
        cnt_frac = c / total_top
        param_frac = p / total_params if total_params else 0
        return (cnt_frac / param_frac) if param_frac else 0.0

    read_rows = [r for r in count_rows if r["role"] == "read"]
    write_rows = [r for r in count_rows if r["role"] == "write"]
    readmid_rows = [r for r in count_rows if r["role"] == "read" and r["bucket"] == "middle"]
    topk_enrichment = {
        str(tau): {
            "read": count_enrich(read_rows, tau),
            "write": count_enrich(write_rows, tau),
            "read_middle": count_enrich(readmid_rows, tau),
        }
        for tau in taus
    }

    return {
        "by_role": group_by("role"),
        "by_proj": group_by("proj"),
        "by_bucket": group_by("bucket"),
        "read_middle_mass_enrichment": enrich(
            [r for r in rows if r["role"] == "read" and r["bucket"] == "middle"]
        ),
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "topk_count_enrichment": topk_enrichment,
        "per_matrix": rows,
    }


def print_report(rep):
    print("\n" + "=" * 64)
    print("SNIP SAFETY-CRITICAL LOCALIZATION  (mass enrichment, threshold-free)")
    print("=" * 64)
    print(f"{'group':<14}{'enrichment':>12}{'mass%':>10}{'param%':>10}")
    for role, d in rep["by_role"].items():
        print(f"{role:<14}{d['enrichment']:>12.3f}{100*d['mass_frac']:>9.1f}%{100*d['param_frac']:>9.1f}%")
    print("-" * 46)
    for proj, d in rep["by_proj"].items():
        print(f"{proj:<14}{d['enrichment']:>12.3f}{100*d['mass_frac']:>9.1f}%{100*d['param_frac']:>9.1f}%")
    print("-" * 46)
    for b, d in rep["by_bucket"].items():
        print(f"{b:<14}{d['enrichment']:>12.3f}{100*d['mass_frac']:>9.1f}%{100*d['param_frac']:>9.1f}%")
    rm = rep["read_middle_mass_enrichment"]
    print("-" * 46)
    print(f"{'READ x middle':<14}{rm['enrichment']:>12.3f}{100*rm['mass_frac']:>9.1f}%{100*rm['param_frac']:>9.1f}%")
    print("\nTop-tau COUNT enrichment (read / write / read_middle):")
    for tau, d in rep["topk_count_enrichment"].items():
        print(f"  tau={tau:<8} read={d['read']:.2f}  write={d['write']:.2f}  read_middle={d['read_middle']:.2f}")
    print("=" * 64)
    print("Headline: safety-critical weights are concentrated in reader-side "
          "middle layers iff READ>1, middle>1, and READ x middle is the largest cell.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b", help="alias (config.MODELS) or HF id")
    ap.add_argument("--n_prompts", type=int, default=260, help="harmful prompts for refusal-CE")
    ap.add_argument("--harmful_source", default="advbench",
                    choices=["advbench", "wildjailbreak"])
    ap.add_argument("--wj_config", default="eval", choices=["train", "eval"])
    ap.add_argument("--mode", choices=["refusal", "contrastive"], default="refusal")
    ap.add_argument("--pairs_file", default=None,
                    help="displacement pairs JSON (build_displacement_pairs.py); "
                         "forces contrastive-over-real-pairs and ignores --mode/--n_prompts")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.001, 0.01, 0.1])
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--sal_dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--n_refusals", type=int, default=3, help="how many canonical refusals to average")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    sal_dtype = {"float32": torch.float32, "float16": torch.float16}[args.sal_dtype]

    if args.pairs_file:
        with open(args.pairs_file) as f:
            pairs = json.load(f)["pairs"]
        mode_label = "contrastive_pairs"
        print(f"[data] {len(pairs)} real displacement pairs from {args.pairs_file}  mode={mode_label}")
        model, tok, targets, n_layers = load_model_for_grad(args.model, dtype)
        sal = snip_scores_pairs(model, tok, targets, pairs, sal_dtype)
        n_data = len(pairs)
    else:
        prompts = load_harmful(args.n_prompts, args.harmful_source, args.wj_config)
        refusals = DEFAULT_REFUSALS[:max(1, args.n_refusals)]
        mode_label = f"{args.mode}_{args.harmful_source}"
        print(f"[data] {len(prompts)} harmful prompts x {len(refusals)} refusals  mode={mode_label}")
        model, tok, targets, n_layers = load_model_for_grad(args.model, dtype)
        sal = snip_scores(model, tok, targets, prompts, refusals, args.mode, sal_dtype)
        n_data = len(prompts)

    # Checkpoint per-matrix saliency mass BEFORE the fragile aggregation step.
    # This is tiny (224 scalars) and fully reconstructs the primary mass-enrichment
    # metric offline, so a post-processing bug never costs another GPU pass.
    out = args.output or f"outputs/snip_{args.model}_{mode_label}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ckpt = {n: {"mass": float(s.sum().item()), "n_params": int(s.numel())}
            for n, s in sal.items()}
    ckpt_path = out.replace(".json", "_massckpt.json")
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    print(f"[ckpt] per-matrix saliency mass -> {ckpt_path}")

    rep = aggregate(sal, n_layers, args.taus)
    rep["_meta"] = {
        "model": args.model, "n_data": n_data, "mode": mode_label,
        "taus": args.taus, "dtype": args.dtype,
        "pairs_file": args.pairs_file,
        "read_projs": READ_PROJS, "write_projs": WRITE_PROJS,
    }
    print_report(rep)

    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
