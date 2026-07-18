#!/usr/bin/env python
"""Aggregate every completed experiment JSON into ONE paper-ready markdown file.

Unlike collect_results.py (which dumps raw JSON per file), this emits organized
grids (model x quantizer) for each experiment plus a reproduction header, so the
output can be handed to a model/collaborator to fill paper values -- and anyone
can regenerate it byte-for-byte from the repo:

  python scripts/aggregate_results.py > ~/PAPER_RESULTS.md

Missing cells are printed as "--" (not silently dropped) so gaps are visible.
Pure stdlib. Run from the repo root (where outputs/ lives).
"""
import glob
import json
from pathlib import Path

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]
OUT = "outputs"


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def g(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def num(x, r=3):
    return "--" if x is None else f"{float(x):.{r}f}"


def grid(title, note, getter, models=MODELS, quants=QUANTS):
    """Print a model x quantizer markdown table using getter(model, quant)->str."""
    print(f"\n### {title}\n")
    if note:
        print(f"{note}\n")
    print("| model | " + " | ".join(quants) + " |")
    print("|" + "---|" * (len(quants) + 1))
    for m in models:
        cells = [getter(m, q) for q in quants]
        print(f"| {m} | " + " | ".join(cells) + " |")


def sig(exp, m, q):
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def main():
    print("# SafeQuant — Aggregated Results (paper values)\n")
    print("Regenerate with `python scripts/aggregate_results.py > PAPER_RESULTS.md` "
          "from the repo root after `git pull`. Cells shown as `--` have no result "
          "file (experiment not run for that model/quantizer). All ASR/damage in "
          "percentage points; `damage_pp = quant_asr - fp32_asr` (positive = safety "
          "got WORSE under quantization).\n")

    # -- Reproduction provenance -------------------------------------------
    print("## Reproduction\n")
    print("- **Models (FP32 base):** Llama-3.1-8B-Instruct (alias `llama7b`), "
          "Qwen2.5-7B-Instruct (`qwen7b`), Mistral-7B-Instruct-v0.2 "
          "(`mistral7b-v02`), Phi-3-mini-4k-instruct (`phi3`).")
    print("- **Quantizers:** GPTQ 4/3/2-bit are SELF-GENERATED via AutoGPTQ "
          "(group_size=128, desc_act=True, sym=True, damp=0.1, WikiText-2 "
          "calibration, 128x2048-tok windows). AWQ 4-bit and AQLM 2-bit are "
          "PUBLIC downloads: AWQ = hugging-quants (llama) / Qwen-official / "
          "TheBloke (mistral) / Sreenington (phi3); AQLM = ISTA-DASLab.")
    print("- **Env:** eval in conda `sq_clean` (transformers 4.57); GPTQ "
          "self-quant in `sq_autogptq` (auto-gptq 0.7.1, transformers 4.44.2, "
          "torch 2.2.2, numpy<2).")
    print("- **Eval sets:** n_eval=200 harmful (ASR), ARC-easy (utility), "
          "XSTest-250 (false-refusal). Greedy, left-padded batched generation "
          "(byte-identical to unbatched).")
    print("- **Qwen has no AQLM-2 cell by design; Phi-3 GPTQ pending (see gaps).**")

    # -- Table 1: refusal-direction drift ----------------------------------
    print("\n## Table 1 — Refusal-direction geometry\n")
    grid("1a. Diff-in-means direction drift (cosine median, FP32 vs quant)",
         "Higher = direction preserved. `direction_drift_*`.",
         lambda m, q: num(g(sig("direction_drift", m, q), "summary",
                            "cos_sim_median")))
    grid("1b. Direction drift (cosine MIN over layers)",
         "Worst-layer cosine. `direction_drift_*`.",
         lambda m, q: num(g(sig("direction_drift", m, q), "summary",
                            "cos_sim_min")))
    grid("1c. Linear-probe rotation (cosine median of probe weight vectors)",
         "The readable/separating axis. `probe_rotation_*`.",
         lambda m, q: num(g(sig("probe_rotation", m, q), "summary",
                            "cos_median")))

    # -- Table 2: probe accuracy -------------------------------------------
    print("\n## Table 2 — Linear separability of the safety concept\n")
    grid("2a. Probe accuracy (quant mean over layers)",
         "FP32 mean ~0.994-0.997 for all. `probe_accuracy_*`.",
         lambda m, q: num(g(sig("probe_accuracy", m, q), "summary",
                            "quant_mean_acc")))
    grid("2b. Probe accuracy (quant MIN over layers)",
         "Worst-layer separability. `probe_accuracy_*`.",
         lambda m, q: num(g(sig("probe_accuracy", m, q), "summary",
                            "quant_min_acc")))
    grid("2c. # layers with >5pp accuracy drop",
         "0 everywhere except qwen/gptq2. `probe_accuracy_*`.",
         lambda m, q: (lambda d: "--" if d is None else
                       str(g(d, "summary", "n_layers_with_drop", default="--")))
                      (sig("probe_accuracy", m, q)))

    # -- Table 3: restoration / damage -------------------------------------
    print("\n## Table 3 — Damage and projection-restoration\n")
    grid("3a. Safety damage (damage_pp = quant_asr - fp32_asr)",
         "Positive = safety got WORSE. `restoration_sweep_*`.",
         lambda m, q: num(g(restore(m, q), "damage_pp"), 1))
    grid("3b. Quantized ASR (% harmful complied)",
         "`restoration_sweep_*` quantized_asr.",
         lambda m, q: num(g(restore(m, q), "quantized_asr"), 1))
    grid("3c. ASR after READ-projection restoration",
         "`restoration_sweep_*` asr_per_condition.read_restored.",
         lambda m, q: num(g(restore(m, q), "asr_per_condition", "read_restored"), 1))
    grid("3d. ASR after WRITE-projection restoration",
         "`restoration_sweep_*` asr_per_condition.write_restored.",
         lambda m, q: num(g(restore(m, q), "asr_per_condition", "write_restored"), 1))
    grid("3e. ASR after BOTH restored",
         "`restoration_sweep_*` asr_per_condition.both_restored.",
         lambda m, q: num(g(restore(m, q), "asr_per_condition", "both_restored"), 1))
    grid("3f. swap_valid (activation-swap sanity gate)",
         "False = swap didn't localize; exclude that cell from restore claims.",
         lambda m, q: (lambda d: "--" if d is None else
                       str(g(d, "swap_valid", default="--")))(restore(m, q)))

    # -- FP32 / quant ASR reference row ------------------------------------
    print("\n### 3g. FP32 baseline ASR per model (reference)\n")
    print("| model | fp32_asr |")
    print("|---|---|")
    for m in MODELS:
        d = next((restore(m, q) for q in QUANTS if restore(m, q)), None)
        print(f"| {m} | {num(g(d, 'fp32_asr'), 1)} |")

    # -- damage-vs-displacement correlation --------------------------------
    print("\n## §5 — Does displacement predict damage?\n")
    rows = []
    for m in MODELS:
        for q in QUANTS:
            dd, pr, rs = (sig("direction_drift", m, q),
                          sig("probe_rotation", m, q), restore(m, q))
            if dd and rs and g(rs, "swap_valid", default=True) \
                    and g(rs, "damage_pp") is not None:
                rows.append((g(rs, "damage_pp"),
                             g(dd, "summary", "cos_sim_median"),
                             g(pr, "summary", "cos_median")))

    def corr(xs, ys):
        n = len(xs)
        if n < 3:
            return None
        mx, my = sum(xs) / n, sum(ys) / n
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        if sx == 0 or sy == 0:
            return None
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)

    cm = [(d, 1 - c) for d, c, _ in rows if c is not None]
    pm = [(d, 1 - p) for d, _, p in rows if p is not None]
    c1 = corr([a for a, _ in cm], [b for _, b in cm]) if cm else None
    c2 = corr([a for a, _ in pm], [b for _, b in pm]) if pm else None
    print(f"- swap-valid cells with damage: **{len(rows)}**")
    print(f"- corr(damage, 1 - diff-in-means cosmed) = "
          f"**{num(c1)}**  (n={len(cm)})")
    print(f"- corr(damage, 1 - probe rotation)       = "
          f"**{num(c2)}**  (n={len(pm)})")
    print("- NOTE: sign is confounded — GPTQ-2/3 cells show large displacement "
          "AND *negative* damage (safety improved via generic degradation), while "
          "AQLM/Phi-3 cells show positive damage at moderate displacement. Report "
          "the causal read-restoration result (Table 3c) as the primary evidence, "
          "not this correlation.")

    # -- Behavioral (Arditi), boundary, localization, dequant, repair ------
    print("\n## Behavioral direction (Arditi ablation)\n")
    print("| cell | baseline_asr | ablated_asr | delta_pp | layer | pos | operative |")
    print("|---|---|---|---|---|---|---|")
    for f in sorted(glob.glob(f"{OUT}/behavioral_direction/arditi_repro_*.json")):
        d = load(f)
        print(f"| {Path(f).stem.replace('arditi_repro_','')} "
              f"| {num(g(d,'baseline_asr'),1)} | {num(g(d,'ablated_asr'),1)} "
              f"| {num(g(d,'delta_asr_pp'),1)} | {g(d,'selected_layer')} "
              f"| {g(d,'selected_pos')} | {g(d,'operative')} |")

    print("\n## Boundary — activation swap (llama/aqlm2)\n")
    for f in sorted(glob.glob(f"{OUT}/boundary/activation_swap_*.json")):
        d = load(f)
        print(f"- {Path(f).stem.replace('activation_swap_','')}: "
              f"{json.dumps(g(d,'swap_asr', default={}))}")

    print("\n## Localization (per-projection / depth-band recovery)\n")
    for f in sorted(glob.glob(f"{OUT}/localization/*.json")):
        d = load(f)
        rec = (g(d, "recovery_pp_per_projection")
               or g(d, "recovery_pp_per_range") or {})
        print(f"- **{Path(f).stem}** (damage {num(g(d,'damage_pp'),1)}): "
              f"{json.dumps(rec)}")

    print("\n## Targeted dequant (source-agnostic read-restore)\n")
    for f in sorted(glob.glob(f"{OUT}/regime_sweep/targeted_dequant_*.json")):
        d = load(f)
        print(f"- **{Path(f).stem}**: recov_4bit={num(g(d,'recovery_pp_4bit'),1)}, "
              f"recov_fp32={num(g(d,'recovery_pp_fp32'),1)}, "
              f"frac_of_fp32={num(g(d,'frac_of_fp32_recovery'),2)} "
              f"(role={g(d,'role')}, layers={g(d,'layer_range')})")

    print("\n## Trained repair (Q-Resafe DPO / LR-QAT)\n")
    for f in sorted(glob.glob(f"{OUT}/trained_repair/*.json")):
        d = load(f)
        keep = {k: d[k] for k in ("mean_codes_changed", "fp_lora", "folded", "role")
                if isinstance(d, dict) and k in d}
        print(f"- **{Path(f).stem}**: {json.dumps(keep, default=str)}")

    # -- Headline (rich XSTest/ARC/mcnemar) --------------------------------
    print("\n## Headline eval (full XSTest / ARC / McNemar)\n")
    for f in sorted(glob.glob(f"{OUT}/headline/*.json")):
        d = load(f)
        print(f"\n### {Path(f).stem}\n```json")
        print(json.dumps({k: d[k] for k in
              ("damage_pp", "recovery_pp", "asr", "arc", "recovery_ci_95")
              if isinstance(d, dict) and k in d}, indent=2, default=str))
        xs = g(d, "xstest", default={})
        if xs:
            print("xstest false_refusal_rate: " + json.dumps(
                {k: g(v, "false_refusal_rate") for k, v in xs.items()}))
        mc = g(d, "mcnemar", default={})
        if mc:
            print("mcnemar p-values: " + json.dumps(
                {k: g(v, "p_value") for k, v in mc.items()}, default=str))
        print("```")

    # -- Gaps --------------------------------------------------------------
    print("\n## Gaps / caveats (do NOT invent values for these)\n")
    missing = []
    for m in MODELS:
        for q in QUANTS:
            if q == "aqlm2" and m == "qwen":
                continue  # by design
            if not sig("direction_drift", m, q):
                missing.append(f"{m}/{q} (direction_drift)")
    for item in missing:
        print(f"- MISSING: {item}")
    print("- Phi-3 GPTQ 4/3/2 cells pending (rerun sweep with SWEEP_MODELS=phi3).")
    print("- Cross-model mechanistic suite (arditi/swap/localization/headline) "
          "currently llama-only; mistral/phi3 not yet launched.")
    print("- LR-QAT cells still running at snapshot time — treat as provisional.")


if __name__ == "__main__":
    main()
