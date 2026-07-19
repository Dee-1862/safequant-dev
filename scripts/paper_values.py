#!/usr/bin/env python
"""Emit ONE machine-readable file of every paper value, keyed by table label.

Each of the 6 tables + 1 figure in the paper maps to a block here, and every
number is pulled from its exact source JSON under outputs/ -- so a collaborator
(or a model editing the manuscript) can read one file instead of hunting across
60+ result files, and anyone can regenerate it byte-for-byte:

  python scripts/paper_values.py > paper_values.json     # machine-readable
  python scripts/paper_values.py --md                    # human-readable dump

Table inventory (label -> source):
  tab:geometry      RQ1  dmed,a_min,cosmed,coswmed,dASR x 19 cells
                         (direction_drift_* [cosmed + per-layer sep],
                          probe_rotation_* [coswmed], probe_accuracy_* [a_min],
                          restoration_sweep_* [dASR])
  tab:regime        RQ2  ASR/ARC per condition x screen cells (restoration_sweep_*)
  tab:f3a_proj      RQ3  per-projection recovery (per_projection_llama_aqlm2_L11-22)
  tab:f3a_range     RQ3  depth-band recovery (depth_bands_llama_aqlm2_read/write)
  tab:f3a           RQ3  static restoration (headline_llama_aqlm2)
  tab:trained_repair RQ3 Q-Resafe folded (qresafe_dpo_*)
  tab:lrqat         RQ3  LR-QAT cells (lrqat_*)
  fig:readside      RQ3  read vs write restored ASR (headline_llama_aqlm2)

Nulls (not "--") mark genuinely-absent cells so JSON stays valid. Pure stdlib.
Run from the repo root (where outputs/ lives).
"""
import glob
import json
import statistics
import sys
from pathlib import Path

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]
OUT = "outputs"
CONDS = ["fp32", "quant", "write_restored", "read_restored", "both_restored"]


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


def rnd(x, n=3):
    return None if x is None else round(float(x), n)


def sig(exp, m, q):
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def dmed_quant(m, q):
    """Median per-layer separation_sigma of the quantized model (decodability)."""
    d = sig("direction_drift", m, q)
    if not d:
        return None
    qk = [k for k in d if k.endswith("_separation_per_layer")
          and not k.startswith("fp32")]
    if not qk:
        return None
    vals = [v.get("separation_sigma") for v in d[qk[0]].values()
            if isinstance(v, dict) and v.get("separation_sigma") is not None]
    return rnd(statistics.median(vals), 2) if vals else None


def cond_map(d, block):
    """Normalize a restoration_sweep asr/arc block to fixed condition keys.

    The per-condition dict keys the quantized cell by the quantizer name
    (e.g. 'gptq4'); remap that to 'quant' so every cell has the same schema.
    """
    src = g(d, block, default={}) or {}
    qtz = next((k for k in src if k not in
                ("fp32", "write_restored", "read_restored", "both_restored")), None)
    out = {}
    for c in CONDS:
        key = qtz if c == "quant" else c
        out[c] = rnd(src.get(key), 1) if src.get(key) is not None else None
    return out


def build():
    pv = {
        "meta": {
            "generated_by": "scripts/paper_values.py",
            "note": ("All ASR/ARC/damage in percentage points. dASR = "
                     "quant_asr - fp32_asr (positive = safety worse). Null = "
                     "cell absent (e.g. qwen/aqlm2 by design; phi3/awq "
                     "restoration is arch-mismatch N/A)."),
            "seed": 42,
        }
    }

    # -- tab:geometry (RQ1) : 19 cells x 5 measures ------------------------
    geo = {}
    for m in MODELS:
        for q in QUANTS:
            dd = sig("direction_drift", m, q)
            if not dd:
                continue  # skip cells with no geometry (qwen/aqlm2)
            rs = restore(m, q)
            geo[f"{m}/{q}"] = {
                "dmed": dmed_quant(m, q),
                "a_min": rnd(g(sig("probe_accuracy", m, q), "summary",
                              "quant_min_acc")),
                "cosmed": rnd(g(dd, "summary", "cos_sim_median")),
                "coswmed": rnd(g(sig("probe_rotation", m, q), "summary",
                               "cos_median")),
                "dASR": rnd(g(rs, "damage_pp"), 1),
                "swap_valid": g(rs, "swap_valid"),
                "restore_status": g(rs, "restore_status", default="ok"),
            }
    pv["tab:geometry"] = {
        "columns": ["dmed", "a_min", "cosmed", "coswmed", "dASR"],
        "source": ("direction_drift_* (cosmed + per-layer separation_sigma), "
                   "probe_rotation_* (coswmed), probe_accuracy_* (a_min), "
                   "restoration_sweep_* (dASR)"),
        "cells": geo,
    }

    # -- tab:regime (RQ2) : ASR/ARC per condition --------------------------
    reg = {}
    for m in MODELS:
        for q in QUANTS:
            rs = restore(m, q)
            if not rs:
                continue
            reg[f"{m}/{q}"] = {
                "asr": cond_map(rs, "asr_per_condition"),
                "arc": cond_map(rs, "arc"),
                "damage_pp": rnd(g(rs, "damage_pp"), 1),
                "swap_valid": g(rs, "swap_valid"),
                "restore_status": g(rs, "restore_status", default="ok"),
            }
    pv["tab:regime"] = {
        "conditions": CONDS,
        "source": "restoration_sweep_* asr_per_condition + arc",
        "note": ("ARC taken from restoration_sweep (per-cell, complete) rather "
                 "than headline (llama/aqlm2 only) for self-consistency."),
        "cells": reg,
    }

    # -- tab:f3a_proj (RQ3) : per-projection recovery ----------------------
    pp = load(f"{OUT}/localization/per_projection_llama_aqlm2_L11-22.json")
    pv["tab:f3a_proj"] = {
        "source": "per_projection_llama_aqlm2_L11-22.json",
        "layer_range": g(pp, "layer_range"),
        "damage_pp": rnd(g(pp, "damage_pp"), 1),
        "recovery_pp_per_projection": g(pp, "recovery_pp_per_projection"),
    }

    # -- tab:f3a_range (RQ3) : depth-band recovery -------------------------
    dbr = load(f"{OUT}/localization/depth_bands_llama_aqlm2_read.json")
    dbw = load(f"{OUT}/localization/depth_bands_llama_aqlm2_write.json")
    pv["tab:f3a_range"] = {
        "source": "depth_bands_llama_aqlm2_read/write.json",
        "read": g(dbr, "recovery_pp_per_range"),
        "write": g(dbw, "recovery_pp_per_range"),
    }

    # -- tab:f3a (RQ3) : static restoration --------------------------------
    hl = load(f"{OUT}/headline/headline_llama_aqlm2.json")
    pv["tab:f3a"] = {
        "source": "headline_llama_aqlm2.json",
        "damage_pp": rnd(g(hl, "damage_pp"), 1),
        "asr": g(hl, "asr"),
        "arc": g(hl, "arc"),
        "recovery_pp": g(hl, "recovery_pp"),
        "recovery_ci_95": g(hl, "recovery_ci_95"),
        "xstest_false_refusal": {k: g(v, "false_refusal_rate")
                                 for k, v in (g(hl, "xstest", default={}) or {}).items()},
        "mcnemar_p": {k: g(v, "p_value")
                      for k, v in (g(hl, "mcnemar", default={}) or {}).items()},
    }

    # -- tab:trained_repair (RQ3) : Q-Resafe folded ------------------------
    qr = {}
    for f in sorted(glob.glob(f"{OUT}/trained_repair/qresafe_dpo_*.json")):
        d = load(f)
        if not isinstance(d, dict):
            continue
        qr[Path(f).stem.replace("qresafe_dpo_", "")] = {
            "mean_codes_changed": g(d, "mean_codes_changed"),
            "fp_lora": g(d, "fp_lora"),
            "folded": g(d, "folded"),
        }
    pv["tab:trained_repair"] = {"source": "qresafe_dpo_*.json", "cells": qr}

    # -- tab:lrqat (RQ3) : LR-QAT cells ------------------------------------
    lr = {}
    for f in sorted(glob.glob(f"{OUT}/trained_repair/lrqat_*.json")):
        d = load(f)
        if not isinstance(d, dict) or "asr_lrqat_trained" not in d:
            lr[Path(f).stem.replace("lrqat_", "")] = {"status": "incomplete"}
            continue
        lr[Path(f).stem.replace("lrqat_", "")] = {
            "role": g(d, "role"), "n_bits": g(d, "n_bits"),
            "asr_fp16": rnd(g(d, "asr_fp16"), 1),
            "asr_aqlm2": rnd(g(d, "asr_aqlm2"), 1),
            "asr_ptq_untrained": rnd(g(d, "asr_ptq_untrained"), 1),
            "asr_lrqat_trained": rnd(g(d, "asr_lrqat_trained"), 1),
            "arc_ptq_untrained": rnd(g(d, "arc_ptq_untrained"), 1),
            "arc_lrqat_trained": rnd(g(d, "arc_lrqat_trained"), 1),
            "rho_lrqat_trained": rnd(g(d, "rho_lrqat_trained"), 2),
            "repaired_effective_bits": rnd(g(d, "repaired_effective_bits"), 2),
            "coherence_collapse": g(d, "coherence_collapse_lrqat_trained"),
        }
    pv["tab:lrqat"] = {"source": "lrqat_*.json", "cells": lr}

    # -- fig:readside (RQ3) : read vs write restored ASR -------------------
    pv["fig:readside"] = {
        "source": "headline_llama_aqlm2.json asr",
        "fp32": rnd(g(hl, "asr", "fp32"), 1),
        "aqlm2": rnd(g(hl, "asr", "aqlm2"), 1),
        "read_restored": rnd(g(hl, "asr", "read_restored"), 1),
        "write_restored": rnd(g(hl, "asr", "write_restored"), 1),
        "both_restored": rnd(g(hl, "asr", "both_restored"), 1),
    }
    return pv


def as_markdown(pv):
    lines = ["# Paper values (per table label)\n"]
    lines.append(pv["meta"]["note"] + "\n")
    geo = pv["tab:geometry"]["cells"]
    lines.append("## tab:geometry\n")
    lines.append("| cell | dmed | a_min | cosmed | coswmed | dASR | swap_valid |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in geo.items():
        lines.append(f"| {k} | {v['dmed']} | {v['a_min']} | {v['cosmed']} | "
                     f"{v['coswmed']} | {v['dASR']} | {v['swap_valid']} |")
    lines.append("\n## tab:regime (ASR / ARC)\n")
    lines.append("| cell | fp32 | quant | +write | +read | +both | swap |")
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in pv["tab:regime"]["cells"].items():
        a, r = v["asr"], v["arc"]
        cells = " | ".join(f"{a[c]}/{r[c]}" for c in CONDS)
        lines.append(f"| {k} | {cells} | {v['swap_valid']} |")
    for lab in ("tab:f3a_proj", "tab:f3a_range", "tab:f3a",
                "tab:trained_repair", "tab:lrqat", "fig:readside"):
        lines.append(f"\n## {lab}\n```json")
        lines.append(json.dumps(pv[lab], indent=2, default=str))
        lines.append("```")
    return "\n".join(lines)


def main():
    pv = build()
    if "--md" in sys.argv[1:]:
        print(as_markdown(pv))
    else:
        print(json.dumps(pv, indent=2, default=str))


if __name__ == "__main__":
    main()
