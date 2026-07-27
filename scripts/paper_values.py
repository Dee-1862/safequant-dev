# Pull paper-table values from outputs/ JSON files (one block per LaTeX table label).
import glob
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.stats import holm_bonferroni, mcnemar_exact, paired_delta_asr

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "aqlm2", "gptq3", "gptq2"]
F3A_MODELS = ["llama", "phi3"]
CONDS = ["fp32", "quant", "write_restored", "read_restored", "both_restored"]
OUT = "outputs"
QRESAFE_RANKS = [128, 256, 512, 1024, 2048]
QRESAFE_PATCHES = {
    "read": "Read",
    "write": "Writer",
    "full": "Both",
}

MODEL_LABEL = {
    "llama": "Llama-3.1-8B",
    "mistral": "Mistral-7B-v0.2",
    "qwen": "Qwen-2.5-7B",
    "phi3": "Phi-3-mini",
}

QUANT_LABEL = {
    "awq": "AWQ-4",
    "gptq4": "GPTQ-4",
    "aqlm2": "AQLM-2",
    "gptq3": "GPTQ-3",
    "gptq2": "GPTQ-2",
}

PROJ_ORDER_LLAMA = [
    ("self_attn.q_proj", "W_q", "Attention routing", 1),
    ("self_attn.k_proj", "W_k", "Attention routing", 1),
    ("mlp.up_proj", "W_up", "MLP feature lift", 1),
    ("mlp.down_proj", "W_down", "MLP output", 1),
    ("self_attn.v_proj", "W_v", "Attention content", 1),
    ("mlp.gate_proj", "W_gate", "MLP gating", 1),
    ("self_attn.o_proj", "W_o", "Attention output", 1),
]

PROJ_ORDER_PHI3 = [
    ("self_attn.qkv_proj", "W_qkv", "Attention q,k,v (fused)", 1),
    ("self_attn.o_proj", "W_o", "Attention output", 1),
    ("mlp.gate_up_proj", "W_gate_up", "MLP gate+lift (fused)", 1),
    ("mlp.down_proj", "W_down", "MLP output", 1),
]

DEPTH_RANGES = [
    ("early", "Early", "[0,11)"),
    ("middle", "Middle", "[11,22)"),
    ("late", "Late", "[22,32)"),
]

# AQLM inference is nondeterministic at +-1-2pp on 200 prompts, so the
# restoration re-run's ASR for AQLM cells can differ from the headline run.
# For the two studiable AQLM cells (which also appear in tab:f3a and carry the
# significance stats), source ASR/ARC/damage from the single headline run so
# every table agrees by construction. gptq/awq cells are deterministic -> use
# restoration directly.
STUDIABLE_HEADLINE = {("llama", "aqlm2"), ("phi3", "aqlm2")}


def load(p):
    """
    Load one JSON result file from disk.

    Inputs:
        - p (str | Path): Path to a result JSON

    Outputs:
        - data (dict | None): Parsed JSON, or None on missing/corrupt data
    """
    try:
        with open(p, encoding = "utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def g(d, *path, default = None):
    """
    Walk nested dict keys safely.

    Inputs:
        - d (dict | None): Root object
        - path: Sequence of keys
        - default: Fallback when a key is missing

    Outputs:
        - value: Nested value or default
    """
    for k in path:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
    return default if d is None else d


def rnd(x, n = 3):
    """
    Round a numeric value for paper display.

    Inputs:
        - x: Numeric value (or None)
        - n (int): Decimal places

    Outputs:
        - rounded (float | None): Rounded value, or None if x is None
    """
    if x is None:
        return None
    return round(float(x), n)


def sig(exp, m, q):
    """
    Load a signal_geometry experiment JSON.

    Inputs:
        - exp (str): Experiment stem (e.g. "direction_drift")
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    """
    Load a restoration_sweep experiment JSON.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def headline(m, q):
    """
    Load a headline eval JSON.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Parsed JSON
    """
    return load(f"{OUT}/headline/headline_{m}_{q}.json")


def cell_json(m, q):
    """
    Pick the canonical JSON for one model/quant cell.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - data (dict | None): Headline JSON for studiable AQLM cells, else restoration
    """
    if (m, q) in STUDIABLE_HEADLINE:
        hl = headline(m, q)
        if hl:
            return hl
    return restore(m, q)


def flags_for_cell(m, q):
    """
    Extract paired fp32/quant compliance flags for McNemar stats.

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - fp_flags (list[int] | None): Full-precision flags
        - quant_flags (list[int] | None): Quantized flags (same prompt order)
    """
    if m == "qwen" and q == "aqlm2":
        return None, None
    d = cell_json(m, q)
    if not d:
        return None, None
    fbc = g(d, "flags_by_cond", default = {}) or {}
    if "fp32" not in fbc or q not in fbc:
        return None, None
    to_int = lambda xs: [int(bool(x)) for x in xs]
    return to_int(fbc["fp32"]), to_int(fbc[q])


def bootstrap_delta_asr(flags_a, flags_b, n_boot = 2000, seed = 42):
    """
    Bootstrap a 95% CI for paired delta ASR (a minus b).

    Inputs:
        - flags_a (list[int]): Compliance flags for condition A
        - flags_b (list[int]): Compliance flags for condition B
        - n_boot (int): Bootstrap replicates
        - seed (int): RNG seed

    Outputs:
        - ci95 (list[float]): [lower, upper] percentile bounds in pp
    """
    a = np.asarray(flags_a, dtype = int)
    b = np.asarray(flags_b, dtype = int)
    n = a.size
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = 100.0 * (a[idx].mean() - b[idx].mean())
    return [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def dmed_quant(m, q):
    """
    Median per-layer separation_sigma on the quantized model (Table 1 dmed).

    Inputs:
        - m (str): Model key
        - q (str): Quantizer key

    Outputs:
        - dmed (float | None): Median separation_sigma across layers
    """
    d = sig("direction_drift", m, q)
    if not d:
        return None
    qk = [k for k in d if k.endswith("_separation_per_layer") and not k.startswith("fp32")]
    if not qk:
        return None
    vals = [
        v.get("separation_sigma")
        for v in d[qk[0]].values()
        if isinstance(v, dict) and v.get("separation_sigma") is not None
    ]
    if not vals:
        return None
    return rnd(statistics.median(vals), 2)


def cond_map(d, block):
    """
    Normalize restoration/headline ASR/ARC blocks to fixed condition keys.

    Inputs:
        - d (dict): Result JSON
        - block (str): Top-level key ("asr", "arc", or "asr_per_condition")

    Outputs:
        - out (dict): Map from CONDS keys to rounded percentages
    """
    src = g(d, block, default = {}) or {}
    qtz = next(
        (k for k in src if k not in ("fp32", "write_restored", "read_restored", "both_restored")),
        None,
    )
    out = {}
    for c in CONDS:
        if c == "quant":
            key = qtz
        else:
            key = c
        val = src.get(key)
        if val is None:
            out[c] = None
        else:
            out[c] = rnd(val, 1)
    return out


def screen_verdict(rs):
    """
    Heuristic damaged-coherent screen label (Table 2 verdict column).

    Inputs:
        - rs (dict | None): Restoration sweep JSON

    Outputs:
        - verdict (str | None): Human-readable screen label
    """
    if not rs:
        return None
    if g(rs, "restore_status") == "arch_mismatch_na":
        return "N/A (arch mismatch)"
    damage = g(rs, "damage_pp")
    fp_asr = g(rs, "fp32_asr")
    arc_q = cond_map(rs, "arc").get("quant")
    read_rec = g(rs, "read_recovery_pp")

    if arc_q is not None and arc_q < 35:
        return "fails coherence"
    if fp_asr is not None and fp_asr >= 70:
        return "fails baseline"
    if damage is None:
        return None
    if damage < 5:
        return "no damage"
    if read_rec is not None and read_rec >= 5:
        return "studiable, damaged"
    if damage >= 5:
        return "damaged (read recovery weak)"
    return "no damage"


def recovery_rho(recovery_pp, damage_pp):
    """
    Share of safety damage recovered (percent).

    Inputs:
        - recovery_pp (float | None): Recovery in percentage points
        - damage_pp (float | None): Baseline damage in percentage points

    Outputs:
        - rho_pct (int | None): Integer percent recovered
    """
    if recovery_pp is None or not damage_pp:
        return None
    pct = 100.0 * float(recovery_pp) / float(damage_pp)
    return int(round(pct))


def build_geometry():
    """
    Build tab:geometry: dmed, a_min, cosmed, coswmed per model/quant cell.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:geometry
    """
    by_model = {}
    for m in MODELS:
        rows = []
        for q in QUANTS:
            if m == "qwen" and q == "aqlm2":
                continue
            dd = sig("direction_drift", m, q)
            if not dd:
                continue
            rows.append({
                "quant": q,
                "quant_label": QUANT_LABEL[q],
                "dmed": dmed_quant(m, q),
                "a_min": rnd(g(sig("probe_accuracy", m, q), "summary", "quant_min_acc"), 3),
                "cosmed": rnd(g(dd, "summary", "cos_sim_median"), 2),
                "coswmed": rnd(g(sig("probe_rotation", m, q), "summary", "cos_median"), 2),
            })
        if rows:
            by_model[m] = {"model_label": MODEL_LABEL[m], "rows": rows}
    return {
        "columns": ["dmed", "a_min", "cosmed", "coswmed"],
        "source": (
            "direction_drift_* (cosmed + per-layer separation_sigma), "
            "probe_rotation_* (coswmed), probe_accuracy_* (a_min)"
        ),
        "by_model": by_model,
    }


def build_regime():
    """
    Build tab:regime: ASR/ARC per condition, paired screen stats, verdict.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:regime
    """
    cells = {}
    paired_raw = []

    for m in MODELS:
        for q in QUANTS:
            rs = restore(m, q)
            if not rs:
                continue
            hl = headline(m, q) if (m, q) in STUDIABLE_HEADLINE else None
            if hl:
                asr = cond_map(hl, "asr")
                arc = cond_map(hl, "arc")
                damage = rnd(g(hl, "damage_pp"), 1)
                src = "headline"
            else:
                asr = cond_map(rs, "asr_per_condition")
                arc = cond_map(rs, "arc")
                damage = rnd(g(rs, "damage_pp"), 1)
                src = "restoration_sweep"

            key = f"{m}/{q}"
            cell = {
                "model_label": MODEL_LABEL[m],
                "quant_label": QUANT_LABEL[q],
                "asr": asr,
                "arc": arc,
                "damage_pp": damage,
                "swap_valid": g(rs, "swap_valid"),
                "restore_status": g(rs, "restore_status", default = "ok"),
                "screen_verdict": screen_verdict(rs),
                "source": src,
                "paired": None,
            }

            fp_flags, q_flags = flags_for_cell(m, q)
            if fp_flags is not None and q_flags is not None:
                pr = paired_delta_asr(fp_flags, q_flags, n_boot = 5000, seed = 42)
                cell["paired"] = {
                    "delta_asr": rnd(pr["delta_asr"], 1),
                    "ci95": [rnd(pr["ci95"][0], 1), rnd(pr["ci95"][1], 1)],
                    "b": pr["b"],
                    "c": pr["c"],
                    "p_raw": pr["p_mcnemar"],
                    "p_holm": None,
                }
                paired_raw.append((key, pr["p_mcnemar"]))

            cells[key] = cell

    if paired_raw:
        keys, pvals = zip(*paired_raw)
        adj = holm_bonferroni(list(pvals))
        for key, p_holm in zip(keys, adj):
            cells[key]["paired"]["p_holm"] = p_holm

    return {
        "conditions": CONDS,
        "source": (
            "restoration_sweep_* (ARC n = 100); AQLM studiable cells "
            "llama/aqlm2 + phi3/aqlm2 sourced from headline_* for "
            "ASR/ARC consistency with tab:f3a; paired stats from flags_by_cond"
        ),
        "cells": cells,
    }


def _build_f3a_proj_model(m):
    """
    Per-projection rho for one damaged-coherent model.

    Inputs:
        - m (str): Model key ("llama" or "phi3")

    Outputs:
        - block (dict): Per-projection rows for tab:f3a_proj
    """
    if m == "phi3":
        proj_order = PROJ_ORDER_PHI3
    else:
        proj_order = PROJ_ORDER_LLAMA
    pp = load(f"{OUT}/localization/per_projection_{m}_aqlm2_L11-22.json")
    damage = g(pp, "damage_pp")
    rec = g(pp, "recovery_pp_per_projection", default = {}) or {}
    rows = []
    for path, sym, func, n_mat in proj_order:
        rec_pp = rec.get(path)
        rows.append({
            "projection": sym,
            "function": func,
            "matrices": n_mat,
            "recovery_pp": rnd(rec_pp, 1),
            "rho_pct": recovery_rho(rec_pp, damage),
        })
    return {
        "model_label": MODEL_LABEL[m],
        "source": f"per_projection_{m}_aqlm2_L11-22.json",
        "layer_range": g(pp, "layer_range"),
        "damage_pp": rnd(damage, 1),
        "rows": rows,
    }


def build_f3a_proj():
    """
    Build tab:f3a_proj: per-projection recovery rho (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a_proj
    """
    return {
        "source": "per_projection_{llama,phi3}_aqlm2_L11-22.json",
        "by_model": {m: _build_f3a_proj_model(m) for m in F3A_MODELS},
    }


def _side_rows(db, side, damage):
    """
    Depth-band rows for one read or write side.

    Inputs:
        - db (dict): depth_bands JSON
        - side (str): "read" or "write"
        - damage (float | None): Cell damage_pp for rho

    Outputs:
        - rows (list[dict]): Depth-band table rows
    """
    asr = g(db, "asr_per_condition", default = {}) or {}
    arc = g(db, "arc_per_condition", default = {}) or {}
    rec = g(db, "recovery_pp_per_range", default = {}) or {}
    out = []
    for key, label, layer_str in DEPTH_RANGES:
        asr_key = f"layers_{key}"
        out.append({
            "side": side,
            "range": label,
            "layers": layer_str,
            "asr": rnd(asr.get(asr_key), 1),
            "arc": rnd(arc.get(asr_key), 1),
            "recovery_pp": rnd(rec.get(key), 1),
            "rho_pct": recovery_rho(rec.get(key), damage),
        })
    return out


def _build_f3a_range_model(m):
    """
    Depth-band recovery rho for one model.

    Inputs:
        - m (str): Model key

    Outputs:
        - block (dict): Depth-band rows for tab:f3a_range
    """
    dbr = load(f"{OUT}/localization/depth_bands_{m}_aqlm2_read.json")
    dbw = load(f"{OUT}/localization/depth_bands_{m}_aqlm2_write.json")
    damage = g(dbr, "damage_pp") or g(dbw, "damage_pp")
    return {
        "model_label": MODEL_LABEL[m],
        "damage_pp": rnd(damage, 1),
        "read": _side_rows(dbr, "Read", damage),
        "write": _side_rows(dbw, "Write", damage),
    }


def build_f3a_range():
    """
    Build tab:f3a_range: depth-band recovery rho (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a_range
    """
    return {
        "source": "depth_bands_{llama,phi3}_aqlm2_read/write.json",
        "by_model": {m: _build_f3a_range_model(m) for m in F3A_MODELS},
    }


def _build_f3a_model(m):
    """
    Combined per-projection and depth-band block for one model.

    Inputs:
        - m (str): Model key

    Outputs:
        - block (dict): Rows for tab:f3a
    """
    hl = headline(m, "aqlm2")
    damage = g(hl, "damage_pp")
    rec = g(hl, "recovery_pp", default = {}) or {}
    xs = g(hl, "xstest", default = {}) or {}

    def row(restoration, asr_key, rec_key = None, arc_key = None, xs_key = None):
        """
        Build one tab:f3a row from headline JSON fields.

        Inputs:
            - restoration (str): Display label for the restoration condition
            - asr_key (str): Key under hl["asr"]
            - rec_key (str | None): Key under recovery_pp
            - arc_key (str | None): Key under hl["arc"]
            - xs_key (str | None): Key under xstest block

        Outputs:
            - row (dict): One table row with ASR, rho, ARC, and XSTest FR
        """
        rec_pp = None
        if rec_key:
            rec_pp = rec.get(rec_key)
        delta = None
        if rec_pp is not None:
            delta = rnd(-rec_pp, 1)
        xs_entry = xs.get(xs_key or asr_key, {})
        return {
            "restoration": restoration,
            "asr": rnd(g(hl, "asr", asr_key), 1),
            "delta_quant": delta,
            "rho_pct": recovery_rho(rec_pp, damage),
            "arc": rnd(g(hl, "arc", arc_key or asr_key), 0),
            "xstest_fr": rnd(g(xs_entry, "false_refusal_rate"), 1),
        }

    rows = [
        row("FP (bf16)", "fp32"),
        row("quant", "aqlm2"),
        row("+ Write", "write_restored", rec_key = "write"),
        row("+ Read", "read_restored", rec_key = "read"),
        row("+ Both", "both_restored", rec_key = "both"),
    ]
    mcnemar = g(hl, "mcnemar", default = {}) or {}
    return {
        "model_label": MODEL_LABEL[m],
        "source": f"headline_{m}_aqlm2.json",
        "damage_pp": rnd(damage, 1),
        "rows": rows,
        "mcnemar_p": {k: g(v, "p_value") for k, v in mcnemar.items()},
    }


def build_f3a():
    """
    Build tab:f3a: localization recovery summary (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:f3a
    """
    return {
        "source": "headline_{llama,phi3}_aqlm2.json",
        "by_model": {m: _build_f3a_model(m) for m in F3A_MODELS},
    }


def build_controls():
    """
    Build tab:controls: probe refit and leakage control numbers.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:controls
    """
    out = {}
    for m, label in (("llama", "Llama-AQLM-2"), ("phi3", "Phi-3-AQLM-2")):
        pm = load(f"{OUT}/regime_sweep/param_matched_restore_{m}_aqlm2.json")
        refit = load(f"{OUT}/boundary/probe_refit_control_{m}.json")
        rot = sig("probe_rotation", m, "aqlm2")
        cosmed = load(f"{OUT}/signal_geometry/cosmed_after_restore_{m}_aqlm2.json")
        dd = sig("direction_drift", m, "aqlm2")

        cell = {
            "label": label,
            "param_matched_pp": {
                "full_write": rnd(g(pm, "recovery_pp", "write_full"), 1),
                "matched_read_structured": rnd(g(pm, "recovery_pp", "read_matched_whole"), 1),
                "matched_read_random": rnd(g(pm, "recovery_pp", "read_matched_random_mean"), 1),
            },
            "probe_rotation": {
                "refit_floor": rnd(g(refit, "summary", "cos_floor_median"), 3),
                "observed_rotation": rnd(g(rot, "summary", "cos_median"), 3),
            },
            "direction_cosine": {
                "quantized": rnd(g(dd, "summary", "cos_sim_median"), 3),
            },
        }
        if cosmed and g(cosmed, "summary"):
            sm = cosmed["summary"]
            cell["direction_cosine"]["read_restored"] = rnd(
                sm.get("cosmed_read_restored") or sm.get("read_restored_cos_median"), 3)
            cell["direction_cosine"]["write_restored"] = rnd(
                sm.get("cosmed_write_restored") or sm.get("write_restored_cos_median"), 3)
            cell["direction_cosine"]["both_restored"] = rnd(
                sm.get("cosmed_both_restored") or sm.get("both_restored_cos_median"), 3)
        else:
            cell["direction_cosine"]["note"] = (
                "cosmed_after_restore missing or empty; run "
                f"experiments.signal_geometry.cosmed_after_restore --model {m}")
        out[m] = cell
    return {"source": "param_matched_restore, probe_refit_control, cosmed_after_restore", "cells": out}


def build_trained_repair():
    """
    Build tab:trained_repair: Q-Resafe DPO sweep across ranks and patches.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:trained_repair
    """
    grid = {}
    rows = []
    missing = []
    for patch_key, patch_label in QRESAFE_PATCHES.items():
        grid[patch_label] = {}
        for rank in QRESAFE_RANKS:
            fname = f"qresafe_dpo_llama_aqlm2_{patch_key}_r{rank}.json"
            d = load(f"{OUT}/trained_repair/{fname}")
            if not d:
                missing.append(f"{patch_label}/r{rank}")
                grid[patch_label][str(rank)] = None
                continue
            asr = rnd(g(d, "folded", "asr"), 1)
            grid[patch_label][str(rank)] = asr
            rows.append({
                "patch": patch_label,
                "rank": rank,
                "asr": asr,
                "arc": rnd(g(d, "folded", "arc"), 0),
                "xstest_fr": rnd(g(d, "folded", "xstest_fr"), 1),
                "mean_codes_changed": g(d, "mean_codes_changed"),
                "status": "complete",
            })
    return {
        "source": "qresafe_dpo_llama_aqlm2_{read,write,full}_r{128..2048}.json",
        "reference": {"fp_asr": 41.0, "aqlm2_asr": 62.5, "fp_xstest": 6.0, "aqlm2_arc": 46},
        "grid": grid,
        "rows": rows,
        "missing": missing,
    }


def build_lrqat():
    """
    Build tab:lrqat: LR-QAT int2/int3/int4 repair results.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:lrqat
    """
    rows = []
    refs = {
        "fp16": {"asr": 41.0, "arc": 55, "coherent": True},
        "aqlm2": {"asr": 62.5, "arc": 46, "coherent": None},
    }
    rows.append({"role": "FP reference", "bits": 16, **refs["fp16"]})
    rows.append({"role": "AQLM-2", "bits": 2.0, **refs["aqlm2"]})

    for f in sorted(glob.glob(f"{OUT}/trained_repair/lrqat_llama_aqlm2_*.json")):
        d = load(f)
        if not d or "asr_lrqat_trained" not in d:
            continue
        role = g(d, "role", default = "unknown")
        n_bits = g(d, "n_bits")
        ptq_coll = g(d, "coherence_collapse_ptq_untrained", default = False)
        train_coll = g(d, "coherence_collapse_lrqat_trained", default = False)
        bits_val = float(n_bits) if n_bits is not None else None
        coherent_ptq = "no" if ptq_coll else "yes"
        coherent_train = "no" if train_coll else "yes"
        rows.append({
            "role": f"{role} (PTQ untrained)",
            "bits": bits_val,
            "asr": rnd(g(d, "asr_ptq_untrained"), 1),
            "arc": rnd(g(d, "arc_ptq_untrained"), 0),
            "xstest_fr": rnd(g(d, "false_refusal_ptq_untrained"), 1),
            "coherent": coherent_ptq,
        })
        rows.append({
            "role": f"{role} (LR-QAT trained)",
            "bits": bits_val,
            "asr": rnd(g(d, "asr_lrqat_trained"), 1),
            "arc": rnd(g(d, "arc_lrqat_trained"), 0),
            "xstest_fr": rnd(g(d, "false_refusal_lrqat_trained"), 1),
            "coherent": coherent_train,
            "effective_bits": rnd(g(d, "repaired_effective_bits"), 2),
        })
    return {"source": "lrqat_llama_aqlm2_*.json", "rows": rows}


def build_ppl():
    """
    Build tab:ppl: WikiText perplexity under restoration conditions.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:ppl
    """
    by_model = {}
    for m in MODELS:
        rows = []
        fp_path = f"{OUT}/regime_sweep/wikitext_ppl_{m}_fp32_wikitext2_ppl.json"
        fp_j = load(fp_path)
        fp_ppl = g(fp_j, "wikitext2_ppl")
        if fp_ppl is not None:
            rows.append({
                "quant": "fp32",
                "quant_label": "FP",
                "ppl": rnd(fp_ppl, 2),
                "ratio_fp": 1.0,
            })
        for q in QUANTS:
            if m == "qwen" and q == "aqlm2":
                continue
            j = load(f"{OUT}/regime_sweep/wikitext_ppl_{m}_{q}_wikitext2_ppl.json")
            if not j:
                continue
            rows.append({
                "quant": q,
                "quant_label": QUANT_LABEL[q],
                "ppl": rnd(g(j, "wikitext2_ppl"), 2),
                "ratio_fp": rnd(g(j, "ppl_ratio"), 2),
            })
        if rows:
            by_model[m] = {"model_label": MODEL_LABEL[m], "rows": rows}
    return {"source": "wikitext_ppl_*_wikitext2_ppl.json", "by_model": by_model}


def _dequant_row_from_json(j, label):
    """
    Format one targeted-dequant row from a result JSON.

    Inputs:
        - j (dict): targeted_dequant JSON
        - label (str): Row label for the table

    Outputs:
        - row (dict): Formatted ASR/ARC/recovery fields
    """
    if not j:
        return None
    eff = g(j, "effective_bits")
    if isinstance(eff, dict):
        bits = rnd(eff.get("effective_bits"), 2)
    else:
        bits = rnd(eff, 2)
    asr_block = g(j, "asr_per_condition", default = {}) or {}
    arc_block = g(j, "arc_per_condition", default = {}) or {}
    xs = g(j, "xstest_false_refusal", default = {}) or {}
    restore_key = next(
        (k for k in asr_block if k not in ("fp32", "aqlm2")),
        None,
    )
    return {
        "label": label,
        "bits": bits,
        "asr": rnd(asr_block.get(restore_key) if restore_key else g(j, "quantized_asr"), 1),
        "arc": rnd(arc_block.get(restore_key) if restore_key else None, 0),
        "xstest_fr": rnd(xs.get(restore_key) if restore_key else None, 1),
    }


def build_dequant():
    """
    Build tab:dequant: targeted dequantization baselines.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:dequant
    """
    out = {}
    for m in F3A_MODELS:
        hl = headline(m, "aqlm2")
        rs = restore(m, "aqlm2")
        rows = [
            {
                "label": "full precision",
                "bits": 16,
                "asr": rnd(g(hl, "asr", "fp32"), 1),
                "arc": rnd(g(hl, "arc", "fp32"), 0),
                "xstest_fr": rnd(g(g(hl, "xstest", default = {}).get("fp32", {}), "false_refusal_rate"), 1),
            },
            {
                "label": "AQLM-2",
                "bits": 2.0,
                "asr": rnd(g(rs, "quantized_asr"), 1),
                "arc": rnd(g(rs, "arc", "aqlm2"), 0),
                "xstest_fr": rnd(g(g(hl, "xstest", default = {}).get("aqlm2", {}), "false_refusal_rate"), 1),
            },
        ]
        mid = load(f"{OUT}/regime_sweep/targeted_dequant_{m}_aqlm2_read_gptq4_L11-22.json")
        all_j = load(f"{OUT}/regime_sweep/targeted_dequant_{m}_aqlm2_read_gptq4_all.json")
        for j, label in ((mid, "GPTQ-4, middle"), (all_j, "GPTQ-4, all")):
            row = _dequant_row_from_json(j, label)
            if row:
                rows.append(row)
        awq = load(f"{OUT}/regime_sweep/targeted_dequant_{m}_aqlm2_read_awq_all.json")
        if awq and m == "llama":
            rows.append({
                "label": "AWQ-4, all (collapse control)",
                "bits": rnd(g(awq, "effective_bits", "effective_bits"), 2),
                "asr": rnd(g(awq, "asr_per_condition", "awq4_restore"), 1),
                "arc": rnd(g(awq, "arc_per_condition", "awq4_restore"), 0),
                "xstest_fr": None,
            })
        out[m] = {"model_label": MODEL_LABEL[m], "rows": rows}
    return {"source": "targeted_dequant_* + headline/restoration baselines", "by_model": out}


def build_leakage():
    """
    Build tab:leakage: probe leakage control summary.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for tab:leakage
    """
    cells = {}
    for m in F3A_MODELS:
        j = load(f"{OUT}/boundary/probe_leakage_control_{m}.json")
        sm = g(j, "summary", default = {}) or {}
        cells[m] = {
            "model_label": MODEL_LABEL[m],
            "specificity": rnd(sm.get("residual_xstest_safe_specificity"), 3),
            "unsafe_recall": rnd(sm.get("residual_xstest_unsafe_recall"), 3),
            "harmbench_transfer_recall": rnd(sm.get("residual_harmbench_transfer_recall"), 3),
            "surface_specificity": rnd(sm.get("surface_xstest_safe_specificity"), 3),
            "surface_unsafe_recall": rnd(sm.get("surface_xstest_unsafe_recall"), 3),
        }
    return {"source": "probe_leakage_control_{llama,phi3}.json", "cells": cells}


def build_table_stats():
    """
    Build table:stats: headline paired McNemar replication.

    Inputs:
        - None

    Outputs:
        - block (dict): LaTeX table block for table:stats
    """
    hl = headline("llama", "aqlm2")
    fbc = g(hl, "flags_by_cond", default = {}) or {}
    q_flags = [int(bool(x)) for x in fbc.get("aqlm2", [])]
    read_flags = [int(bool(x)) for x in fbc.get("read_restored", [])]
    write_flags = [int(bool(x)) for x in fbc.get("write_restored", [])]

    rows = []
    specs = [
        ("quant vs. read", q_flags, read_flags),
        ("quant vs. write", q_flags, write_flags),
        ("read vs. write", write_flags, read_flags),
    ]
    for label, a_flags, b_flags in specs:
        mc = mcnemar_exact(a_flags, b_flags)
        delta = rnd(100.0 * (np.mean(a_flags) - np.mean(b_flags)), 1)
        ci = bootstrap_delta_asr(a_flags, b_flags, n_boot = 2000, seed = 42)
        rows.append({
            "comparison": label,
            "b": mc["b"],
            "c": mc["c"],
            "p": mc["p_value"],
            "delta_asr_pp": delta,
            "ci95": [rnd(ci[0], 1), rnd(ci[1], 1)],
        })
    return {"source": "headline_llama_aqlm2.json flags_by_cond", "rows": rows}


def build():
    """
    Assemble all paper-table blocks into one JSON document.

    Inputs:
        - None

    Outputs:
        - paper_values (dict): All LaTeX table blocks keyed by label
    """
    return {
        "meta": {
            "generated_by": "scripts/paper_values.py",
            "note": (
                "ASR/ARC/damage in percentage points. dASR = quant_asr - fp32_asr. "
                "Studiable AQLM cells use headline JSON for tab:f3a/regime ASR; "
                "restoration JSON for swap_valid and dequant AQLM baseline."
            ),
            "seed": 42,
        },
        "tab:geometry": build_geometry(),
        "tab:regime": build_regime(),
        "tab:f3a_proj": build_f3a_proj(),
        "tab:f3a_range": build_f3a_range(),
        "tab:f3a": build_f3a(),
        "tab:controls": build_controls(),
        "tab:dequant": build_dequant(),
        "tab:ppl": build_ppl(),
        "tab:leakage": build_leakage(),
        "table:stats": build_table_stats(),
        "tab:trained_repair": build_trained_repair(),
        "tab:lrqat": build_lrqat(),
    }


def fmt_cell(x):
    """
    Format one table cell for markdown output.

    Inputs:
        - x: Cell value (any)

    Outputs:
        - text (str): Display string ("--" when x is None)
    """
    if x is None:
        return "--"
    return str(x)


def as_markdown(pv):
    """
    Render paper_values JSON as human-readable markdown tables.

    Inputs:
        - pv (dict): paper_values document from build()

    Outputs:
        - markdown (str): Markdown table text
    """
    lines = ["# Paper values (LaTeX table alignment)\n", pv["meta"]["note"] + "\n"]

    lines.append("\n## tab:geometry\n")
    lines.append("| Model | Quant | dmed | a_min | cosmed | coswmed |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for m, block in pv["tab:geometry"]["by_model"].items():
        for row in block["rows"]:
            lines.append(
                f"| {block['model_label']} | {row['quant_label']} | "
                f"{fmt_cell(row['dmed'])} | {fmt_cell(row['a_min'])} | "
                f"{fmt_cell(row['cosmed'])} | {fmt_cell(row['coswmed'])} |")

    lines.append("\n## tab:regime\n")
    lines.append("| Model | Quant | FP | Quant | dASR | p(Holm) | +Write | +Read | +Both |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key, v in pv["tab:regime"]["cells"].items():
        a, r = v["asr"], v["arc"]
        cells = " | ".join(f"{fmt_cell(a[c])}/{fmt_cell(r[c])}" for c in CONDS[:2])
        rest = " | ".join(f"{fmt_cell(a[c])}/{fmt_cell(r[c])}" for c in CONDS[2:])
        paired = v.get("paired") or {}
        dasr = fmt_cell(paired.get("delta_asr"))
        ph = paired.get("p_holm")
        ph_str = fmt_cell(f"{ph:.2e}" if ph is not None else None)
        lines.append(
            f"| {v['model_label']} | {v['quant_label']} | {cells} | {dasr} | {ph_str} | {rest} |")

    for section, key in (
        ("tab:f3a_proj", "rows"),
        ("tab:f3a", "rows"),
    ):
        block = pv.get(section, {})
        if "by_model" in block:
            lines.append(f"\n## {section}\n")
            for m, mb in block["by_model"].items():
                lines.append(f"\n### {mb.get('model_label', m)}\n")
                if section == "tab:f3a_proj":
                    lines.append("| Projection | Function | Matrices | rho % |")
                    lines.append("|---|---|---:|---:|")
                    for row in mb["rows"]:
                        lines.append(
                            f"| {row['projection']} | {row['function']} | {row['matrices']} | "
                            f"{fmt_cell(row['rho_pct'])} |")
                else:
                    lines.append("| Restoration | ASR | delta quant | rho % | ARC | XSTest |")
                    lines.append("|---|---|---:|---:|---:|---:|")
                    for row in mb["rows"]:
                        lines.append(
                            f"| {row['restoration']} | {fmt_cell(row['asr'])} | "
                            f"{fmt_cell(row['delta_quant'])} | {fmt_cell(row['rho_pct'])} | "
                            f"{fmt_cell(row['arc'])} | {fmt_cell(row['xstest_fr'])} |")

    lines.append("\n## tab:f3a_range\n")
    for m, mb in pv["tab:f3a_range"]["by_model"].items():
        lines.append(f"\n### {mb['model_label']}\n")
        lines.append("| Side | Range | Layers | ASR | Recovery pp | ARC |")
        lines.append("|---|---|---|---:|---:|---:|")
        for side_key in ("read", "write"):
            for row in mb[side_key]:
                lines.append(
                    f"| {row['side']} | {row['range']} | {row['layers']} | "
                    f"{fmt_cell(row['asr'])} | {fmt_cell(row['recovery_pp'])} | "
                    f"{fmt_cell(row['arc'])} |")

    lines.append("\n## tab:controls\n")
    for m, cell in pv["tab:controls"]["cells"].items():
        lines.append(f"\n### {cell['label']}\n")
        pm = cell["param_matched_pp"]
        pr = cell["probe_rotation"]
        dc = cell["direction_cosine"]
        lines.append("| Control | Value |")
        lines.append("|---|---:|")
        lines.append(f"| Full write (pp) | {fmt_cell(pm['full_write'])} |")
        lines.append(f"| Matched read structured (pp) | {fmt_cell(pm['matched_read_structured'])} |")
        lines.append(f"| Matched read random (pp) | {fmt_cell(pm['matched_read_random'])} |")
        lines.append(f"| Refit floor | {fmt_cell(pr['refit_floor'])} |")
        lines.append(f"| Observed rotation | {fmt_cell(pr['observed_rotation'])} |")
        for k in ("quantized", "read_restored", "write_restored", "both_restored"):
            if k in dc:
                lines.append(f"| Direction cos {k} | {fmt_cell(dc[k])} |")

    lines.append("\n## tab:dequant\n")
    for m, block in pv["tab:dequant"]["by_model"].items():
        lines.append(f"\n### {block['model_label']}\n")
        lines.append("| Replacement | Bits | ASR | ARC | XSTest |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in block["rows"]:
            lines.append(
                f"| {row['label']} | {fmt_cell(row['bits'])} | {fmt_cell(row['asr'])} | "
                f"{fmt_cell(row['arc'])} | {fmt_cell(row['xstest_fr'])} |")

    lines.append("\n## tab:ppl\n")
    for m, block in pv["tab:ppl"]["by_model"].items():
        lines.append(f"\n### {block['model_label']}\n")
        lines.append("| Quant | PPL | x FP |")
        lines.append("|---|---:|---:|")
        for row in block["rows"]:
            lines.append(
                f"| {row['quant_label']} | {fmt_cell(row['ppl'])} | {fmt_cell(row['ratio_fp'])} |")

    lines.append("\n## tab:leakage\n")
    lines.append("| Model | Specificity | Unsafe recall | HarmBench transfer |")
    lines.append("|---|---:|---:|---:|")
    for m, cell in pv["tab:leakage"]["cells"].items():
        lines.append(
            f"| {cell['model_label']} | {fmt_cell(cell['specificity'])} | "
            f"{fmt_cell(cell['unsafe_recall'])} | {fmt_cell(cell['harmbench_transfer_recall'])} |")

    lines.append("\n## table:stats\n")
    lines.append("| Comparison | b | c | p | dASR pp | 95% CI |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in pv["table:stats"]["rows"]:
        ci = row.get("ci95") or [None, None]
        ci_str = f"[{fmt_cell(ci[0])}, {fmt_cell(ci[1])}]"
        p = row.get("p")
        p_str = fmt_cell(f"{p:.2e}" if p is not None else None)
        lines.append(
            f"| {row['comparison']} | {row['b']} | {row['c']} | {p_str} | "
            f"{fmt_cell(row['delta_asr_pp'])} | {ci_str} |")

    lines.append("\n## tab:trained_repair\n")
    lines.append("| Patch | Rank | ASR | ARC | XSTest |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in pv["tab:trained_repair"]["rows"]:
        lines.append(
            f"| {row['patch']} | {row['rank']} | {fmt_cell(row.get('asr'))} | "
            f"{fmt_cell(row.get('arc'))} | {fmt_cell(row.get('xstest_fr'))} |")

    lines.append("\n## tab:lrqat\n")
    lines.append("| Role | Bits | ASR | ARC | XSTest | Coherent |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in pv["tab:lrqat"]["rows"]:
        lines.append(
            f"| {row['role']} | {fmt_cell(row.get('bits'))} | {fmt_cell(row.get('asr'))} | "
            f"{fmt_cell(row.get('arc'))} | {fmt_cell(row.get('xstest_fr'))} | "
            f"{fmt_cell(row.get('coherent'))} |")

    return "\n".join(lines)


def main():
    """
    CLI entry point: write paper_values.json and optional markdown.

    Inputs:
        - None (pass --md for markdown instead of JSON)

    Outputs:
        - None (prints JSON or markdown to stdout)
    """
    pv = build()
    if "--md" in sys.argv[1:]:
        print(as_markdown(pv))
    else:
        print(json.dumps(pv, indent = 2, default = str))


if __name__ == "__main__":
    main()
