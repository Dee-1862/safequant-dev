# Pull paper-table values from outputs/ JSON files (one block per LaTeX table label).
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "aqlm2", "gptq3", "gptq2"]
CONDS = ["fp32", "quant", "write_restored", "read_restored", "both_restored"]
OUT = "outputs"

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

PROJ_ORDER = [
    ("self_attn.q_proj", "W_q", "Attention routing", 1),
    ("self_attn.k_proj", "W_k", "Attention routing", 1),
    ("mlp.up_proj", "W_up", "MLP feature lift", 1),
    ("mlp.down_proj", "W_down", "MLP output", 1),
    ("self_attn.v_proj", "W_v", "Attention content", 1),
    ("mlp.gate_proj", "W_gate", "MLP gating", 1),
    ("self_attn.o_proj", "W_o", "Attention output", 1),
]

DEPTH_RANGES = [
    ("early", "Early", "[0,11)"),
    ("middle", "Middle", "[11,22)"),
    ("late", "Late", "[22,32)"),
]

PHI3_PROJ_ORDER = [
    ("self_attn.qkv_proj", "W_qkv", "Attention q,k,v (fused)", 1),
    ("self_attn.o_proj", "W_o", "Attention output", 1),
    ("mlp.gate_up_proj", "W_gate_up", "MLP gate+lift (fused)", 1),
    ("mlp.down_proj", "W_down", "MLP output", 1),
]

QRESAFE_RANKS = (128, 256, 512, 1024, 2048)
QRESAFE_PATCHES = (("Read", "read"), ("Write", "write"), ("Both", "full"))
DEQUANT_CELLS = (
    ("llama", "all"),
    ("llama", "L11-22"),
    ("phi3", "all"),
    ("phi3", "L11-22"),
)

def load(p):
    """
    Load one JSON result file, returning None on missing or corrupt data.

    Inputs:
        - p (str | Path): Path to a result JSON

    Outputs:
        - data (dict | None): Parsed JSON or None
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
        - x: Numeric or None
        - n (int): Decimal places

    Outputs:
        - out (float | None): Rounded float or None
    """
    return None if x is None else round(float(x), n)


def sig(exp, m, q):
    """
    Path helper for signal_geometry JSON.

    Inputs:
        - exp (str): Experiment stem
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - data (dict | None): Loaded JSON
    """
    return load(f"{OUT}/signal_geometry/{exp}_{m}_{q}.json")


def restore(m, q):
    """
    Path helper for restoration_sweep JSON.

    Inputs:
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - data (dict | None): Loaded JSON
    """
    return load(f"{OUT}/regime_sweep/restoration_sweep_{m}_{q}.json")


def dmed_quant(m, q):
    """
    Median per-layer separation_sigma on the quantized model (Table 1 dmed).

    Inputs:
        - m (str): Filename model token
        - q (str): Quantizer id

    Outputs:
        - dmed (float | None): Median separation across layers
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
    return rnd(statistics.median(vals), 2) if vals else None


def cond_map(d, block):
    """
    Normalize restoration_sweep ASR/ARC blocks to fixed condition keys.

    Inputs:
        - d (dict): restoration_sweep JSON
        - block (str): Top-level key ('asr_per_condition' or 'arc')

    Outputs:
        - out (dict): Keys fp32, quant, write_restored, read_restored, both_restored
    """
    src = g(d, block, default = {}) or {}
    qtz = next(
        (k for k in src if k not in ("fp32", "write_restored", "read_restored", "both_restored")),
        None,
    )
    out = {}
    for c in CONDS:
        key = qtz if c == "quant" else c
        val = src.get(key)
        out[c] = rnd(val, 1) if val is not None else None
    return out


def screen_verdict(rs):
    """
    Heuristic damaged-coherent screen label (Table 2 verdict column).

    Inputs:
        - rs (dict): restoration_sweep JSON

    Outputs:
        - verdict (str | None): Screen label or None if file incomplete
    """
    if not rs:
        return None
    if g(rs, "restore_status") == "arch_mismatch_na":
        return "N/A (arch mismatch)"
    damage = g(rs, "damage_pp")
    fp_asr = g(rs, "fp32_asr")
    arc_q = cond_map(rs, "arc").get("quant")
    read_rec = g(rs, "read_recovery_pp")

    # Coherence collapse: utility near chance
    if arc_q is not None and arc_q < 35:
        return "fails coherence"
    # High baseline refusal already — hard to interpret damage
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
        - recovery_pp (float | None): Points recovered
        - damage_pp (float | None): Baseline damage in pp

    Outputs:
        - rho (int | None): Integer percent or None
    """
    if recovery_pp is None or not damage_pp:
        return None
    return int(round(100.0 * float(recovery_pp) / float(damage_pp)))


def build_geometry():
    """
    Build tab:geometry — dmed, a_min, cosmed, coswmed per model/quant cell.

    Inputs:
        - None

    Outputs:
        - block (dict): Table payload keyed by model then quant
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


# AQLM inference is nondeterministic at +-1-2pp on 200 prompts, so the
# restoration re-run's ASR for AQLM cells can differ from the headline run.
# For the two studiable AQLM cells (which also appear in tab:f3a and carry the
# significance stats), source ASR/ARC/damage from the single headline run so
# every table agrees by construction. gptq/awq cells are deterministic -> use
# restoration directly.
STUDIABLE_HEADLINE = {("llama", "aqlm2"), ("phi3", "aqlm2")}


def _paired_screen_map():
    """
    Holm-adjusted paired Delta ASR screen for all 19 regime cells.

    Inputs:
        - None

    Outputs:
        - out (dict): cell key -> McNemar stats
    """
    from experiments.regime_sweep.paired_screen import MODELS as PM, QUANTS as PQ, sweep

    out = {}
    for r in sweep([(m, q) for m in PM for q in PQ]):
        if r.get("status") != "ok":
            continue
        out[f"{r['model']}/{r['quant']}"] = {
            "delta_asr": rnd(r["delta_asr"], 1),
            "ci95": [rnd(r["ci95"][0], 1), rnd(r["ci95"][1], 1)],
            "p_mcnemar": r["p_mcnemar"],
            "p_holm": r["p_holm"],
            "b": r["b"],
            "c": r["c"],
        }
    return out


def build_regime():
    """
    Build tab:regime — ASR/ARC per condition + screen verdict.

    Inputs:
        - None

    Outputs:
        - block (dict): Table payload for damaged-coherent screen
    """
    paired_map = _paired_screen_map()
    cells = {}
    for m in MODELS:
        for q in QUANTS:
            rs = restore(m, q)
            if not rs:
                continue
            hl = (load(f"{OUT}/headline/headline_{m}_{q}.json")
                  if (m, q) in STUDIABLE_HEADLINE else None)
            if hl:
                asr, arc = cond_map(hl, "asr"), cond_map(hl, "arc")
                damage, src = rnd(g(hl, "damage_pp"), 1), "headline"
            else:
                asr, arc = cond_map(rs, "asr_per_condition"), cond_map(rs, "arc")
                damage, src = rnd(g(rs, "damage_pp"), 1), "restoration_sweep"
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
            }
            if f"{m}/{q}" in paired_map:
                cell["paired"] = paired_map[f"{m}/{q}"]
            cells[f"{m}/{q}"] = cell
    return {
        "conditions": CONDS,
        "source": ("restoration_sweep_* (ARC n=100); AQLM studiable cells "
                   "llama/aqlm2 + phi3/aqlm2 sourced from headline_* for "
                   "ASR/ARC consistency with tab:f3a"),
        "cells": cells,
    }


def _f3a_proj_block(model, proj_order, fname):
    """
    Build per-projection rho rows for one model middle-band JSON.

    Inputs:
        - model (str): Filename model token
        - proj_order (list): Projection metadata tuples
        - fname (str): JSON filename under localization/

    Outputs:
        - block (dict): Table payload
    """
    pp = load(f"{OUT}/localization/{fname}")
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
        "source": fname,
        "layer_range": g(pp, "layer_range"),
        "damage_pp": rnd(damage, 1),
        "rows": rows,
    }


def build_f3a_proj():
    """
    Build tab:f3a_proj — per-projection recovery rho (Llama + Phi-3, middle band).

    Inputs:
        - None

    Outputs:
        - block (dict): Per-model projection tables
    """
    return {
        "llama": _f3a_proj_block(
            "llama", PROJ_ORDER, "per_projection_llama_aqlm2_L11-22.json"),
        "phi3": _f3a_proj_block(
            "phi3", PHI3_PROJ_ORDER, "per_projection_phi3_aqlm2_L11-22.json"),
    }


def _f3a_range_block(model):
    """
    Build depth-band read/write recovery for one model.

    Inputs:
        - model (str): Filename model token

    Outputs:
        - block (dict): Read and write side rows
    """
    dbr = load(f"{OUT}/localization/depth_bands_{model}_aqlm2_read.json")
    dbw = load(f"{OUT}/localization/depth_bands_{model}_aqlm2_write.json")
    damage = g(dbr, "damage_pp") or g(dbw, "damage_pp")

    def side_rows(db, side):
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

    return {
        "source": f"depth_bands_{model}_aqlm2_read/write.json",
        "damage_pp": rnd(damage, 1),
        "read": side_rows(dbr, "Read"),
        "write": side_rows(dbw, "Write"),
    }


def build_f3a_range():
    """
    Build tab:f3a_range — depth-band read/write recovery (Llama + Phi-3).

    Inputs:
        - None

    Outputs:
        - block (dict): Per-model depth tables
    """
    return {
        "llama": _f3a_range_block("llama"),
        "phi3": _f3a_range_block("phi3"),
    }


def _f3a_block(model, quant_key="aqlm2"):
    """
    Build static restoration headline rows for one studiable cell.

    Inputs:
        - model (str): Filename model token
        - quant_key (str): Quantizer id in headline JSON

    Outputs:
        - block (dict): FP / quant / restoration rows
    """
    hl = load(f"{OUT}/headline/headline_{model}_{quant_key}.json")
    if not hl:
        return {"source": f"headline_{model}_{quant_key}.json", "status": "missing"}
    damage = g(hl, "damage_pp")
    rec = g(hl, "recovery_pp", default = {}) or {}
    xs = g(hl, "xstest", default = {}) or {}
    rows = [
        {"restoration": "FP32", "asr": rnd(g(hl, "asr", "fp32"), 1),
         "delta_quant": None, "rho_pct": None,
         "arc": rnd(g(hl, "arc", "fp32"), 0),
         "xstest_fr": rnd(g(xs.get("fp32", {}), "false_refusal_rate"), 1)},
        {"restoration": "quant", "asr": rnd(g(hl, "asr", quant_key), 1),
         "delta_quant": None, "rho_pct": None,
         "arc": rnd(g(hl, "arc", quant_key), 0),
         "xstest_fr": rnd(g(xs.get(quant_key, {}), "false_refusal_rate"), 1)},
        {"restoration": "+ Write-side", "asr": rnd(g(hl, "asr", "write_restored"), 1),
         "delta_quant": rnd(-g(rec, "write"), 1) if rec.get("write") else None,
         "rho_pct": recovery_rho(rec.get("write"), damage),
         "arc": rnd(g(hl, "arc", "write_restored"), 0),
         "xstest_fr": rnd(g(xs.get("write_restored", {}), "false_refusal_rate"), 1)},
        {"restoration": "+ Read-side", "asr": rnd(g(hl, "asr", "read_restored"), 1),
         "delta_quant": rnd(-g(rec, "read"), 1) if rec.get("read") else None,
         "rho_pct": recovery_rho(rec.get("read"), damage),
         "arc": rnd(g(hl, "arc", "read_restored"), 0),
         "xstest_fr": rnd(g(xs.get("read_restored", {}), "false_refusal_rate"), 1)},
        {"restoration": "+ Both", "asr": rnd(g(hl, "asr", "both_restored"), 1),
         "delta_quant": rnd(-g(rec, "both"), 1) if rec.get("both") else None,
         "rho_pct": recovery_rho(rec.get("both"), damage),
         "arc": rnd(g(hl, "arc", "both_restored"), 0),
         "xstest_fr": rnd(g(xs.get("both_restored", {}), "false_refusal_rate"), 1)},
    ]
    return {
        "source": f"headline_{model}_{quant_key}.json",
        "damage_pp": rnd(damage, 1),
        "rows": rows,
        "mcnemar_p": {k: g(v, "p_value") for k, v in (g(hl, "mcnemar", default = {}) or {}).items()},
    }


def build_f3a():
    """
    Build tab:f3a — static restoration headline (Llama + Phi-3 AQLM-2).

    Inputs:
        - None

    Outputs:
        - block (dict): Per-model headline tables
    """
    return {
        "llama": _f3a_block("llama"),
        "phi3": _f3a_block("phi3"),
    }


def build_controls():
    """
    Build tab:controls — localization controls for Llama and Phi-3 AQLM-2.

    Inputs:
        - None

    Outputs:
        - block (dict): Param-matched restore, probe refit floor, cosmed after restore
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
            cell["direction_cosine"]["read_restored"] = rnd(
                g(cosmed, "summary", "read_restored_cos_median"), 3)
            cell["direction_cosine"]["write_restored"] = rnd(
                g(cosmed, "summary", "write_restored_cos_median"), 3)
            cell["direction_cosine"]["both_restored"] = rnd(
                g(cosmed, "summary", "both_restored_cos_median"), 3)
        else:
            cell["direction_cosine"]["note"] = (
                "cosmed_after_restore missing or empty. Run "
                f"experiments.signal_geometry.cosmed_after_restore --model {m}")
        out[m] = cell
    return {"source": "param_matched_restore, probe_refit_control, cosmed_after_restore", "cells": out}


def build_trained_repair():
    """
    Build tab:trained_repair — Q-Resafe folded ASR at ranks 128 through 2048.

    Inputs:
        - None

    Outputs:
        - block (dict): Patch x rank grid
    """
    grid = {}
    for patch, stem in QRESAFE_PATCHES:
        grid[patch] = {}
        for rank in QRESAFE_RANKS:
            fname = f"qresafe_dpo_llama_aqlm2_{stem}_r{rank}.json"
            d = load(f"{OUT}/trained_repair/{fname}")
            if not d:
                grid[patch][str(rank)] = {"status": "missing"}
                continue
            grid[patch][str(rank)] = {
                "asr": rnd(g(d, "folded", "asr"), 1),
                "arc": rnd(g(d, "folded", "arc"), 0),
                "xstest_fr": rnd(g(d, "folded", "xstest_fr"), 1),
                "mean_codes_changed": g(d, "mean_codes_changed"),
                "status": "complete",
            }
    return {
        "source": "qresafe_dpo_llama_aqlm2_{read,write,full}_r{128..2048}.json",
        "reference": {"fp_asr": 41.0, "aqlm2_asr": 62.5, "fp_xstest": 6.0, "aqlm2_arc": 46},
        "ranks": list(QRESAFE_RANKS),
        "grid": grid,
    }


def build_lrqat():
    """
    Build tab:lrqat — LR-QAT cells for Llama AQLM-2 read/write x int2/int3/int4.

    Inputs:
        - None

    Outputs:
        - block (dict): Role x bits grid with coherence flag
    """
    rows = []
    refs = {"fp16": {"asr": 41.0, "arc": 55, "coherent": True},
            "aqlm2": {"asr": 62.5, "arc": 46, "coherent": None}}
    rows.append({"role": "FP reference", "bits": 16, **refs["fp16"]})
    rows.append({"role": "AQLM-2", "bits": 2.0, **refs["aqlm2"]})

    for f in sorted(glob.glob(f"{OUT}/trained_repair/lrqat_llama_aqlm2_*.json")):
        stem = Path(f).stem
        if "L11-22" in stem or "_both_" in stem:
            continue
        d = load(f)
        if not d or "asr_lrqat_trained" not in d:
            continue
        role = g(d, "role", default = "unknown")
        n_bits = g(d, "n_bits")
        ptq_coll = g(d, "coherence_collapse_ptq_untrained", default = False)
        train_coll = g(d, "coherence_collapse_lrqat_trained", default = False)
        rows.append({
            "role": f"{role} (PTQ untrained)",
            "bits": float(n_bits) if n_bits is not None else None,
            "asr": rnd(g(d, "asr_ptq_untrained"), 1),
            "arc": rnd(g(d, "arc_ptq_untrained"), 0),
            "coherent": "no" if ptq_coll else "yes",
        })
        rows.append({
            "role": f"{role} (LR-QAT trained)",
            "bits": float(n_bits) if n_bits is not None else None,
            "asr": rnd(g(d, "asr_lrqat_trained"), 1),
            "arc": rnd(g(d, "arc_lrqat_trained"), 0),
            "coherent": "no" if train_coll else "yes",
            "effective_bits": rnd(g(d, "repaired_effective_bits"), 2),
        })
    return {"source": "lrqat_llama_aqlm2_read/write_int{2,3,4}.json", "rows": rows}


def build_ppl():
    """
    Build tab:ppl — WikiText-2 perplexity per model/quant cell.

    Inputs:
        - None

    Outputs:
        - block (dict): PPL absolute and ratio to FP32
    """
    by_model = {}
    for m in MODELS:
        rows = []
        fp = load(f"{OUT}/regime_sweep/wikitext_ppl_{m}_fp32_wikitext2_ppl.json")
        if fp and g(fp, "wikitext2_ppl") is not None:
            rows.append({
                "quant": "fp32",
                "quant_label": "FP",
                "ppl": rnd(g(fp, "wikitext2_ppl"), 2),
                "ratio_fp": 1.0,
                "status": g(fp, "status", default = "ok"),
            })
        for q in QUANTS:
            if m == "qwen" and q == "aqlm2":
                continue
            d = load(f"{OUT}/regime_sweep/wikitext_ppl_{m}_{q}_wikitext2_ppl.json")
            if not d:
                continue
            rows.append({
                "quant": q,
                "quant_label": QUANT_LABEL[q],
                "ppl": rnd(g(d, "wikitext2_ppl"), 2),
                "ratio_fp": rnd(g(d, "ppl_ratio"), 2),
                "status": g(d, "status", default = "ok"),
            })
        if rows:
            by_model[m] = {"model_label": MODEL_LABEL[m], "rows": rows}
    return {"source": "wikitext_ppl_*_wikitext2_ppl.json", "by_model": by_model}


def build_dequant():
    """
    Build tab:dequant — GPTQ-4 read-side replacement on AQLM-2 cells.

    Inputs:
        - None

    Outputs:
        - block (dict): Per-model all/middle band rows
    """
    out = {}
    for model, band in DEQUANT_CELLS:
        suffix = "all" if band == "all" else band
        fname = f"targeted_dequant_{model}_aqlm2_read_gptq4_{suffix}.json"
        d = load(f"{OUT}/regime_sweep/{fname}")
        if not d:
            out[f"{model}/{band}"] = {"status": "missing", "source": fname}
            continue
        eb = g(d, "effective_bits", default = {}) or {}
        out[f"{model}/{band}"] = {
            "source": fname,
            "model_label": MODEL_LABEL[model],
            "band": band,
            "asr": rnd(g(d, "asr_per_condition", "gptq44_restore"), 1),
            "arc": rnd(g(d, "arc_per_condition", "gptq44_restore"), 0),
            "xstest_fr": rnd(g(d, "xstest_false_refusal", "gptq44_restore"), 1),
            "effective_bits": rnd(g(eb, "effective_bits"), 2),
            "aqlm2_asr": rnd(g(d, "asr_per_condition", "aqlm2"), 1),
        }
    awq = load(f"{OUT}/regime_sweep/targeted_dequant_llama_aqlm2_read_awq_all.json")
    if awq:
        out["llama/awq_collapse"] = {
            "source": "targeted_dequant_llama_aqlm2_read_awq_all.json",
            "asr": rnd(g(awq, "asr_per_condition", "awq4_restore"), 1),
            "arc": rnd(g(awq, "arc_per_condition", "awq4_restore"), 0),
            "note": "AWQ substitution collapse (swap-invalid control)",
        }
    return out


def build_leakage():
    """
    Build probe leakage control summary (surface vs residual probe).

    Inputs:
        - None

    Outputs:
        - block (dict): Per-model XSTest specificity/recall
    """
    cells = {}
    for m in ("llama", "phi3"):
        d = load(f"{OUT}/boundary/probe_leakage_control_{m}.json")
        if not d:
            continue
        s = g(d, "summary", default = {}) or {}
        cells[m] = {
            "label": MODEL_LABEL[m],
            "residual_xstest_safe_specificity": rnd(g(s, "residual_xstest_safe_specificity"), 3),
            "residual_xstest_unsafe_recall": rnd(g(s, "residual_xstest_unsafe_recall"), 3),
            "residual_harmbench_transfer_recall": rnd(g(s, "residual_harmbench_transfer_recall"), 3),
            "surface_xstest_unsafe_recall": rnd(g(s, "surface_xstest_unsafe_recall"), 3),
        }
    return {"source": "probe_leakage_control_{llama,phi3}.json", "cells": cells}


def build_stats():
    """
    Build table:stats — paired McNemar on Llama AQLM-2 headline run.

    Inputs:
        - None

    Outputs:
        - block (dict): quant vs read/write and read vs write
    """
    hl = load(f"{OUT}/headline/headline_llama_aqlm2.json")
    if not hl:
        return {"status": "missing headline_llama_aqlm2.json"}
    rec = g(hl, "recovery_pp", default = {}) or {}
    rows = [
        {"comparison": "quant vs. read", "b": g(hl, "mcnemar", "aqlm2_vs_read", "b"),
         "c": g(hl, "mcnemar", "aqlm2_vs_read", "c"),
         "p": g(hl, "mcnemar", "aqlm2_vs_read", "p_value"),
         "delta_asr_pp": rnd(g(rec, "read"), 1)},
        {"comparison": "quant vs. write", "b": g(hl, "mcnemar", "aqlm2_vs_write", "b"),
         "c": g(hl, "mcnemar", "aqlm2_vs_write", "c"),
         "p": g(hl, "mcnemar", "aqlm2_vs_write", "p_value"),
         "delta_asr_pp": rnd(g(rec, "write"), 1)},
        {"comparison": "read vs. write", "b": g(hl, "mcnemar", "read_vs_write", "b"),
         "c": g(hl, "mcnemar", "read_vs_write", "c"),
         "p": g(hl, "mcnemar", "read_vs_write", "p_value"),
         "delta_asr_pp": rnd((rec.get("read") or 0) - (rec.get("write") or 0), 1)},
    ]
    boot = g(hl, "recovery_ci_95", default = {}) or {}
    return {
        "source": "headline_llama_aqlm2.json",
        "rows": rows,
        "recovery_ci_95_pp": {
            "read": boot.get("read"),
            "write": boot.get("write"),
            "read_minus_write": boot.get("read_minus_write"),
        },
    }


def build():
    """
    Assemble all paper-table blocks into one JSON document.

    Inputs:
        - None

    Outputs:
        - pv (dict): Full paper-values payload
    """
    return {
        "meta": {
            "generated_by": "scripts/paper_values.py",
            "note": (
                "ASR/ARC/damage in percentage points. dASR = quant_asr - fp32_asr. "
                "Null = cell absent. Incomplete Q-Resafe read/write r128 jobs stay "
                "marked 'missing or running' until JSON lands."
            ),
            "seed": 42,
        },
        "tab:geometry": build_geometry(),
        "tab:regime": build_regime(),
        "tab:f3a_proj": build_f3a_proj(),
        "tab:f3a_range": build_f3a_range(),
        "tab:f3a": build_f3a(),
        "tab:controls": build_controls(),
        "tab:trained_repair": build_trained_repair(),
        "tab:lrqat": build_lrqat(),
        "tab:ppl": build_ppl(),
        "tab:dequant": build_dequant(),
        "tab:leakage": build_leakage(),
        "table:stats": build_stats(),
    }


def fmt_cell(x):
    """
    Format one table cell for markdown output.

    Inputs:
        - x: Value or None

    Outputs:
        - s (str): Display string
    """
    return "--" if x is None else str(x)


def as_markdown(pv):
    """
    Render paper_values JSON as human-readable markdown tables.

    Inputs:
        - pv (dict): Output of build()

    Outputs:
        - text (str): Markdown document
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
    lines.append("| Model | Quant | FP | Quant | +Write | +Read | +Both | Verdict |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for key, v in pv["tab:regime"]["cells"].items():
        a, r = v["asr"], v["arc"]
        cells = " | ".join(f"{fmt_cell(a[c])}/{fmt_cell(r[c])}" for c in CONDS)
        lines.append(
            f"| {v['model_label']} | {v['quant_label']} | {cells} | "
            f"{fmt_cell(v['screen_verdict'])} |")

    lines.append("\n## tab:regime (paired Delta ASR)\n")
    lines.append("| Model | Quant | dASR | 95% CI | p Holm |")
    lines.append("|---|---|---:|---|---:|")
    for key, v in pv["tab:regime"]["cells"].items():
        p = v.get("paired")
        if not p:
            continue
        ci = p.get("ci95") or [None, None]
        lines.append(
            f"| {v['model_label']} | {v['quant_label']} | {fmt_cell(p.get('delta_asr'))} | "
            f"[{fmt_cell(ci[0])}, {fmt_cell(ci[1])}] | {fmt_cell(p.get('p_holm'))} |")

    for model_key, label in (("llama", "Llama-3.1-8B"), ("phi3", "Phi-3-mini")):
        proj = pv["tab:f3a_proj"].get(model_key, {})
        if not proj.get("rows"):
            continue
        lines.append(f"\n## tab:f3a_proj ({label})\n")
        lines.append("| Projection | Function | Matrices | rho % |")
        lines.append("|---|---|---:|---:|")
        for row in proj["rows"]:
            lines.append(
                f"| {row['projection']} | {row['function']} | {row['matrices']} | "
                f"{fmt_cell(row['rho_pct'])} |")

    for model_key, label in (("llama", "Llama-3.1-8B"), ("phi3", "Phi-3-mini")):
        depth = pv["tab:f3a_range"].get(model_key, {})
        if not depth.get("read"):
            continue
        lines.append(f"\n## tab:f3a_range ({label})\n")
        lines.append("| Side | Range | Layers | ASR | rho % | ARC |")
        lines.append("|---|---|---|---:|---:|---:|")
        for side_key in ("read", "write"):
            for row in depth[side_key]:
                lines.append(
                    f"| {row['side']} | {row['range']} | {row['layers']} | "
                    f"{fmt_cell(row['asr'])} | {fmt_cell(row['rho_pct'])} | "
                    f"{fmt_cell(row['arc'])} |")

    for model_key, label in (("llama", "Llama-3.1-8B"), ("phi3", "Phi-3-mini")):
        f3 = pv["tab:f3a"].get(model_key, {})
        if not f3.get("rows"):
            continue
        lines.append(f"\n## tab:f3a ({label})\n")
        lines.append("| Restoration | ASR | delta quant | rho % | ARC | XSTest |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for row in f3["rows"]:
            lines.append(
                f"| {row['restoration']} | {fmt_cell(row['asr'])} | "
                f"{fmt_cell(row['delta_quant'])} | {fmt_cell(row['rho_pct'])} | "
                f"{fmt_cell(row['arc'])} | {fmt_cell(row['xstest_fr'])} |")

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

    lines.append("\n## tab:trained_repair\n")
    tr = pv["tab:trained_repair"]
    header = "| Patch | " + " | ".join(str(r) for r in tr["ranks"]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(tr["ranks"]) + 1))
    for patch, ranks in tr["grid"].items():
        cells = []
        for rank in tr["ranks"]:
            cell = ranks.get(str(rank), {})
            if cell.get("status") == "missing":
                cells.append("--")
            else:
                cells.append(fmt_cell(cell.get("asr")))
        lines.append(f"| {patch} | " + " | ".join(cells) + " |")

    lines.append("\n## tab:ppl\n")
    lines.append("| Model | Quant | PPL | x FP |")
    lines.append("|---|---|---:|---:|")
    for m, block in pv["tab:ppl"]["by_model"].items():
        for row in block["rows"]:
            lines.append(
                f"| {block['model_label']} | {row['quant_label']} | "
                f"{fmt_cell(row['ppl'])} | {fmt_cell(row['ratio_fp'])} |")

    lines.append("\n## tab:dequant\n")
    lines.append("| Cell | ASR | ARC | XSTest | Bits |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, row in pv["tab:dequant"].items():
        if row.get("status") == "missing":
            continue
        lines.append(
            f"| {key} | {fmt_cell(row.get('asr'))} | {fmt_cell(row.get('arc'))} | "
            f"{fmt_cell(row.get('xstest_fr'))} | {fmt_cell(row.get('effective_bits'))} |")

    lines.append("\n## tab:leakage\n")
    lines.append("| Model | Residual spec | Residual unsafe rec | HarmBench transfer | Surface unsafe rec |")
    lines.append("|---|---:|---:|---:|---:|")
    for m, cell in pv["tab:leakage"]["cells"].items():
        lines.append(
            f"| {cell['label']} | {fmt_cell(cell['residual_xstest_safe_specificity'])} | "
            f"{fmt_cell(cell['residual_xstest_unsafe_recall'])} | "
            f"{fmt_cell(cell['residual_harmbench_transfer_recall'])} | "
            f"{fmt_cell(cell['surface_xstest_unsafe_recall'])} |")

    lines.append("\n## table:stats\n")
    lines.append("| Comparison | b | c | p | delta ASR pp |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in pv["table:stats"].get("rows", []):
        lines.append(
            f"| {row['comparison']} | {fmt_cell(row['b'])} | {fmt_cell(row['c'])} | "
            f"{fmt_cell(row['p'])} | {fmt_cell(row['delta_asr_pp'])} |")

    lines.append("\n## tab:lrqat\n")
    lines.append("| Role | Bits | ASR | ARC | Coherent |")
    lines.append("|---|---:|---:|---:|---|")
    for row in pv["tab:lrqat"]["rows"]:
        lines.append(
            f"| {row['role']} | {fmt_cell(row.get('bits'))} | {fmt_cell(row.get('asr'))} | "
            f"{fmt_cell(row.get('arc'))} | {fmt_cell(row.get('coherent'))} |")

    return "\n".join(lines)


def main():
    # Build payload and emit JSON or markdown
    pv = build()
    if "--md" in sys.argv[1:]:
        print(as_markdown(pv))
    else:
        print(json.dumps(pv, indent = 2, default = str))


if __name__ == "__main__":
    main()
