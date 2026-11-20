# Summaries for the reviewer-item runs (X1, X2, held-out AQLM-2, X3, X4, X6).
#
# Reads the new result names. The one exception is X6, which also READS (never
# writes) the original headline_<model>_aqlm2.json and lists it as the "paper
# run" (outside the repeat spread):
#   X1        outputs/headline/headline_llama_gptq4_xstest.json
#   held-out  outputs/headline/headline_llama3-8b_aqlm2_heldout.json
#   X3        outputs/signal_geometry/probe_accuracy_<model>_<quant>_paired.json  (14 coherent cells;
#             real + shuffled-label probe on the same activations; Table 1 a_min read from the
#             untagged probe_accuracy_<model>_<quant>.json for the consistency check)
#   collapsed outputs/signal_geometry/probe_accuracy_<model>_<quant>_paired_rebuilt.json,
#             outputs/boundary/probe_leakage_control_<model>_<quant>_rebuilt.json,
#             outputs/regime_sweep/arc_check_<model>_<quant>_rebuilt.json (else _rebuilt_gptqmodel)
#             (5 collapsed cells on rebuilt dense checkpoints; reported separately)
#   X4        outputs/boundary/probe_refit_control_<model>_splits1000.json if present, else _splits100.json
#             (FP-vs-FP refit floor),
#             tested against outputs/signal_geometry/probe_rotation_<model>_<quant>.json (observed
#             FP-vs-quant probe cosines; Table 1/7 coswmed = their median over layers):
#               split-median test: per split, median over layers of the refit cosine; empirical
#                 p = (#{split-medians <= observed coswmed} + 1) / (n_splits + 1)
#                 and z = (median of split-medians - coswmed) / SD(split-medians)
#               per-layer test: floor = 5th percentile of that layer's refit cosine over splits;
#                 count layers whose observed cosine is below floor - X4_FLOOR_MARGIN (rounding ties
#                 in low-variance early layers do not count)
#             The pooled 95th-percentile refit distance is descriptive only, not a test threshold.
#   X2        outputs/boundary/probe_leakage_control_<model>_<quant>.json  (14 coherent cells)
#             + outputs/boundary/probe_leakage_control_mistral_fp32_rerun.json (FP32 Mistral reference)
#   X6        outputs/headline/headline_<model>_aqlm2_rep{1,2,3}.json   (repeats, same
#             prompts) + the two paper runs, reported separately: headline_<model>_aqlm2.json and
#             outputs/regime_sweep/restoration_sweep_<model>_aqlm2.json (Table 3)
# The one measurement skipped on purpose (ARC on the rebuilt Mistral GPTQ-3) reports status
# "not run" (NOT_RUN); any other absent result reports status "missing" with the expected path.
# Used by scripts/report_all.py (section 5) and scripts/paper_values.py ("reviewer_runs").
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"

HELDOUT_MODELS = ("llama3-8b",)
# Measurements skipped on purpose: shown as "not run" instead of "missing"
NOT_RUN = {
    "collapsed_arc/mistral/gptq3": "not run",
}
# The 14 coherent Table 2 cells (quantized ARC >= 40); filename model tokens. X2 and X3 use the
# same cells (X3: matched real/shuffled pair per cell). Keep in sync with
# scripts/boundary/probe_leakage_control_quant.sbatch.
COHERENT_CELLS = (
    ("llama", "awq"), ("llama", "gptq4"), ("llama", "gptq3"), ("llama", "aqlm2"),
    ("qwen", "awq"), ("qwen", "gptq4"), ("qwen", "gptq3"),
    ("mistral", "awq"), ("mistral", "gptq4"), ("mistral", "aqlm2"),
    ("phi3", "awq"), ("phi3", "gptq4"), ("phi3", "gptq3"), ("phi3", "aqlm2"),
)
X2_CELLS = X3_CELLS = COHERENT_CELLS
X2_FP32_REFS = (("mistral", "probe_leakage_control_mistral_fp32_rerun.json"),)
# Collapsed cells (rebuilt checkpoints; never compared against Table 1's collapsed numbers)
COLLAPSED_CELLS = (("llama", "gptq2"), ("qwen", "gptq2"), ("mistral", "gptq2"),
                   ("phi3", "gptq2"), ("mistral", "gptq3"))
X4_MODELS = ("llama", "phi3", "mistral", "qwen")
# X4 tests: every Table 1 cell (qwen has no AQLM-2 cell)
X4_QUANTS = ("aqlm2", "awq", "gptq4", "gptq3", "gptq2")
X4_FLOOR_MARGIN = 0.002
CONDS = ("fp32", "QUANT", "write_restored", "read_restored", "both_restored")
X6_CELLS = ("llama", "phi3")
X6_REPS = (("rep1", "_rep1"), ("rep2", "_rep2"), ("rep3", "_rep3"))   # all in the reviewer-run environment


# Reviewer-run result files (new names only): paper tables (report_all.py, collect_results.py)
# leave these out or label them; local archives (pieces, superseded) are never read.
_REVIEWER_RE = re.compile(
    r"_(?:rerun|rebuilt|rebuilt_gptqmodel|flags|paired|paired_rebuilt|shuffled|splits\d+|heldout"
    r"|xstest|rep\d+)\.json$"
    r"|^(?:arc_check|x2_activation_diag)_"
    r"|^probe_leakage_control_[a-z0-9-]+_(?:aqlm2|awq|gptq\d)(?:_\w+)?\.json$")
ARCHIVE_DIRS = ("pieces", "superseded")


def is_archived(path):
    """
    Whether a path lies in a local archive folder (pieces, superseded; not in git).

    Inputs:
        - path (str or Path): File path

    Outputs:
        - archived (bool): True for outputs/**/pieces/ and outputs/**/superseded/
    """
    return any(part in ARCHIVE_DIRS for part in Path(path).parts)


def is_reviewer_file(path):
    """
    Whether a result file belongs to the reviewer runs rather than the paper.

    Inputs:
        - path (str or Path): File path

    Outputs:
        - reviewer (bool): True for reviewer-run names (docs/REVIEWER_RUNS.md) and archives
    """
    return is_archived(path) or bool(_REVIEWER_RE.search(Path(path).name))


def _load(path):
    """Parse a JSON file, or None if it is absent or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


def _rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def _missing(path):
    return {"status": "missing", "expected": _rel(path)}


def _r(x, nd = 1):
    return None if x is None else round(float(x), nd)


def _headline_summary(d, quant):
    """Pull ASR / ARC / XSTest / recovery out of a headline_eval JSON."""
    conds = [quant if c == "QUANT" else c for c in CONDS]
    asr, arc, xs = d.get("asr", {}), d.get("arc", {}), d.get("xstest", {})
    return {
        "status": "ok",
        "n_eval": d.get("manifest", {}).get("n_eval"),
        "asr": {c: _r(asr.get(c)) for c in conds},
        "arc": {c: _r(arc.get(c)) for c in conds},
        "xstest_false_refusal": {
            c: {
                # rate is stored in percent; the Wilson CI is stored as a fraction
                "rate": _r(xs[c]["false_refusal_rate"]) if c in xs else None,
                "ci95": ([_r(100 * v) for v in xs[c]["false_refusal_ci"]]
                         if c in xs and xs[c].get("false_refusal_ci") else None),
            }
            for c in conds
        },
        "damage_pp": _r(d.get("damage_pp")),
        "recovery_pp": {k: _r(v) for k, v in d.get("recovery_pp", {}).items()},
    }


def _paired(cells, variant):
    """Paired McNemar screen on tagged files; {} if the screen cannot run."""
    try:
        from experiments.regime_sweep.paired_screen import sweep
        rows = sweep(cells, variant = variant)
    except Exception as e:  # missing deps or labels: report, don't crash the report
        return {"status": f"unavailable ({type(e).__name__}: {e})"}
    out = {}
    for r in rows:
        key = f"{r['model']}/{r['quant']}"
        if r.get("status") != "ok":
            out[key] = {"status": r.get("status")}
        else:
            out[key] = {
                "status": "ok", "n": r["n"], "b": r["b"], "c": r["c"],
                "delta_asr": _r(r["delta_asr"]),
                "ci95": [_r(v) for v in r["ci95"]],
                "p_mcnemar": r["p_mcnemar"], "p_holm": r.get("p_holm"),
            }
    return out


def x1():
    path = OUT / "headline" / "headline_llama_gptq4_xstest.json"
    d = _load(path)
    if d is None:
        return _missing(path)
    res = _headline_summary(d, "gptq4")
    res["source"] = _rel(path)
    return res


def heldout():
    out = {}
    for model in HELDOUT_MODELS:
        path = OUT / "headline" / f"headline_{model}_aqlm2_heldout.json"
        d = _load(path)
        if d is None:
            out[model] = _missing(path)
            continue
        res = _headline_summary(d, "aqlm2")
        res["source"] = _rel(path)
        res["paired_screen"] = _paired([(model, "aqlm2")], "heldout").get(f"{model}/aqlm2")
        out[model] = res
    return out


def _x3_pair(path):
    """
    Matched real / shuffled probe pair (probe_accuracy --paired) for one cell.

    Inputs:
        - path (Path): probe_accuracy_<model>_<quant>_paired[_rebuilt].json

    Outputs:
        - res (dict): Accuracies, selectivity and the Table 1 consistency check, or a
          missing record
    """
    d = _load(path)
    if d is None:
        return _missing(path)
    q = d.get("quant", {}).get("summary", {})
    t1 = d.get("table1_check", {})
    return {
        "status": "ok", "source": _rel(path),
        "real_mean_acc": _r(q.get("real_mean_acc"), 3), "real_min_acc": _r(q.get("real_min_acc"), 3),
        "real_min_layer": q.get("real_min_layer"),
        "shuffled_mean_acc": _r(q.get("shuffled_mean_acc"), 3),
        "shuffled_max_acc": _r(q.get("shuffled_max_acc"), 3),
        "shuffled_max_layer": q.get("shuffled_max_layer"),
        "selectivity_mean": _r(q.get("selectivity_mean"), 3),
        "selectivity_min_layer": _r(q.get("selectivity_min_layer"), 3),
        "table1_a_min": _r(t1.get("table1_a_min"), 3),
        "delta_vs_table1_pt": _r(100 * t1["delta_real_a_min"], 1) if t1.get("delta_real_a_min") is not None else None,
        "fp32_shuffled_mean_acc": _r(d.get("fp32", {}).get("summary", {}).get("shuffled_mean_acc"), 3),
    }


def x3():
    return {f"{m}/{q}": _x3_pair(OUT / "signal_geometry" / f"probe_accuracy_{m}_{q}_paired.json")
            for m, q in X3_CELLS}


def collapsed():
    """
    X3 pair, X2 and ARC on the 5 collapsed cells, all on rebuilt dense checkpoints.

    Inputs:
        - None

    Outputs:
        - out (dict): model/quant -> {x3, x2, arc}
    """
    out = {}
    for m, q in COLLAPSED_CELLS:
        res = {"x3": _x3_pair(OUT / "signal_geometry" / f"probe_accuracy_{m}_{q}_paired_rebuilt.json")}
        res["x2"] = _x2_row(OUT / "boundary" / f"probe_leakage_control_{m}_{q}_rebuilt.json")
        res["arc"] = ({"status": "not run", "note": NOT_RUN[f"collapsed_arc/{m}/{q}"]}
                      if f"collapsed_arc/{m}/{q}" in NOT_RUN else {"status": "missing"})
        for tag in ("rebuilt", "rebuilt_gptqmodel"):
            pa = OUT / "regime_sweep" / f"arc_check_{m}_{q}_{tag}.json"
            da = _load(pa)
            if da is not None:
                res["arc"] = {"status": "ok", "source": _rel(pa), "arc": _r(da.get("arc")),
                              "ci95": [_r(v) for v in da.get("arc_wilson95") or []], "tag": tag}
                break
        out[f"{m}/{q}"] = res
    return out


def _x4_test(cos_by_split, observed):
    """
    Compare one cell's observed FP-vs-quant probe cosines with the FP-vs-FP refit floor.

    Inputs:
        - cos_by_split (dict): layer -> list of refit cosines, one per split (X4)
        - observed (dict): layer -> observed FP-vs-quant cosine (probe_rotation)

    Outputs:
        - res (dict): split-median distribution + empirical p, per-layer floor count
    """
    import numpy as np
    layers = sorted(cos_by_split, key = int)
    C = np.array([cos_by_split[L] for L in layers], dtype = np.float64)   # (layers, splits)
    split_medians = np.median(C, axis = 0)
    obs = np.array([observed[L] for L in layers], dtype = np.float64)
    coswmed = float(np.median(obs))
    floors = np.percentile(C, 5, axis = 1)
    below = [int(L) for L, o, f in zip(layers, obs, floors) if o < f - X4_FLOOR_MARGIN]
    n = len(split_medians)
    n_le = int((split_medians <= coswmed).sum())
    sd = float(np.std(split_medians, ddof = 1))
    return {
        "coswmed": _r(coswmed, 3),
        "split_median_min": _r(float(split_medians.min()), 3),
        "split_median_p5": _r(float(np.percentile(split_medians, 5)), 3),
        "split_median_median": _r(float(np.median(split_medians)), 3),
        "p_value": _r((n_le + 1) / (n + 1), 4),
        "p_label": f"p < {1 / (n + 1):.1g} ({n} splits)" if n_le == 0 else f"p = {(n_le + 1) / (n + 1):.3g}",
        "n_splits": n,
        "split_median_sd": _r(sd, 4),
        "z": _r((float(np.median(split_medians)) - coswmed) / sd, 1) if sd > 0 else None,
        "floor_margin": X4_FLOOR_MARGIN,
        "n_layers": len(layers),
        "n_below_floor": len(below),
        "layers_below_floor": below,
    }


def x4():
    out = {"floors": {}, "tests": {}}
    for model in X4_MODELS:
        # the 1000-split floor (empirical p down to 0.001) replaces the 100-split one once it exists
        path = OUT / "boundary" / f"probe_refit_control_{model}_splits1000.json"
        if not path.exists():
            path = OUT / "boundary" / f"probe_refit_control_{model}_splits100.json"
        d = _load(path)
        if d is None:
            out["floors"][model] = _missing(path)
            continue
        s = d.get("summary_splits", {})
        out["floors"][model] = {
            "status": "ok", "source": _rel(path),
            "n_splits": s.get("n_splits"),
            # descriptive only (not a test threshold)
            "refit_distance_p95_pooled": _r(s.get("refit_distance_p95_pooled"), 3),
            "cos_floor_p5_pooled": _r(s.get("cos_floor_p5_pooled"), 3),
            "cos_median_pooled": _r(s.get("cos_median_pooled"), 3),
            "refit_distance_p95_per_layer_median": _r(s.get("refit_distance_p95_per_layer_median"), 3),
        }
        for q in X4_QUANTS:
            if model == "qwen" and q == "aqlm2":
                continue
            rpath = OUT / "signal_geometry" / f"probe_rotation_{model}_{q}.json"
            r = _load(rpath)
            if r is None:
                continue
            res = {"status": "ok", "floor_source": _rel(path), "observed_source": _rel(rpath)}
            res.update(_x4_test(d["cos_by_split_per_layer"], r["cos_probe_weight_per_layer"]))
            out["tests"][f"{model}/{q}"] = res
    return out


def _spread(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": _r(statistics.mean(vals)),
        "sd": _r(statistics.stdev(vals)) if len(vals) > 1 else None,
        "min": _r(min(vals)), "max": _r(max(vals)),
    }


def _best_layer(d):
    """
    X2 best layer, marked when it is an argmax tie (probe_leakage_control takes the first of
    the tied layers, so a flat profile reports layer 0).

    Inputs:
        - d (dict): probe_leakage_control result

    Outputs:
        - label (str or None): e.g. "14", or "0 (tie, 32 of 32 layers)"
    """
    best = d.get("summary", {}).get("best_layer")
    per = d.get("residual_per_layer") or {}
    if best is None or not per:
        return None if best is None else str(best)
    bal = {L: 0.5 * (v["xstest_safe_specificity"] + v["xstest_unsafe_recall"]) for L, v in per.items()}
    top = max(bal.values())
    n_top = sum(1 for v in bal.values() if v == top)
    return str(best) if n_top == 1 else f"{best} (tie, {n_top} of {len(bal)} layers)"


def _x2_row(path):
    """
    One X2 row from a probe_leakage_control result.

    Inputs:
        - path (Path): Result file

    Outputs:
        - row (dict): Best layer and residual / surface probe scores, or a missing record
    """
    d = _load(path)
    if d is None:
        return _missing(path)
    sm = d.get("summary", {})
    return {
        "status": "ok", "source": _rel(path),
        "best_layer": _best_layer(d),
        "residual_xstest_safe_specificity": _r(sm.get("residual_xstest_safe_specificity"), 3),
        "residual_xstest_unsafe_recall": _r(sm.get("residual_xstest_unsafe_recall"), 3),
        "residual_harmbench_transfer_recall": _r(sm.get("residual_harmbench_transfer_recall"), 3),
        "surface_xstest_unsafe_recall": _r(sm.get("surface_xstest_unsafe_recall"), 3),
    }


def x2():
    """
    X2 surface-feature control on the coherent quantized cells.

    Inputs:
        - None

    Outputs:
        - out (dict): model/quant -> best layer and residual / surface probe scores
    """
    out = {f"{m}/{q}": _x2_row(OUT / "boundary" / f"probe_leakage_control_{m}_{q}.json")
           for m, q in X2_CELLS}
    for m, name in X2_FP32_REFS:
        out[f"{m}/fp32 (reference)"] = _x2_row(OUT / "boundary" / name)
    return out


def _rho(d, side):
    """
    Recovery ratio rho from a headline JSON.

    Inputs:
        - d (dict): Headline result
        - side (str): read, write or both

    Outputs:
        - rho (float or None): 100 recovery / damage, None if undefined
    """
    dmg = d.get("damage_pp")
    rec = (d.get("recovery_pp") or {}).get(side, d.get(f"{side}_recovery_pp"))
    return None if not dmg or rec is None else 100.0 * rec / dmg


X6_METRICS = ("quant_asr", "write_restored_asr", "read_restored_asr",
              "both_restored_asr", "rho_read_pct", "rho_write_pct")


def _x6_row(path):
    """
    One X6 row (quantized and restored ASRs, rho) from a headline JSON.

    Inputs:
        - path (Path): headline_<model>_aqlm2[_repK].json

    Outputs:
        - row (dict): ASRs and rho, or a missing record
    """
    d = _load(path)
    if d is None:
        return _missing(path)
    asr = d.get("asr") or d.get("asr_per_condition") or {}   # headline / restoration_sweep schema
    return {
        "status": "ok", "source": _rel(path),
        "quant_asr": _r(asr.get("aqlm2")),
        "write_restored_asr": _r(asr.get("write_restored")),
        "read_restored_asr": _r(asr.get("read_restored")),
        "both_restored_asr": _r(asr.get("both_restored")),
        "rho_read_pct": _r(_rho(d, "read")),
        "rho_write_pct": _r(_rho(d, "write")),
    }


def x6():
    """
    Same-prompt repeats: rep1-rep3 (one environment) give the spread; the two original runs
    (headline, and the restoration sweep behind Table 3) are listed separately as paper runs
    and are not in the spread.

    Inputs:
        - None

    Outputs:
        - out (dict): model -> paper_run, paper_sweep, reps, n_present, across_reps
    """
    out = {}
    for model in X6_CELLS:
        reps = {label: _x6_row(OUT / "headline" / f"headline_{model}_aqlm2{suffix}.json")
                for label, suffix in X6_REPS}
        ok = [r for r in reps.values() if r.get("status") == "ok"]
        out[model] = {
            "paper_run": _x6_row(OUT / "headline" / f"headline_{model}_aqlm2.json"),
            "paper_sweep": _x6_row(OUT / "regime_sweep" / f"restoration_sweep_{model}_aqlm2.json"),
            "reps": reps,
            "n_present": len(ok),
            "across_reps": {m: _spread([r[m] for r in ok]) for m in X6_METRICS} if ok else None,
        }
    return out


def collect():
    """All reviewer-run summaries in one dict (statuses say what is missing)."""
    return {"x1": x1(), "x2": x2(), "heldout": heldout(), "x3": x3(), "x4": x4(),
            "x6": x6(), "collapsed": collapsed()}


# ---------------------------------------------------------------- rendering --
def _f(x):
    return "--" if x is None else str(x)


def _ci(x):
    return "--" if not x else f"[{x[0]}, {x[1]}]"


def to_markdown(rv):
    """Markdown block for PAPER_TABLES.md. Missing runs show as 'missing'."""
    L = ["\n## reviewer runs (X1, X2, held-out, X3, X4, X6)\n",
         "Read only from the new result names.\n"]

    L.append("\n### X1: XSTest false refusal, Llama GPTQ-4\n")
    r = rv["x1"]
    if r["status"] != "ok":
        L.append(f"missing: `{r['expected']}`")
    else:
        L.append("XSTest false-refusal rate per condition (one condition per job, merged).\n")
        L.append("| Condition | XSTest FRR % | 95% CI |")
        L.append("|---|---:|---|")
        for c, v in r["xstest_false_refusal"].items():
            L.append(f"| {c} | {_f(v['rate'])} | {_ci(v['ci95'])} |")

    L.append("\n### X2: surface-feature control on the 14 coherent quantized cells\n")
    L.append("Best layer = argmax of balanced XSTest accuracy; \"tie\" means several layers share the "
             "maximum and the first (layer 0) is reported.\n")
    L.append("| Cell | best layer | residual spec (XSTest-safe) | residual recall (XSTest-unsafe) "
             "| HarmBench transfer | surface recall (XSTest-unsafe) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, r in rv["x2"].items():
        if r["status"] != "ok":
            L.append(f"| {k} | missing | `{r['expected']}` | | | |")
            continue
        L.append(f"| {k} | {_f(r['best_layer'])} | {_f(r['residual_xstest_safe_specificity'])} | "
                 f"{_f(r['residual_xstest_unsafe_recall'])} | "
                 f"{_f(r['residual_harmbench_transfer_recall'])} | "
                 f"{_f(r['surface_xstest_unsafe_recall'])} |")

    L.append("\n### Held-out AQLM-2 screen\n")
    L.append("| Model | FP ASR/ARC | Quant ASR/ARC | dASR | 95% CI | p (McNemar) | Source |")
    L.append("|---|---|---|---:|---|---:|---|")
    for m, r in rv["heldout"].items():
        if r["status"] != "ok":
            L.append(f"| {m} | missing | | | | | `{r['expected']}` |")
            continue
        ps = r.get("paired_screen") or {}
        L.append(f"| {m} | {_f(r['asr']['fp32'])}/{_f(r['arc']['fp32'])} | "
                 f"{_f(r['asr']['aqlm2'])}/{_f(r['arc']['aqlm2'])} | {_f(ps.get('delta_asr'))} | "
                 f"{_ci(ps.get('ci95'))} | {_f(ps.get('p_mcnemar'))} | `{r['source']}` |")

    L.append("\n### X3: matched real / shuffled-label probe on the 14 coherent cells\n")
    L.append("Same activations, layers, folds and regularisation for both fits (shuffle seed 42); "
             "selectivity = real - shuffled from that pair. Table 1 a_min = the paper's real-label "
             "run (consistency check, expect within ~1 pt).\n")
    L.append("| Cell | real acc mean / min (layer) | shuffled acc mean / max (layer) | selectivity mean / min-layer "
             "| Table 1 a_min | Delta vs Table 1 (pt) |")
    L.append("|---|---|---|---|---:|---:|")
    for k, r in rv["x3"].items():
        if r["status"] != "ok":
            L.append(f"| {k} | missing | `{r['expected']}` | | | |")
            continue
        L.append(f"| {k} | {_f(r['real_mean_acc'])} / {_f(r['real_min_acc'])} (L{_f(r['real_min_layer'])}) | "
                 f"{_f(r['shuffled_mean_acc'])} / {_f(r['shuffled_max_acc'])} (L{_f(r['shuffled_max_layer'])}) | "
                 f"{_f(r['selectivity_mean'])} / {_f(r['selectivity_min_layer'])} | "
                 f"{_f(r['table1_a_min'])} | {_f(r['delta_vs_table1_pt'])} |")

    splits = sorted({r["n_splits"] for r in rv["x4"]["floors"].values() if r.get("status") == "ok"})
    L.append(f"\n### X4: FP-vs-FP refit floor ({'/'.join(map(str, splits)) or '--'} splits)\n")
    L.append("Floor runs (pooled p95 is descriptive only, not a test threshold):\n")
    L.append("| Model | splits | refit distance p95 (pooled) | cos p5 (pooled) | cos median | per-layer p95 (median) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m, r in rv["x4"]["floors"].items():
        if r["status"] != "ok":
            L.append(f"| {m} | missing | `{r['expected']}` | | | |")
            continue
        L.append(f"| {m} | {_f(r['n_splits'])} | {_f(r['refit_distance_p95_pooled'])} | "
                 f"{_f(r['cos_floor_p5_pooled'])} | {_f(r['cos_median_pooled'])} | "
                 f"{_f(r['refit_distance_p95_per_layer_median'])} |")
    L.append("\nTests per Table 1 cell. Split-median: median over layers of the refit cosine, one per "
             "split; p = (#{split-medians <= coswmed} + 1) / (splits + 1). Per-layer: floor = 5th "
             "percentile of that layer's refit cosine; count of layers with observed FP-vs-quant "
             "cosine below floor - 0.002. z = (median split-median - coswmed) / SD(split-medians).\n")
    L.append("| Cell | coswmed | split-median min / p5 / median (SD) | p | z | layers below floor |")
    L.append("|---|---:|---:|---|---:|---:|")
    for c, r in rv["x4"]["tests"].items():
        L.append(f"| {c} | {_f(r['coswmed'])} | {_f(r['split_median_min'])} / {_f(r['split_median_p5'])} / "
                 f"{_f(r['split_median_median'])} ({_f(r['split_median_sd'])}) | {r['p_label']} | {_f(r['z'])} | "
                 f"{r['n_below_floor']} / {r['n_layers']} |")

    L.append("\n### X6: same-prompt repeats (AQLM-2; rep1-rep3)\n")
    L.append("Spread = rep1-rep3 only (one environment). The two paper runs (original environment: the headline "
             "run and the restoration sweep behind Table 3) are listed separately and are not part of "
             "the mean +/- sd.\n")
    L.append("| Cell | Run | quant ASR | +Write ASR | +Read ASR | +Both ASR | rho read % | rho write % |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")

    def row(m, label, r):
        if r["status"] != "ok":
            return f"| {m} | {label} | missing | `{r['expected']}` | | | | |"
        return (f"| {m} | {label} | {_f(r['quant_asr'])} | {_f(r['write_restored_asr'])} | "
                f"{_f(r['read_restored_asr'])} | {_f(r['both_restored_asr'])} | "
                f"{_f(r['rho_read_pct'])} | {_f(r['rho_write_pct'])} |")

    for m, block in rv["x6"].items():
        for label, r in block["reps"].items():
            L.append(row(m, label, r))
        agg = block["across_reps"]
        if agg:
            def ms(k):
                s = agg[k]
                return "--" if not s else f"{s['mean']} +/- {_f(s['sd'])}"
            L.append(f"| {m} | **repeats mean +/- sd ({block['n_present']} of 3)** | {ms('quant_asr')} | "
                     f"{ms('write_restored_asr')} | {ms('read_restored_asr')} | "
                     f"{ms('both_restored_asr')} | {ms('rho_read_pct')} | {ms('rho_write_pct')} |")
        else:
            L.append(f"| {m} | repeats mean +/- sd (0 of 3) | -- | -- | -- | -- | -- | -- |")
        L.append(row(m, "paper headline run (original environment, not in spread)", block["paper_run"]))
        L.append(row(m, "paper restoration sweep, Table 3 (original environment, not in spread)", block["paper_sweep"]))
    L.append("\n### Collapsed cells (rebuilt checkpoints)\n")
    L.append("Rebuilds on the dense GPTQ path, NOT the paper's original checkpoints (the GPTQ-2 "
             "rebuilds differ a lot). Selectivity comes only from each cell's own real/shuffled pair; "
             "Table 1 a_min is shown for reference, a large gap is expected.\n")
    L.append("| Cell (rebuilt) | ARC % [95% CI] | real acc mean / min | shuffled acc mean / max | selectivity mean "
             "| X2 best layer | XSTest spec / recall (residual) | surface recall | Table 1 a_min |")
    L.append("|---|---|---|---|---:|---|---|---:|---:|")
    for k, r in rv["collapsed"].items():
        a, x3r, x2r = r["arc"], r["x3"], r["x2"]
        arc = (f"{_f(a.get('arc'))} {_ci(a.get('ci95'))}" + ("" if a.get("tag") == "rebuilt" else f" ({a.get('tag')})")
               if a["status"] == "ok" else "--")
        if x3r["status"] == "ok":
            p3 = (f"{_f(x3r['real_mean_acc'])} / {_f(x3r['real_min_acc'])} | "
                  f"{_f(x3r['shuffled_mean_acc'])} / {_f(x3r['shuffled_max_acc'])} | {_f(x3r['selectivity_mean'])}")
            t1 = _f(x3r["table1_a_min"])
        else:
            p3, t1 = "missing | | ", "--"
        if x2r["status"] == "ok":
            p2 = (f"{_f(x2r['best_layer'])} | "
                  f"{_f(x2r['residual_xstest_safe_specificity'])} / {_f(x2r['residual_xstest_unsafe_recall'])} | "
                  f"{_f(x2r['surface_xstest_unsafe_recall'])}")
        else:
            p2 = "missing | | "
        L.append(f"| {k} | {arc} | {p3} | {p2} | {t1} |")
    not_run = [k for k, r in rv["collapsed"].items() if r["arc"]["status"] == "not run"]
    if not_run:
        L.append(f"\n-- = ARC not run on the rebuilt checkpoint ({', '.join(not_run)}).")
    return "\n".join(L)


def print_report(rv = None):
    """Plain-text report (report_all.py section 5)."""
    rv = rv or collect()
    print(to_markdown(rv))


if __name__ == "__main__":
    if "--json" in sys.argv[1:]:
        print(json.dumps(collect(), indent = 2, default = str))
    else:
        print_report()
