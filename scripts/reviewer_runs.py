# Summaries for the reviewer-item runs (X1, X2, held-out AQLM-2, X3, X4, X5, X6).
#
# Reads the new result names. The one exception is X6, which also READS (never
# writes) the original headline_<model>_aqlm2.json as repeat 1:
#   X1        outputs/headline/headline_llama_gptq4_x1_n200.json
#   held-out  outputs/headline/headline_<model>_aqlm2_heldout.json   (llama3-8b, else mistral-nemo)
#   X3        outputs/signal_geometry/probe_accuracy_<model>_<quant>_shuffled.json
#   X4        outputs/boundary/probe_refit_control_<model>_splits100.json
#   X5        outputs/headline/headline_<model>_aqlm2_subset<k>.json   (k = 1..5)
#   X2        outputs/boundary/probe_leakage_control_<model>_<quant>.json  (14 coherent cells)
#   X6        outputs/headline/headline_<model>_aqlm2{,_rep2,_rep3}.json   (same prompts)
# Every entry reports status "missing" (with the expected path) until its JSON exists.
# Used by scripts/report_all.py (section 5) and scripts/paper_values.py ("reviewer_runs").
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs"

HELDOUT_MODELS = ("llama3-8b", "mistral-nemo")
X3_CELLS = (("llama", "aqlm2"), ("phi3", "aqlm2"))
X4_MODELS = ("llama", "phi3")
X5_CELLS = ("llama", "phi3")          # the two significant AQLM-2 cells
X5_SEEDS = (1, 2, 3, 4, 5)
CONDS = ("fp32", "QUANT", "write_restored", "read_restored", "both_restored")
# X2: the 14 coherent Table 2 cells (quantized ARC >= 40); filename model tokens.
# Keep in sync with scripts/boundary/probe_leakage_control_quant.sbatch.
X2_CELLS = (
    ("llama", "awq"), ("llama", "gptq4"), ("llama", "gptq3"), ("llama", "aqlm2"),
    ("qwen", "awq"), ("qwen", "gptq4"), ("qwen", "gptq3"),
    ("mistral", "awq"), ("mistral", "gptq4"), ("mistral", "aqlm2"),
    ("phi3", "awq"), ("phi3", "gptq4"), ("phi3", "gptq3"), ("phi3", "aqlm2"),
)
X6_CELLS = ("llama", "phi3")
X6_REPS = (("rep1 (original)", ""), ("rep2", "_rep2"), ("rep3", "_rep3"))


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
    path = OUT / "headline" / "headline_llama_gptq4_x1_n200.json"
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


def x3():
    out = {}
    for model, quant in X3_CELLS:
        path = OUT / "signal_geometry" / f"probe_accuracy_{model}_{quant}_shuffled.json"
        d = _load(path)
        if d is None:
            out[f"{model}/{quant}"] = _missing(path)
            continue
        res = {"status": "ok", "source": _rel(path)}
        for side in ("fp32", "quant"):
            accs = [v["mean_acc"] for v in d.get(f"{side}_probe_per_layer", {}).values()]
            res[side] = {
                "mean_acc": _r(statistics.mean(accs), 3) if accs else None,
                "min_acc": _r(min(accs), 3) if accs else None,
                "max_acc": _r(max(accs), 3) if accs else None,
                "max_abs_dev_from_0.5": _r(max(abs(a - 0.5) for a in accs), 3) if accs else None,
            }
        out[f"{model}/{quant}"] = res
    return out


def x4():
    out = {}
    for model in X4_MODELS:
        path = OUT / "boundary" / f"probe_refit_control_{model}_splits100.json"
        d = _load(path)
        if d is None:
            out[model] = _missing(path)
            continue
        s = d.get("summary_splits", {})
        out[model] = {
            "status": "ok", "source": _rel(path),
            "n_splits": s.get("n_splits"),
            "refit_distance_p95_pooled": _r(s.get("refit_distance_p95_pooled"), 3),
            "cos_floor_p5_pooled": _r(s.get("cos_floor_p5_pooled"), 3),
            "cos_median_pooled": _r(s.get("cos_median_pooled"), 3),
            "refit_distance_p95_per_layer_median": _r(s.get("refit_distance_p95_per_layer_median"), 3),
        }
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


def x5():
    out = {}
    for model in X5_CELLS:
        seeds = {}
        for k in X5_SEEDS:
            path = OUT / "headline" / f"headline_{model}_aqlm2_subset{k}.json"
            d = _load(path)
            if d is None:
                seeds[k] = _missing(path)
                continue
            res = _headline_summary(d, "aqlm2")
            res["source"] = _rel(path)
            res["paired_screen"] = _paired([(model, "aqlm2")], f"subset{k}").get(f"{model}/aqlm2")
            seeds[k] = res
        ok = [s for s in seeds.values() if s.get("status") == "ok"]
        out[model] = {
            "seeds": seeds,
            "n_present": len(ok),
            "across_seeds": {
                "damage_pp": _spread([s["damage_pp"] for s in ok]),
                "read_recovery_pp": _spread([s["recovery_pp"].get("read") for s in ok]),
                "write_recovery_pp": _spread([s["recovery_pp"].get("write") for s in ok]),
                "both_recovery_pp": _spread([s["recovery_pp"].get("both") for s in ok]),
            } if ok else None,
        }
    return out


def x2():
    out = {}
    for model, quant in X2_CELLS:
        path = OUT / "boundary" / f"probe_leakage_control_{model}_{quant}.json"
        d = _load(path)
        if d is None:
            out[f"{model}/{quant}"] = _missing(path)
            continue
        sm = d.get("summary", {})
        out[f"{model}/{quant}"] = {
            "status": "ok", "source": _rel(path),
            "best_layer": sm.get("best_layer"),
            "residual_xstest_safe_specificity": _r(sm.get("residual_xstest_safe_specificity"), 3),
            "residual_xstest_unsafe_recall": _r(sm.get("residual_xstest_unsafe_recall"), 3),
            "residual_harmbench_transfer_recall": _r(sm.get("residual_harmbench_transfer_recall"), 3),
            "surface_xstest_unsafe_recall": _r(sm.get("surface_xstest_unsafe_recall"), 3),
        }
    return out


def _rho(d, side):
    """rho % = recovery / damage from a headline JSON (None if undefined)."""
    dmg = d.get("damage_pp")
    rec = (d.get("recovery_pp") or {}).get(side)
    return None if not dmg or rec is None else 100.0 * rec / dmg


def x6():
    out = {}
    for model in X6_CELLS:
        reps = {}
        for label, suffix in X6_REPS:
            path = OUT / "headline" / f"headline_{model}_aqlm2{suffix}.json"
            d = _load(path)
            if d is None:
                reps[label] = _missing(path)
                continue
            asr = d.get("asr", {})
            reps[label] = {
                "status": "ok", "source": _rel(path),
                "quant_asr": _r(asr.get("aqlm2")),
                "write_restored_asr": _r(asr.get("write_restored")),
                "read_restored_asr": _r(asr.get("read_restored")),
                "both_restored_asr": _r(asr.get("both_restored")),
                "rho_read_pct": _r(_rho(d, "read")),
                "rho_write_pct": _r(_rho(d, "write")),
            }
        ok = [r for r in reps.values() if r.get("status") == "ok"]
        metrics = ("quant_asr", "write_restored_asr", "read_restored_asr",
                   "both_restored_asr", "rho_read_pct", "rho_write_pct")
        out[model] = {
            "reps": reps,
            "n_present": len(ok),
            "across_reps": {m: _spread([r[m] for r in ok]) for m in metrics} if ok else None,
        }
    return out


def collect():
    """All reviewer-run summaries in one dict (statuses say what is missing)."""
    return {"x1": x1(), "x2": x2(), "heldout": heldout(), "x3": x3(), "x4": x4(),
            "x5": x5(), "x6": x6()}


# ---------------------------------------------------------------- rendering --
def _f(x):
    return "--" if x is None else str(x)


def _ci(x):
    return "--" if not x else f"[{x[0]}, {x[1]}]"


def to_markdown(rv):
    """Markdown block for PAPER_TABLES.md. Missing runs show as 'missing'."""
    L = ["\n## reviewer runs (X1, X2, held-out, X3, X4, X5, X6)\n",
         "Read only from the new result names; 'missing' until the GPU job lands.\n"]

    L.append("\n### X1: XSTest false refusal, Llama GPTQ-4 (n_eval 200)\n")
    r = rv["x1"]
    if r["status"] != "ok":
        L.append(f"missing: `{r['expected']}`")
    else:
        L.append("| Condition | ASR | ARC | XSTest FRR % | 95% CI |")
        L.append("|---|---:|---:|---:|---|")
        for c, v in r["xstest_false_refusal"].items():
            L.append(f"| {c} | {_f(r['asr'][c])} | {_f(r['arc'][c])} | {_f(v['rate'])} | {_ci(v['ci95'])} |")

    L.append("\n### X2: surface-feature control on the 14 coherent quantized cells\n")
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

    L.append("\n### X3: shuffled-label probe control (expect ~0.5)\n")
    L.append("| Cell | FP mean acc | FP min-max | Quant mean acc | Quant min-max | max abs dev from 0.5 |")
    L.append("|---|---:|---|---:|---|---:|")
    for k, r in rv["x3"].items():
        if r["status"] != "ok":
            L.append(f"| {k} | missing | `{r['expected']}` | | | |")
            continue
        fp, q = r["fp32"], r["quant"]
        dev = max(x for x in (fp["max_abs_dev_from_0.5"], q["max_abs_dev_from_0.5"]) if x is not None)
        L.append(f"| {k} | {_f(fp['mean_acc'])} | {_f(fp['min_acc'])}-{_f(fp['max_acc'])} | "
                 f"{_f(q['mean_acc'])} | {_f(q['min_acc'])}-{_f(q['max_acc'])} | {_f(dev)} |")

    L.append("\n### X4: 100-split FP-vs-FP refit floor\n")
    L.append("| Model | splits | refit distance p95 (pooled) | cos floor p5 | cos median | per-layer p95 (median) |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for m, r in rv["x4"].items():
        if r["status"] != "ok":
            L.append(f"| {m} | missing | `{r['expected']}` | | | |")
            continue
        L.append(f"| {m} | {_f(r['n_splits'])} | {_f(r['refit_distance_p95_pooled'])} | "
                 f"{_f(r['cos_floor_p5_pooled'])} | {_f(r['cos_median_pooled'])} | "
                 f"{_f(r['refit_distance_p95_per_layer_median'])} |")

    L.append("\n### X5: 5 random prompt subsets (AQLM-2, n_eval 200 each)\n")
    L.append("| Cell | seed | dASR (paired) | 95% CI | damage | read rec | write rec | both rec |")
    L.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for m, block in rv["x5"].items():
        for k, r in block["seeds"].items():
            if r["status"] != "ok":
                L.append(f"| {m} | {k} | missing | `{r['expected']}` | | | | |")
                continue
            ps = r.get("paired_screen") or {}
            rec = r["recovery_pp"]
            L.append(f"| {m} | {k} | {_f(ps.get('delta_asr'))} | {_ci(ps.get('ci95'))} | "
                     f"{_f(r['damage_pp'])} | {_f(rec.get('read'))} | {_f(rec.get('write'))} | "
                     f"{_f(rec.get('both'))} |")

    L.append("\nAcross seeds (present seeds only):\n")
    L.append("| Cell | Metric | seeds | mean | sd | min | max |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for m, block in rv["x5"].items():
        agg = block["across_seeds"]
        if not agg:
            L.append(f"| {m} | missing | 0 | | | | |")
            continue
        for name, s in agg.items():
            if s:
                L.append(f"| {m} | {name} | {s['n']} | {s['mean']} | {_f(s['sd'])} | "
                         f"{s['min']} | {s['max']} |")

    L.append("\n### X6: same-prompt repeats (AQLM-2; original + rep2 + rep3)\n")
    L.append("| Cell | Run | quant ASR | +Write ASR | +Read ASR | +Both ASR | rho read % | rho write % |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for m, block in rv["x6"].items():
        for label, r in block["reps"].items():
            if r["status"] != "ok":
                L.append(f"| {m} | {label} | missing | `{r['expected']}` | | | | |")
                continue
            L.append(f"| {m} | {label} | {_f(r['quant_asr'])} | {_f(r['write_restored_asr'])} | "
                     f"{_f(r['read_restored_asr'])} | {_f(r['both_restored_asr'])} | "
                     f"{_f(r['rho_read_pct'])} | {_f(r['rho_write_pct'])} |")
        agg = block["across_reps"]
        if agg:
            def ms(k):
                s = agg[k]
                return "--" if not s else f"{s['mean']} +/- {_f(s['sd'])}"
            L.append(f"| {m} | mean +/- sd ({block['n_present']} runs) | {ms('quant_asr')} | "
                     f"{ms('write_restored_asr')} | {ms('read_restored_asr')} | "
                     f"{ms('both_restored_asr')} | {ms('rho_read_pct')} | {ms('rho_write_pct')} |")
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
