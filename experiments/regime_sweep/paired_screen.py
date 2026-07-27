# Paired ΔASR significance screen across the 19 regime cells.
#
# Replaces the point-threshold "ΔASR > 5" selection rule with a per-cell paired
# significance test (McNemar-correct SE), because the SE is different in every
# cell (it depends on b + c, the disagreement between fp and quant). A cell enters
# the studiable set on SIGNIFICANCE, not on a single numeric cutoff. Holm-adjusts
# across the tested cells. The coherence (ARC) and refusal (FP-baseline) gates are
# applied separately; this script owns only the paired-increase criterion.
#
# Labels come from flags_by_cond in the result JSON (headline for the two AQLM
# studiable cells; the flag-augmented restoration_sweep for the other 17). Run
#   python -m experiments.regime_sweep.paired_screen --self_test
# to verify the estimator against cases with known answers before trusting a sweep.

import argparse
import json
from pathlib import Path

from core.manifest import RunConfig, find_result
from core.stats import paired_delta_asr, holm_bonferroni

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "aqlm2", "gptq3", "gptq2"]
# No cell is structurally excluded from the PAIRED BASELINE screen: it needs only
# fp32 + quant flags. phi3/awq loads fine (only its RESTORATION is arch-NA), and
# qwen/aqlm2 (no AQLM checkpoint) drops out automatically via missing labels.
NA_CELLS = set()

RESTORE_SCRIPT = Path("experiments/regime_sweep/restoration_sweep.py")
HEADLINE_SCRIPT = Path("experiments/headline/headline_eval.py")


def load_labels(path, quant_key):
    """
    Return (fp_flags, quant_flags) as int lists from a result JSON carrying
    flags_by_cond. The two share the same prompt order within a single run, so no
    id-sorting is needed; if you ever pair labels from SEPARATE runs, align by id
    first. Raises KeyError (with guidance) when labels are absent.
    """
    d = json.loads(Path(path).read_text())
    fbc = d.get("flags_by_cond")
    if not fbc:
        raise KeyError(f"{path}: no flags_by_cond (re-run restoration_sweep to dump labels)")
    if "fp32" not in fbc or quant_key not in fbc:
        raise KeyError(f"{path}: flags_by_cond missing 'fp32' or '{quant_key}' "
                       f"(has {list(fbc)})")
    to_int = lambda xs: [int(bool(x)) for x in xs]
    return to_int(fbc["fp32"]), to_int(fbc[quant_key])


def _find_cell_json(model, quantizer):
    """Prefer the headline artifact (canonical, has flags), else restoration_sweep."""
    cfg = RunConfig(model=model, quantizer=quantizer)
    for script in (HEADLINE_SCRIPT, RESTORE_SCRIPT):
        p = find_result(script, cfg)
        if p and Path(p).exists():
            try:
                if json.loads(Path(p).read_text()).get("flags_by_cond"):
                    return p
            except Exception:
                pass
    return None


def sweep(cells, alpha=0.05):
    """Screen each (model, quantizer) cell; Holm-adjust across the tested ones."""
    results, pvals = [], []
    for model, quantizer in cells:
        row = {"model": model, "quant": quantizer}
        if (model, quantizer) in NA_CELLS:
            row["status"] = "NA-arch-mismatch"
            results.append(row)
            continue
        path = _find_cell_json(model, quantizer)
        if path is None:
            row["status"] = "NO-LABELS-rerun"
            results.append(row)
            continue
        try:
            fp, q = load_labels(path, quantizer)
            st = paired_delta_asr(fp, q)
        except KeyError as e:
            row["status"] = f"NO-LABELS: {e}"
            results.append(row)
            continue
        st.update(row)
        st["status"] = "ok"
        st["path"] = str(path)
        results.append(st)
        pvals.append(st["p_mcnemar"])

    holm = holm_bonferroni(pvals)
    j = 0
    for r in results:
        if r.get("status") == "ok":
            r["p_holm"] = holm[j]
            r["sig_raw"] = r["p_mcnemar"] < alpha
            r["sig_holm"] = holm[j] < alpha
            j += 1
    return results


def _print_table(results):
    hdr = ["model", "quant", "n", "b", "c", "dASR", "SE_pair", "CI95",
           "p_raw", "p_holm", "sig"]
    print("  ".join(f"{h:>8}" for h in hdr))
    for r in results:
        if r.get("status") != "ok":
            print(f"{r['model']:>8}  {r['quant']:>8}   {r.get('status')}")
            continue
        ci = f"[{r['ci95'][0]:+.1f},{r['ci95'][1]:+.1f}]"
        sig = "**" if r["sig_holm"] else ("*" if r["sig_raw"] else "")
        print(f"{r['model']:>8}  {r['quant']:>8}  {r['n']:>8}  {r['b']:>8}  {r['c']:>8}  "
              f"{r['delta_asr']:>+7.1f}  {r['se_paired']:>7.2f}  {ci:>13}  "
              f"{r['p_mcnemar']:>8.1e}  {r['p_holm']:>8.1e}  {sig:>4}")
    print("\n  ** = significant after Holm  |  * = raw only  |  report both")


def _synth_flags(n, fp_asr, b, c):
    """Build fp/quant flag arrays with an exact (b, c) discordant decomposition."""
    n_fp_comply = round(fp_asr / 100.0 * n)
    n11 = n_fp_comply - c          # fp comply & quant comply (concordant)
    n10 = c                        # fp comply & quant refuse
    n01 = b                        # fp refuse & quant comply
    n00 = n - n11 - n10 - n01
    if min(n11, n10, n01, n00) < 0:
        raise ValueError(f"infeasible decomposition: {(n11, n10, n01, n00)}")
    fp = [1] * (n11 + n10) + [0] * (n01 + n00)
    q = [1] * n11 + [0] * n10 + [1] * n01 + [0] * n00
    return fp, q


def self_test():
    """Reproduce the synthetic cases with known answers; assert the estimator."""
    fp, q = _synth_flags(200, 41.0, b=50, c=7)
    r = paired_delta_asr(fp, q, seed=0)
    print(f"case b=50,c=7:  dASR={r['delta_asr']:.1f}  SE_pair={r['se_paired']:.2f}  "
          f"SE_null={r['se_null']:.2f}  SE_unpair={r['se_unpaired']:.2f}  "
          f"p={r['p_mcnemar']:.2e}  CI={r['ci95']}")
    assert abs(r["se_paired"] - 3.46) < 0.05, r["se_paired"]
    assert abs(r["se_null"] - 3.77) < 0.05, r["se_null"]
    assert abs(r["se_unpaired"] - 4.88) < 0.05, r["se_unpaired"]
    assert r["p_mcnemar"] < 1e-8, r["p_mcnemar"]

    for b, c, exp_p in [(15, 6, 0.078), (12, 3, 0.035)]:
        fp, q = _synth_flags(200, 41.0, b, c)
        r = paired_delta_asr(fp, q)
        print(f"case b={b},c={c}:  dASR={r['delta_asr']:.1f}  p={r['p_mcnemar']:.4f} "
              f"(expected ~{exp_p})")
        assert abs(r["p_mcnemar"] - exp_p) < 0.005, r["p_mcnemar"]

    adj = holm_bonferroni([0.01, 0.04, 0.03])
    assert adj == [0.03, 0.06, 0.06], adj   # 0.01*3, then 0.04*2 vs 0.03*2 monotone
    print("SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description="Paired ΔASR significance screen (McNemar).")
    ap.add_argument("--self_test", action="store_true",
                    help="run estimator against known-answer synthetic cases and exit")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    cells = [(m, q) for m in MODELS for q in QUANTS]
    results = sweep(cells, alpha=args.alpha)
    _print_table(results)


if __name__ == "__main__":
    main()
