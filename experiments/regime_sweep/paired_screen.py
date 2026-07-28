# Paired ΔASR significance screen across the 19 regime cells.

#Front Matter
import argparse
import json
from pathlib import Path

from core.manifest import RunConfig, find_result
from core.stats import paired_delta_asr, holm_bonferroni

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "aqlm2", "gptq3", "gptq2"]
# No cell is structurally excluded from the PAIRED BASELINE screen: it needs only fp32 + quant flags.
# phi3/awq loads fine (only its RESTORATION is arch-NA), and qwen/aqlm2 (no AQLM checkpoint) drops out automatically via missing labels.
NA_CELLS = set()

RESTORE_SCRIPT = Path("experiments/regime_sweep/restoration_sweep.py")
HEADLINE_SCRIPT = Path("experiments/headline/headline_eval.py")
STUDIABLE_HEADLINE = {("llama", "aqlm2"), ("phi3", "aqlm2")}


def load_labels(path, quant_key):
    """Extract paired fp32/quant compliance flags from a result JSON."""
    d = json.loads(Path(path).read_text(encoding = "utf-8"))
    fbc = d.get("flags_by_cond")
    if not fbc:
        raise KeyError(f"{path}: no flags_by_cond (re-run restoration_sweep to dump labels)")
    if "fp32" not in fbc or quant_key not in fbc:
        raise KeyError(
            f"{path}: flags_by_cond missing 'fp32' or '{quant_key}' (has {list(fbc)})")
    to_int = lambda xs: [int(bool(x)) for x in xs]
    return to_int(fbc["fp32"]), to_int(fbc[quant_key])


def _find_cell_json(model, quantizer):
    """Prefer headline JSON for studiable AQLM cells, else restoration_sweep."""
    if (model, quantizer) in STUDIABLE_HEADLINE:
        hl = Path(f"outputs/headline/headline_{model}_{quantizer}.json")
        if hl.is_file():
            try:
                if json.loads(hl.read_text(encoding = "utf-8")).get("flags_by_cond"):
                    return hl
            except Exception:
                pass
    cfg = RunConfig(model = model, quantizer = quantizer)
    p = find_result(RESTORE_SCRIPT, cfg)
    if p and Path(p).is_file():
        try:
            if json.loads(Path(p).read_text(encoding = "utf-8")).get("flags_by_cond"):
                return p
        except Exception:
            pass
    return None


def sweep(cells, alpha = 0.05):
    """Run paired ΔASR screen with Holm adjustment across cells."""
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


def main():
    ap = argparse.ArgumentParser(description = "Paired ΔASR significance screen (McNemar).")
    ap.add_argument("--alpha", type = float, default = 0.05)
    args = ap.parse_args()

    cells = [(m, q) for m in MODELS for q in QUANTS]
    results = sweep(cells, alpha = args.alpha)
    _print_table(results)


if __name__ == "__main__":
    main()
