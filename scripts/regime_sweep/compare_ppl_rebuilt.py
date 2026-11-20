# Compare a rebuilt checkpoint's WikiText-2 PPL with the paper value.
#
#   python scripts/regime_sweep/compare_ppl_rebuilt.py --model phi3 --quant gptq4 --tag rebuilt
#
# Reads outputs/regime_sweep/wikitext_ppl_<m>_<q>_wikitext2_ppl_<tag>.json (the rebuilt
# run) and the untagged paper file wikitext_ppl_<m>_<q>_wikitext2_ppl.json, prints both
# with the relative difference, and warns if it exceeds --tol percent (default 2). The
# fp32 row is printed too when the tagged run measured fp32 (environment control: if
# fp32 also moves, the difference is the Delta environment, not the checkpoint).
# Read-only; always exits 0 unless a file is missing (exit 1).
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.manifest import RunConfig, result_path  # noqa: E402

_PPL_SCRIPT = Path("experiments/regime_sweep/wikitext_ppl.py")


def _load(model, quant, variant):
    """
    Load one wikitext_ppl result.

    Inputs:
        - model (str): Model alias
        - quant (str): Quantizer key
        - variant (str): wikitext2_ppl[_<tag>]

    Outputs:
        - d (dict or None): Parsed result
        - path (Path): File looked up
    """
    p = result_path(_PPL_SCRIPT, RunConfig(model = model, quantizer = quant, variant = variant))
    try:
        return json.loads(p.read_text()), p
    except (OSError, ValueError):
        return None, p


def main():
    ap = argparse.ArgumentParser(description = "Rebuilt vs paper WikiText-2 PPL.")
    ap.add_argument("--model", required = True)
    ap.add_argument("--quant", required = True)
    ap.add_argument("--tag", default = "rebuilt")
    ap.add_argument("--tol", type = float, default = 2.0, help = "warn above this %% difference")
    ap.add_argument("--ref_tag", default = None,
                    help = "compare with wikitext2_ppl_<ref_tag> (e.g. rebuilt_gptqmodel, the same "
                         "checkpoint via gptqmodel) instead of the paper file")
    args = ap.parse_args()

    rows, warn, missing = [], [], False
    for q in (args.quant, "fp32"):
        new, pn = _load(args.model, q, f"wikitext2_ppl_{args.tag}")
        old, po = _load(args.model, q, f"wikitext2_ppl_{args.ref_tag}" if args.ref_tag else "wikitext2_ppl")
        if new is None and q == "fp32":
            continue  # fp32 is only measured in some rebuilt jobs
        if new is None or old is None:
            print(f"  !! missing: {pn if new is None else po}")
            missing = True
            continue
        a, b = new.get("wikitext2_ppl"), old.get("wikitext2_ppl")
        if a is None or b is None:
            print(f"  !! {q}: PPL is null (rebuilt status={new.get('status')})")
            missing = True
            continue
        if not (math.isfinite(a) and math.isfinite(b)):
            # NaN/inf compares False against the tolerance, so it would pass silently
            print(f"  !! {q}: PPL is not finite (rebuilt {a}, paper {b})")
            warn.append(f"{q} PPL not finite ({a})")
            missing = True
            continue
        d = 100.0 * (a - b) / b
        flag = abs(d) > args.tol
        rows.append((q, a, b, d, flag))
        if flag:
            warn.append(f"{q} {a:.3f} vs {b:.3f} ({d:+.2f}%)")
        if q == args.quant:
            bi = new.get("checkpoint_build") or {}
            v = bi.get("versions", {})
            print(f"  checkpoint backend={bi.get('backend')}  torch={v.get('torch')}  "
                  f"transformers={v.get('transformers')}  auto_gptq={v.get('auto_gptq')}  "
                  f"gptqmodel={v.get('gptqmodel')}")

    ref = args.ref_tag or "paper"
    print(f"  {args.model:14s} {'cell':6s} {args.tag[:9]:>9s} {ref[:13]:>13s} {'diff':>8s}")
    for q, a, b, d, flag in rows:
        print(f"  {args.model:14s} {q:6s} {a:9.3f} {b:13.3f} {d:+7.2f}%"
              f"{'  <-- WARN >' + format(args.tol, 'g') + '%' if flag else ''}")
    if warn:
        print(f"  WARNING: rebuilt PPL differs from {ref} by more than {args.tol:g}%: "
              + "; ".join(warn))
    elif rows and not missing:
        print(f"  OK: within {args.tol:g}% of {ref}.")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
