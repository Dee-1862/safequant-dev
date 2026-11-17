#!/usr/bin/env python
"""rho intervals for the localization tables (Tables 5, 6, 7). CPU only.

    python scripts/localization_intervals.py            # markdown tables
    python scripts/localization_intervals.py --json     # machine-readable

rho = share of the quantization damage a restoration removes:
    rho = 100 * (ASR_quant - ASR_cond) / (ASR_quant - ASR_fp32)

  Table 5  per-projection, middle band L11-22   (tab:f3a_proj)
  Table 6  depth bands, read and write           (tab:f3a_range)
  Table 7  parameter-matched read restoration    (tab:controls)

Interval per row:
  * paired:   the rerun with per-prompt labels exists (*_flags.json from
              scripts/localization/localization_with_flags.sbatch) ->
              core.stats.bootstrap_recovery(flags, quant, n_boot=2000, seed=42).
  * unpaired (conservative): no flags file -> normal approximation from the ASRs alone
              (each condition an independent binomial sample of n=200, delta method for
              the ratio). Ignoring the positive pairing correlation makes these wider.
Point estimates are read from the file each row's interval comes from; the paired
rerun can differ slightly from the published run (AQLM kernels are not
bit-deterministic).
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.stats import bootstrap_recovery  # noqa: E402  (numpy only)

OUT = ROOT / "outputs"
N_BOOT, SEED, Z = 2000, 42, 1.959964
MODELS = (("llama", "Llama-3.1-8B"), ("phi3", "Phi-3-mini"))
TAG = "flags"


def _load(p):
    try:
        return json.loads(Path(p).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None


def _pick(base, flagged):
    """Prefer the flags rerun; else the published file. Returns (data, path, paired)."""
    d = _load(flagged)
    if d and d.get("flags_by_cond"):
        return d, flagged, True
    d = _load(base)
    return d, base, False


def rho_point(q, c, fp):
    dmg = q - fp
    return None if dmg == 0 else 100.0 * (q - c) / dmg


def rho_unpaired(q, c, fp, n):
    """Delta-method normal CI for rho from ASRs (percent) of independent samples."""
    pq, pc, pf = q / 100.0, c / 100.0, fp / 100.0
    vq, vc, vf = (p * (1 - p) / n for p in (pq, pc, pf))
    num, den = pq - pc, pq - pf
    if den == 0:
        return None
    r = num / den
    # var(N)=vq+vc, var(D)=vq+vf, cov(N,D)=vq (q is shared by numerator and denominator)
    var_r = (vq + vc - 2 * r * vq + r * r * (vq + vf)) / (den * den)
    half = Z * math.sqrt(max(var_r, 0.0))
    return [100.0 * (r - half), 100.0 * (r + half)]


def rho_paired(flags, qkey, cond_flags):
    """bootstrap_recovery CI for one condition: pass it in the slot the function reads."""
    fb = {"fp32": flags["fp32"], qkey: flags[qkey],
          "read_restored": cond_flags, "write_restored": cond_flags, "both_restored": cond_flags}
    return bootstrap_recovery(fb, qkey, n_boot = N_BOOT, seed = SEED)["read"]


def _row(label, d, paired, qkey, cond, n, cond_flags = None, asr_override = None):
    asr = d["asr_per_condition"]
    q, fp = asr[qkey], asr["fp32"]
    c = asr_override if asr_override is not None else asr[cond]
    if paired:
        cf = cond_flags if cond_flags is not None else d["flags_by_cond"][cond]
        ci, method = rho_paired(d["flags_by_cond"], qkey, cf), "paired bootstrap"
    else:
        ci, method = rho_unpaired(q, c, fp, n), "unpaired (conservative)"
    pt = rho_point(q, c, fp)
    return {
        "row": label, "asr": round(c, 1), "rho_pct": None if pt is None else round(pt),
        "ci95": None if ci is None else [round(ci[0]), round(ci[1])], "method": method,
    }


def _n(d):
    return int((d.get("manifest") or {}).get("n_eval") or 200)


def table5():
    out = {}
    for m, label in MODELS:
        base = OUT / "localization" / f"per_projection_{m}_aqlm2_L11-22.json"
        flg = OUT / "localization" / f"per_projection_{m}_aqlm2_L11-22_{TAG}.json"
        d, src, paired = _pick(base, flg)
        if not d:
            out[m] = {"label": label, "status": "missing", "expected": base.relative_to(ROOT).as_posix()}
            continue
        rows = []
        for proj in d["recovery_pp_per_projection"]:
            cond = f"only_{proj.replace('.', '_')}_middle"
            rows.append(_row(proj, d, paired, "aqlm2", cond, _n(d)))
        out[m] = {"label": label, "source": Path(src).relative_to(ROOT).as_posix(),
                  "damage_pp": round(d["damage_pp"], 1), "rows": rows}
    return out


def table6():
    out = {}
    for m, label in MODELS:
        for side in ("read", "write"):
            base = OUT / "localization" / f"depth_bands_{m}_aqlm2_{side}.json"
            flg = OUT / "localization" / f"depth_bands_{m}_aqlm2_{side}_{TAG}.json"
            d, src, paired = _pick(base, flg)
            key = f"{m}/{side}"
            if not d:
                out[key] = {"label": f"{label} {side}", "status": "missing",
                            "expected": base.relative_to(ROOT).as_posix()}
                continue
            rows = [_row(band, d, paired, "aqlm2", f"layers_{band}", _n(d))
                    for band in ("early", "middle", "late")]
            out[key] = {"label": f"{label} {side}", "source": Path(src).relative_to(ROOT).as_posix(),
                        "damage_pp": round(d["damage_pp"], 1), "rows": rows}
    return out


def table7():
    out = {}
    for m, label in MODELS:
        base = OUT / "regime_sweep" / f"param_matched_restore_{m}_aqlm2.json"
        flg = OUT / "regime_sweep" / f"param_matched_restore_{m}_aqlm2_{TAG}.json"
        d, src, paired = _pick(base, flg)
        if not d:
            out[m] = {"label": label, "status": "missing", "expected": base.relative_to(ROOT).as_posix()}
            continue
        asr, n = d["asr_per_condition"], _n(d)
        seeds = sorted((k for k in asr if k.startswith("read_matched_random_s")),
                       key = lambda k: int(k.rsplit("s", 1)[1]))
        rows = [_row("full write", d, paired, "quant", "write_full", n),
                _row("matched read, structured", d, paired, "quant", "read_matched_whole", n)]
        for k in seeds:
            rows.append(_row(f"matched read, random {k.rsplit('_', 1)[1]}", d, paired, "quant", k, n))
        # Mean over random subsets: per-prompt mean of the seed labels when paired
        mean_asr = statistics.mean(asr[k] for k in seeds)
        mean_flags = None
        if paired:
            fb = d["flags_by_cond"]
            mean_flags = [sum(fb[k][i] for k in seeds) / len(seeds) for i in range(len(fb["quant"]))]
        rows.append(_row(f"matched read, random mean ({len(seeds)} seeds)", d, paired, "quant",
                         seeds[0], n, cond_flags = mean_flags, asr_override = mean_asr))
        rho_seeds = [rho_point(asr["quant"], asr[k], asr["fp32"]) for k in seeds]
        out[m] = {
            "label": label, "source": Path(src).relative_to(ROOT).as_posix(),
            "damage_pp": round(asr["quant"] - asr["fp32"], 1), "rows": rows,
            "random_subsets": {
                "n_seeds": len(seeds),
                "asr_mean": round(mean_asr, 1),
                "asr_sd": round(statistics.stdev(asr[k] for k in seeds), 1) if len(seeds) > 1 else None,
                "rho_sd": round(statistics.stdev(rho_seeds), 1) if len(seeds) > 1 else None,
            },
        }
    return out


def collect():
    return {"table5_per_projection": table5(), "table6_depth_bands": table6(),
            "table7_param_matched": table7(),
            "meta": {"n_boot": N_BOOT, "seed": SEED, "flags_tag": TAG}}


def to_markdown(res):
    L = ["# Localization rho intervals (Tables 5-7)\n",
         f"Paired = bootstrap_recovery on *_{TAG}.json labels (n_boot={N_BOOT}, seed={SEED}); "
         "unpaired (conservative) = normal approximation from ASRs (n=200) when no labels exist.\n"]
    titles = (("table5_per_projection", "Table 5: per-projection restore, middle band L11-22"),
              ("table6_depth_bands", "Table 6: depth-band restore"),
              ("table7_param_matched", "Table 7: parameter-matched read restoration"))
    for key, title in titles:
        L.append(f"\n## {title}\n")
        for blk in res[key].values():
            if blk.get("status") == "missing":
                L.append(f"\n### {blk['label']}\nmissing: `{blk['expected']}`")
                continue
            L.append(f"\n### {blk['label']} (damage {blk['damage_pp']} pp)\nSource: `{blk['source']}`\n")
            L.append("| Restoration | ASR | rho % | 95% CI | interval |")
            L.append("|---|---:|---:|---|---|")
            for r in blk["rows"]:
                ci = "--" if not r["ci95"] else f"[{r['ci95'][0]}, {r['ci95'][1]}]"
                L.append(f"| {r['row']} | {r['asr']} | {r['rho_pct']} | {ci} | {r['method']} |")
            rs = blk.get("random_subsets")
            if rs:
                L.append(f"\nRandom subsets: {rs['n_seeds']} seeds, ASR {rs['asr_mean']} "
                         f"(sd {rs['asr_sd']}), rho sd {rs['rho_sd']} pp")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description = "rho intervals for Tables 5-7 (CPU only).")
    ap.add_argument("--json", action = "store_true", help = "print JSON instead of markdown")
    args = ap.parse_args()
    res = collect()
    print(json.dumps(res, indent = 2) if args.json else to_markdown(res))


if __name__ == "__main__":
    main()
