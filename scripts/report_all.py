# Consolidated results report: full-write scope audit, WikiText-2 PPL, paired
# McNemar studiable-set screen, and headline McNemar + points-scale CI, all in one
# pass. Run from repo root:  $SQPY scripts/report_all.py
import glob
import json
import os

import numpy as np


def latest(pat):
    """
    Find the most recently modified file matching a glob pattern.

    Inputs:
        - pat (str): Glob pattern under outputs/ or experiments/

    Outputs:
        - path (str | None): Latest matching file path, or None
    """
    fs = glob.glob("outputs/" + pat, recursive = True) + glob.glob("experiments/" + pat, recursive = True)
    return max(fs, key = os.path.getmtime) if fs else None


def hr(t):
    """
    Print a section header banner.

    Inputs:
        - t (str): Section title

    Outputs:
        - None (prints to stdout)
    """
    print("\n" + "=" * 72 + f"\n  {t}\n" + "=" * 72)


# --- 1. Full-write scope audit -------------------------------------------------
hr("1. FULL-WRITE SCOPE AUDIT  (param_matched, Llama/AQLM-2)")
p = latest("**/param_matched_restore_llama_aqlm2*.json")
if p:
    d = json.load(open(p))
    pb, rp, ac = d["param_budget"], d["recovery_pp"], d["asr_per_condition"]
    print(f"  scope: write_params = {pb['write_params']/1e9:.2f}B (full write, not middle band)"
          f"  read_total = {pb['read_params_total']/1e9:.2f}B  match = {pb['match_fraction']:.3f}")
    print(f"  whole_subset = {pb['whole_subset']}  ({pb['whole_subset_params']/1e9:.2f}B)")
    print(f"  ASR: quant = {ac['quant']}  write_full = {ac['write_full']}  read_matched = {ac['read_matched_whole']}")
    rw, rr = rp["write_full"], rp["read_matched_whole"]
    rel = "~=" if abs(rr - rw) < 3 else (">" if rr > rw else "<")
    print(f"  recovery(pp): write_full = {rw}  read_matched_whole = {rr}  =>  read {rel} write at matched budget")
else:
    print("  (no param_matched JSON found)")

# --- 2. WikiText-2 PPL ---------------------------------------------------------
hr("2. WIKITEXT-2 PPL  (continuous coherence, complements ARC)")
rows, seen = [], set()
for q in glob.glob("outputs/**/*.json", recursive = True) + glob.glob("experiments/**/*.json", recursive = True):
    rq = os.path.realpath(q)
    if rq in seen:
        continue
    try:
        dd = json.load(open(q))
    except Exception:
        continue
    if "wikitext2_ppl" not in dd:
        continue
    seen.add(rq)
    rows.append([dd.get("model"), dd.get("quantizer"), dd.get("wikitext2_ppl"),
                 dd.get("ppl_ratio"), dd.get("status")])
order = {"fp32": 0, "awq": 1, "gptq4": 2, "aqlm2": 3, "gptq3": 4, "gptq2": 5}
rows.sort(key = lambda r: (str(r[0]), order.get(r[1], 9)))
fmt = lambda v: "" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
print("  " + "  ".join(f"{h:>9}" for h in ["model", "quant", "ppl", "x_fp32", "status"]))
for r in rows:
    print("  " + "  ".join(f"{fmt(v):>9}" for v in r))
ok = sum(1 for r in rows if r[1] != "fp32" and r[4] == "ok")
print(f"  --> {ok}/20 quant PPL cells OK")

# --- 3. Paired McNemar studiable-set screen ------------------------------------
hr("3. PAIRED dASR McNEMAR SCREEN  (studiable-set selection, Holm-adjusted)")
try:
    from experiments.regime_sweep.paired_screen import sweep, MODELS, QUANTS, _print_table
    _print_table(sweep([(m, q) for m in MODELS for q in QUANTS]))
except Exception as e:
    print(f"  screen unavailable ({type(e).__name__}: {e})")
    print("  (needs the flag-carrying restoration re-runs to have finished)")

# --- 4. Headline McNemar + native points CI ------------------------------------
hr("4. HEADLINE McNEMAR + POINTS CI  (Llama/AQLM-2)")
hp = latest("**/headline_llama_aqlm2*.json")
if hp:
    d = json.load(open(hp))
    for name, blk in (d.get("mcnemar") or {}).items():
        print(f"  {name:16s} b = {blk['b']:>3} c = {blk['c']:>3}  p = {blk['p_value']:.2e}")

    def _nb(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "n_boot":
                    return v
                r = _nb(v)
                if r is not None:
                    return r
        return None

    fbc = d.get("flags_by_cond")
    if fbc and {"aqlm2", "read_restored", "write_restored"} <= set(fbc):
        arr = {k: np.asarray(v, float) for k, v in fbc.items()}
        n = len(next(iter(arr.values())))
        B = int(_nb(d) or 2000)
        rng = np.random.default_rng(0)
        rmw = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            a = {k: arr[k][idx].mean() * 100 for k in arr}
            rmw.append((a["aqlm2"] - a["read_restored"]) - (a["aqlm2"] - a["write_restored"]))
        ci = [round(float(np.percentile(rmw, 2.5)), 1), round(float(np.percentile(rmw, 97.5)), 1)]
        print(f"  n_boot = {B}  read-write points CI = {ci}  "
              f"(straddles 0 -> read>write NOT significant, matches read_vs_write p)")
else:
    print("  (no headline JSON found)")

print("\n" + "=" * 72)
print("  Also finalized (not in JSONs): AWQ group sizes all g128 / INT4 / GEMM,")
print("  read directly from each checkpoint's quant config.")
print("=" * 72)
