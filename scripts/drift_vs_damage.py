#!/usr/bin/env python
"""Which displacement measure tracks safety damage -- diff-in-means drift or
probe rotation? This is the Phi-3 question: Phi-3 is damaged (+20pp) with a high
diff-in-means cosmed (0.935, "no drift") but a low probe-rotation cos (0.696,
"big rotation"). If probe rotation tracks damage across cells where diff-in-means
doesn't, the mechanism should be framed on the readable/separating axis.

Reads existing outputs; reports a per-cell table + damage-vs-displacement
correlations. Pure stdlib + numpy.
  python scripts/drift_vs_damage.py
"""
import glob
import json
from pathlib import Path

import numpy as np

MODELS = ["llama", "qwen", "mistral", "phi3"]
QUANTS = ["awq", "gptq4", "gptq3", "gptq2", "aqlm2"]


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def f(x, r=3):
    return "-" if x is None else round(float(x), r)


def main():
    rows = []
    for m in MODELS:
        for q in QUANTS:
            dd = load(f"outputs/signal_geometry/direction_drift_{m}_{q}.json")
            pr = load(f"outputs/signal_geometry/probe_rotation_{m}_{q}.json")
            rs = load(f"outputs/regime_sweep/restoration_sweep_{m}_{q}.json")
            if not (dd and rs):
                continue
            rows.append({
                "cell": f"{m}/{q}",
                "damage": rs.get("damage_pp"),
                "cosmed": dd.get("summary", {}).get("cos_sim_median"),
                "probe_rot": (pr or {}).get("summary", {}).get("cos_median"),
                "read_rec": rs.get("recovery_pp_read"),
                "swap_ok": rs.get("swap_valid", True),
            })

    if not rows:
        print("no matched direction_drift + restoration_sweep cells found.")
        return

    print(f"{'cell':16} {'damage':>7} {'cosmed':>7} {'probe_rot':>9} "
          f"{'read_rec':>8} {'swap_ok':>7}")
    print("-" * 60)
    for r in rows:
        print(f"{r['cell']:16} {str(f(r['damage'])):>7} {str(f(r['cosmed'])):>7} "
              f"{str(f(r['probe_rot'])):>9} {str(f(r['read_rec'])):>8} "
              f"{str(r['swap_ok']):>7}")

    # Correlate damage with displacement = (1 - similarity). Only swap-valid,
    # damaged-or-not cells with a real damage number.
    valid = [r for r in rows if r["swap_ok"] and r["damage"] is not None]
    print(f"\n{len(valid)} swap-valid cells with damage numbers.")

    cm = [r for r in valid if r["cosmed"] is not None]
    if len(cm) >= 3:
        d = np.array([r["damage"] for r in cm])
        disp = np.array([1 - r["cosmed"] for r in cm])
        c = np.corrcoef(d, disp)[0, 1] if d.std() and disp.std() else float("nan")
        print(f"  corr(damage, 1 - diff-in-means cosmed) = {c:.3f}  (n={len(cm)})")

    prm = [r for r in valid if r["probe_rot"] is not None]
    if len(prm) >= 3:
        d = np.array([r["damage"] for r in prm])
        disp = np.array([1 - r["probe_rot"] for r in prm])
        c = np.corrcoef(d, disp)[0, 1] if d.std() and disp.std() else float("nan")
        print(f"  corr(damage, 1 - probe rotation)       = {c:.3f}  (n={len(prm)})")

    print("\nRead: if corr with probe-rotation >> corr with cosmed, the operative "
          "displacement is the readable/separating axis, and Phi-3 (probe_rot 0.696, "
          "damaged) CONFIRMS the mechanism. If neither correlates, Phi-3 is a genuine "
          "dissociation -> lead with the causal read-restoration claim instead.")


if __name__ == "__main__":
    main()
