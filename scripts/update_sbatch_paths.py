#!/usr/bin/env python3
"""Repoint scripts/*.sbatch to experiments/ paths; move orphaned launchers to attic/scripts/."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
ATTIC = ROOT / "attic" / "scripts"

REPLACEMENTS = [
    ("diagnostics/direction_drift.py", "experiments/signal_geometry/direction_drift.py"),
    ("diagnostics/refusal_probe.py", "experiments/signal_geometry/probe_accuracy.py"),
    ("diagnostics/probe_geometry.py", "experiments/signal_geometry/probe_rotation.py"),
    ("diagnostics/readwrite_restoration.py", "experiments/regime_sweep/restoration_sweep.py"),
    ("diagnostics/readwrite_per_layer_range.py", "experiments/localization/depth_bands.py"),
    ("diagnostics/readwrite_per_projection.py", "experiments/localization/per_projection.py"),
    ("diagnostics/per_projection_middle.py", "experiments/localization/per_projection.py"),
    ("diagnostics/layer_routing.py", "experiments/boundary/routing_null.py"),
    ("diagnostics/headline_eval_full.py", "experiments/headline/headline_eval.py"),
    ("diagnostics/lrqat_repair.py", "experiments/trained_repair/lrqat.py"),
    ("diagnostics/activation_swap.py", "experiments/boundary/activation_swap.py"),
    ("diagnostics/probe_causal_ablation.py", "experiments/boundary/probe_causal_ablation.py"),
    ("diagnostics/arditi_official_repro.py", "experiments/behavioral_direction/arditi_repro.py"),
    ("-m diagnostics.layer_routing", "-m experiments.boundary.routing_null"),
    ("-m diagnostics.headline_eval_full", "-m experiments.headline.headline_eval"),
]

ORPHAN_PATTERNS = [
    "routing_random_control",
    "random_ablation_dist",
    "f2_directional_ablation",
    "behavioral_outcome_probe",
    "arditi_llama",
    "arditi_official",
    "run_arditi",
    "snip_",
    "eq7_",
    "mignull_",
    "refold_gate",
    "phi3_routing_control",
    "phi3_routing_ctrl",
    "_ctrl_norm",
]


def main():
    ATTIC.mkdir(parents=True, exist_ok=True)
    moved = updated = 0
    for sbatch in sorted(SCRIPTS.glob("*.sbatch")):
        text = sbatch.read_text(encoding="utf-8", errors="replace")
        if any(p in sbatch.name for p in ORPHAN_PATTERNS):
            dst = ATTIC / sbatch.name
            if not dst.exists():
                shutil.move(str(sbatch), str(dst))
                moved += 1
                print(f"archived {sbatch.name}")
            continue
        new_text = text
        for old, new in REPLACEMENTS:
            new_text = new_text.replace(old, new)
        if new_text != text:
            sbatch.write_text(new_text, encoding="utf-8")
            updated += 1
            print(f"updated {sbatch.name}")
    print(f"Updated {updated} sbatch files; archived {moved} orphaned launchers.")


if __name__ == "__main__":
    main()
