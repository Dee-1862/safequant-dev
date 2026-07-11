#!/usr/bin/env python3
"""Bulk-move flat outputs/*.json into claim-mirrored subfolders."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

# prefix -> (folder, new_stem or None to keep stem)
RULES: list[tuple[str, str, str | None]] = [
    ("direction_drift_", "signal_geometry", None),
    ("refusal_probe_mlp_", "signal_geometry", "probe_accuracy_mlp_"),
    ("refusal_probe_", "signal_geometry", "probe_accuracy_"),
    ("probe_geometry_", "signal_geometry", "probe_rotation_"),
    ("readwrite_per_layer_range_", "localization", "depth_bands_"),
    ("readwrite_per_projection_", "localization", "per_projection_"),
    ("per_projection_middle_", "localization", "per_projection_"),
    ("readwrite_", "regime_sweep", "restoration_sweep_"),
    ("headline_full_", "headline", "headline_eval_"),
    ("layer_routing_", "boundary", "routing_null_"),
    ("probe_causal_ablation_", "boundary", None),
    ("probe_refit_control_", "boundary", None),
    ("activation_swap_", "boundary", None),
    ("displacement_pairs_", "trained_repair", "qresafe_pairs_"),
    ("qresafe_", "trained_repair", None),
    ("lrqat_", "trained_repair", None),
    ("directional_ablation_", "attic", None),
    ("routing_random_control_", "attic", None),
    ("routing_eq7_", "attic", None),
    ("routing_migration_null_", "attic", None),
    ("snip_", "attic", None),
    ("behavioral_outcome_probe_", "attic", None),
]


def target_for(name: str) -> Path | None:
    for prefix, folder, new_stem in RULES:
        if name.startswith(prefix):
            stem = name if new_stem is None else new_stem + name[len(prefix) :]
            return OUT / folder / stem
    return None


def main():
    moved = 0
    for src in sorted(OUT.glob("*.json")):
        dst = target_for(src.name)
        if dst is None:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        moved += 1
        print(f"  {src.name} -> {dst.relative_to(ROOT)}")
    print(f"Moved {moved} files.")


if __name__ == "__main__":
    main()
