#!/usr/bin/env python3
"""Move legacy diagnostics/attic/stale scripts into v1/; repoint paper sbatch to experiments/."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "v1"

# Whole trees -> v1/
MOVE_DIRS = [
    (ROOT / "diagnostics", V1 / "diagnostics"),
    (ROOT / "attic", V1 / "attic"),
    (ROOT / "outputs" / "attic", V1 / "outputs" / "attic"),
    (ROOT / "outputs" / "v1_archive", V1 / "outputs" / "v1_archive"),
    (ROOT / "outputs" / "_provisional_backup", V1 / "outputs" / "_provisional_backup"),
    (ROOT / "scripts" / "logs", V1 / "scripts" / "logs"),
]

# Loose files at outputs/ root (not claim-mirrored paper JSONs)
OUTPUTS_LOOSE = [
    "arditi_selection_llama_aqlm2.json",
    "arditi_selection_llama_fp32.json",
    "f3a_three_criteria.json",
    "random_ablation_dist_llama_awq.json",
    "random_ablation_dist_mistral_aqlm2.json",
    "random_ablation_dist_qwen_gptq3.json",
    "run_activation_swap.log",
    "run_behavioral_outcome_probe.log",
    "run_direction_drift.log",
    "run_f1_f2.log",
    "run_harmbench_check.log",
    "run_probe_causal_ablation.log",
    "run_refusal_probes.log",
]

# scripts/ -> v1/scripts/ (non-paper launchers and one-off tools)
SCRIPTS_MOVE = [
    "phi3_diag.sbatch",
    "phi3_probe_rotation.sbatch",
    "phi3_routing_z.sbatch",
    "phi3_routing_z_tp3.sbatch",
    "phi3_route_norm.sbatch",
    "phi3_headline_stats.sbatch",
    "run_remaining.sbatch",
    "pv_qwen.sbatch",
    "download_qwen.sbatch",
    "quant_qwen.sbatch",
    "tokenize_qwen.sbatch",
    "quantize_missing_gptq.sbatch",
    "quantize_aqlm_llama_calibvar.sbatch",
    "compare_jbb_sources.py",
    "compare_xstest_sources.py",
    "compare_all_dataset_sources.py",
    "migrate_outputs.py",
    "rename_outputs_short.py",
    "update_sbatch_paths.py",
    "download_datasets.sh",
    "cleanup.sh",
    "screen_qwen.sh",
    "convert_qwen.sh",
    "download_models.py",
    "download_repo.py",
    "quantize_missing_gptq.py",
    "build_aqlm_calibration_variants.py",
    "archive_to_v1.py",
]

SBATCH_REPLACEMENTS = [
    ("-m diagnostics.qresafe_dpo", "-m experiments.trained_repair.qresafe_dpo"),
    ("-m diagnostics.qresafe_fold_writeback", "-m experiments.trained_repair.qresafe_fold"),
    ("-m diagnostics.build_displacement_pairs", "-m experiments.trained_repair.qresafe_pairs"),
    ("-m diagnostics.probe_refit_control", "-m experiments.boundary.probe_refit_control"),
    ("-m diagnostics.probe_geometry", "-m experiments.signal_geometry.probe_rotation"),
    ("-m diagnostics.readwrite_per_layer_range", "experiments/localization/depth_bands.py"),
    ("--proj_group read", "--projections read"),
    ("--proj_group write", "--projections write"),
    (
        "--output outputs/readwrite_per_layer_range_read_llama7b_aqlm2.json",
        "--output outputs/localization/depth_bands_read_llama_aqlm2.json",
    ),
    (
        "--output outputs/readwrite_per_layer_range_write_llama7b_aqlm2.json",
        "--output outputs/localization/depth_bands_write_llama_aqlm2.json",
    ),
    (
        "--output outputs/displacement_pairs_llama7b_wj.json",
        "--output outputs/trained_repair/qresafe_pairs_llama_aqlm2_wj.json",
    ),
]


def move_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"skip (exists): {dst.relative_to(ROOT)}")
        return
    shutil.move(str(src), str(dst))
    print(f"moved {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def main() -> None:
    V1.mkdir(exist_ok=True)
    (V1 / "outputs").mkdir(exist_ok=True)
    (V1 / "scripts").mkdir(exist_ok=True)

    for src, dst in MOVE_DIRS:
        move_path(src, dst)

    loose_dst = V1 / "outputs" / "loose"
    loose_dst.mkdir(parents=True, exist_ok=True)
    for name in OUTPUTS_LOOSE:
        move_path(ROOT / "outputs" / name, loose_dst / name)

    for name in SCRIPTS_MOVE:
        move_path(ROOT / "scripts" / name, V1 / "scripts" / name)

    for log in ROOT.glob("scripts/*.log"):
        move_path(log, V1 / "scripts" / "logs" / log.name)

    updated = 0
    for sbatch in sorted((ROOT / "scripts").glob("*.sbatch")):
        text = sbatch.read_text(encoding="utf-8", errors="replace")
        new_text = text
        for old, new in SBATCH_REPLACEMENTS:
            new_text = new_text.replace(old, new)
        new_text = new_text.replace(
            "python -u -m experiments/localization/depth_bands.py",
            "python -u experiments/localization/depth_bands.py",
        )
        if new_text != text:
            sbatch.write_text(new_text, encoding="utf-8")
            updated += 1
            print(f"updated {sbatch.name}")

    readme = V1 / "README.md"
    if not readme.exists():
        readme.write_text(
            """# v1 — archived pre-consolidation code

Legacy diagnostics, dropped experiments, stale launchers, and old outputs.
**Not used for paper reproducibility** — see `RESULTS.md` at repo root.

| Path | Contents |
|------|----------|
| `diagnostics/` | Original monolithic experiment scripts (superseded by `experiments/` + `core/`) |
| `attic/` | Dropped routing nulls, directional ablation, SNIP, superseded Arditi runners |
| `scripts/` | Phi-3 one-offs, compare/migrate utilities, quantize/download helpers, job logs |
| `outputs/` | `attic/`, `v1_archive/`, loose JSONs from pre-claim-mirror layout |

Active paper pipeline: `core/`, `experiments/`, `tests/`, `scripts/*.sbatch` (keepers), `outputs/<claim>/`.
""",
            encoding="utf-8",
        )
        print("wrote v1/README.md")

    print(f"Done. Updated {updated} sbatch files.")


if __name__ == "__main__":
    main()
