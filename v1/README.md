# v1 — archived pre-consolidation code

Legacy diagnostics, dropped experiments, stale launchers, and old outputs.
**Not used for paper reproducibility** — see `RESULTS.md` at repo root.

| Path | Contents |
|------|----------|
| `diagnostics/` | Original monolithic experiment scripts (superseded by `experiments/` + `core/`) |
| `attic/` | Dropped routing nulls, directional ablation, SNIP, superseded Arditi runners |
| `scripts/` | Phi-3 one-offs, compare/migrate utilities, quantize/download helpers, job logs |
| `outputs/` | `attic/`, `v1_archive/`, loose JSONs from pre-claim-mirror layout |

Active paper pipeline: `core/`, `experiments/`, `tests/`, `scripts/*.sbatch` (keepers), `outputs/<claim>/`.
