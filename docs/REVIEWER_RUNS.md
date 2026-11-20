# Reviewer runs: index

Reviewer analyses were run on GH200 nodes with the software versions in docs/ENV.md; the original analyses on H100 nodes. Held-out and X1 were executed one condition per job and merged; a determinism check found identical outputs across jobs.

One row per item: the code that computes it, the launcher that ran it, the result files
(merged results only), and the script that reads them into tables. All result files are
new; the paper's result files are unchanged. `scripts/reviewer_runs.py` prints every
item below (and `missing` for anything absent); `scripts/paper_values.py --md` appends
that block to `PAPER_TABLES.md`.

## Items

| Item | Code | Launcher | Output | Reader | Status |
|---|---|---|---|---|---|
| X1: XSTest false refusal, Llama GPTQ-4 | `experiments/headline/headline_piece.py --parts xstest` (one condition per job), merged by `experiments/headline/merge_pieces.py` | `scripts/headline/headline_piece.sbatch` | `outputs/headline/headline_llama_gptq4_xstest.json` | `reviewer_runs.py` (`x1`) | done; XSTest only (no ASR / ARC) |
| Held-out AQLM-2 screen (Meta-Llama-3-8B-Instruct) | `headline_piece.py --parts wj,arc,xstest` per condition, merged by `merge_pieces.py` | `scripts/headline/headline_piece.sbatch` | `outputs/headline/headline_llama3-8b_aqlm2_heldout.json` | `reviewer_runs.py` (`heldout`), `experiments/regime_sweep/paired_screen.py --variant heldout` | done |
| X2: surface-feature control, 14 coherent cells | `experiments/boundary/probe_leakage_control.py --quantizer <q>` | `scripts/probe_pool.sbatch` / `scripts/probe_interactive_chain.sh`, pieces `x2:*` in `scripts/probe_pieces.sh` | `outputs/boundary/probe_leakage_control_<model>_<quant>.json` | `reviewer_runs.py` (`x2`) | done |
| X2: FP32 Mistral reference | `probe_leakage_control.py` (no `--quantizer`) | piece `x2fp:mistral7b-v02:fp32_rerun` | `outputs/boundary/probe_leakage_control_mistral_fp32_rerun.json` | `reviewer_runs.py` (`x2`) | done |
| X2: activation diagnostic, Mistral AQLM-2 | `experiments/boundary/x2_activation_diag.py` | piece `diag:mistral7b-v02:aqlm2:rerun` | `outputs/boundary/x2_activation_diag_mistral_aqlm2_rerun.json` | printed by the script | done |
| X3: matched real / shuffled-label probe, 14 coherent cells | `experiments/signal_geometry/probe_accuracy.py --paired` | pieces `x3p:*` | `outputs/signal_geometry/probe_accuracy_<model>_<quant>_paired.json` | `reviewer_runs.py` (`x3`) | done |
| X3 (earlier shuffled-only run, Llama / Phi-3 AQLM-2) | `probe_accuracy.py --shuffle_labels` | pieces `x3:*` (superseded) | `outputs/signal_geometry/probe_accuracy_<model>_aqlm2_shuffled.json` | not read (the `_paired` files contain it) | superseded |
| X4: FP-vs-FP refit floor, 100 splits | `experiments/boundary/probe_refit_control.py --n_splits 100` | pieces `x4:*` | `outputs/boundary/probe_refit_control_<model>_splits100.json` | `reviewer_runs.py` (`x4`, tested against `probe_rotation_<model>_<quant>.json`) | done (1000-split run not done) |
| X6: same-prompt repeats, Llama / Phi-3 AQLM-2 | `headline_eval.py --tag repK` | `scripts/headline/headline_repeats_aqlm2.sbatch` | `outputs/headline/headline_<model>_aqlm2_rep{1,2,3}.json` | `reviewer_runs.py` (`x6`) | done |
| Collapsed cells (GPTQ-2 x4, Mistral GPTQ-3): X3 pair, X2, ARC on rebuilt checkpoints | `probe_accuracy.py --paired --tag rebuilt`, `probe_leakage_control.py --tag rebuilt`, `scripts/regime_sweep/arc_check.py` | pieces `x3p:*:rebuilt`, `x2:*:rebuilt`; `scripts/regime_sweep/arc_check.sbatch` | `outputs/signal_geometry/probe_accuracy_<model>_<quant>_paired_rebuilt.json`, `outputs/boundary/probe_leakage_control_<model>_<quant>_rebuilt.json`, `outputs/regime_sweep/arc_check_<model>_gptq2_rebuilt.json` | `reviewer_runs.py` (`collapsed`) | done; ARC not run for Mistral GPTQ-3 |
| Tables 5-7 intervals (per-projection, depth bands, param-matched) | `experiments/localization/per_projection.py`, `depth_bands.py`, `experiments/regime_sweep/param_matched_restore.py` with `--tag flags` (per-prompt labels) | `scripts/localization/localization_with_flags.sbatch` | `outputs/localization/per_projection_<model>_aqlm2_L11-22_flags.json`, `outputs/localization/depth_bands_<model>_aqlm2_{read,write}_flags.json`, `outputs/regime_sweep/param_matched_restore_<model>_aqlm2_flags.json` | `scripts/localization_intervals.py` (point estimates and intervals both from the rerun) | done |
| WikiText-2 PPL, public checkpoints (AQLM-2, AWQ, FP32) | `experiments/regime_sweep/wikitext_ppl.py --tag rerun` | `scripts/regime_sweep/wikitext_ppl_rebuilt.sbatch` via `scripts/queue.sh` | `outputs/regime_sweep/wikitext_ppl_<model>_<quant>_wikitext2_ppl_rerun.json` | `scripts/regime_sweep/compare_ppl_rebuilt.py`, `scripts/report_all.py` (section 2) | done |
| WikiText-2 PPL, rebuilt GPTQ checkpoints, dense path | `wikitext_ppl.py --tag rebuilt` with `SQ_GPTQ_DENSE=1` | `scripts/dense_gptq.sbatch` | `outputs/regime_sweep/wikitext_ppl_<model>_gptq<b>_wikitext2_ppl_rebuilt.json` | `compare_ppl_rebuilt.py` | done |
| WikiText-2 PPL, rebuilt GPTQ checkpoints, gptqmodel (loader cross-check) | `wikitext_ppl.py --tag rebuilt_gptqmodel` | `wikitext_ppl_rebuilt.sbatch` | `outputs/regime_sweep/wikitext_ppl_<model>_<quant>_wikitext2_ppl_rebuilt_gptqmodel.json` | `compare_ppl_rebuilt.py --ref_tag rebuilt_gptqmodel` | done; the three GPTQ-3 files are failed loads (PPL nan, the gptqmodel 3-bit fault in docs/MODELS_QUANTIZATION.md) |
| ARC, Llama GPTQ-2 via gptqmodel | `arc_check.py --tag rebuilt_gptqmodel` | `arc_check.sbatch` | `outputs/regime_sweep/arc_check_llama_gptq2_rebuilt_gptqmodel.json` | `reviewer_runs.py` (`collapsed`, fallback) | done |
| Determinism check (first 32 prompts, Llama FP32, three jobs, two nodes) | `headline_piece.py --prompt_hi 32`, compared token by token | `headline_piece.sbatch` | kept outside git (raw generations): `outputs/headline/pieces/dettest_run{1,2,3}/` | `experiments/headline/compare_pieces.py` | done: 32/32 token-identical |

## Conventions

- File names: `<stem>_<model>_<quant>[_paper-variant][_analysis][_provenance][_repK]`.
  Provenance tokens: `rerun` (public checkpoint, GH200 stack), `rebuilt` (self-built GPTQ
  checkpoint, dense dequantized load path), `rebuilt_gptqmodel` (same checkpoint loaded by
  gptqmodel). Reviewer item numbers are never part of a file name.
- Per-condition pieces (`outputs/headline/pieces/`) hold raw generations and are not in git.
  They stay on disk with a `MANIFEST.sha256`; the merged files carry every statistic and
  list their pieces (job, node, timings) under `pieces`, with the paths used at run time
  (`outputs/headline_pieces/...`). `merge_pieces.py` refuses pieces from different commits,
  environments or settings.
- Two superseded Phi-3 PPL files from an earlier gptqmodel build
  (`wikitext_ppl_phi3_{fp32,gptq4}_wikitext2_ppl_rebuilt.json`) are kept on disk in
  `outputs/regime_sweep/superseded/` (not in git, `MANIFEST.sha256`).
