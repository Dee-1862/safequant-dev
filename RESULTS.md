# Experiment results provenance

Table numbers live here only — experiment folders use claim names so renumbering does not require directory renames.

Filename convention: `outputs/<claim_folder>/<stem>_<model>_<quantizer>[_variant].json` with short model tokens (`llama`, `phi3`, `mistral`, `qwen`). CLI accepts long aliases (`llama7b`, etc.).

| Table / artifact | Claim folder | Script | Example command | Output JSON |
|------------------|--------------|--------|-----------------|-------------|
| 1 direction drift | `signal_geometry` | `direction_drift.py` | `python experiments/signal_geometry/direction_drift.py --model llama --quantizer aqlm2 --token_pos -5` | `outputs/signal_geometry/direction_drift_llama_aqlm2.json` |
| 1 probe accuracy | `signal_geometry` | `probe_accuracy.py` | `python experiments/signal_geometry/probe_accuracy.py --model llama --quantizer aqlm2` | `outputs/signal_geometry/probe_accuracy_llama_aqlm2.json` |
| 1 probe rotation | `signal_geometry` | `probe_rotation.py` | `python experiments/signal_geometry/probe_rotation.py --model llama --quantizer aqlm2` | `outputs/signal_geometry/probe_rotation_llama_aqlm2.json` |
| Sec. 5 Arditi repro | `behavioral_direction` | `arditi_repro.py` | `python experiments/behavioral_direction/arditi_repro.py --model llama --quantizer aqlm2` | `outputs/behavioral_direction/arditi_repro_llama_aqlm2.json` |
| 2 restoration matrix | `regime_sweep` | `restoration_sweep.py` | `python experiments/regime_sweep/restoration_sweep.py --model llama --quantizer aqlm2` | `outputs/regime_sweep/restoration_sweep_llama_aqlm2.json` |
| 3 per-projection | `localization` | `per_projection.py` | `python experiments/localization/per_projection.py --model llama --quantizer aqlm2` | `outputs/localization/per_projection_llama_aqlm2.json` |
| 4 depth bands | `localization` | `depth_bands.py` | `python experiments/localization/depth_bands.py --model llama --quantizer aqlm2 --projections read` | `outputs/localization/depth_bands_llama_aqlm2_read.json` |
| 5 headline | `headline` | `headline_eval.py` | `python experiments/headline/headline_eval.py --model llama --quantizer aqlm2 --n_eval 400` | `outputs/headline/headline_llama_aqlm2.json` |
| 6–7 Q-Resafe pairs | `trained_repair` | `qresafe_pairs.py` | `python experiments/trained_repair/qresafe_pairs.py --model llama --wj_config eval` | `outputs/trained_repair/qresafe_pairs_llama_aqlm2_wj.json` |
| 6–7 Q-Resafe DPO | `trained_repair` | `qresafe_dpo.py` | `python experiments/trained_repair/qresafe_dpo.py --mask read --model llama` | `outputs/trained_repair/qresafe_dpo_llama_aqlm2_read.json` |
| 6–7 LR-QAT | `trained_repair` | `lrqat.py` | `python experiments/trained_repair/lrqat.py --model llama --role read --n_bits 4` | `outputs/trained_repair/lrqat_llama_aqlm2_read_int4.json` |
| Appendix probe ablation | `boundary` | `probe_causal_ablation.py` | `python experiments/boundary/probe_causal_ablation.py --model llama --quantizer aqlm2` | `outputs/boundary/probe_causal_ablation_llama_aqlm2.json` |
| Appendix refit control | `boundary` | `probe_refit_control.py` | `python experiments/boundary/probe_refit_control.py --model llama` | `outputs/boundary/probe_refit_control_llama.json` |
| Appendix activation swap | `boundary` | `activation_swap.py` | `python experiments/boundary/activation_swap.py --model llama --quantizer aqlm2` | `outputs/boundary/activation_swap_llama_aqlm2.json` |
| Appendix routing null | `boundary` | `routing_null.py` | `python experiments/boundary/routing_null.py --model llama --normalize` | `outputs/boundary/routing_null_llama_aqlm2_norm.json` |
| Discussion targeted dequant | `repair_deploy` | `targeted_dequant.py` | `python experiments/repair_deploy/targeted_dequant.py --dry_run` | (planned) |

## Regression anchors

```bash
pytest tests/test_anchors.py -v
```

Frozen metrics: `tests/anchors/frozen_metrics.json`.

## Raw checkpoints

Intermediate `*_raw.json` files are gitignored (`outputs/**/*_raw.json`).
