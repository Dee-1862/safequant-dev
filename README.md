# SafeQuant v2

Mechanistic-interpretability study of how quantization affects safety in instruction-tuned LLMs. Reproducibility-hardened version of the original `safequant/` codebase. Every random operation routes through one seed (42); see `SEEDS.md` for the seed strategy.

## Findings the active code supports

The repo is pruned to the 14 diagnostics whose outputs feed the v9 dashboard. Everything else lives in `/jet/home/bhoomire/projects/safequant/_trash/v2_inactive/` and is retrievable.

| Finding (v9 dashboard) | Diagnostic | Output prefix |
|---|---|---|
| F1 phase transition (cosine, three architectures) | `direction_drift.py` | `direction_drift_*.json` |
| F2a directional ablation + 20-trial random null | `directional_ablation.py` + `random_ablation_distribution.py` | `directional_ablation_*.json`, `random_ablation_dist_*.json` |
| F2b operative-layer routing | `layer_routing.py` | `layer_routing_*.json` |
| F2b routing random baseline (migration vs collapse) | `routing_random_control.py` | `routing_random_control_*.json` |
| Linear probe preservation (linear + MLP variants) | `refusal_probe.py` | `refusal_probe_*.json`, `refusal_probe_mlp_*.json` |
| Probe geometry / rotation | `probe_geometry.py` | `probe_geometry_*.json` |
| Probe causal ablation null #1 (harmful-vs-harmless) | `probe_causal_ablation.py` | `probe_causal_ablation_*.json` |
| Probe causal ablation null #2 (behavioral outcome) | `behavioral_outcome_probe.py` | `behavioral_outcome_probe_*.json` |
| F3a spatial localization (whole-network read/write) | `readwrite_restoration.py` | `readwrite_*.json` |
| F3a depth axis (early / middle / late) | `readwrite_per_layer_range.py` | `readwrite_per_layer_range_*.json` |
| F3a per-matrix axis (single projection across all layers) | `readwrite_per_projection.py` | `readwrite_per_projection_*.json` |
| F3a per-projection at middle layers | `per_projection_middle.py` | `per_projection_middle_*.json` |
| Activation swap (weight vs activation damage) | `activation_swap.py` | `activation_swap_*.json` |

The three-conditions analysis (Section 4 of the v9 paper) is a synthesis of F1 + F2b + F4 outputs and does not require a dedicated script.

## Repository layout

```
safequant_v2/
  README.md                  this file
  SEEDS.md                   seed strategy and reproducibility guarantees
  config.py                  paths, model aliases, defaults
  core/
    seeds.py                 set_global_seed(), DEFAULT_SEED
    datasets.py              WildJailbreak, JBB, Alpaca, ARC, C4 loaders
    model_loader.py          FP32/AQLM/GPTQ/AWQ loaders
    evaluation.py            ASR via HarmBench judge + ARC via logprob
    refusal_direction.py     Arditi-protocol helpers
  diagnostics/               14 active diagnostics (see table above)
  scripts/                   sbatch files + utility .py
  outputs/                   per-run JSON results
  figs/                      paper figures
  docs/                      paper plan, finding tables, slides notes
```

## Quantizer and model coverage

| Family | Llama-3.1-8B | Mistral-7B-v0.2 | Qwen-2.5-7B |
|---|---|---|---|
| FP16 baseline | yes | yes | yes |
| AWQ-4 | yes | yes | yes |
| GPTQ-4 | yes | yes | yes |
| GPTQ-3 | yes | yes | yes |
| GPTQ-2 | yes | yes | yes |
| AQLM-2 | yes | yes | (no checkpoint available) |

FP16 ASR baselines: Llama 41%, Mistral 75%, Qwen 57%.

## Quick start

```bash
conda activate /ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean
export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface

# F1 phase transition (per architecture)
sbatch scripts/f1_phase_transition_llama.sbatch
sbatch scripts/f1_phase_transition_mistral.sbatch
sbatch scripts/f1_phase_transition_qwen.sbatch

# F2 operative layer + random baseline
sbatch scripts/layer_routing_llama.sbatch                    # fast quantizers
sbatch scripts/layer_routing_llama_slow.sbatch               # GPTQ-3/2 (Triton)
sbatch scripts/routing_random_control_llama.sbatch           # 20-trial baseline

# F2a directional ablation + random null
sbatch scripts/f2_directional_ablation_llama_fast.sbatch
sbatch scripts/random_ablation_dist_llama_awq.sbatch         # K=20 trials, n=100

# F3 input-class probe (linear + MLP variants)
sbatch scripts/refusal_probe_llama_linear.sbatch
sbatch scripts/refusal_probe_llama_mlp.sbatch

# Probe causal-ablation nulls (two distinct probes, both null on Llama-AQLM-2)
sbatch scripts/add2_probe_causal_ablation.sbatch       # harmful-vs-harmless probe ablation
sbatch scripts/behavioral_outcome_probe_pilot.sbatch   # response-label probe ablation

# F4 probe direction cosine
sbatch scripts/probe_geometry_llama.sbatch

# F3a static restoration + decomposition axes
sbatch scripts/f3a_static_restoration_llama.sbatch
sbatch scripts/f3a_decomposition_per_layer_range.sbatch       # depth axis
sbatch scripts/perproj_llama_gptq2.sbatch                     # per-matrix axis
sbatch scripts/add4_per_projection_middle_llama.sbatch        # per-projection x middle

# Activation swap (Llama-AQLM-2, 6 swap points)
sbatch scripts/add5_activation_swap.sbatch
```

Mistral and Qwen sbatches follow the same naming pattern.

## Smoke-test mode

To validate the pipeline without burning GPU-hours:

```bash
python diagnostics/direction_drift.py --model llama7b --quantizer aqlm2 --n_calib 20
python diagnostics/directional_ablation.py --model llama7b --quantizer aqlm2 --n_eval 20 --n_arc 10
```

Every diagnostic accepts `--n_eval`, `--n_arc`, `--n_calib` overrides. Defaults are n_eval=200, n_arc=50 in most scripts (lower than v1's 1000/1000 to keep slow-kernel cells like Qwen-GPTQ-3 tractable).

## Reproducibility

See `SEEDS.md` for full detail. Briefly:
- Calibration prompts are identical run-to-run via `random.Random(42)` shuffle.
- Refusal-direction extraction is deterministic.
- Random unit-vector controls (archived F2a scripts) use an isolated `torch.Generator` seeded explicitly.
- The only unavoidable non-determinism is tiny CUDA atomic-add noise; below +/- 1 prompt on n=200.

## Archived material

Anything not in the active diagnostics list above has been moved to `/jet/home/bhoomire/projects/safequant/_trash/` for retrieval if needed:

| Archive folder | Contents |
|---|---|
| `v2_inactive/diagnostics/` | F3b parametric tuning, srbc (F4 input-dependence), Appendix C failed-static methods, behavioral-outcome probe pilot, probe causal ablation, three-criteria analysis, sigma-vs-probe diagnostics, variance-explained, rhat-propagation, build-routing-table |
| `v2_inactive/scripts/` | sbatches and helper .py for the above; quantize/download utilities for Gemma-7B and Qwen3-8B; disabled `quantize_aqlm_qwen.sbatch` |
| `v2_inactive/outputs/` | JSONs from inactive diagnostics |
| `v2_singletons/` | Earlier dead-end outputs (bootstrap_gap, dsq_prototype, codebook_safety_tune, etc.) |
| `v2_stale_sbatches/` | Superseded sbatches from earlier iterations |
| `v2_logs/` | Old job logs |

To restore any of these:

```bash
mv /jet/home/bhoomire/projects/safequant/_trash/v2_inactive/diagnostics/<name>.py diagnostics/
```
