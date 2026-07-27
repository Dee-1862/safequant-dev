# SafeQuant v2 -- Reproduction Guide

This file maps each numerical claim in the paper to the exact command that produces it. All commands assume:

- Working directory: `/jet/home/bhoomire/projects/safequant_v2/`
- Conda env activated: `/ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean`
- HF cache pointed at Ocean: `export HF_HOME=/ocean/projects/cis240116p/bhoomire/.cache/huggingface`
- SLURM account: `cis250287p`
- Seed: 42 (default everywhere; never override unless estimating variance)
- ASR n: 1000 (was 100 in v1)
- ARC n: 1000 (was 50 in v1)

## One-sbatch-per-finding map

The six paper findings/sections each correspond to exactly one sbatch file in `scripts/`. Each script loops through every model/quantizer/condition variant in its scope.

| Section / Finding | sbatch | Diagnostic(s) inside | Wall-time |
|---|---|---|---|
| Finding 1 (Figure 1) | `finding1_phase_transition.sbatch` | `direction_drift.py` x {AQLM-2, GPTQ-4, GPTQ-3, GPTQ-2} | ~2h |
| Finding 2 (Table 1) | `finding2_directional_ablation.sbatch` | `directional_ablation.py` x 4 quantizers x 4 conditions | ~18h |
| Finding 3a (Fig 3 panel A) | `finding3a_static_restoration.sbatch` | `readwrite_restoration.py` x {Llama, Mistral} x 5 conditions | ~4h |
| Finding 3b (Fig 3 panel B + ?8) | `finding3b_parametric_tuning.sbatch` | `codebook_safety_tune.py` x 4 conditions + `gptq_scale_safety_tune.py` x 2 conditions | ~5h |
| Finding 4 (?7) | `finding4_input_dependence.sbatch` | `srbc.py` (CV measurement + 3 SRBC variants) | ~1.5h |
| Appendix C + ?8 Pred-1 | `appendixC_failed_static.sbatch` | 4 failed-static methods | ~3.5h |

The ?8 Prediction-3 corroboration result (single-direction alpha=1 ablation) is a column of the Finding 2 output -- no separate sbatch needed; see "?8 Prediction-3 corroboration" below.

## Finding 1 -- Cos-sim phase transition

**Claim:** GPTQ-4 cos-sim ~ 0.97, AQLM-2 ~ 0.82, GPTQ-3 ~ 0.02, GPTQ-2 ~ 0.03 on Llama-3.1-8B.

```bash
sbatch scripts/finding1_phase_transition.sbatch
```

This one job loops through all 4 quantizers internally. Output: `outputs/direction_drift_llama7b_{aqlm2,gptq4,gptq3,gptq2}.json` with `cos_sim_per_layer` field per quantizer.

## Finding 2 -- Direction dissociation

**Claim:** cross-r_hat ablation drops ASR by ~9pp on AQLM-2 / ~15pp on GPTQ-4; self-r_hat drops by ~1pp / ~6pp; random by <=5pp. At GPTQ-3/GPTQ-2 neither cross nor self moves ASR more than random (mechanism broken).

```bash
sbatch scripts/finding2_directional_ablation.sbatch
```

One job loops 4 quantizers x 4 conditions. AQLM-2 and GPTQ-4 use n_eval=1000 for precise asymmetry magnitudes; GPTQ-3 and GPTQ-2 use n_eval=200 (CI +/-7pp, sufficient for the "neither moves > random" claim and avoiding the 50 sec/prompt triton kernel wall).

Output: `outputs/directional_ablation_llama7b_{aqlm2,gptq2,gptq3,gptq4}.json` with `results.{baseline,self_ablation,cross_ablation,random_control}.asr` fields. The `decision` field summarizes the regime.

## Finding 3a -- Static read/write asymmetry

**Claim:** static read-side restoration captures ~90% of damage on Llama AQLM-2; write-side captures ~67%. Pattern replicates on Mistral.

```bash
sbatch scripts/finding3a_static_restoration.sbatch
```

One job runs `readwrite_restoration.py` on Llama-AQLM-2 then Mistral-AQLM-2. Output: `outputs/readwrite_{llama7b,mistral7b-v02}_aqlm2.json`. The summary table at the end of each run gives FP32, AQLM-2 baseline, write_restored, read_restored, both_restored ASRs.

## Finding 3b -- Parametric write >> read (asymmetry inversion)

**Claim:** Llama AQLM-2 full-codebook tuning gives ASR 63->23 (-40pp); write-only (29% of modules) gives 63->31 (-32pp); read-only (71% of modules) gives 63->52 (-11pp). Asymmetry direction is *opposite* of static restoration. GPTQ-4 cross-quantizer replicates the pattern. Mistral validates faithful distillation.

```bash
sbatch scripts/finding3b_parametric_tuning.sbatch
```

One job runs 6 conditions sequentially:
1. Llama AQLM-2, `--side all` -> `outputs/codebook_safety_tune_llama7b_aqlm2.json`
2. Llama AQLM-2, `--side read` -> `outputs/codebook_safety_tune_read_llama7b_aqlm2.json`
3. Llama AQLM-2, `--side write` -> `outputs/codebook_safety_tune_write_llama7b_aqlm2.json`
4. Llama GPTQ-4, `--side all` -> `outputs/gptq_scale_safety_tune_llama7b_gptq4.json`
5. Llama GPTQ-4, `--side read` -> `outputs/gptq_scale_safety_tune_read_llama7b_gptq4.json`
6. Mistral AQLM-2, `--side all` -> `outputs/codebook_safety_tune_mistral7b-v02_aqlm2.json`

Each JSON has `baseline`, `tuned`, `delta_asr_pp`, `delta_arc_pp`, `n_quantized_linear_modules_patched` (or `n_quant_modules_patched`), `n_trainable_params`, `side`.

## Finding 4 -- Input dependence (CV + 3 SRBC variants)

**Claim:** median per-prompt CV of safety-scalar attenuation ~ 2.03 on Llama-AQLM-2. None of the three SRBC variants recover safety: they either collapse ARC alongside ASR or have no effect at all.

```bash
sbatch scripts/finding4_input_dependence.sbatch
```

Output: `outputs/srbc_llama7b_aqlm2.json` with `median_delta_cv` (the 2.03 number) plus `results.{baseline,srbc_full,srbc_harmful,srbc_projected}.{asr,arc}`.

## Appendix C -- Failed static-correction methods (?8 Prediction-1)

**Claim:** Four other static methods (SVD low-rank, sparse top-k, k-means CRC, selection-by-participation) also fail to recover safety on Llama-AQLM-2. Together with the SRBC variants in Finding 4, these form the "5 failed static methods" evidence cited in ?8 for Prediction-1.

```bash
sbatch scripts/appendixC_failed_static.sbatch
```

One job runs all 4 methods sequentially. Outputs: `outputs/{svd_diagnostic,super_weight_discovery,crc_validation,dsq_prototype}_*.json`.

## ?8 Prediction-3 corroboration (single-direction alpha=1 ablation)

**Claim:** single-direction Arditi-style ablation at alpha=1 with FP16-extracted r_hat gives ASR -5 to -9pp / ARC stable on Llama AQLM-2. This corroborates Prediction-3 (the operative direction has a natural scale; activation interventions are bounded by it).

**No separate script needed.** This result is the `cross_ablation` condition in Finding 2's `directional_ablation.py`. To extract it:

```bash
# Already produced by the Finding 2 runs above. Pull from JSON:
python -c "
import json
for q in ['aqlm2', 'gptq4']:
    d = json.load(open(f'outputs/directional_ablation_llama7b_{q}.json'))
    base = d['results']['baseline']['asr']
    cross = d['results']['cross_ablation']['asr']
    print(f'{q}: baseline={base}%, cross_alpha=1={cross}%, DeltaASR={cross-base:+.1f}pp')
"
```

We deliberately do not run an alpha-sweep or per-layer variants here. The v1 explorations of those variants (per-layer fixed, per-layer gated, single-direction with alpha?1, negative alpha) all either matched the alpha=1 result or destabilized the model. ?8 of the paper documents this as a methodological constraint in one sentence; the supporting evidence is the cross_ablation row of Finding 2's table.

## Producing a paper table from the JSON outputs

After all the above sbatch jobs complete, each result table in the paper can be regenerated:

```bash
python -m scripts.build_tables                          # walks outputs/ and emits LaTeX rows
```

(If `scripts/build_tables.py` doesn't exist yet, add it -- straightforward parse of the JSON schemas described in `SEEDS.md`.)

## Verification checklist

When porting these results to another machine:

1. `python -c "from core.seeds import set_global_seed, fixed_rng; set_global_seed(); print(fixed_rng().integers(0,100,5).tolist())"` should print `[8, 77, 65, 43, 43]`.
2. `python -c "from core.datasets import load_wildjailbreak_prompts; print(len(load_wildjailbreak_prompts(n_samples=1000, config='eval')))"` should print `1000`.
3. Tokenizer chat template hash matches: `python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct'); print(hash(t.chat_template))"` -- should be stable across machines for the same HF model revision.
4. HarmBench judge produces identical verdicts on a fixed (prompt, response) pair across runs (it's greedy).

Any discrepancy in 1-4 means the run will not reproduce paper numbers.

## Compute budget

For a complete reproduction of all 4 findings at n_eval=1000, n_arc=1000:

| Finding | Wall time | GPU-hours |
|---|---|---|
| 1 (direction_drift x 4 quantizers) | 2 hr | 2 |
| 2 (directional_ablation x 4 quantizers, GPTQ-3 is slow) | 12 hr | 12 |
| 3a (readwrite_restoration x 2 archs) | 4 hr | 4 |
| 3b (codebook_safety_tune x 3 sides + GPTQ-4 x 2 sides + Mistral) | 5 hr | 5 |
| 4 (srbc) | 1.5 hr | 1.5 |
| Appendix C (4 failed-static methods) | 3.5 hr | 3.5 |
| **Total** | ~28 hr | ~28 H100-hours |

Wall time on Bridges-2 with parallel SLURM dispatch: ~5-7 hours real time.

?8 Prediction-3 corroboration (single-direction alpha=1 ablation) requires no additional compute -- it's a column of the Finding 2 output.
