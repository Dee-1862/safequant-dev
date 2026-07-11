# Phase 1 — post-submission consolidation

Post-submission refactor plan. Do NOT start before the paper is submitted; the regression
harness below is the gate for every step.

## Target layout
```
safequant_v2/
  core/
    seeds.py  model_loader.py  datasets.py  evaluation.py  lrqat.py   (existing)
    hooks.py        residual-stream hooks as context managers      [Phase 0: seeded, see below]
    residuals.py    activation extraction + token-position policy + chat formatting
    directions.py   diff-in-means, r_hat, separation, cosine drift
    probes.py       absorbs probe_utils + the two fit_probe_* variants
    restoration.py  proj sets per architecture, weight extract/inject, AWQ transform, measure_arc
    stats.py        mcnemar_exact, wilson_ci, bootstrap_recovery
    manifest.py
  experiments/      one runnable script per paper artifact, thin, importing only core
  tests/            existing self-test gates, runnable via pytest
  attic/
```

## Duplication to kill (each a divergence risk)
- `_get_inner` / `_get_layers` / `_get_inner_model` — 4 variants; GPTQ-wrapper vs plain unwrap
  differs. -> one impl in `core/residuals.py`.
- `extract_last_token_residuals` — in `layer_routing.py` AND `direction_drift.py`; token-position
  differs (see landmine). -> `core/residuals.py`.
- chat-template formatting (`format_prompt` / `_format_prompt` / `chat_text`) — 3+, field-priority
  differs. -> `core/residuals.py`.
- ablation/addition hook install — 4 copies (directional_ablation, probe_causal_ablation, both
  arditi scripts); dtype + layer-targeting differ. -> `core/hooks.py` (Phase 0 seeds this).
- `REFUSAL_SUBSTRINGS` — 2 copies, identical today. -> one home.
- seed try/except boilerplate + `sys.path.insert` — every file; pure noise.
- flatten-then-`classify_batch`-once — 5+ scripts; -> `evaluation.py:classify_conditions(behaviors, responses_by_cond)`.
- `compute_r_hat`, `per_layer_separation` — 2 identical copies. -> `core/directions.py`.

## TOKEN-POSITION LANDMINE (flag before consolidating)
`direction_drift.py` extracts at **token_pos = -5** (Llama); `probe_geometry.py` and
`probe_refit_control.py` extract at the default **-1**. Table 1's `cosmed` and its probe-rotation
column were therefore measured at DIFFERENT token positions. A single `token_pos(model)` policy
will silently change one published number on rerun. Requirements:
- the consolidated extractor takes an EXPLICIT `token_pos` arg (policy = documented default);
- `RESULTS.md` records which token position each published number used. Do not let cleanup
  rewrite history.

## Classes that earn their keep (functions + dataclasses otherwise)
- `HookContext` (`core/hooks.py`) — context manager, `with ablate(model, dir, layers):` guarantees
  handle removal. Leaked hooks silently corrupt every later condition. **Highest value.**
- `Restorer` (`core/restoration.py`) — holds FP32 weight cache, per-arch proj sets, AWQ transform;
  `restore(model, side)`, `restore_range(...)`, `restore_projection(...)`. Puts Phi-3 fused-layout
  handling in ONE place.
- `EvalSuite` dataclass — ASR-generate / ARC / XSTest triple per condition (headline_eval_full,
  lrqat_repair, qresafe_dpo each reimplement this).
- `RunConfig` dataclass — replaces argparse, serialized into the manifest (config == provenance).

## Regression harness (gate for the whole refactor)
Before merging Phase 1, one script reruns two cheap anchors and asserts equality with the frozen
JSONs within float tolerance:
- per-layer probe accuracy at `n_calib=50`;
- `cosmed` for the headline cell.
If anchors match, consolidation did not perturb the science. Move the existing self-test gates
(qresafe_repair, qresafe_dpo, qresafe_fold_writeback round-trip) under `tests/` so `pytest tests/`
runs everything with no GPU except the anchors.

## Release hygiene
- Pin the environment in `requirements.txt` + README note — INCLUDING the transformers-vs-AQLM
  version conflict already hit (a reviewer trying the artifact hits this in the first 10 min).
- README top section = the `RESULTS.md` provenance table. For a paper repo, that table is the
  product; the code is supporting material.

## Orphaned sbatch launchers -> attic (this pass left them; move them here in Phase 1)
Their target `.py` is already in `attic/`. Move each to `attic/scripts/` after confirming it calls
only an archived module (the `arditi_official_*` ones may call the KEEPER `arditi_official_repro.py`
— verify per file, do NOT archive those):
- routing: `routing_random_control_{llama,mistral,qwen}.sbatch`, `eq7_{llama,phi3}.sbatch`,
  `mignull_*.sbatch`, `phi3_routing_control.sbatch`, `phi3_routing_ctrl_tp3.sbatch`,
  `{llama,phi3}_ctrl_norm.sbatch`
- directional ablation: `f2_directional_ablation_{llama,mistral,qwen}_{fast,slow}.sbatch`
- random ablation: `random_ablation_dist_*.sbatch`
- snip: `snip_llama*.sbatch`, `snip_wj.sbatch`
- refold: `refold_gate.sbatch`
- arditi (VERIFY FIRST): `arditi_llama_*.sbatch`, `arditi_official_*.sbatch` — keep any that call
  `arditi_official_repro.py`.

## Next action (now, not Phase 1)
1. Phase 0 dependency fix — hooks in `core/hooks.py`, `probe_causal_ablation` import updated. [DONE]
2. Pull the activation-swap JSON; that decision gates the paper text and whether the swap script
   joins the keep list.
