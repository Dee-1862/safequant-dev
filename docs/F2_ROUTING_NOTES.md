# F2 (routing) -- methodological notes for Section 3.2

This file collects the framing language, controls, and limitations for the
operative-layer routing analysis. Drop these into the paper as needed.

## What the analysis measures

For each (model, quantizer) condition we extract per-prompt last-token
residual-stream activations from the **quantized** model at every layer L,
then project them onto two unit vectors:

  * r_hat_FP -- the refusal direction recovered from the FP32 model at the
    sigma-best layer L_FP.
  * r_hat_q  -- the refusal direction recovered from the quantized model at the
    sigma-best layer L_q.

For each prompt we report the **operative layer** = argmax_L |projection|.
The table reports the mode operative layer (opFP(peak), opq(peak)) over the
harmful calibration set.

## Acknowledged limitation: sigma-inheritance

The operative-layer analysis takes the sigma-best extraction layer as input. Both
r_hat_FP and r_hat_q are estimated at the sigma-chosen layer; opFP(peak) and opq(peak)
therefore characterize **where the sigma-chosen direction is operatively most
active**, not where refusal lives in some methodology-free sense.

Concretely:

  - The sigma statistic ranks layers by harmful/harmless separation in
    last-token activation space. If sigma peaks at L=8 but peak-projection peaks
    at L=29, the standard pipeline reports L=8 as "the refusal layer" while
    the extracted vector is most operatively active 21 layers later.
  - The Mistral / AQLM-2 row of the table makes this concrete: sigma-best layer
    is L=4, peak-projection layer is L=31 -- a 27-layer discrepancy within
    a single condition.

We report this as a methodological observation: sigma-statistics and
peak-projection magnitudes can diverge under quantization. The interpretation
of operative-layer divergence (across quantizer families) should be read in
that context -- it is a property of the standard pipeline's outputs, not
an unconditional claim about refusal localization.

## Random-direction control

To defend the operative-layer signal against the structural-magnitude
objection ("late layers have larger activations, so any direction peaks
late"), we run a control: for each condition we sample K=20 random unit
vectors from the residual hidden-dim and run the same operative-layer
analysis on each.

  - If random directions peak at the same late layer (low std across K
    trials, mode-of-modes close to opFP/opq), then late-layer peak is
    structural -- operative-layer divergence loses interpretive weight for
    that condition.
  - If random directions are spread across layers (high std, many distinct
    mode layers), then a clustered r_hat peak is direction-specific.

The random baseline appears in the routing table as the `rand_peak
(mean+/-std)` and `rand_distinct` columns.

Implementation: `diagnostics/routing_random_control.py`,
sbatch scripts `scripts/routing_random_control_{llama,mistral,qwen}.sbatch`,
output `outputs/routing_random_control_<model>_<quant>.json`.

Reproducibility: seeds are fixed (`SEED=42` for prompt sampling,
`rand_seed=12345` for random-direction sampling -- separate so the control
seed can vary without perturbing the rest of the pipeline).

## Completed coverage matrix

After matrix completion, the F2 table covers:

  * Llama-3.1-8B   x {GPTQ-4, AWQ-4, GPTQ-3, GPTQ-2, AQLM-2}   -- 5
  * Mistral-7B-v02 x {GPTQ-4, AWQ-4, GPTQ-3, GPTQ-2, AQLM-2}   -- 5
  * Qwen2.5-7B     x {GPTQ-4, AWQ-4, GPTQ-3, GPTQ-2}           -- 4

Total: 14 conditions, forecloses the "cherry-picked working cases"
objection. The 6 conditions added relative to the v1 table (Llama-GPTQ-3/2,
Mistral-GPTQ-3/2, Qwen-GPTQ-3/2) are mostly destroyed-regime per the F1
phase-transition figure; we report them for completeness, not as
quantitatively load-bearing.

## How to regenerate the table

```bash
# After routing_random_control_*.sbatch jobs complete:
cd /jet/home/bhoomire/projects/safequant_v2
python diagnostics/build_routing_table.py \
    --write_to docs/F2_routing_table.md
```

The script reads `outputs/layer_routing_*.json` and (if present)
`outputs/routing_random_control_*.json`. The random-baseline columns show
`--` for any condition whose random control has not yet been run, so the
table can be regenerated incrementally.
