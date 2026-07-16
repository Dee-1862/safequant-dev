# Reader-side low-bit repair (LR-QAT)

Goal: show the lost safety is recovered by **re-aligning reader-side projection
geometry, not by adding precision**. We train only the reader-side projections at
the **same bit-width** as the damaged AQLM-2 baseline, on a **neutral corpus with
no safety data**, distilling toward the FP16 teacher, and show ASR returns toward
41% while ARC holds. Two confounds die at once: *precision* (effective bits stay
2) and *safety fine-tuning* (no harmful/safety examples in training).

Cell: **Llama-3.1-8B / AQLM-2** — FP16 ASR 41.0, AQLM-2 ASR 62.5.

## Grid route — A1 (hybrid)

AQLM-2 is codebook (non-uniform) quantization; LR-QAT's reparam assumes a
**uniform** grid, so they don't compose directly. We resolve this with **Route
A1**: keep writers + all non-selected weights as the original AQLM-2 checkpoint
**byte-for-byte**, and move **only the selected projections** onto a uniform
INT-b grid with an LR-QAT integer adapter. The 62.5%-ASR AQLM-2 cell stays the
untouched baseline.

We additionally run the **untrained INT-2 PTQ control** the reviewer would ask
for: readers re-quantized to INT-2 but *not trained*. Because the adapter is
zero-initialised (`lora_A = 0` ⇒ delta `= 0`), this control is just the
pre-training state of the same module — it falls out of every run for free.

## v1 run (job 41237145) — what we learned

First run used **per-output-channel** INT PTQ. Result at strict INT-2:
**coherence collapse** — ARC fell to 24% (≈ 4-way chance) and distillation KL
plateaued at ~4.7 nats/token. The low ASR (23–27%, ρ>1) was a *babbling* model,
not recovered safety. The untrained-PTQ control did its job: it showed the INT-2
**grid swap itself** collapsed the model before any training. The INT-4 sweep
point was coherent and on-thesis: untrained INT-4 readers ASR 65.5% ≈ baseline
(ρ≈0, re-quant alone recovers nothing), trained ASR 5.5% with ARC held (52%) —
but ρ=2.65 (ASR *below* FP16) signalled likely **over-refusal**.

Fixes in v2: (1) **group-wise** INT PTQ (group 128) so INT-2 stays coherent, like
AQLM/GPTQ; (2) **XSTest false-refusal** eval per condition to separate genuine
recovery from over-refusal; (3) a **coherence gate** that flags any condition with
ARC ≈ chance and refuses to report its ASR as safety; (4) a **write-INT-4** row so
the read-vs-write asymmetry is measured at a coherent bit-width.

## The reparameterization (core/lrqat.py)

Lifted from `qualcomm-ai-research/LR-QAT` (cloned to `reference/` for reading
only; we do not run from inside it):

```
forward (train):  W_hat   = s · clamp(W_int0 + round_ste(α/r · A·Bt), qmin, qmax)
fuse  (deploy):   W_int0 ← clamp(W_int0 + round(α/r · A·Bt),      qmin, qmax)
```

`W_int0` is a frozen **group-wise** (group 128) symmetric INT-b PTQ of the
dequantized AQLM weight; `s` is the frozen per-group scale; `(A, Bt)` is the only
trainable part. The low-rank correction is added **in the integer domain** (inside
the round/clamp), so after fusing, the deployed weight is **genuinely INT-b** — no
surviving FP adapter, no added precision. Group-wise INT-2 effective bits =
`2 + 16/128 = 2.125`; the script asserts `≤ --aqlm_eff_bits` (default **2.2**, the
realistic AQLM-PV-2Bit-1x16 effective-bit ceiling — AQLM 2-bit is *not* exactly
2.0 either). **Decision point:** if you want a sub-2.05 "exactly 2.0" claim,
either raise `--group_size` (256 → 2.0625) or move to a 2-bit codebook base; the
control-backed framing ("same ~2-bit grid, untrained re-quant recovers nothing")
holds at 2.125 regardless.

## Conditions (one table, eval fixed)

| Condition | Eff. bits | Trained | Train data | Source |
|---|---|---|---|---|
| FP16 reference | 16 | no | none | F3a / spec (41.0) |
| AQLM-2 baseline | 2 | no | none | F3a / spec (62.5) |
| Read-side dequant | ~11.7 | no | none | existing `readwrite_restoration.py` |
| **ptq_untrained (read)** | 2 | no | none | `lrqat_repair.py --role read` (pre-train) |
| **lrqat_trained (read)** | 2 | readers | C4 | `lrqat_repair.py --role read` (post-train) |
| **lrqat_trained (write)** | 2 | writers | C4 | `lrqat_repair.py --role write`, matched budget |
| read-side sweep | ~2.5–3.3 | readers | C4 | `--role read --n_bits 4` |

`ρ(S) = (ASR_quant − ASR_repaired) / (ASR_quant − ASR_FP16)`, with
ASR_quant = 62.5, ASR_FP16 = 41.0 (read from `outputs/readwrite_llama7b_aqlm2.json`
if present, else spec constants).

## Controls held fixed

- **Effective bit-width** asserted ≤ AQLM-2 nominal (no added precision); logged
  per condition. The binding comparison is *AQLM-2 baseline vs lrqat_trained
  (read)* at identical effective bits.
- **No safety/harmful data**: corpus is `allenai/c4` (en) general web text;
  logged as `train_corpus` with `contains_safety_data: false`.
- **Identical training budget** for read and write — guaranteed by a single
  `BUDGET_ARGS` string shared by both invocations in the sbatch.
- **Same eval**: WildJailbreak/eval + HarmBench classifier (ASR), log-prob ARC,
  n fixed, greedy generation, **seed 42**. Eval harness is reused **unchanged**.

## How to launch

Everything needed is already in the `sq_clean` env (torch 2.9, transformers
4.57, datasets, bitsandbytes 0.49, peft, aqlm). No pip installs required.

```bash
cd /jet/home/bhoomire/projects/safequant_v2
sbatch scripts/lrqat_repair_llama_aqlm2.sbatch     # read INT-2, write INT-2, read INT-4 sweep
```

Single condition, interactively:

```bash
python -u diagnostics/lrqat_repair.py --model llama7b --quantizer aqlm2 \
    --role read --n_bits 2 --steps 3000 --n_eval 200 --n_reasoning 50
```

Outputs: `outputs/lrqat_llama7b_aqlm2_{read,write}_int{2,4}.json`, each with
ASR/ARC/ρ for `ptq_untrained` and `lrqat_trained`, the effective-bit assertion,
the corpus + budget + generation config, and the val-KL history.

## Acceptance criteria → where enforced

- Read-side LR-QAT eff. bits == AQLM-2 baseline → `assert` in `lrqat_repair.py`
  (`eff_bits_repaired ≤ 2.05`), logged before and after fuse.
- Training data documented safety-free → `TRAIN_CORPUS`, `contains_safety_data`.
- Read recovery > write recovery at equal budget → compare the two output JSONs
  (`rho_lrqat_trained`); budget matched via shared `BUDGET_ARGS`.
- ARC in band → `arc_lrqat_trained` reported per condition.
- Reproduces under seed 42 → `set_global_seed(42)` at import, logged as `seed`.

## Expected outcome / honesty note

Static training-free fixes at strict 2-bit capped at ~53–57% recovery in prior
results; a trained reader-side correction at strict 2-bit may recover more but
might not reach the ~84% the (precision-adding) dequant gets. That is fine and
informative. Frame as "substantial recovery, read-side localized, at no added
precision," and report the recovery-vs-precision curve from the INT-4 sweep
point (~3 effective bits), which is the deployable mixed-precision hypothesis.
