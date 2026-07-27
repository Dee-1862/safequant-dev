# Paper fidelity & disclosure matrix

Validation of our implementations against the original papers' official code, produced by
cross-reading each of our files against the corresponding reference repo. Reference repos are
cloned (shallow) under `paper_refs/` (git-ignored) for side-by-side comparison:

| Method | Our code | Reference repo (`paper_refs/`) | Paper |
|---|---|---|---|
| Refusal direction | `experiments/behavioral_direction/arditi_repro.py`, `core/directions.py`, `core/hooks.py` | `refusal_direction/` | Arditi et al., *Refusal in LLMs is mediated by a single direction* (NeurIPS 2024, arXiv 2406.11717) |
| Q-Resafe repair | `experiments/trained_repair/qresafe_{repair,dpo,pairs,fold}.py` | `Qresafe/` | Q-resafe (ICML 2025, arXiv 2506.20251) |
| LR-QAT repair | `core/lrqat.py`, `experiments/trained_repair/lrqat.py` | `lr-qat/` | Low-Rank QAT for LLMs (Qualcomm AI Research, arXiv 2406.06385) |

**Overall pattern (all three):** the mathematical scaffolding is a faithful port; the deviations
live in setup / objective / selection internals; and in each case at least one thing we attribute
to the paper is actually **ours** or **absent from the released code**.

---

## Per-method verdict

| Method | Faithfully reused (High conf.) | Where we diverge | Load-bearing correction |
|---|---|---|---|
| **Arditi** | diff-in-means formula, pos×layer grid search, KL≤0.1 + induce>0 constraints, prune-20%, N=128/32, ablation formula `h−(h·r̂)r̂`, **logit refusal-score selection** (`core.evaluation.refusal_score_logit`) | ablation only at **resid_post** not resid_pre+attn+mlp; direction extracted at resid_post (off-by-one layer); no train/val refusal-filtering; eval on **WildJailbreak+HarmBench** not JBB+LlamaGuard | **"Chhabra App. B" is misattributed**  -  those constants are Arditi's own defaults; residual hook sites still diverge from full Arditi recipe |
| **Q-Resafe** | DPO sigmoid loss, β=0.01, r=128, adapter-disabled reference model, all-7-projection target set (= our `full`) | lr **5e-6 vs 5e-7 (10×)**, α **256 vs 128 (2×)**, epochs **3 vs 1**; bespoke WildJailbreak displacement pairs vs their ultrafeedback+swap; AQLM-2 vs their AWQ-4/NF4 | **their released code has no SNIP and no fold**  -  SNIP is paper-only; the code-fold is **ours** |
| **LR-QAT** | **bit-exact** integer reparam, STE round, integer clamp, fuse-to-INT-b, α/r scaling, zero-A/kaiming-B init, group-128 (faithful port of `lr_qat_int`) | we implement `lr_qat_int` (an **ablation**), not the headline `lr_qat_fixed_point8` φ-variant; **scale frozen** (they learn ranges); **KD-on-C4** objective vs their next-token CE; hybrid AQLM base (mixed-format) | call it **"LR-QAT (`lr_qat_int` variant), fixed-scale, no φ-downcasting"**; the quant-domain-adapter claim **is verified and holds** |

---

## Action list

### I. Fix / decide in CODE
1. **Q-Resafe hyperparameter drift** (concrete): reconcile `lr=5e-6→5e-7`, `alpha=256→128`,
   `epochs=3→1` in `qresafe_dpo.py`  -  or add a one-line justification in the docstring.
   Directly affects the fold-survival result; highest-priority code item.
2. *(Decision, larger)* Arditi: selection now uses the logit refusal-score
   (`refusal_score_logit`). Remaining fidelity gap is ablation sites
   (resid_post only vs resid_pre+attn+mlp). Disclose residual-site divergence;
   do not re-engineer unless claiming full-site reproduction.

### II. Correct CITATION / WORDING (cheap, low-risk)
3. Arditi: replace **"Chhabra App. B"** → Arditi et al. defaults, in code comments and manuscript
   (unless the actual Chhabra paper can be produced  -  **still unverified**).
4. Arditi: **"exact reproduction"** → "adaptation of / reimplementation following Arditi et al."
5. Q-Resafe: reframe **"override Q-Resafe's SNIP"** → "Algorithm 1 *as described in the paper*
   prescribes saliency masking; the released code omits it and trains all projections  -  we treat
   the mask as a controlled variable (read/write/full)."
6. Q-Resafe: present the **code-fold** as *our* contribution, not Q-Resafe's.
7. LR-QAT: qualify as the **`lr_qat_int` variant without φ-downcasting, fixed ranges**.

### III. DISCLOSE in manuscript (per-method deviation notes)
- **Arditi:** eval swap (WJ+HarmBench vs JBB+LlamaGuard), no data filtering, resid_post-only
  ablation, K=5 positions.
- **Q-Resafe:** bespoke displacement data vs ultrafeedback+swap, AQLM-2 vs AWQ-4, fold-back is
  novel, "Q-Resafe-**based**."
- **LR-QAT:** hybrid AQLM mixed-format base, frozen RTN ranges, KD-on-C4 objective,
  `lr_qat_int` ≠ headline φ-variant.

### Open item
- **The "Chhabra" reference** is the one unresolved external fact: either it's a real paper being
  cited (produce it) or a mislabel (correct it). Not independently verifiable from the code.

---

## Side-by-side file map (for detailed comparison)

### Arditi  -  refusal direction
| Ours | Reference |
|---|---|
| `experiments/behavioral_direction/arditi_repro.py` (selection ~L173-199, scoring L62-66) | `paper_refs/refusal_direction/pipeline/submodules/select_direction.py` (L17-49 logit score, L106-191) |
| `core/hooks.py` (ablation L25-54) | `paper_refs/refusal_direction/pipeline/utils/hook_utils.py` (L50-88) |
| `core/directions.py` (diff-in-means) | `paper_refs/refusal_direction/pipeline/submodules/generate_directions.py` (L42-54) |

### Q-Resafe
| Ours | Reference |
|---|---|
| `experiments/trained_repair/qresafe_dpo.py` (loss L66-69, hyperparams L150-155) | `paper_refs/Qresafe/quant-with-ft/quant.py` (L187-208) + `configs/llama7b.yaml` (L11-59) |
| `experiments/trained_repair/qresafe_pairs.py` (pair build L115-119) | `paper_refs/Qresafe/quant-with-ft/alignment/data.py` (L70-73, chosen/rejected swap) |
| `qresafe_fold.py` (our fold  -  **no counterpart**, it is our extension) | *(none  -  Q-Resafe keeps adapters separate)* |

### LR-QAT
| Ours | Reference |
|---|---|
| `core/lrqat.py` (reparam L166-206) | `paper_refs/lr-qat/quantization/hijacker.py` (L101-126 `lr_qat_int`; L128-180 the φ-variant we do NOT use) |
| `experiments/trained_repair/lrqat.py` (KD loss L106-115) | `paper_refs/lr-qat/clm_main.py` (L553-577 CE objective) |

---

## Notes on this validation
- Line numbers are as of the reference-repo HEADs recorded on clone:
  `refusal_direction 9d852fa`, `Qresafe fc46653`, `lr-qat 8795afe`.
- `experiments/trained_repair/lrqat.py` is the live LR-QAT driver; the now-deleted
  `diagnostics/lrqat_repair.py` was an identical-logic mirror, so its findings carry over.
- Produced 2026-07 by cross-reading both codebases; treat confidence as noted inline above
  (kernel-level claims are High; training-dynamics impact is Medium).

---

## Anomaly audit (2026-07)  -  status

Two-cell context: the core localization result **replicates on Phi-3** as well as Llama
(`outputs/headline/headline_eval_phi3_aqlm2.json`: ASR 41.5→65.0 under quant; read recovery
78.7% vs write 47.9%; read−write 95% CI [9.3, 55.2] excludes 0; McNemar read-vs-write
p=0.0076). So the fixes below are hygiene on a sound two-cell result, not load-bearing.

Core method math verified CLEAN in both (DPO loss/polarity, LoRA, fold, read/write split;
LR-QAT reparam, STE, merge, zero-init). No Critical/High math anomaly.

### Applied (safe  -  no paper number moves)
- **L1-Q** `attach_masked_lora` default `alpha` 256.0 → 128.0 (was a silent 2× trap).
- **LR-QAT stale paths**  -  `diagnostics/lrqat_repair.py` header/usage → `experiments/trained_repair/lrqat.py`.
- **M1-L comment only**  -  KD-loss docstring now states the true behavior (batch=1 → loss is
  SUMMED over positions). The `F.kl_div(reduction="batchmean")` call itself is UNCHANGED.

### STAGED  -  run-and-compare before applying (moves a paper number; never silent)
- **M1-L normalization**  -  switching KD loss to a per-token mean changes grad-clip binding and
  early-stop, so it alters LR-QAT training dynamics → potentially different **Table 7** ASR.
  Decision: run with per-token mean, compare to current Table 7, report corrected version + note.
- **M2-Q recipe**  -  official TRL config also uses cosine LR + 0.1 warmup, effective batch 16
  (8×2 accum), LoRA dropout 0.05; our loop uses constant-LR AdamW, batch 1, dropout 0. Matching
  these moves **Table 6** → run-and-compare, same discipline as the lr/alpha/epochs reconciliation.

### VERIFY FIRST (do not repoint blindly)
- **H1 pairs handshake**  -  producer writes `..._eval`, consumer seeks `..._wj`, on-disk canonical
  is `qresafe_pairs_llama_wj.json` (231 pairs). No `qresafe_dpo` output exists in `outputs/` to
  confirm which pairs file Table 6 trained on. Confirm the canonical pairs file against the paper
  BEFORE aligning the default, or a rerun silently trains on different data.

### DEFERRED
- **M1-Q** hardcoded `AQLM_REPO = /jet/home/bhoomire/...`  -  works on Bridges-2 (where the fold
  runs). Wire to env/config when setting up the fold elsewhere; not biting now.
