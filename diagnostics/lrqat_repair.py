"""
diagnostics/lrqat_repair.py
===========================
Reader-side (vs writer-side) low-bit REPAIR via quant-domain LR-QAT.

Kills the "safety came back because you added bits" objection. We train ONLY the
selected projections (readers q/k/v/gate/up, or writers o/down) at the SAME
bit-width as the damaged AQLM-2 baseline, on a NEUTRAL corpus (C4 web text, no
harmful/safety data), distilling toward the FP16 teacher. After training the
adapter is fused back into the integer grid, so the deployed model is genuinely
INT-b -- no added precision.

Grid route: A1 (hybrid). Only the selected projections are moved AQLM-2 ->
uniform INT-b + LR-QAT integer adapter; every other weight stays the original
AQLM-2 checkpoint, byte-for-byte. The 62.5%-ASR AQLM-2 cell is the untouched
baseline. See core/lrqat.py for the reparameterization.

Three conditions come out of ONE run (per --role):
  * "ptq_untrained"  -- readers requantized to INT-b, adapter delta = 0
                        (the control: re-quantization without training).
  * "lrqat_trained"  -- same INT-b grid, adapter trained + fused.
  * baselines (fp16 41.0, aqlm2 62.5) are read from the existing F3a output
    or fall back to the spec constants; they are NOT recomputed here.

Eval is the EXISTING harness, unchanged: WildJailbreak + HarmBench classifier
for ASR, log-prob ARC-Challenge for coherence.

Usage:
    python -u diagnostics/lrqat_repair.py --model llama7b --quantizer aqlm2 \
        --role read --n_bits 2 --steps 3000 --n_eval 200 --n_reasoning 50
"""

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Reproducibility (identical protocol to the rest of SafeQuant v2) ----------
from core.seeds import set_global_seed, DEFAULT_SEED          # noqa: E402
SEED = set_global_seed(DEFAULT_SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from core.model_loader import load_fp32, load_quantized, get_weight   # noqa: E402
from core.datasets import (                                    # noqa: E402
    load_wildjailbreak_prompts, load_c4_calibration, load_arc_challenge,
)
from core.evaluation import (                                  # noqa: E402
    generate_only, classify_batch, _extract_behavior, eval_xstest,
)
from core.datasets import load_xstest                          # noqa: E402
from core.lrqat import (                                       # noqa: E402
    inject_lrqat_adapters, merge_all, injected_effective_bits,
)
# Reuse the EXACT projection lists and ARC scorer from the static-restoration
# experiment so read/write definitions and coherence eval are identical.
from diagnostics.readwrite_restoration import (                # noqa: E402
    READ_PROJS, WRITE_PROJS, measure_arc,
)

# This cell's anchors (Llama-3.1-8B / AQLM-2), from the F3a result.
ASR_FP16_DEFAULT = 41.0
ASR_QUANT_DEFAULT = 62.5

# Documented training corpus -- contains NO harmful or safety examples.
TRAIN_CORPUS = "allenai/c4 (en) -- general web text, no harmful/safety data"


# -----------------------------------------------------------------------------
# Neutral corpus -> fixed-length LM token blocks (no chat template, no safety)
# -----------------------------------------------------------------------------
def build_token_blocks(tokenizer, n_docs, seq_len, device, max_chars=4000):
    """Stream C4, concatenate, and chunk into [seq_len] LM blocks."""
    docs = load_c4_calibration(n_samples=n_docs, max_chars=max_chars)
    ids = []
    for d in docs:
        ids.extend(tokenizer(d, add_special_tokens=False).input_ids)
        ids.append(tokenizer.eos_token_id)
    n_blocks = len(ids) // seq_len
    if n_blocks == 0:
        raise RuntimeError(f"C4 produced only {len(ids)} tokens (< seq_len={seq_len}); "
                           f"raise --n_train_docs.")
    blocks = torch.tensor(ids[: n_blocks * seq_len], dtype=torch.long).view(n_blocks, seq_len)
    print(f"  Built {n_blocks} blocks of {seq_len} tokens "
          f"({n_blocks * seq_len:,} tokens) from {len(docs)} C4 docs")
    return blocks  # CPU tensor; moved per-step


# -----------------------------------------------------------------------------
# Distillation loss: forward KL from FP16 teacher onto the student
# -----------------------------------------------------------------------------
def kl_distill_loss(student_logits, teacher_logits, temperature=1.0):
    """
    KL( p_teacher || p_student ) averaged over tokens.
    F.kl_div expects log-probs for the input (student) and probs for the target
    (teacher). reduction='batchmean' divides by the number of token positions.
    """
    t = temperature
    s_logp = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_logp, t_prob, reduction="batchmean") * (t * t)


@torch.no_grad()
def eval_val_kl(student, teacher, val_blocks, device, max_batches=8):
    student.eval()
    total, n = 0.0, 0
    for i in range(min(max_batches, val_blocks.size(0))):
        x = val_blocks[i:i + 1].to(device)
        s = student(input_ids=x).logits
        t = teacher(input_ids=x).logits
        total += kl_distill_loss(s, t).item()
        n += 1
    student.train()
    return total / max(n, 1)


# -----------------------------------------------------------------------------
# Training loop (8-bit AdamW, grad checkpointing, grad accumulation)
# -----------------------------------------------------------------------------
def train_adapters(student, teacher, trainable, train_blocks, val_blocks, args, device):
    try:
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(trainable, lr=args.lr, weight_decay=0.0)
        opt_name = "bitsandbytes.AdamW8bit"
    except Exception as e:  # pragma: no cover - fallback if bnb unavailable
        print(f"  [warn] bitsandbytes optimizer unavailable ({e}); using torch.AdamW")
        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
        opt_name = "torch.AdamW"
    print(f"  Optimizer: {opt_name} | lr={args.lr} | grad_accum={args.grad_accum}")

    if args.grad_checkpointing:
        # Only the adapters are trainable, so the (frozen) embedding output does
        # not require grad. Make inputs require grad and use the non-reentrant
        # checkpoint so gradients still reach the adapter params -- the standard
        # PEFT recipe for partial fine-tuning under checkpointing.
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    student.train()
    teacher.eval()

    n_blocks = train_blocks.size(0)
    order = np.random.permutation(n_blocks)
    ptr = 0

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    tokens_seen = 0
    t0 = time.time()
    history = []

    optim.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            if ptr >= n_blocks:
                order = np.random.permutation(n_blocks)
                ptr = 0
            x = train_blocks[order[ptr]:order[ptr] + 1].to(device)
            ptr += 1
            tokens_seen += x.numel()

            with torch.no_grad():
                t_logits = teacher(input_ids=x).logits
            s_logits = student(input_ids=x).logits
            loss = kl_distill_loss(s_logits, t_logits, temperature=args.temperature)
            if args.ce_weight > 0:  # optional small next-token CE on the corpus
                ce = F.cross_entropy(s_logits[:, :-1].reshape(-1, s_logits.size(-1)),
                                     x[:, 1:].reshape(-1))
                loss = loss + args.ce_weight * ce
            (loss / args.grad_accum).backward()
            accum_loss += loss.item() / args.grad_accum

        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            print(f"    step {step:>5}/{args.steps} | KL+CE {accum_loss:.4f} "
                  f"| tokens {tokens_seen:,} | {time.time() - t0:.0f}s")

        if step % args.val_every == 0 or step == args.steps:
            val = eval_val_kl(student, teacher, val_blocks, device)
            history.append({"step": step, "train_loss": accum_loss, "val_kl": val})
            print(f"    [val] step {step} | val_KL {val:.4f} (best {best_val:.4f})")
            if val < best_val - 1e-4:
                best_val = val
                best_state = {id(p): p.detach().clone() for p in trainable}
                patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"    early stop at step {step} (no val improvement)")
                    break

    # restore best adapter state
    if best_state is not None:
        with torch.no_grad():
            for p in trainable:
                if id(p) in best_state:
                    p.copy_(best_state[id(p)])
    student.config.use_cache = True
    wall = time.time() - t0
    return {
        "steps_run": step,
        "tokens_seen": tokens_seen,
        "best_val_kl": best_val,
        "wall_clock_s": wall,
        "optimizer": opt_name,
        "history": history,
    }


# -----------------------------------------------------------------------------
def load_baseline_asr(model, quantizer):
    """Read fp16/aqlm baseline ASR from the F3a output; fall back to spec consts."""
    p = Path(f"outputs/readwrite_{model}_{quantizer}.json")
    if p.exists():
        try:
            d = json.loads(p.read_text())
            return float(d["fp32_asr"]), float(d["quantized_asr"]), str(p)
        except Exception:
            pass
    return ASR_FP16_DEFAULT, ASR_QUANT_DEFAULT, "spec-constants"


def recovery_fraction(asr_repaired, asr_quant, asr_fp16):
    denom = (asr_quant - asr_fp16)
    if denom == 0:
        return float("nan")
    return (asr_quant - asr_repaired) / denom


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama7b")
    ap.add_argument("--quantizer", default="aqlm2")
    ap.add_argument("--role", choices=["read", "write"], required=True)
    ap.add_argument("--n_bits", type=int, default=2, help="2 for the strict story, 4 for sweep")
    ap.add_argument("--group_size", type=int, default=128,
                    help="input-dim group width for INT scales (coherence at low bits)")
    ap.add_argument("--aqlm_eff_bits", type=float, default=2.2,
                    help="realistic AQLM-2 effective-bit ceiling for the no-added-precision assert")
    # adapter
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64.0)
    # training budget (KEEP IDENTICAL across read and write)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--n_train_docs", type=int, default=4000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--ce_weight", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--grad_checkpointing", action="store_true", default=True)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--n_val_blocks", type=int, default=16)
    # eval budget (matches F3a)
    ap.add_argument("--n_eval", type=int, default=200, help="WildJailbreak prompts (ASR)")
    ap.add_argument("--n_reasoning", type=int, default=50, help="ARC-Challenge questions")
    ap.add_argument("--n_xstest", type=int, default=50,
                    help="XSTest prompts for the over-refusal check (0 to skip)")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proj_paths = READ_PROJS if args.role == "read" else WRITE_PROJS
    if args.output is None:
        args.output = f"outputs/lrqat_{args.model}_{args.quantizer}_{args.role}_int{args.n_bits}.json"

    print("=" * 74)
    print(f"  LR-QAT low-bit repair  |  role={args.role}  |  INT-{args.n_bits}")
    print(f"  {args.model} / {args.quantizer}  (Route A1 hybrid)")
    print(f"  projections: {proj_paths}")
    print(f"  budget: steps={args.steps} accum={args.grad_accum} seq_len={args.seq_len} "
          f"lr={args.lr} rank={args.rank} alpha={args.alpha} group_size={args.group_size}")
    print(f"  train corpus: {TRAIN_CORPUS}")
    print(f"  eval: n_eval={args.n_eval}  n_reasoning={args.n_reasoning}  "
          f"n_xstest={args.n_xstest}  seed={SEED}")
    print("=" * 74)

    # -- datasets (eval) ------------------------------------------------------
    items = load_wildjailbreak_prompts(n_samples=args.n_eval, config="eval")
    behaviors = [_extract_behavior(it) for it in items]
    arc_items = load_arc_challenge(n_samples=args.n_reasoning)
    # ARC chance level (used to flag coherence collapse): 1/num_choices.
    arc_chance = 100.0 / max(2, len(arc_items[0]["choices"]))
    xstest_items = load_xstest(n_samples=args.n_xstest) if args.n_xstest > 0 else []

    # -- student: AQLM-2 + injected INT-b adapters on the selected role -------
    print("\n[1] Loading AQLM-2 student and injecting adapters ...")
    student, tokenizer = load_quantized(args.model, quantizer=args.quantizer)
    injected, trainable = inject_lrqat_adapters(
        student, proj_paths, n_bits=args.n_bits, rank=args.rank, alpha=args.alpha,
        get_weight_fn=get_weight, group_size=args.group_size)
    eff_bits_repaired = injected_effective_bits(injected)
    print(f"  Repaired-matrix effective bits/weight: {eff_bits_repaired:.4f} "
          f"(INT-{args.n_bits}, group_size={args.group_size})")

    # -- effective-bit-width ASSERTION (core control) -------------------------
    # Group-wise INT-b adds one fp16 scale per group: INT-2 @ group 128 = 2.125
    # bits. The "no added precision" claim is vs the REALISTIC AQLM-2 effective
    # bits (PV-2Bit-1x16 is ~2.0-2.2, NOT exactly 2.0). We assert the repaired
    # readers do not exceed that ceiling. The binding comparison is at n_bits=2;
    # n_bits=4 is the intentionally-higher mixed-precision sweep point.
    aqlm_nominal_bits = 2.0
    bit_assert_ok = None
    if args.n_bits == 2:
        bit_assert_ok = eff_bits_repaired <= args.aqlm_eff_bits + 1e-3
        assert bit_assert_ok, (
            f"Effective-bit control VIOLATED: repaired={eff_bits_repaired:.4f} bits "
            f"> AQLM-2 ceiling {args.aqlm_eff_bits}. The repair must not add precision. "
            f"(Lower --group_size raises overhead; raise --aqlm_eff_bits only with evidence.)")
        print(f"  ASSERT OK: repaired {eff_bits_repaired:.4f} <= AQLM-2 ceiling "
              f"{args.aqlm_eff_bits} bits (no added precision)")
    else:
        print(f"  [sweep] n_bits={args.n_bits}: effective bits {eff_bits_repaired:.4f} "
              f"(mixed-precision point, not bit-matched to baseline)")

    # -- condition A: PTQ untrained (adapter delta = 0) -----------------------
    print("\n[2] Eval condition: ptq_untrained (INT-b readers, NO training) ...")
    resp_ptq = generate_only(student, tokenizer, items, max_new_tokens=args.max_new_tokens,
                             desc="PTQ-untrained")
    arc_ptq = measure_arc(student, tokenizer, arc_items)
    xstest_ptq = (eval_xstest(student, tokenizer, xstest_items, desc="XSTest-ptq")
                  if xstest_items else None)
    print(f"  ARC(ptq_untrained) = {arc_ptq:.1f}%"
          + (f" | false-refusal = {xstest_ptq['false_refusal_rate']:.1f}%"
             if xstest_ptq else ""))

    # -- training -------------------------------------------------------------
    print("\n[3] Building neutral C4 token blocks ...")
    blocks = build_token_blocks(tokenizer, args.n_train_docs, args.seq_len, device)
    val_blocks = blocks[: args.n_val_blocks]
    train_blocks = blocks[args.n_val_blocks:]

    print("\n[4] Loading FP16 teacher and distilling adapters ...")
    teacher, _ = load_fp32(args.model)
    for p in teacher.parameters():
        p.requires_grad_(False)
    train_stats = train_adapters(student, teacher, trainable,
                                 train_blocks, val_blocks, args, device)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    # -- fuse + requantize (deployed model is genuinely INT-b) ----------------
    print("\n[5] Fusing adapters into INT-b grid (requantize) ...")
    merge_all(injected)
    eff_bits_fused = injected_effective_bits(injected)
    if args.n_bits == 2:
        assert eff_bits_fused <= args.aqlm_eff_bits + 1e-3, "post-fuse bit-width drifted"
    print(f"  Post-fuse effective bits/weight: {eff_bits_fused:.4f}")

    # -- condition B: LR-QAT trained -----------------------------------------
    print("\n[6] Eval condition: lrqat_trained (INT-b, trained + fused) ...")
    resp_lrqat = generate_only(student, tokenizer, items, max_new_tokens=args.max_new_tokens,
                               desc="LRQAT-trained")
    arc_lrqat = measure_arc(student, tokenizer, arc_items)
    xstest_lrqat = (eval_xstest(student, tokenizer, xstest_items, desc="XSTest-lrqat")
                    if xstest_items else None)
    print(f"  ARC(lrqat_trained) = {arc_lrqat:.1f}%"
          + (f" | false-refusal = {xstest_lrqat['false_refusal_rate']:.1f}%"
             if xstest_lrqat else ""))
    del student
    gc.collect()
    torch.cuda.empty_cache()

    # -- classify both conditions in ONE classifier session -------------------
    print("\n[7] Classifying responses (single HarmBench load) ...")
    n = args.n_eval
    flat_beh = behaviors + behaviors
    flat_resp = resp_ptq + resp_lrqat
    flags = classify_batch(flat_beh, flat_resp, device=device)
    asr_ptq = 100.0 * sum(flags[:n]) / n
    asr_lrqat = 100.0 * sum(flags[n:2 * n]) / n

    # -- baselines + recovery -------------------------------------------------
    asr_fp16, asr_quant, base_src = load_baseline_asr(args.model, args.quantizer)
    rho_ptq = recovery_fraction(asr_ptq, asr_quant, asr_fp16)
    rho_lrqat = recovery_fraction(asr_lrqat, asr_quant, asr_fp16)

    # -- coherence gate -------------------------------------------------------
    # If ARC is within 5pp of chance the model has collapsed; its ASR reflects
    # incoherence (it produces no usable text), NOT recovered safety, so rho is
    # meaningless. Flag it loudly instead of reporting a misleading rho.
    collapse_ptq = arc_ptq <= arc_chance + 5.0
    collapse_lrqat = arc_lrqat <= arc_chance + 5.0
    fr_ptq = xstest_ptq["false_refusal_rate"] if xstest_ptq else None
    fr_lrqat = xstest_lrqat["false_refusal_rate"] if xstest_lrqat else None

    def _fr(v):
        return f"{v:>6.0f}%" if v is not None else f"{'-':>7}"

    # -- report ---------------------------------------------------------------
    print("\n" + "=" * 86)
    print(f"  RESULT  |  {args.model} / {args.quantizer}  |  role={args.role}  INT-{args.n_bits}")
    print(f"  baselines from: {base_src}  (FP16 {asr_fp16:.1f} | AQLM-2 {asr_quant:.1f})  "
          f"| ARC chance={arc_chance:.0f}%")
    print("-" * 86)
    print(f"  {'Condition':<16}{'EffBits':>9}{'Trained':>9}{'ASR':>8}{'ARC':>7}"
          f"{'FalseRef':>9}{'rho':>8}{'  flag':<10}")
    print(f"  {'fp16_reference':<16}{16.0:>9.2f}{'no':>9}{asr_fp16:>7.1f}%{'-':>7}{'-':>9}{'-':>8}")
    print(f"  {'aqlm2_baseline':<16}{aqlm_nominal_bits:>9.2f}{'no':>9}{asr_quant:>7.1f}%"
          f"{'-':>7}{'-':>9}{'-':>8}")
    print(f"  {'ptq_untrained':<16}{eff_bits_fused:>9.4f}{'no':>9}{asr_ptq:>7.1f}%"
          f"{arc_ptq:>6.0f}%{_fr(fr_ptq):>9}{rho_ptq:>8.2f}"
          f"{'  COLLAPSE' if collapse_ptq else '':<10}")
    print(f"  {'lrqat_trained':<16}{eff_bits_fused:>9.4f}{args.role:>9}{asr_lrqat:>7.1f}%"
          f"{arc_lrqat:>6.0f}%{_fr(fr_lrqat):>9}{rho_lrqat:>8.2f}"
          f"{'  COLLAPSE' if collapse_lrqat else '':<10}")
    print("=" * 86)
    if collapse_lrqat:
        print("  WARNING: lrqat_trained ARC ~ chance -> coherence COLLAPSE. Its low ASR is "
              "incoherence,\n           not safety recovery; rho is not interpretable. "
              "Raise --group_size / --n_bits.")
    elif rho_lrqat > 1.0 + 1e-6:
        print(f"  NOTE: rho={rho_lrqat:.2f} > 1 -> ASR fell BELOW FP16. Check false-refusal "
              f"({fr_lrqat}) for over-refusal vs genuine recovery.")

    out = {
        "model": args.model,
        "quantizer": args.quantizer,
        "route": "A1_hybrid",
        "role": args.role,
        "proj_paths": proj_paths,
        "n_bits": args.n_bits,
        "seed": SEED,
        # bit-width control
        "aqlm_nominal_bits": aqlm_nominal_bits,
        "aqlm_eff_bits_ceiling": args.aqlm_eff_bits,
        "group_size": args.group_size,
        "repaired_effective_bits": eff_bits_fused,
        "bit_width_assert_ok": bit_assert_ok,
        # training provenance / reproducibility
        "train_corpus": TRAIN_CORPUS,
        "contains_safety_data": False,
        "adapter": {"rank": args.rank, "alpha": args.alpha, "scaling": args.alpha / args.rank},
        "budget": {
            "steps_requested": args.steps,
            "steps_run": train_stats["steps_run"],
            "grad_accum": args.grad_accum,
            "seq_len": args.seq_len,
            "tokens_seen": train_stats["tokens_seen"],
            "wall_clock_s": train_stats["wall_clock_s"],
            "lr": args.lr, "optimizer": train_stats["optimizer"],
        },
        "generation_config": {"do_sample": False, "max_new_tokens": args.max_new_tokens,
                              "greedy": True},
        "eval": {"asr_prompts": args.n_eval, "asr_set": "wildjailbreak/eval",
                 "judge": "harmbench", "arc_questions": args.n_reasoning,
                 "arc_chance": arc_chance, "n_xstest": args.n_xstest},
        # results
        "baseline_source": base_src,
        "asr_fp16": asr_fp16, "asr_aqlm2": asr_quant,
        "asr_ptq_untrained": asr_ptq, "asr_lrqat_trained": asr_lrqat,
        "arc_ptq_untrained": arc_ptq, "arc_lrqat_trained": arc_lrqat,
        "rho_ptq_untrained": rho_ptq, "rho_lrqat_trained": rho_lrqat,
        # over-refusal (XSTest false-refusal rate on safe prompts) + coherence gate
        "false_refusal_ptq_untrained": fr_ptq,
        "false_refusal_lrqat_trained": fr_lrqat,
        "xstest_refusal_on_unsafe_lrqat": (xstest_lrqat["refusal_rate_on_unsafe"]
                                           if xstest_lrqat else None),
        "coherence_collapse_ptq_untrained": bool(collapse_ptq),
        "coherence_collapse_lrqat_trained": bool(collapse_lrqat),
        "best_val_kl": train_stats["best_val_kl"],
        "best_val_kl_per_token": train_stats["best_val_kl"] / args.seq_len,
        "train_history": train_stats["history"],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n> Saved {out_path}")


if __name__ == "__main__":
    main()
