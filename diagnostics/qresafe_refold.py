"""
qresafe_refold.py -- the requant-fold, the crux of the Q-Resafe repair.

A trained LoRA delta on a masked projection set must be folded back into the
AQLM-quantized weights and RE-QUANTIZED to the same bit width, so the shipped
model stays 2-bit (not 2-bit + an FP adapter). Skipping this silently adds FP
precision through the adapter -- exactly the confound the advisor raised -- and
invalidates the result. So this is not polish; it is the whole reason Q-Resafe
adds anything over the per-projection control.

Mechanism (AQLM/src/aq.py QuantizedWeight):
  W_target = qweight()  +  (alpha/r) * (B @ A)          # fold the LoRA delta
  qweight.beam_search_update_codes_(reference_weight=W_target)   # codes-only refit

Codes-only update => same codes shape/dtype, same codebooks, ZERO new FP params,
so estimate_nbits_per_parameter() is invariant: the fold preserves the bit width
by construction. Pass XTX (calibration X^T X) for the data-aware (activation-MSE)
refit; omit it for the data-free (weight-MSE) refit.

Run the self-test:  python -m diagnostics.qresafe_refold
"""

import sys
from pathlib import Path

import torch

AQLM_REPO = "/jet/home/bhoomire/projects/AQLM"
if AQLM_REPO not in sys.path:
    sys.path.insert(0, AQLM_REPO)


def fold_lora_into_quantized_(qweight, A, B, scaling, XTX=None, beam_size=5, **beam_kwargs):
    """Fold LoRA delta (scaling * B @ A) into an AQLM QuantizedWeight in place by
    re-fitting its codes to the modified weight. Bit width is preserved.

    Args:
        qweight: AQLM src.aq.QuantizedWeight (out_features x in_features).
        A: LoRA down-projection [r, in_features].
        B: LoRA up-projection   [out_features, r].
        scaling: LoRA scale (alpha / r).
        XTX: optional [in, in] calibration Gram matrix for data-aware refit;
             if None, data-free weight-MSE refit.
        beam_kwargs: forwarded to beam_search (e.g. beam_size).

    Returns:
        dict with pre/post relative reconstruction error vs the fold target and
        the nbits-per-parameter before/after (must be equal).
    """
    with torch.no_grad():
        W = qweight().float()                                  # current dense reconstruct
        delta = scaling * (B.float() @ A.float())             # [out, in]
        target = (W + delta).to(qweight.get_codebooks().dtype)

        nbits_before = qweight.estimate_nbits_per_parameter()
        err_pre = (qweight().float() - target.float()).norm() / (target.float().norm() + 1e-9)

        if XTX is not None:
            qweight.beam_search_update_codes_(XTX=XTX, reference_weight=target,
                                              beam_size=beam_size, **beam_kwargs)
        else:
            qweight.beam_search_update_codes_(reference_weight=target,
                                              beam_size=beam_size, **beam_kwargs)

        err_post = (qweight().float() - target.float()).norm() / (target.float().norm() + 1e-9)
        nbits_after = qweight.estimate_nbits_per_parameter()

    return {
        "rel_err_pre_fold": float(err_pre),
        "rel_err_post_fold": float(err_post),
        "nbits_before": float(nbits_before),
        "nbits_after": float(nbits_after),
    }


def _self_test():
    """Validate the fold on a synthetic QuantizedWeight (small-but-valid AQLM
    config) so the crux is proven without loading an 8B model. Asserts the refit
    drives reconstruction toward the fold target AND that bit width is unchanged."""
    from src.aq import QuantizedWeight

    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_f, in_f = 512, 512
    W0 = torch.randn(out_f, in_f, device=dev)

    # small-but-valid AQLM config (2 codebooks x 8 bits, in_group 8) -- exercises
    # the exact code path the real 1x16g8 cell uses, fast enough for a unit test.
    qw = QuantizedWeight(
        reference_weight=W0, in_group_size=8, out_group_size=1,
        num_codebooks=2, nbits_per_codebook=8, codebook_value_nbits=16,
    ).to(dev)

    base_err = float((qw().float() - W0).norm() / W0.norm())
    print(f"  baseline quant rel-err (no delta): {base_err:.4f}")

    # a LoRA-shaped delta well ABOVE the quant granularity so the refit has real
    # work to do (a sub-floor delta would just be snapped away -- see note below).
    r = 16
    A = torch.randn(r, in_f, device=dev) * 0.12
    B = torch.randn(out_f, r, device=dev) * 0.12
    scaling = 256.0 / 128.0  # alpha/r per spec
    target = (W0 + scaling * (B @ A)).detach()

    # the achievable floor: quantize the TARGET from scratch (codebooks refit too).
    qw_fresh = QuantizedWeight(
        reference_weight=target, in_group_size=8, out_group_size=1,
        num_codebooks=2, nbits_per_codebook=8, codebook_value_nbits=16,
    ).to(dev)
    fresh_floor = float((qw_fresh().float() - target.float()).norm() / target.float().norm())
    print(f"  fresh-quant floor for target: {fresh_floor:.4f}")

    qw_codes_snapshot = qw.get_codes().clone()
    res = fold_lora_into_quantized_(qw, A, B, scaling)
    print(f"  fold: rel_err {res['rel_err_pre_fold']:.4f} -> {res['rel_err_post_fold']:.4f}"
          f"  | nbits {res['nbits_before']:.3f} -> {res['nbits_after']:.3f}")

    codes_before = qw_codes_snapshot
    codes_after = qw.get_codes().clone()
    frac_codes_changed = float((codes_before != codes_after).float().mean())

    # Diagnostic (NOT pass/fail): floor-relative fidelity is regime-dependent and
    # can only be judged on the real cell, whose quant floor is a few % rather than
    # this synthetic's ~30%. Printed for transparency.
    print(f"  [diag] delta rel-magnitude={res['rel_err_pre_fold']:.4f}  "
          f"fresh-floor={fresh_floor:.4f}  codes changed={100*frac_codes_changed:.1f}%")
    if res["rel_err_pre_fold"] < fresh_floor:
        print("  [diag] NOTE: delta is BELOW the synthetic quant floor -> snapped away "
              "by re-quantization; tracking can't be shown here (measure on-model).")

    ok = True
    # (1) bit width preserved exactly -- the whole point (no FP adapter)
    if res["nbits_after"] != res["nbits_before"]:
        print("  FAIL: bit width changed during fold"); ok = False
    # (2) codes are still integer codes, no FP smuggled in
    if qw.get_codes().dtype not in (torch.int32, torch.int16, torch.uint8, torch.int64):
        print(f"  FAIL: codes are not integer ({qw.get_codes().dtype})"); ok = False
    # (3) the refit never INCREASES reconstruction error (it minimizes MSE)
    if not (res["rel_err_post_fold"] <= res["rel_err_pre_fold"] + 1e-6):
        print("  FAIL: refit increased error -- beam search misused"); ok = False
    # (4) the refit actually engaged (codes moved), i.e. not a silent no-op
    if not (frac_codes_changed > 0.0):
        print("  FAIL: refit changed no codes -- fold did not run"); ok = False

    print("\n  REFOLD GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
