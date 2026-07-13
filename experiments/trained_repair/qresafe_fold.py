"""Fold trained LoRA delta back into AQLM quantized weights.

CONNECTED — do not remove.
  Imported by: qresafe_dpo.py (fold_and_writeback_, quant_cfg, dequantize_module)
  Also run directly by: roundtrip_gate.sbatch, fold_delta_smoke.sbatch (dev gates)
  Not a RESULTS.md paper row, but required by the Q-Resafe DPO pipeline.
"""
import sys
from pathlib import Path

import torch

AQLM_REPO = "/jet/home/bhoomire/projects/AQLM"
if AQLM_REPO not in sys.path:
    sys.path.insert(0, AQLM_REPO)

from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data  # noqa: E402


def quant_cfg(model):
    """
    Get the quantization configuration.
    Inputs:
        model (nn.Module): The model to get the quantization configuration from.

    Outputs:
        cfg (dict): The quantization configuration.
    """
    # Get the quantization configuration
    qc = model.config.quantization_config

    # Check if the quantization configuration is a dictionary
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)

    # Return the quantization configuration
    return {
        "nbits": int(qc["nbits_per_codebook"]),
        "num_codebooks": int(qc["num_codebooks"]),
        "in_group_size": int(qc["in_group_size"]),
        "out_group_size": int(qc["out_group_size"]),
    }


def is_quantized_linear(module):
    """
    Check if the module is a quantized linear.
    Inputs:
        module (nn.Module): The module to check if it is a quantized linear.

    Outputs:
        is_quantized_linear (bool): True if the module is a quantized linear, False otherwise.
    """
    # Check if the module is a quantized linear
    return (hasattr(module, "codes") and hasattr(module, "codebooks")
            and hasattr(module, "scales"))


def dequantize_module(module, nbits):
    """
    Dequantize the module.
    Inputs:
        module (nn.Module): The module to dequantize.
        nbits (int): The number of bits to dequantize the module to.

    Outputs:
        dequantized_module (torch.Tensor): The dequantized module.
    """
    # Dequantize the module
    codes_u = unpack_int_data(module.codes.data, nbits)
    return _dequantize_weight(codes_u, module.codebooks.data, module.scales.data)


def fold_and_writeback_(module, delta, cfg, beam_size=5):
    """
    Fold `delta` (dense [out,in], may be None/zero) into the module's quantized
    weight and write it back in place. Returns fraction of codes changed.
    Inputs:
        module (nn.Module): The module to fold and write back.
        delta (torch.Tensor): The delta to fold into the module.
        cfg (dict): The configuration of the module.
        beam_size (int): The number of beams to use for the beam search.

    Outputs:
        changed (float): The fraction of codes changed.
    """
    # Get the number of bits
    nbits = cfg["nbits"]
    codes_u = unpack_int_data(module.codes.data, nbits)          # [out_g,in_g,ncb] >=0
    codebooks = module.codebooks.data
    scales = module.scales.data

    # Initialize the changed variable
    changed = 0.0
    if delta is not None and float(delta.abs().sum()) > 0.0:
        
        # Create the QuantizedWeight
        from src.aq import QuantizedWeight
        W = _dequantize_weight(codes_u, codebooks, scales)
        W_target = (W + delta.to(W.dtype)).float()
        qw = QuantizedWeight.__new__(QuantizedWeight)   # bypass kmeans re-init
        torch.nn.Module.__init__(qw)
        qw.out_features, qw.in_features = module.out_features, module.in_features
        qw.out_group_size, qw.in_group_size = cfg["out_group_size"], cfg["in_group_size"]
        qw.num_codebooks = cfg["num_codebooks"]
        qw.nbits_per_codebook = nbits
        qw.codebook_size = 2 ** nbits
        qw.codebook_value_nbits = 16
        qw.codebook_value_num_groups = 1
        qw.codebook_value_clusters = None
        qw.scales_clusters = qw.scales_indices = None
        qw.scale_nbits = 0
        qw.straight_through_gradient = False
        qw.scales_are_lossless = True
        qw.codes_storage = None
        qw.codes = torch.nn.Parameter(codes_u.to(torch.int32), requires_grad=False)
        qw.codebooks = torch.nn.Parameter(codebooks.float(), requires_grad=False)
        qw.scales = torch.nn.Parameter(scales.float(), requires_grad=False)
        new_codes = qw.beam_search_update_codes_(reference_weight=W_target, beam_size=beam_size)
        new_codes_u = new_codes.to(torch.int64)
        changed = float((new_codes_u != codes_u).float().mean())
        codes_u = new_codes_u
        codebooks = qw.get_codebooks().to(codebooks.dtype)
        scales = qw.get_scales().to(scales.dtype)

    # write back (re-pack codes to the module's signed int dtype)
    module.codes.data = pack_int_data(codes_u, nbits).to(module.codes.dtype)
    module.codebooks.data = codebooks.to(module.codebooks.dtype)
    module.scales.data = scales.to(module.scales.dtype)
    return changed


def roundtrip_gate(model, target_names, cfg, atol=0.0):
    """Zero-training round trip: for each target module, dequantize, write back with delta=0, dequantize again, assert the dense weight is bit-identical. Returns (n_ok, n_total, max_abs_diff).
    Inputs:
        model (nn.Module): The model to run the roundtrip gate on.
        target_names (list[str]): The names of the target modules.
        cfg (dict): The configuration of the model.
        atol (float): The absolute tolerance for the roundtrip gate.

    Outputs:
        n_ok (int): The number of target modules that passed the roundtrip gate.
    """
    # Initialize the variables
    n_ok, n_total, max_diff = 0, 0, 0.0
    mods = dict(model.named_modules())

    # Run the roundtrip gate for each target module
    for name in target_names:

        # Get the module
        m = mods[name]
        if not is_quantized_linear(m):
            continue
        n_total += 1

        # Dequantize the module
        W0 = dequantize_module(m, cfg["nbits"]).float()
        codes0 = m.codes.data.clone()

        # Fold the module
        fold_and_writeback_(m, delta=None, cfg=cfg)          # pure writeback
        W1 = dequantize_module(m, cfg["nbits"]).float()
        d = float((W1 - W0).abs().max())

        # Check if the codes are identical and update the maximum difference
        codes_identical = bool(torch.equal(m.codes.data, codes0))
        max_diff = max(max_diff, d)
        if d <= atol and codes_identical:
            n_ok += 1
        else:
            print(f"    [mismatch] {name}: max|dW|={d:.2e} codes_identical={codes_identical}")
    return n_ok, n_total, max_diff


def run_gate(model_alias="llama7b", quantizer="aqlm2", n_arc=100):
    """Zero-training round-trip gate on the real headline checkpoint: bit-identity of every target module's dense weight AND ARC unchanged through the bridge.
    Inputs:
        model_alias (str): The alias of the model to run the roundtrip gate on.
        quantizer (str): The quantizer to use for the model.
        n_arc (int): The number of ARC items to use for the roundtrip gate.

    Outputs:
        ok (bool): True if the roundtrip gate passed, False otherwise.
    """
    # Insert the path to the core modules
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.model_loader import load_quantized
    from core.datasets import load_arc_challenge
    from core.restoration import measure_arc
    from experiments.trained_repair.qresafe_repair import select_target_modules

    # Load the model and tokenizer
    model, tok = load_quantized(model_alias, quantizer=quantizer)
    cfg = quant_cfg(model)
    print("quant cfg:", cfg)

    # Select the target modules
    targets = select_target_modules(model, "full")
    print(f"target modules: {len(targets)}")

    # Load the ARC items
    arc_items = load_arc_challenge(n_samples=n_arc)
    arc_before = measure_arc(model, tok, arc_items)

    # Run the roundtrip gate
    n_ok, n_total, max_diff = roundtrip_gate(model, targets, cfg)
    arc_after = measure_arc(model, tok, arc_items)

    # Print the results
    print(f"\n  round-trip: {n_ok}/{n_total} modules bit-identical | max|dW|={max_diff:.2e}")
    print(f"  ARC through bridge: {arc_before:.1f}% -> {arc_after:.1f}% "
          f"(headline aqlm2 ARC was 46%)")
    ok = (n_ok == n_total and n_total > 0 and max_diff == 0.0
          and abs(arc_before - arc_after) < 1e-9)
    print("\n  ROUND-TRIP GATE:", "PASS" if ok else "FAIL")
    return ok


def delta_fold_smoke(model_alias="llama7b", quantizer="aqlm2"):
    """Exercise the delta!=0 fold path (QuantizedWeight.__new__ + beam_search +
    writeback) on ONE real module, before any DPO run depends on it.
    Inputs:
        model_alias (str): The alias of the model to run the delta fold smoke test on.
        quantizer (str): The quantizer to use for the model.

    Outputs:
        ok (bool): True if the delta fold smoke test passed, False otherwise.
    """
    # Insert the path to the core modules
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    from core.model_loader import load_quantized
    from experiments.trained_repair.qresafe_repair import select_target_modules

    # Load the model and tokenizer
    model, tok = load_quantized(model_alias, quantizer=quantizer)
    cfg = quant_cfg(model)

    # Select the target module
    name = select_target_modules(model, "read")[0]

    # Get the module
    m = dict(model.named_modules())[name]

    # Dequantize the module
    W0 = dequantize_module(m, cfg["nbits"]).float()

    # Initialize the variables
    nb0 = None
    out_f, in_f = m.out_features, m.in_features

    # Create the delta
    r = 16
    A = torch.randn(r, in_f, device=W0.device) * 0.01
    B = torch.randn(out_f, r, device=W0.device) * 0.01
    delta = (B @ A)
    rel = float(delta.norm() / W0.norm())
    dtype0 = m.codes.dtype
    frac = fold_and_writeback_(m, delta, cfg, beam_size=5)

    # Dequantize the module
    W1 = dequantize_module(m, cfg["nbits"]).float()

    # Print the results
    moved = float((W1 - W0).norm() / W0.norm())
    print(f"  module={name} delta_rel={rel:.4f} codes_changed={100*frac:.2f}% "
          f"recon_moved={moved:.4f} codes_dtype={m.codes.dtype}")
    ok = (frac > 0.0) and bool(torch.isfinite(W1).all()) and (m.codes.dtype == dtype0)
    print("  DELTA-FOLD SMOKE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "delta":
        sys.exit(0 if delta_fold_smoke() else 1)
    sys.exit(0 if run_gate() else 1)
