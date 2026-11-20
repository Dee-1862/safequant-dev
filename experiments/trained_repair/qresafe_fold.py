# Fold trained LoRA delta back into AQLM quantized weights.
import os
import sys

import torch

AQLM_REPO = os.environ.get("AQLM_REPO", "")
if AQLM_REPO and AQLM_REPO not in sys.path:
    sys.path.insert(0, AQLM_REPO)

from aqlm.utils import _dequantize_weight, pack_int_data, unpack_int_data  # noqa: E402


def quant_cfg(model):
    """
    Extract AQLM codebook / group sizes from the model config.

    Inputs:
        - model (nn.Module): Quantized model with quantization_config

    Outputs:
        - cfg (dict): nbits, num_codebooks, in_group_size, out_group_size
    """
    # Normalize config object to a plain dict
    qc = model.config.quantization_config
    if not isinstance(qc, dict):
        qc = qc.to_dict() if hasattr(qc, "to_dict") else dict(qc)
    return {
        "nbits": int(qc["nbits_per_codebook"]),
        "num_codebooks": int(qc["num_codebooks"]),
        "in_group_size": int(qc["in_group_size"]),
        "out_group_size": int(qc["out_group_size"]),
    }


def dequantize_module(module, nbits):
    """
    Dequantize an AQLM linear module to a dense weight matrix.

    Inputs:
        - module (nn.Module): Quantized linear module
        - nbits (int): Bits per codebook entry

    Outputs:
        - weight (Tensor): Dense dequantized weight
    """
    # Unpack signed codes then reconstruct with codebooks/scales
    codes_u = unpack_int_data(module.codes.data, nbits)
    return _dequantize_weight(codes_u, module.codebooks.data, module.scales.data)


def fold_and_writeback_(module, delta, cfg, beam_size = 5):
    """
    Fold a dense delta into AQLM codes and write codes/codebooks/scales back.

    Inputs:
        - module (nn.Module): Quantized linear module (mutated in place)
        - delta (Tensor | None): Dense [out, in] update, or None/zero for pure writeback
        - cfg (dict): Quantization configuration from quant_cfg
        - beam_size (int): Beam width for code updates

    Outputs:
        - changed (float): Fraction of codes that changed
    """
    # Unpack current codes and storage tensors
    nbits = cfg["nbits"]
    codes_u = unpack_int_data(module.codes.data, nbits)          # [out_g,in_g,ncb] >=0
    codebooks = module.codebooks.data
    scales = module.scales.data

    # Non-zero delta: rebuild Codes via QuantizedWeight beam search
    changed = 0.0
    if delta is not None and float(delta.abs().sum()) > 0.0:
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
        qw.codes = torch.nn.Parameter(codes_u.to(torch.int32), requires_grad = False)
        qw.codebooks = torch.nn.Parameter(codebooks.float(), requires_grad = False)
        qw.scales = torch.nn.Parameter(scales.float(), requires_grad = False)
        new_codes = qw.beam_search_update_codes_(reference_weight = W_target, beam_size = beam_size)
        new_codes_u = new_codes.to(torch.int64)
        changed = float((new_codes_u != codes_u).float().mean())
        codes_u = new_codes_u
        codebooks = qw.get_codebooks().to(codebooks.dtype)
        scales = qw.get_scales().to(scales.dtype)

    # Re-pack codes to the module's signed int dtype and write back
    module.codes.data = pack_int_data(codes_u, nbits).to(module.codes.dtype)
    module.codebooks.data = codebooks.to(module.codebooks.dtype)
    module.scales.data = scales.to(module.scales.dtype)
    return changed
