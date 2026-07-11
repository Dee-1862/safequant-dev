"""Diagnose the broken Phi-3 read/write restoration: is the AQLM checkpoint a
faithful quantization of the SAME FP weights we inject, or a different version?

For each restored projection, compare the FP weight against the AQLM-dequantized
weight. ~quant-error (rel_err ~0.1-0.3, cos ~0.95+) => same model, bug is elsewhere.
Large error / low cos / shape mismatch => FP and AQLM are different versions, and
injecting FP weights corrupts the model (the observed both_restored != FP32)."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.model_loader import load_fp32, load_quantized, get_weight
from diagnostics.readwrite_restoration import get_proj_sets, _get_submodule, _get_inner_model


def main():
    fp, tok = load_fp32("phi3")
    q, _ = load_quantized("phi3", quantizer="aqlm2")
    read, write = get_proj_sets(fp)
    fp_inner = fp.model
    q_inner = _get_inner_model(q)
    print(f"\nFP config _name_or_path: {getattr(fp.config,'_name_or_path','?')}")
    print(f"AQLM config _name_or_path: {getattr(q.config,'_name_or_path','?')}")
    print(f"FP n_layers={fp.config.num_hidden_layers}  AQLM n_layers={q.config.num_hidden_layers}\n")
    for L in (0, fp.config.num_hidden_layers // 2):
        print(f"--- layer {L} ---")
        for proj in read + write:
            fw = get_weight(_get_submodule(fp_inner.layers[L], proj)).float()
            qw = get_weight(_get_submodule(q_inner.layers[L], proj)).float().to(fw.device)
            if fw.shape != qw.shape:
                print(f"  {proj:22s} SHAPE MISMATCH  FP{tuple(fw.shape)} vs AQLM{tuple(qw.shape)}")
                continue
            rel = (fw - qw).norm() / (fw.norm() + 1e-9)
            cos = F.cosine_similarity(fw.flatten(), qw.flatten(), dim=0)
            verdict = "same-version(quant err)" if rel < 0.45 else "DIFFERENT VERSION"
            print(f"  {proj:22s} FP{tuple(fw.shape)} rel_err={float(rel):.3f} cos={float(cos):.3f}  {verdict}")
    print("\n=> If rel_err is large / cos low across projections, the AQLM checkpoint is a "
          "different Phi-3 version than the FP base; pin the FP revision to match.")


if __name__ == "__main__":
    main()
