# Dense fp16 copy of a GPTQ-3 checkpoint, dequantized by unmodified auto-gptq 0.7.1.
#
#   $AGENV/bin/python scripts/gptq_dequant_dense.py --model phi3 [--bits 3]
#
# Why: gptqmodel 5.7.0 (sq_clean's GPTQ loader) unpacks 3-bit weights wrongly (PPL nan
# in fp16, ~1e6 in bf16), while auto-gptq 0.7.1, which wrote the checkpoints, loads them
# correctly (phi3 8.419 / llama7b 8.766 on 20 WikiText-2 windows). This writes the exact
# fp16 weights auto-gptq computes, so sq_clean can evaluate GPTQ-3 cells as a plain fp16
# model (core/model_loader.py, opt-in SQ_GPTQ_DENSE=1 for all bit widths, SQ_GPTQ3_DENSE=1 for 3-bit only).
#
# Method: load with AutoGPTQForCausalLM.from_quantized (qlinear_cuda torch path: weights =
# scales[g_idx] * (q - zeros[g_idx]) in fp16, then x @ weights + bias). For each QuantLinear
# an identity matrix is pushed through with the bias detached, which returns those weights
# exactly; the module is replaced by nn.Linear(weight = weights.T, bias = bias).
#
# Checks (printed, and stored in sq_dequant_info.json):
#   - per decoder layer + logits: max |auto-gptq output - dense output| on 4 real inputs
#     (WikiText-2 test windows of 512 tokens); expected ~0
#   - PPL on the first 20 WikiText-2 test windows (2048 tokens) with the auto-gptq model
#     (reference for the sq_clean dense check)
# Output: $QM_ROOT/gptq_dense/<model>-gptq<bits>bit-g128 (written to .partial, then renamed;
# never overwrites).
import argparse
import datetime
import json
import math
import os
import platform
import shutil
import sys
import time
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QM_ROOT  # noqa: E402
from core.datasets import load_wikitext2  # noqa: E402


def ckpt_dir(model, bits):
    """
    Self-built GPTQ checkpoint folder.

    Inputs:
        - model (str): Model alias
        - bits (int): Bit width

    Outputs:
        - path (Path): $QM_ROOT/gptq/<model>-gptq<bits>bit-g128
    """
    return Path(QM_ROOT) / "gptq" / f"{model}-gptq{bits}bit-g128"


def load_autogptq(path, model_name):
    """
    Load a GPTQ checkpoint through auto-gptq (with the Phi-3 registration).

    Inputs:
        - path (Path): Checkpoint folder
        - model_name (str): Model alias

    Outputs:
        - model (nn.Module): Model in eval mode
        - tok: Tokenizer
    """
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM
    if model_name == "phi3":  # same Phi-3 registration as scripts/quantize_gptq.py
        from auto_gptq.modeling._base import BaseGPTQForCausalLM
        from auto_gptq.modeling import _const as _agc, auto as _aga
        if "phi3" not in _agc.SUPPORTED_MODELS:
            class Phi3GPTQForCausalLM(BaseGPTQForCausalLM):
                layer_type = "Phi3DecoderLayer"
                layers_block_name = "model.layers"
                outside_layer_modules = ["model.embed_tokens", "model.norm"]
                inside_layer_modules = [["self_attn.qkv_proj"], ["self_attn.o_proj"],
                                        ["mlp.gate_up_proj"], ["mlp.down_proj"]]
            _agc.SUPPORTED_MODELS.append("phi3")
            _aga.GPTQ_CAUSAL_LM_MODEL_MAP["phi3"] = Phi3GPTQForCausalLM
    base = sorted(p.stem for p in path.glob("gptq_model-*.safetensors"))[0]
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoGPTQForCausalLM.from_quantized(
        str(path), model_basename = base, device = "cuda:0", use_safetensors = True,
        use_triton = False, disable_exllama = True, disable_exllamav2 = True,
        inject_fused_attention = False, inject_fused_mlp = False)
    kinds = sorted({type(m).__name__ for m in model.modules() if "QuantLinear" in type(m).__name__})
    print(f"  quant kernels in model: {kinds}  (module {next(m for m in model.modules() if 'QuantLinear' in type(m).__name__).__module__})")
    return model.eval(), tok


def ppl(model, tok, text, n_windows, seq_len = 2048):
    """
    Perplexity over the first n_windows windows, same arithmetic as core.evaluation.measure_perplexity.

    Inputs:
        - model (nn.Module): Model under test
        - tok: Tokenizer
        - text (str): WikiText-2 test text
        - n_windows (int): Windows to score
        - seq_len (int): Window length

    Outputs:
        - ppl (float): Perplexity (nan if no tokens were scored)
        - bad (int): Windows with a non-finite loss
        - n (int): Windows scored
        - enc (Tensor): Encoded text
    """
    enc = tok(text, return_tensors = "pt").input_ids.to("cuda:0")
    n = min(enc.shape[1] // seq_len, n_windows)
    nll, cnt, bad = 0.0, 0, 0
    with torch.no_grad():
        for i in range(n):
            chunk = enc[:, i * seq_len:(i + 1) * seq_len]
            logits = model(input_ids = chunk).logits[0]
            if not torch.isfinite(logits).all():
                bad += 1
            lp = torch.log_softmax(logits[:-1].float(), dim = -1)
            nll += -lp.gather(1, chunk[0, 1:].unsqueeze(1)).sum().item()
            cnt += chunk.shape[1] - 1
    return math.exp(nll / cnt) if cnt else float("nan"), bad, n, enc


def is_qlinear(m):
    """
    Whether a module is an auto-gptq quantized linear layer.

    Inputs:
        - m (nn.Module): Module

    Outputs:
        - is_q (bool): True for QuantLinear modules with packed weights
    """
    return hasattr(m, "qweight") and "QuantLinear" in type(m).__name__


@torch.no_grad()
def dense_weight(mod):
    """
    Exact fp16 weight that auto-gptq computes: identity through the layer, bias detached.

    Inputs:
        - mod (nn.Module): auto-gptq QuantLinear

    Outputs:
        - weight (Tensor): (out, in) fp16 weight
        - bias (Tensor or None): Copy of the bias
    """
    bias, mod.bias = mod.bias, None
    try:
        eye = torch.eye(mod.infeatures, device = mod.qweight.device, dtype = torch.float16)
        w_t = mod(eye)                      # (in, out) = scales[g_idx] * (q - zeros[g_idx])
    finally:
        mod.bias = bias
    return w_t.t().contiguous(), (bias.detach().clone() if bias is not None else None)


@torch.no_grad()
def layer_outputs(hf, inputs):
    """
    Per-decoder-layer hidden states and logits for each input (kept on CPU, fp32).

    Inputs:
        - hf (nn.Module): Hugging Face causal LM
        - inputs (list[Tensor]): Token id batches

    Outputs:
        - outs (list[tuple]): Per input, (hidden states per layer, logits)
    """
    layers = hf.model.layers
    outs = []
    for ids in inputs:
        cap = {}
        hooks = [l.register_forward_hook(
            (lambda i: lambda m, a, o: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o).float().cpu()))(i))
            for i, l in enumerate(layers)]
        logits = hf(input_ids = ids).logits.float().cpu()
        for h in hooks:
            h.remove()
        outs.append(([cap[i] for i in range(len(layers))], logits))
    return outs


def main():
    ap = argparse.ArgumentParser(description = "Dense fp16 copy of a GPTQ checkpoint via auto-gptq.")
    ap.add_argument("--model", required = True)
    ap.add_argument("--bits", type = int, default = 3)
    ap.add_argument("--n_compare", type = int, default = 4)
    ap.add_argument("--compare_len", type = int, default = 512)
    ap.add_argument("--windows", type = int, default = 20)
    a = ap.parse_args()

    src = ckpt_dir(a.model, a.bits)
    out = Path(QM_ROOT) / "gptq_dense" / src.name
    if (out / "config.json").exists():
        print(f"ERROR: {out} already exists; not overwriting.", file = sys.stderr)
        sys.exit(1)
    tmp = out.with_name(f"{out.name}.partial-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    if tmp.exists():
        shutil.rmtree(tmp)
    print(f"=== dequantize {src} -> {out}")

    t0 = time.time()
    qmodel, tok = load_autogptq(src, a.model)       # unmodified auto-gptq 0.7.1
    hf = qmodel.model
    text = load_wikitext2("test")

    # Reference PPL (auto-gptq) on the same 20 windows the sq_clean check uses
    ppl_ag, bad_ag, nwin, enc = ppl(qmodel, tok, text, a.windows)
    print(f"  auto-gptq PPL({nwin} windows) = {ppl_ag:.4f}  non-finite windows {bad_ag}")

    # Real inputs for the per-layer comparison: spaced WikiText-2 test windows
    step = enc.shape[1] // (a.n_compare + 1)
    inputs = [enc[:, (k + 1) * step:(k + 1) * step + a.compare_len] for k in range(a.n_compare)]
    ref = layer_outputs(hf, inputs)

    # Replace every QuantLinear with nn.Linear holding auto-gptq's exact fp16 weights
    names = [n for n, m in hf.named_modules() if is_qlinear(m)]
    for n in names:
        parent_name, _, child = n.rpartition(".")
        parent = hf.get_submodule(parent_name)
        q = getattr(parent, child)
        w, b = dense_weight(q)
        lin = nn.Linear(q.infeatures, q.outfeatures, bias = b is not None,
                        device = w.device, dtype = torch.float16)
        lin.weight.data.copy_(w)
        if b is not None:
            lin.bias.data.copy_(b.to(torch.float16))
        setattr(parent, child, lin)
        del q
    torch.cuda.empty_cache()
    assert not any(is_qlinear(m) for m in hf.modules()), "QuantLinear left after replacement"
    print(f"  replaced {len(names)} QuantLinear modules with nn.Linear")

    new = layer_outputs(hf, inputs)
    per_layer = []
    for i in range(len(hf.model.layers)):
        d = max(float((r[0][i] - n_[0][i]).abs().max()) for r, n_ in zip(ref, new))
        s = max(float(r[0][i].abs().max()) for r in ref)
        per_layer.append({"layer": i, "max_abs_diff": d, "max_abs_ref": s})
    logit_diff = max(float((r[1] - n_[1]).abs().max()) for r, n_ in zip(ref, new))
    worst = max(per_layer, key = lambda x: x["max_abs_diff"])
    print(f"  per-layer max |auto-gptq - dense| over {len(inputs)} inputs x {a.compare_len} tokens:")
    for p in per_layer:
        print(f"    layer {p['layer']:2d}: {p['max_abs_diff']:.3g}   (max |ref| {p['max_abs_ref']:.3g})")
    print(f"  logits max |diff| = {logit_diff:.3g};  worst layer {worst['layer']} = {worst['max_abs_diff']:.3g}")

    # Save as a plain fp16 model (no quantization_config)
    for attr in ("quantization_config",):
        if hasattr(hf.config, attr):
            delattr(hf.config, attr)
    hf.save_pretrained(str(tmp), safe_serialization = True, max_shard_size = "5GB")
    tok.save_pretrained(str(tmp))
    src_info = {}
    if (src / "sq_build_info.json").exists():
        src_info = json.loads((src / "sq_build_info.json").read_text())
    info = {
        "what": f"dense fp16 copy of {src.name}, dequantized by auto-gptq",
        "source_checkpoint": str(src),
        "source_build_info": src_info,
        "method": ("AutoGPTQForCausalLM.from_quantized (qlinear_cuda torch path, no CUDA ext); "
                   "each QuantLinear -> identity forward with bias detached = "
                   "scales[g_idx]*(q-zeros[g_idx]) in fp16 -> nn.Linear(weight=W.T, bias)"),
        "n_quantlinear_replaced": len(names),
        "checks": {
            "per_layer_max_abs_diff": per_layer, "logits_max_abs_diff": logit_diff,
            "compare_inputs": f"{len(inputs)} WikiText-2 test windows x {a.compare_len} tokens",
            "autogptq_ppl_20_windows": ppl_ag,
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torch_cuda": torch.version.cuda, "transformers": pkg_version("transformers"),
                     "auto_gptq": pkg_version("auto-gptq")},
        "host": {"machine": platform.machine(), "node": platform.node(),
                 "gpu": torch.cuda.get_device_name(0), "slurm_job_id": os.environ.get("SLURM_JOB_ID")},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (tmp / "sq_dequant_info.json").write_text(json.dumps(info, indent = 2))
    tmp.rename(out)
    print(f"  saved {out}  ({time.time() - t0:.0f} s)")
    print(f"RESULT {a.model} gptq{a.bits} dequant: layers max|diff| {worst['max_abs_diff']:.3g} "
          f"(layer {worst['layer']}), logits {logit_diff:.3g}, auto-gptq PPL20 {ppl_ag:.4f}")


if __name__ == "__main__":
    main()
