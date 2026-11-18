# Diagnose NaN PPL on rebuilt GPTQ-3 checkpoints (prints only; writes nothing to outputs/).
#
#   python scripts/tests/diag_gptq3.py --model phi3 --method gptqmodel --dtype fp16   (sq_clean)
#   python scripts/tests/diag_gptq3.py --model phi3 --method gptqmodel --dtype bf16   (sq_clean)
#   $AGENV/bin/python scripts/tests/diag_gptq3.py --model phi3 --method autogptq      (sq_autogptq)
#
# Same 20 WikiText-2 test windows (2048 tokens) for every method. gptqmodel = the
# pipeline's path (transformers GPTQConfig -> optimum -> gptqmodel, act_group_aware
# shim as in core/model_loader.py); autogptq = auto-gptq 0.7.1's own from_quantized
# (checks the checkpoint independently of gptqmodel). On window 0 every module is
# hooked to report the first one whose output is non-finite and the largest |value|
# entering it.
import argparse
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import HF_HOME, QM_ROOT  # noqa: E402
from core.datasets import load_wikitext2  # noqa: E402

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def ckpt_dir(model, bits):
    return Path(QM_ROOT) / "gptq" / f"{model}-gptq{bits}bit-g128"


def load_gptqmodel(path, bits, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
    import optimum.gptq.quantizer as _oq  # same act_group_aware shim as core/model_loader.py
    if not getattr(_oq.GPTQQuantizer, "_safequant_agaw_patch", False):
        _orig = _oq.GPTQQuantizer.__init__

        def _init(self, *a, **kw):
            kw["act_group_aware"] = False
            _orig(self, *a, **kw)
        _oq.GPTQQuantizer.__init__ = _init
        _oq.GPTQQuantizer._safequant_agaw_patch = True
    tok = AutoTokenizer.from_pretrained(str(path), cache_dir=HF_HOME)
    model = AutoModelForCausalLM.from_pretrained(
        str(path), quantization_config=GPTQConfig(bits=bits, use_exllama=False),
        device_map="cuda:0", cache_dir=HF_HOME, torch_dtype=dtype)
    kinds = sorted({type(m).__name__ for m in model.modules() if "QuantLinear" in type(m).__name__})
    import gptqmodel
    from importlib.metadata import version as _v
    print(f"  gptqmodel {_v('gptqmodel')} from {Path(gptqmodel.__file__).parent.parent}")
    print(f"  quant kernels in model: {kinds}")
    return model.eval(), tok


def load_dense(model_name, bits):
    """Dense fp16 copy written by scripts/gptq_dequant_dense.py, loaded as a plain model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(QM_ROOT) / "gptq_dense" / f"{model_name}-gptq{bits}bit-g128"
    tok = AutoTokenizer.from_pretrained(str(path))
    model = AutoModelForCausalLM.from_pretrained(str(path), device_map="cuda:0", torch_dtype=torch.float16)
    n_q = sum("QuantLinear" in type(m).__name__ for m in model.modules())
    print(f"  dense model from {path}; QuantLinear modules: {n_q}")
    return model.eval(), tok


def load_autogptq(path, model_name):
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
        str(path), model_basename=base, device="cuda:0", use_safetensors=True,
        use_triton=False, disable_exllama=True, disable_exllamav2=True,
        inject_fused_attention=False, inject_fused_mlp=False)
    kinds = sorted({type(m).__name__ for m in model.modules() if "QuantLinear" in type(m).__name__})
    print(f"  quant kernels in model: {kinds}  (module {next(m for m in model.modules() if 'QuantLinear' in type(m).__name__).__module__})")
    return model.eval(), tok


def first_nonfinite(model, ids):
    """Run one window with hooks; return (module, max|input|) for the first non-finite output."""
    found, last_ok = [], [None]
    hooks = []

    def mk(name):
        def hook(mod, inp, out):
            if found:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(t):
                return
            if not torch.isfinite(t).all():
                xin = inp[0] if inp and torch.is_tensor(inp[0]) else None
                amax = float(xin.float().abs().max()) if xin is not None and torch.isfinite(xin).all() else float("nan")
                found.append((name, type(mod).__name__, amax, str(t.dtype)))
            else:
                last_ok[0] = (name, float(t.float().abs().max()))
        return hook
    for n, m in model.named_modules():
        if n:
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        model(input_ids=ids)
    for h in hooks:
        h.remove()
    return (found[0] if found else None), last_ok[0]


def ppl(model, tok, text, n_windows, seq_len=2048):
    """Identical arithmetic to core.evaluation.measure_perplexity, first n_windows windows."""
    enc = tok(text, return_tensors="pt").input_ids.to("cuda:0")
    n = min(enc.shape[1] // seq_len, n_windows)
    nll, cnt, bad = 0.0, 0, 0
    with torch.no_grad():
        for i in range(n):
            chunk = enc[:, i * seq_len:(i + 1) * seq_len]
            logits = model(input_ids=chunk).logits[0]
            if not torch.isfinite(logits).all():
                bad += 1
            lp = torch.log_softmax(logits[:-1].float(), dim=-1)
            nll += -lp.gather(1, chunk[0, 1:].unsqueeze(1)).sum().item()
            cnt += chunk.shape[1] - 1
    return math.exp(nll / cnt) if cnt else float("nan"), bad, n, enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--method", choices=["gptqmodel", "autogptq", "dense"], required=True)
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp16")
    ap.add_argument("--windows", type=int, default=20)
    a = ap.parse_args()
    path = ckpt_dir(a.model, a.bits)
    gv = ""
    if a.method == "gptqmodel":
        try:
            from importlib.metadata import version as _v
            gv = f" {_v('gptqmodel')}"
        except Exception:
            gv = " ?"
    tag = f"{a.model} gptq{a.bits} | {a.method}{gv}" + (f" {a.dtype}" if a.method == "gptqmodel" else " fp16")
    print(f"=== {tag} | {path}")
    t0 = time.time()
    try:
        if a.method == "gptqmodel":
            model, tok = load_gptqmodel(path, a.bits, DTYPES[a.dtype])
        elif a.method == "dense":
            model, tok = load_dense(a.model, a.bits)
        else:
            model, tok = load_autogptq(path, a.model)
        text = load_wikitext2("test")
        p, bad, n, enc = ppl(model, tok, text, a.windows)
        hit, last = first_nonfinite(model, enc[:, :2048])
        print(f"RESULT {tag}: PPL({n} windows) = {p:.4f}   windows with non-finite logits: {bad}/{n}"
              f"   ({time.time() - t0:.0f} s)")
        if hit:
            print(f"  first non-finite output: {hit[0]} ({hit[1]}, out dtype {hit[3]}); max|input| = {hit[2]:.4g}")
            print(f"  last finite module before it: {last}")
        else:
            print("  no non-finite activations on window 0")
    except Exception as e:  # report and let the next method run
        print(f"RESULT {tag}: FAILED {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    main()
