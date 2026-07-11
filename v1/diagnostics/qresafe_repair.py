"""
qresafe_repair.py -- Q-Resafe forced-mask repair harness (Tier 2).

Reimplements Algorithm 1 as a 3-way comparison: forced-READER vs forced-WRITER
vs FULL masked LoRA-DPO, then requant-folds and re-evaluates. The mask is the
experimental knob: we OVERRIDE Q-Resafe's SNIP selection on purpose, because
SNIP pointed write-side and the thesis says read-side, so the test is forced.

This file currently implements the foundation (mask selector + masked LoRA);
the DPO loop, fold-with-writeback, and in-place eval build on top of it and on
fold_lora_into_quantized_ in qresafe_refold.py.

Architecture decision (validated 2026-06-13): run entirely from the HF-inference
checkpoint (core.model_loader.load_quantized(...,'aqlm2')). The fold uses a fresh
AQLM QuantizedWeight as the re-quantization engine and packs back into the
inference tensors -- no repo/FSDP trainable loader needed.

Self-test:  python -m diagnostics.qresafe_repair
"""

import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from diagnostics.readwrite_restoration import READ_PROJS, WRITE_PROJS  # noqa: E402

MASK_SETS = {
    "read": READ_PROJS,                  # forced reader-side {q,k,v,gate,up}
    "write": WRITE_PROJS,                # forced writer-side {o,down}
    "full": READ_PROJS + WRITE_PROJS,    # the published (all-projection) setting
}


def select_target_modules(model, mask: str):
    """Return the names of Linear-like modules whose projection type is in the
    chosen mask set. Matches the same suffixes the diagnosis uses, so the repair
    lives in the diagnosis coordinate system."""
    if mask not in MASK_SETS:
        raise ValueError(f"mask must be one of {list(MASK_SETS)}, got {mask!r}")
    wanted = MASK_SETS[mask]
    names = []
    for name, module in model.named_modules():
        if any(name.endswith(p) for p in wanted) and (
            hasattr(module, "weight") or hasattr(module, "in_features")
            or type(module).__name__ in ("QuantizedLinear", "WQLinear")
        ):
            names.append(name)
    return names


def _infeat_outfeat(module):
    inf = getattr(module, "in_features", None) or getattr(module, "infeatures", None)
    out = getattr(module, "out_features", None) or getattr(module, "outfeatures", None)
    return inf, out


class LoRALinear(nn.Module):
    """Wrap a (possibly quantized) linear with a frozen base + trainable LoRA
    delta:  y = base(x) + scaling * (dropout(x) @ A^T) @ B^T.  B is zero-init so
    the initial delta is exactly zero (the policy starts == the frozen base)."""

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = _infeat_outfeat(base)
        self.in_features, self.out_features = in_f, out_f
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = next(base.parameters()).device
        dt = torch.float32
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.enabled = True   # toggle off to get the DPO reference (base, no adapter)

    def forward(self, x):
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        xd = self.drop(x).to(self.lora_A.dtype)
        delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * delta.to(base_out.dtype)

    def delta_weight(self):
        """The dense LoRA delta added to the base weight: scaling * B @ A  [out,in]."""
        return self.scaling * (self.lora_B @ self.lora_A)


def _set_module(model, name, new):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    setattr(parent, parts[-1], new)


def attach_masked_lora(model, mask: str, r: int = 128, alpha: float = 256.0, dropout: float = 0.0):
    """Freeze the whole model, wrap the masked projection set with LoRALinear,
    and return (lora_params, wrapped_names). Only lora_A/lora_B will train."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = select_target_modules(model, mask)
    wrapped = []
    for name in names:
        base = dict(model.named_modules())[name]
        _set_module(model, name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    return lora_params, wrapped


def set_lora_enabled(model, flag: bool):
    """Toggle every LoRALinear. flag=False gives the DPO reference (base, no adapter)."""
    for m in model.modules():
        if type(m).__name__ == "LoRALinear":
            m.enabled = flag


def unwrap_lora(model, wrapped_names):
    """Replace each LoRALinear with its (now folded-into) base module, so the model
    is pure quantized with NO FP adapter. Call AFTER folding the delta into codes."""
    mods = dict(model.named_modules())
    for name in wrapped_names:
        m = mods.get(name)
        if m is not None and type(m).__name__ == "LoRALinear":
            _set_module(model, name, m.base)


# --------------------------------------------------------------------------- #
# self-test (CPU, toy model -- no checkpoint)
# --------------------------------------------------------------------------- #
def _toy_llama(d=32, n_layers=2):
    def block():
        b = nn.Module()
        b.self_attn = nn.Module()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(d, d, bias=False))
        b.mlp = nn.Module()
        for p in ("gate_proj", "up_proj", "down_proj"):
            setattr(b.mlp, p, nn.Linear(d, d, bias=False))
        return b
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([block() for _ in range(n_layers)])
    return m


def _self_test():
    torch.manual_seed(0)
    d, n_layers = 32, 2
    ok = True

    # mask selection counts: reader=5/layer, writer=2/layer, full=7/layer
    m = _toy_llama(d, n_layers)
    counts = {k: len(select_target_modules(m, k)) for k in MASK_SETS}
    exp = {"read": 5 * n_layers, "write": 2 * n_layers, "full": 7 * n_layers}
    print(f"  mask counts: {counts}  expected {exp}")
    if counts != exp:
        print("  FAIL: mask selection counts wrong"); ok = False

    # attach reader LoRA; writer projections must remain plain Linear
    m = _toy_llama(d, n_layers)
    lora_params, wrapped = attach_masked_lora(m, "read", r=8, alpha=16.0)
    q = m.model.layers[0].self_attn.q_proj
    o = m.model.layers[0].self_attn.o_proj
    if type(q).__name__ != "LoRALinear":
        print("  FAIL: reader q_proj not wrapped"); ok = False
    if type(o).__name__ == "LoRALinear":
        print("  FAIL: writer o_proj wrongly wrapped"); ok = False

    # zero-init B => initial delta exactly zero (policy starts == base)
    x = torch.randn(4, d)
    with torch.no_grad():
        d0 = (q(x) - q.base(x)).abs().max().item()
    if d0 > 1e-6:
        print(f"  FAIL: initial LoRA delta not zero ({d0:.2e})"); ok = False

    # backward: grads land ONLY on lora_A/lora_B, never on the frozen base
    y = q(x).sum()
    y.backward()
    if q.lora_A.grad is None or q.lora_B.grad is None:
        print("  FAIL: no grad on LoRA params"); ok = False
    base_w = q.base.weight
    if base_w.grad is not None:
        print("  FAIL: frozen base received a gradient"); ok = False
    n_train = sum(p.numel() for p in lora_params)
    print(f"  reader LoRA: {len(wrapped)} modules wrapped, {n_train} trainable params, "
          f"init-delta={d0:.1e}")

    print("\n  REPAIR FOUNDATION GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
