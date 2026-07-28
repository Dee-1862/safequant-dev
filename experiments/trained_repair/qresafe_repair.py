# Q-Resafe LoRA repair library (mask selection, attach, train helpers).
#Front Matter
import math

import torch
import torch.nn as nn

from core.restoration import READ_PROJS, WRITE_PROJS

MASK_SETS = {
    "read": READ_PROJS,                  # forced reader-side {q,k,v,gate,up}
    "write": WRITE_PROJS,                # forced writer-side {o,down}
    "full": READ_PROJS + WRITE_PROJS,    # the published (all-projection) setting
}


def select_target_modules(model, mask):
    """Select module names matching the requested read/write/full mask."""
    # Validate mask and collect matching linear / quantized linear names
    if mask not in MASK_SETS:
        raise ValueError(f"mask must be one of {list(MASK_SETS)}, got {mask!r}")
    wanted = MASK_SETS[mask]
    names = []
    for name, module in model.named_modules():
        if any(name.endswith(p) for p in wanted) and (
            hasattr(module, "weight")
            or hasattr(module, "in_features")
            or type(module).__name__ in ("QuantizedLinear", "WQLinear")
        ):
            names.append(name)
    return names


def _infeat_outfeat(module):
    """Resolve input and output feature sizes from a linear-like module."""
    # Support both HuggingFace and GPTQ-style attribute names
    inf = getattr(module, "in_features", None) or getattr(module, "infeatures", None)
    out = getattr(module, "out_features", None) or getattr(module, "outfeatures", None)
    return inf, out


class LoRALinear(nn.Module):
    # Frozen base + trainable LoRA delta: y = base(x) + scaling * (dropout(x) @ A^T) @ B^T.

    def __init__(self, base, r, alpha, dropout = 0.0):
        """Wrap a linear module with a zero-init LoRA delta."""
        # Freeze the base and create A/B on the same device
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
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device = dev, dtype = dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device = dev, dtype = dt))

        # Zero-init B so the initial delta is exactly zero
        nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(5))
        self.enabled = True   # toggle off to get the DPO reference (base, no adapter)

    def forward(self, x):
        """Forward through base plus optional LoRA delta."""
        # Base path always runs, delta only when enabled
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        xd = self.drop(x).to(self.lora_A.dtype)
        delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * delta.to(base_out.dtype)

    def delta_weight(self):
        """Dense LoRA weight delta: scaling * B @ A with shape [out, in]."""
        # Materialize the low-rank update as a dense matrix
        return self.scaling * (self.lora_B @ self.lora_A)


def _set_module(model, name, new):
    """Replace a nested submodule by dotted name."""
    # Walk parents, supporting integer ModuleList indices
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    setattr(parent, parts[-1], new)


def attach_masked_lora(model, mask, r = 128, alpha = 128.0, dropout = 0.0):
    """Freeze the model and wrap the masked projection set with LoRALinear."""
    # Freeze everything, then wrap only the mask targets
    for p in model.parameters():
        p.requires_grad_(False)
    names = select_target_modules(model, mask)
    wrapped = []
    for name in names:
        base = dict(model.named_modules())[name]
        _set_module(model, name, LoRALinear(base, r = r, alpha = alpha, dropout = dropout))
        wrapped.append(name)
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    return lora_params, wrapped


def set_lora_enabled(model, flag):
    """Toggle every LoRALinear, False yields the DPO reference (base only)."""
    # Toggle the enabled flag on each LoRALinear
    for m in model.modules():
        if type(m).__name__ == "LoRALinear":
            m.enabled = flag


def unwrap_lora(model, wrapped_names):
    """Replace each LoRALinear with its base module after folding."""
    # Restore the (possibly updated) base modules in place
    mods = dict(model.named_modules())
    for name in wrapped_names:
        m = mods.get(name)
        if m is not None and type(m).__name__ == "LoRALinear":
            _set_module(model, name, m.base)
