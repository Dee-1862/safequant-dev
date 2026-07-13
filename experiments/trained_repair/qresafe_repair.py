"""Q-Resafe LoRA repair library (mask selection, attach, train helpers).

CONNECTED — do not remove.
  Imported by: qresafe_dpo.py, qresafe_fold.py
  Not a standalone paper endpoint (self-test only via __main__).
"""
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from core.restoration import READ_PROJS, WRITE_PROJS  # noqa: E402

MASK_SETS = {
    "read": READ_PROJS,                  # forced reader-side {q,k,v,gate,up}
    "write": WRITE_PROJS,                # forced writer-side {o,down}
    "full": READ_PROJS + WRITE_PROJS,    # the published (all-projection) setting
}


def select_target_modules(model, mask: str):
    """
    Select the target modules for the repair.
    Inputs:
        model (nn.Module): The model to select the target modules from.
        mask (str): The mask to select the target modules from. The mask must be one of the keys in MASK_SETS.

    Outputs:
        names (list[str]): The names of the target modules.
    """
    # Check if the mask is valid
    if mask not in MASK_SETS:
        raise ValueError(f"mask must be one of {list(MASK_SETS)}, got {mask!r}")
    wanted = MASK_SETS[mask]

    # Select the target modules
    names = []
    for name, module in model.named_modules():
        if any(name.endswith(p) for p in wanted) and (hasattr(module, "weight") or hasattr(module, "in_features") or type(module).__name__ in ("QuantizedLinear", "WQLinear")):
            names.append(name)
    return names


def _infeat_outfeat(module):
    inf = getattr(module, "in_features", None) or getattr(module, "infeatures", None)
    out = getattr(module, "out_features", None) or getattr(module, "outfeatures", None)
    return inf, out


class LoRALinear(nn.Module):
    """
    Wrap a (possibly quantized) linear with a frozen base + trainable LoRA delta: y = base(x) + scaling * (dropout(x) @ A^T) @ B^T. B is zero-init so the initial delta is exactly zero (the policy starts == the frozen base).
    
    Inputs:
        base (nn.Module): The base module to wrap.
        r (int): The rank of the LoRA delta.
        alpha (float): The scaling factor of the LoRA delta.
        dropout (float): The dropout rate of the LoRA delta.

    Outputs:
        None
    """

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        """
        Initialize the LoRALinear.
        Inputs:
            base (nn.Module): The base module to wrap.
            r (int): The rank of the LoRA delta.
            alpha (float): The scaling factor of the LoRA delta.
            dropout (float): The dropout rate of the LoRA delta.

        Outputs:
            None
        """
        # Initialize the LoRALinear
        super().__init__()
        self.base = base

        # Freeze the base parameters
        for p in self.base.parameters():
            p.requires_grad_(False)

        # Get the input and output features
        in_f, out_f = _infeat_outfeat(base)
        self.in_features, self.out_features = in_f, out_f

        # Initialize the LoRA parameters
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
        """
        Forward pass of the LoRALinear.
        Inputs:
            x (torch.Tensor): The input tensor.

        Outputs:
            out (torch.Tensor): The output tensor.
        """
        # Forward pass of the base module
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        
        # Forward pass of the LoRA delta
        xd = self.drop(x).to(self.lora_A.dtype)
        delta = (xd @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.scaling * delta.to(base_out.dtype)

    def delta_weight(self):
        """
        The dense LoRA delta added to the base weight: scaling * B @ A  [out,in].
        
        Outputs:
            delta_weight (torch.Tensor): The dense LoRA delta added to the base weight.
        """
        # Return the dense LoRA delta added to the base weight
        return self.scaling * (self.lora_B @ self.lora_A)


def _set_module(model, name, new):
    """
    Set the module in the model.
    Inputs:
        model (nn.Module): The model to set the module in.
        name (str): The name of the module to set.
        new (nn.Module): The new module to set.

    Outputs:
        None
    """
    # Split the name into parts
    parts = name.split(".")

    # Get the parent module
    parent = model

    # Set the module in the model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    setattr(parent, parts[-1], new)


def attach_masked_lora(model, mask: str, r: int = 128, alpha: float = 128.0, dropout: float = 0.0):
    """
    Attach the masked LoRA to the model.
    Inputs:
        model (nn.Module): The model to attach the masked LoRA to.
        mask (str): The mask to attach the masked LoRA to.
        r (int): The rank of the LoRA delta.
        alpha (float): The scaling factor of the LoRA delta.
        dropout (float): The dropout rate of the LoRA delta.

    Outputs:
        lora_params (list[torch.Tensor]): The parameters of the LoRA delta.
        wrapped (list[str]): The names of the wrapped modules.
    """
    # Freeze the model parameters
    for p in model.parameters():
        p.requires_grad_(False)
    names = select_target_modules(model, mask)

    # Wrap the masked projection set with LoRALinear
    wrapped = []
    for name in names:
        base = dict(model.named_modules())[name]
        _set_module(model, name, LoRALinear(base, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    
    # Get the parameters of the LoRA delta
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    return lora_params, wrapped


def set_lora_enabled(model, flag: bool):
    """
    Toggle every LoRALinear. flag=False gives the DPO reference (base, no adapter).
    Inputs:
        model (nn.Module): The model to toggle the LoRALinear in.
        flag (bool): The flag to toggle the LoRALinear in.

    Outputs:
        None
    """
    # Toggle every LoRALinear
    for m in model.modules():

        # Check if the module is a LoRALinear
        if type(m).__name__ == "LoRALinear":
            m.enabled = flag


def unwrap_lora(model, wrapped_names):
    """Replace each LoRALinear with its (now folded-into) base module, so the model is pure qu  antized with NO FP adapter. Call AFTER folding the delta into codes.
    Inputs:
        model (nn.Module): The model to unwrap the LoRALinear from.
        wrapped_names (list[str]): The names of the wrapped modules.

    Outputs:
        None
    """
    # Replace each LoRALinear with its (now folded-into) base module
    mods = dict(model.named_modules())
    for name in wrapped_names:
        m = mods.get(name)

        # Check if the module is a LoRALinear
        if m is not None and type(m).__name__ == "LoRALinear":
            # Set the module in the model
            _set_module(model, name, m.base)


# --------------------------------------------------------------------------- #
# self-test (CPU, toy model -- no checkpoint)
# --------------------------------------------------------------------------- #
def _toy_llama(d=32, n_layers=2):
    """
    Create a toy llama model.
    Inputs:
        d (int): The dimension of the model.
        n_layers (int): The number of layers in the model.

    Outputs:
        m (nn.Module): The toy llama model.
    """
    # Create a block
    def block():
        b = nn.Module()
        b.self_attn = nn.Module()

        # Create the self-attention projection modules
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(b.self_attn, p, nn.Linear(d, d, bias=False))

        # Create the MLP projection modules
        b.mlp = nn.Module()
        for p in ("gate_proj", "up_proj", "down_proj"):
            setattr(b.mlp, p, nn.Linear(d, d, bias=False))
        return b
    
    # Create the model
    m = nn.Module()
    m.model = nn.Module()
    m.model.layers = nn.ModuleList([block() for _ in range(n_layers)])
    return m


def _self_test():
    """
    Self-test the qresafe_repair.
    Inputs:
        None

    Outputs:
        ok (bool): True if the self-test passed, False otherwise.
    """
    # Set the random seed
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

    # attach reader LoRA and writer projections must remain plain Linear
    m = _toy_llama(d, n_layers)
    lora_params, wrapped = attach_masked_lora(m, "read", r=8, alpha=16.0)
    q = m.model.layers[0].self_attn.q_proj
    o = m.model.layers[0].self_attn.o_proj

    # Check if the reader q_proj is wrapped
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

    # Check if the gradients are on the LoRA parameters
    if q.lora_A.grad is None or q.lora_B.grad is None:
        print("  FAIL: no grad on LoRA params"); ok = False
    base_w = q.base.weight

    # Check if the base weight received a gradient
    if base_w.grad is not None:
        print("  FAIL: frozen base received a gradient"); ok = False
    n_train = sum(p.numel() for p in lora_params)
    print(f"  reader LoRA: {len(wrapped)} modules wrapped, {n_train} trainable params, "
          f"init-delta={d0:.1e}")

    print("\n  REPAIR FOUNDATION GATE:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _self_test() else 1)
