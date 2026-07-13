"""
Quant-domain low-rank repair (LR-QAT) for SafeQuant v2.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.residuals import get_inner


# -- straight-through round ---------------------------------------------------
# Forward rounds to nearest integer; backward passes the gradient through
class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


round_ste = _RoundSTE.apply


# -- symmetric group-wise integer PTQ -----------------------------------------
def symmetric_int_ptq(weight, n_bits, group_size=128, eps=1e-8):
    """
    Group-wise symmetric signed INT-b post-training quantization.

    Inputs:
        weight (torch.Tensor): The weight to quantize.
        n_bits (int): The number of bits to quantize the weight to.
        group_size (int): The size of the group to quantize the weight to.
        eps (float): The epsilon to use for the quantization.

    Outputs:
        W_int (torch.Tensor): The quantized weight.
        scale (torch.Tensor): The scale of the quantized weight.
    """
    # Initialize the variables
    qmax = 2 ** (n_bits - 1) - 1  # b=2 -> 1
    qmin = -(2 ** (n_bits - 1))  # b=2 -> -2

    # Get the shape of the weight
    w = weight.detach().float()
    out_f, in_f = w.shape

    # Check if the group size is valid
    if group_size is None or group_size <= 0 or group_size >= in_f:
        group_size = in_f

    # Check if the group size is divisible by the number of input features
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    n_groups = in_f // group_size

    # View the weight as a tensor of shape [out, n_groups, group_size]
    wg = w.view(out_f, n_groups, group_size)
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(eps)  # [out, n_groups, 1]

    # Compute the scale of the quantized weight and the quantized weight
    scale_g = amax / max(qmax, 1)  # symmetric
    W_int = torch.clamp(torch.round(wg / scale_g), qmin, qmax).view(out_f, in_f)
    scale = scale_g.squeeze(2).to(torch.float32)

    # Return the quantized weight, the scale, the minimum and maximum values, and the group size
    return W_int, scale, qmin, qmax, group_size


class LRQATLinear(nn.Module):
    """
    Linear layer whose weight is a frozen INT-b grid plus a trainable integer-domain low-rank adapter (LR-QAT, W_int variant).
    """

    def __init__(
        self,
        weight_fp,
        bias,
        n_bits,
        rank,
        alpha,
        device,
        compute_dtype=torch.float16,
        group_size=128,
    ):
    """
    Initialize the LRQATLinear module.
    Inputs:
        weight_fp (torch.Tensor): The weight to quantize.
        bias (torch.Tensor): The bias to use for the linear layer.
        n_bits (int): The number of bits to quantize the weight to.
        rank (int): The rank of the low-rank adapter.
        alpha (float): The scaling factor for the low-rank adapter.
        device (torch.device): The device to use for the linear layer.
        compute_dtype (torch.dtype): The dtype to use for the linear layer.
        group_size (int): The size of the group to quantize the weight to.
    """
    # Initialize the module
        super().__init__()

        # Get the shape of the weight
        out_f, in_f = weight_fp.shape

        # Initialize the variables
        self.in_features = in_f
        self.out_features = out_f
        self.n_bits = int(n_bits)
        self.rank = int(rank)
        self.lora_scaling = float(alpha) / float(rank)

        # Quantize the weight
        W_int, scale, qmin, qmax, gs = symmetric_int_ptq(
            weight_fp, n_bits, group_size=group_size
        )

        # Initialize the group size
        self.group_size = int(gs)
        self.n_groups = in_f // self.group_size

        # Frozen integer codes + group-wise scale (buffers: saved, not trained).
        self.register_buffer("W_int", W_int.to(device))
        self.register_buffer("scale", scale.to(device))  # [out, n_groups]

        # Initialize the minimum and maximum values
        self.qmin = int(qmin)
        self.qmax = int(qmax)

        # Trainable low-rank adapter, kept in fp32 for stable optimisation.
        self.lora_A = nn.Parameter(
            torch.zeros(out_f, self.rank, device=device, dtype=torch.float32)
        )
        self.lora_Bt = nn.Parameter(
            torch.empty(self.rank, in_f, device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_Bt, a=math.sqrt(5))

        # Initialize the bias
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device).to(compute_dtype))
        else:
            self.bias = None

        self.merged = False

    # -- weight construction --------------------------------------------------
    def _delta(self):
        """
        Compute the delta of the low-rank adapter.
        Inputs:
            None
        Outputs:
            delta (torch.Tensor): The delta of the low-rank adapter.
        """
        # Compute the delta of the low-rank adapter
        return (self.lora_A @ self.lora_Bt) * self.lora_scaling

    def _scale_full(self):
        """
        Expand group-wise scale [out, n_groups] -> [out, in].
        Inputs:
            None
        Outputs:
            scale (torch.Tensor): The expanded scale.
        """
        # Expand the scale
        if self.group_size >= self.in_features:
            return self.scale  # already [out, 1] broadcastable
        return self.scale.repeat_interleave(self.group_size, dim=1)

    def _w_hat(self):
        """
        Differentiable repaired weight (un-merged path).
        Inputs:
            None
        Outputs:
            w_hat (torch.Tensor): The repaired weight.
        """
        # Compute the repaired weight
        W_int_hat = torch.clamp(self.W_int + round_ste(self._delta()), self.qmin, self.qmax)
        return self._scale_full() * W_int_hat

    def forward(self, x):
        """
        Forward pass.
        Inputs:
            x (torch.Tensor): The input tensor.
        Outputs:
            y (torch.Tensor): The output tensor.
        """
        # Compute the weight
        if self.merged:
            weight = (self._scale_full() * self.W_int).to(x.dtype)
        else:
            weight = self._w_hat().to(x.dtype)

        # Compute the bias
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    # -- fuse / requantize ----------------------------------------------------
    @torch.no_grad()
    def merge_and_requantize(self):
        """
        Fold the trained adapter into the integer grid (LR-QAT lr_qat_int merge)
        Inputs:
            None
        Outputs:
            None
        """
        # Check if the adapter is already merged
        if self.merged:
            return

        # Compute the delta and fuse the adapter into the integer grid
        C = self._delta()
        W_int_fused = torch.clamp(self.W_int + torch.round(C), self.qmin, self.qmax)
        self.W_int.copy_(W_int_fused)

        # Zero the adapter so any later forward is pure INT-b regardless of path
        self.lora_A.data.zero_()
        self.merged = True

    # -- bit accounting -------------------------------------------------------
    def effective_bits(self):
        """Effective storage bits/weight for THIS matrix once fused: n_bits per code + one fp16 scale per group. For INT2 with group_size=128 this is 2 + 16/128 = 2.125 bits.
        Inputs:
            None
        Outputs:
            effective_bits (float): The effective storage bits/weight.
        """
        # Compute the effective storage bits/weight
        numel = self.W_int.numel()
        scale_bits = 16 * self.scale.numel()
        return (self.n_bits * numel + scale_bits) / numel

    def trainable_parameters(self):
        """
        Get the trainable parameters.
        Inputs:
            None
        Outputs:
            trainable_parameters (list[torch.Tensor]): The trainable parameters.
        """
        # Get the trainable parameters
        return [self.lora_A, self.lora_Bt]

    def extra_repr(self):
        """
        Get the extra representation.
        Inputs:
            None
        Outputs:
            extra_repr (str): The extra representation.
        """
        # Get the extra representation
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"n_bits={self.n_bits}, group_size={self.group_size}, "
            f"rank={self.rank}, scaling={self.lora_scaling:.3g}, "
            f"merged={self.merged}"
        )


# -- module tree helpers (kept local so core/ has no diagnostics dependency) ---
def _get_submodule(layer, dotted):
    """
    Get the submodule.
    Inputs:
        layer (torch.nn.Module): The layer to get the submodule from.
        dotted (str): The dotted path to the submodule.
    Outputs:
        m (torch.nn.Module): The submodule.
    """
    # Get the submodule
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def _set_submodule(layer, dotted, new_module):
    """
    Set the submodule.
    Inputs:
        layer (torch.nn.Module): The layer to set the submodule on.
        dotted (str): The dotted path to the submodule.
        new_module (torch.nn.Module): The new submodule.
    Outputs:
        None
    """
    parts = dotted.split(".")
    # Set the submodule
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)

    setattr(parent, parts[-1], new_module)


def inject_lrqat_adapters(
    model,
    proj_paths,
    n_bits,
    rank,
    alpha,
    get_weight_fn,
    compute_dtype=torch.float16,
    group_size=128,
    verbose=True,
):
    """
    Replace each selected QuantLinear with an LRQATLinear (Route A1).
    Inputs:
        model (torch.nn.Module): The model to inject the LRQATLinear adapters into.
        proj_paths (list[str]): The paths to the projection modules to inject the LRQATLinear adapters into.
        n_bits (int): The number of bits to quantize the weight to.
        rank (int): The rank of the low-rank adapter.
        alpha (float): The scaling factor for the low-rank adapter.
        get_weight_fn (callable): The function to use to get the weight of the module.
        compute_dtype (torch.dtype): The dtype to use for the linear layer.
        group_size (int): The size of the group to quantize the weight to.
        verbose (bool): Whether to print verbose output.

    Outputs:
        injected (list[LRQATLinear]): The injected LRQATLinear adapters.
        trainable (list[torch.Tensor]): The trainable parameters.
    """
    # Get the inner model and the device
    inner = get_inner(model)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    # Freeze the entire base model
    for p in model.parameters():
        p.requires_grad_(False)

    # Initialize the injected list
    injected = []
    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            mod = _get_submodule(layer, proj)
            w = get_weight_fn(mod, as_dtype=torch.float32).to(device)
            bias = getattr(mod, "bias", None)
            bias = bias if (bias is not None) else None
            new = LRQATLinear(
                w,
                bias,
                n_bits=n_bits,
                rank=rank,
                alpha=alpha,
                device=device,
                compute_dtype=compute_dtype,
                group_size=group_size,
            )
            _set_submodule(layer, proj, new)
            injected.append(new)

    # Initialize the trainable list
    trainable = []
    for m in injected:
        for p in m.trainable_parameters():
            p.requires_grad_(True)
            trainable.append(p)

    # Print the verbose output
    if verbose:
        n_train = sum(p.numel() for p in trainable)
        print(
            f"  Injected {len(injected)} LRQATLinear adapters "
            f"(n_bits={n_bits}, group_size={group_size}, rank={rank}, alpha={alpha})"
        )
        print(
            f"  Trainable adapter params: {n_train:,} "
            f"({n_train / 1e6:.2f}M) across {len(proj_paths)} proj types"
        )
    return injected, trainable


def merge_all(injected, verbose=True):
    """
    Fuse + requantize every injected adapter into its INT-b grid.
    Inputs:
        injected (list[LRQATLinear]): The injected LRQATLinear adapters.
        verbose (bool): Whether to print verbose output.
    Outputs:
        None
    """
    # Fuse and requantize every injected adapter into its INT-b grid
    for m in injected:
        m.merge_and_requantize()
    if verbose:
        print(
            f"  Fused {len(injected)} adapters back into INT-{injected[0].n_bits} grid"
        )


def injected_effective_bits(injected):
    """
    Mean effective bits/weight across all injected (repaired) matrices.
    Inputs:
        injected (list[LRQATLinear]): The injected LRQATLinear adapters.
        
    Outputs:
        effective_bits (float): The mean effective bits/weight across all injected (repaired) matrices.
    """
    # Compute the mean effective bits/weight across all injected (repaired) matrices
    if not injected:
        return float("nan")
    return sum(m.effective_bits() for m in injected) / len(injected)
