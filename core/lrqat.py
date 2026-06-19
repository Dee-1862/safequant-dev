"""
core/lrqat.py -- Quant-domain low-rank repair (LR-QAT) for SafeQuant v2.

Lifts two pieces from qualcomm-ai-research/LR-QAT (cloned under reference/ for
reading only; we do NOT run from inside it):

  (a) the integer-domain low-rank reparameterization
          W_hat = s * clamp(W_int0 + round_ste(alpha/r * A @ Bt), qmin, qmax)
      where W_int0 is a FROZEN per-output-channel symmetric INT-b PTQ of the
      base weight, s is the FROZEN per-channel scale, and (A, Bt) are the only
      trainable parameters. The low-rank correction lives in the *integer*
      domain (it is added before the round/clamp), which is what keeps the
      repaired weight on the low-bit grid instead of drifting to FP.

  (b) the fuse-at-inference step
          W_int_fused = clamp(W_int0 + round(alpha/r * A @ Bt), qmin, qmax)
      which folds the trained adapter back into the integer grid so the
      deployed weight is GENUINELY INT-b: no surviving FP adapter, no added
      precision. This is what kills the "safety came back because you added
      bits" objection.

Route A1 (hybrid): only the selected (reader OR writer) projections of an
AQLM-2 model are re-expressed on a uniform INT-b grid and repaired; every other
weight stays the original AQLM-2 checkpoint, byte-for-byte.

Because lora_A is initialised to ZEROS, the adapter delta C is exactly 0 at
step 0, so an untrained LRQATLinear reproduces the plain INT-b PTQ control
bit-for-bit. The "untrained INT-b PTQ" control row and the "trained LR-QAT" row
are therefore the same module evaluated before vs after training -- no separate
construction, no confound.

References:
    Bondarenko, Nagel, Blankevoort. "Low-Rank Quantization-Aware Training for
    LLMs." 2024.  github.com/qualcomm-ai-research/LR-QAT
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- straight-through round ---------------------------------------------------
# Forward rounds to nearest integer; backward passes the gradient through
# unchanged (identity). Lifted from LR-QAT quantization/quantizers/rounding_utils.py.
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

    Per-output-channel (whole-row) scales are far too coarse at 2 bits and
    collapse the model; AQLM/GPTQ stay coherent at low bits precisely because
    they scale in small input groups. We do the same: one scale per
    (output channel, contiguous input group of `group_size`). group_size <= 0 or
    >= in_features falls back to one group per row (per-channel).

    Args:
        weight: [out_features, in_features] float weight (dequantized AQLM weight).
        n_bits: target bit-width b (2 for the strict story, 4 for the sweep).
        group_size: input-dim group width for the scale (e.g. 128).

    Returns:
        W_int: [out, in] float tensor of integer codes in [qmin, qmax].
        scale: [out, n_groups] float32 group-wise scales.
        qmin, qmax: integer grid bounds.
        group_size: the effective group size actually used.
    """
    qmax = 2 ** (n_bits - 1) - 1     # b=2 -> 1
    qmin = -(2 ** (n_bits - 1))      # b=2 -> -2
    w = weight.detach().float()
    out_f, in_f = w.shape
    if group_size is None or group_size <= 0 or group_size >= in_f:
        group_size = in_f
    if in_f % group_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by group_size={group_size}")
    n_groups = in_f // group_size

    wg = w.view(out_f, n_groups, group_size)
    amax = wg.abs().amax(dim=2, keepdim=True).clamp_min(eps)        # [out, n_groups, 1]
    scale_g = amax / max(qmax, 1)                                   # symmetric
    W_int = torch.clamp(torch.round(wg / scale_g), qmin, qmax).view(out_f, in_f)
    scale = scale_g.squeeze(2).to(torch.float32)                   # [out, n_groups]
    return W_int, scale, qmin, qmax, group_size


class LRQATLinear(nn.Module):
    """
    Linear layer whose weight is a frozen INT-b grid plus a trainable
    integer-domain low-rank adapter (LR-QAT, W_int variant).

    Forward (training / un-merged):
        C        = (alpha/r) * A @ Bt                # low-rank, FP, [out, in]
        W_int_hat = clamp(W_int0 + round_ste(C), qmin, qmax)
        W_hat     = scale * W_int_hat
        y         = x @ W_hat^T + bias

    After merge_and_requantize() the adapter is folded into W_int0 and the
    forward is just  y = x @ (scale * W_int0)^T + bias  -- a true INT-b weight.
    """

    def __init__(self, weight_fp, bias, n_bits, rank, alpha,
                 device, compute_dtype=torch.float16, group_size=128):
        super().__init__()
        out_f, in_f = weight_fp.shape
        self.in_features = in_f
        self.out_features = out_f
        self.n_bits = int(n_bits)
        self.rank = int(rank)
        self.lora_scaling = float(alpha) / float(rank)

        W_int, scale, qmin, qmax, gs = symmetric_int_ptq(
            weight_fp, n_bits, group_size=group_size)
        self.group_size = int(gs)
        self.n_groups = in_f // self.group_size
        # Frozen integer codes + group-wise scale (buffers: saved, not trained).
        self.register_buffer("W_int", W_int.to(device))
        self.register_buffer("scale", scale.to(device))   # [out, n_groups]
        self.qmin = int(qmin)
        self.qmax = int(qmax)

        # Trainable low-rank adapter, kept in fp32 for stable optimisation.
        # A init zeros + Bt init kaiming  =>  C = A @ Bt = 0 at step 0, so an
        # untrained module == the plain INT-b PTQ control (see module docstring).
        self.lora_A = nn.Parameter(torch.zeros(out_f, self.rank,
                                               device=device, dtype=torch.float32))
        self.lora_Bt = nn.Parameter(torch.empty(self.rank, in_f,
                                                device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_Bt, a=math.sqrt(5))

        if bias is not None:
            self.register_buffer("bias", bias.detach().to(device).to(compute_dtype))
        else:
            self.bias = None

        self.merged = False

    # -- weight construction --------------------------------------------------
    def _delta(self):
        return (self.lora_A @ self.lora_Bt) * self.lora_scaling

    def _scale_full(self):
        """Expand group-wise scale [out, n_groups] -> [out, in]."""
        if self.group_size >= self.in_features:
            return self.scale  # already [out, 1] broadcastable
        return self.scale.repeat_interleave(self.group_size, dim=1)

    def _w_hat(self):
        """Differentiable repaired weight (un-merged path)."""
        W_int_hat = torch.clamp(self.W_int + round_ste(self._delta()),
                                self.qmin, self.qmax)
        return self._scale_full() * W_int_hat

    def forward(self, x):
        if self.merged:
            weight = (self._scale_full() * self.W_int).to(x.dtype)
        else:
            weight = self._w_hat().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    # -- fuse / requantize ----------------------------------------------------
    @torch.no_grad()
    def merge_and_requantize(self):
        """
        Fold the trained adapter into the integer grid (LR-QAT lr_qat_int merge):
            W_int0 <- clamp(W_int0 + round((alpha/r) A @ Bt), qmin, qmax)
        After this the deployed weight is exactly INT-b -- the adapter adds no
        residual precision. Idempotent guard via self.merged.
        """
        if self.merged:
            return
        C = self._delta()
        W_int_fused = torch.clamp(self.W_int + torch.round(C), self.qmin, self.qmax)
        self.W_int.copy_(W_int_fused)
        # zero the adapter so any later forward is pure INT-b regardless of path
        self.lora_A.data.zero_()
        self.merged = True

    # -- bit accounting -------------------------------------------------------
    def effective_bits(self):
        """
        Effective storage bits/weight for THIS matrix once fused:
        n_bits per code + one fp16 scale per group.
        For INT2 with group_size=128 this is 2 + 16/128 = 2.125 bits.
        """
        numel = self.W_int.numel()
        scale_bits = 16 * self.scale.numel()
        return (self.n_bits * numel + scale_bits) / numel

    def trainable_parameters(self):
        return [self.lora_A, self.lora_Bt]

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"n_bits={self.n_bits}, group_size={self.group_size}, "
                f"rank={self.rank}, scaling={self.lora_scaling:.3g}, "
                f"merged={self.merged}")


# -- module tree helpers (kept local so core/ has no diagnostics dependency) ---
def _get_submodule(layer, dotted):
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


def _set_submodule(layer, dotted, new_module):
    parts = dotted.split(".")
    parent = layer
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def _get_inner_model(model):
    """Unwrap to the inner model that owns .layers (e.g. LlamaModel)."""
    inner = model.model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


def inject_lrqat_adapters(model, proj_paths, n_bits, rank, alpha,
                          get_weight_fn, compute_dtype=torch.float16,
                          group_size=128, verbose=True):
    """
    Replace each selected QuantLinear with an LRQATLinear (Route A1).

    For every layer and every projection in `proj_paths`:
      1. dequantize the base (AQLM) weight via get_weight_fn,
      2. PTQ it to a uniform INT-b grid,
      3. wrap it in an LRQATLinear with a fresh (zero-init) low-rank adapter.

    ALL base parameters are frozen first; only the injected adapters end up
    trainable. Returns (injected_modules, trainable_params).
    """
    inner = _get_inner_model(model)
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers

    # Freeze the entire base model -- nothing outside the adapters trains.
    for p in model.parameters():
        p.requires_grad_(False)

    injected = []
    for l in range(n_layers):
        layer = inner.layers[l]
        for proj in proj_paths:
            mod = _get_submodule(layer, proj)
            w = get_weight_fn(mod, as_dtype=torch.float32).to(device)
            bias = getattr(mod, "bias", None)
            bias = bias if (bias is not None) else None
            new = LRQATLinear(w, bias, n_bits=n_bits, rank=rank, alpha=alpha,
                              device=device, compute_dtype=compute_dtype,
                              group_size=group_size)
            _set_submodule(layer, proj, new)
            injected.append(new)

    trainable = []
    for m in injected:
        for p in m.trainable_parameters():
            p.requires_grad_(True)
            trainable.append(p)

    if verbose:
        n_train = sum(p.numel() for p in trainable)
        print(f"  Injected {len(injected)} LRQATLinear adapters "
              f"(n_bits={n_bits}, group_size={group_size}, rank={rank}, alpha={alpha})")
        print(f"  Trainable adapter params: {n_train:,} "
              f"({n_train / 1e6:.2f}M) across {len(proj_paths)} proj types")
    return injected, trainable


def merge_all(injected, verbose=True):
    """Fuse + requantize every injected adapter into its INT-b grid."""
    for m in injected:
        m.merge_and_requantize()
    if verbose:
        print(f"  Fused {len(injected)} adapters back into INT-{injected[0].n_bits} grid")


def injected_effective_bits(injected):
    """Mean effective bits/weight across all injected (repaired) matrices."""
    if not injected:
        return float("nan")
    return sum(m.effective_bits() for m in injected) / len(injected)
