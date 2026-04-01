# Simulates NF4 quantization in pure PyTorch without needing bitsandbytes.
#
# Why have this when bitsandbytes works on CPU? Two reasons:
# - BitsAndBytes on CPU gives back weights in a weird packed shape that makes
#   direct row/column manipulation painful
# - This version produces standard float32 tensors with the same rounding error,
#   so you can just index into them like normal numpy arrays
#
# The codebook values are taken directly from the QLoRA paper (Dettmers et al. 2023).

import torch
import torch.nn as nn

# 16-point NF4 codebook, optimized for standard-normal weight distributions
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,
     0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
     0.7229568362236023,   1.0,
], dtype=torch.float32)


def fake_quantize_nf4(weight, block_size=64):
    """Quantize then immediately dequantize using the NF4 codebook.

    Returns a float32 tensor of the same shape with NF4-level rounding noise.
    This is not memory-efficient (the weights stay in float32) but it makes
    the quantization error easy to measure and manipulate directly.
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    w = weight.detach().float().flatten()
    n = w.numel()

    # pad so we get clean blocks
    pad = (block_size - n % block_size) % block_size
    w_padded = torch.cat([w, torch.zeros(pad, dtype=torch.float32)]) if pad else w

    blocks = w_padded.reshape(-1, block_size)

    # normalize each block to [-1, 1] by its absolute max
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    blocks_norm = blocks / absmax

    # nearest-neighbour lookup into the codebook
    codebook = NF4_CODEBOOK.to(w.device)
    dists = (blocks_norm.unsqueeze(-1) - codebook).abs()
    idx = dists.argmin(dim=-1)

    dequantized = codebook[idx] * absmax

    result = dequantized.flatten()[:n].reshape(original_shape)
    return result.to(original_dtype)


def apply_fake_nf4_to_model(model, target_modules=None, block_size=64):
    """Apply fake NF4 in-place to all Linear layers (or just the ones in target_modules).

    After this call the model weights carry the same rounding noise as real NF4
    but everything stays in float32 so you can inspect and modify them normally.

    target_modules is a list of name substrings, e.g. ['o_proj', 'down_proj'].
    Pass None to quantize every Linear layer.
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue
        module.weight.data = fake_quantize_nf4(module.weight.data, block_size)
