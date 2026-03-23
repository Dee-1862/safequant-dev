"""
utils/fake_quant.py

Simulates NF4 quantization in pure PyTorch — no bitsandbytes or CUDA required.

Why:
  Real BitsAndBytes NF4 works on CPU (v0.49+) but weights come back in an
  awkward packed shape. Fake quantization produces standard float32 tensors
  with the same rounding error, making weight surgery (row / column replacement)
  trivially simple.

  Used by experiments/read_write_restoration.py.

Usage:
    from utils.fake_quant import fake_quantize_nf4, apply_fake_nf4_to_model
"""
import torch
import torch.nn as nn

# 16-point NF4 codebook — optimized for standard-normal weight distributions.
# Source: BitsAndBytes paper (Dettmers et al., 2023 QLoRA).
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,
     0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
     0.7229568362236023,   1.0,
], dtype=torch.float32)


def fake_quantize_nf4(weight: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """
    Quantize-then-dequantize a weight tensor using the NF4 codebook.

    Produces a float32 tensor of the same shape with NF4-level rounding noise,
    identical to what real bitsandbytes NF4 introduces (without the memory saving).

    Args:
        weight:     Any-shape float tensor.
        block_size: Elements per quantization block (bitsandbytes default = 64).

    Returns:
        Float32 tensor, same shape as input, with NF4 quantization error applied.
    """
    original_shape = weight.shape
    original_dtype = weight.dtype
    w = weight.detach().float().flatten()
    n = w.numel()

    # Pad to a multiple of block_size so we can reshape cleanly.
    pad = (block_size - n % block_size) % block_size
    w_padded = torch.cat([w, torch.zeros(pad, dtype=torch.float32)]) if pad else w

    blocks = w_padded.reshape(-1, block_size)          # (n_blocks, block_size)

    # Per-block absolute-max normalization → [-1, 1].
    absmax = blocks.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
    blocks_norm = blocks / absmax

    # Nearest-neighbour lookup on the 16-point codebook.
    # Shape: (n_blocks, block_size, 1) - (16,) = (n_blocks, block_size, 16)
    codebook = NF4_CODEBOOK.to(w.device)
    dists = (blocks_norm.unsqueeze(-1) - codebook).abs()
    idx = dists.argmin(dim=-1)                         # (n_blocks, block_size)

    # Dequantize: map indices → codebook values, then rescale.
    dequantized = codebook[idx] * absmax               # (n_blocks, block_size)

    result = dequantized.flatten()[:n].reshape(original_shape)
    return result.to(original_dtype)


def apply_fake_nf4_to_model(
    model: nn.Module,
    target_modules: list = None,
    block_size: int = 64,
) -> None:
    """
    Apply fake NF4 quantization IN-PLACE to all (or selected) Linear layers.

    Args:
        model:          Model to quantize (modified in-place).
        target_modules: Optional list of name substrings; if None, ALL Linear layers
                        are quantized.  Example: ['o_proj', 'down_proj'].
        block_size:     NF4 block size (default 64 matches bitsandbytes default).

    After this call the model is a normal float32 model but its weights carry the
    same rounding noise as real NF4 quantization.
    """
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if target_modules is not None:
            if not any(t in name for t in target_modules):
                continue
        module.weight.data = fake_quantize_nf4(module.weight.data, block_size)
