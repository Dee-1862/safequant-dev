# utils

Shared utilities used across both phases.

**model_loader.py** - CPU-compatible model loading. Use `load_model(model_name, precision)` instead of calling `from_pretrained` directly. Handles the BitsAndBytes float32 compute dtype requirement on CPU and provides `get_fp32_weight()` which works around a BNB CPU bug where `.dequantize()` returns the wrong shape.

**fake_quant.py** - Simulates NF4 quantization in pure PyTorch without BitsAndBytes. Useful when you need to work directly with quantized weights as normal float32 tensors (e.g. for measuring per-row error without the BNB shape complications). Uses the exact NF4 codebook from the QLoRA paper.
