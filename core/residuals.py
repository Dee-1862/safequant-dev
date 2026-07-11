"""Residual-stream extraction and model unwrapping (single canonical impl)."""

from __future__ import annotations

import torch


def get_inner(model):
    """Unwrap to the inner model that has .layers (handles GPTQ wrappers)."""
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner


# Backward-compatible alias used across the codebase during migration.
_get_inner = get_inner


def prompt_text(p, tok) -> str:
    """Format a prompt (str or dict) with the model chat template."""
    if isinstance(p, dict):
        text = p.get("prompt") or p.get("vanilla") or p.get("behavior") or ""
    else:
        text = str(p)
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return text


# Alias for scripts that used format_prompt.
format_prompt = prompt_text


def default_token_pos(model_name: str) -> int:
    """Per-model heuristic for post-instruction residual token position."""
    name = model_name.lower()
    if "llama" in name:
        return -5
    if "phi" in name:
        return -3
    return -1


def extract_last_token_residuals(
    model, tokenizer, prompts, layers, device, token_pos: int = -1
):
    """Returns {layer: list of (hidden,) CPU tensors, one per prompt}."""
    inner = get_inner(model)
    buf = {}

    def make_hook(layer_idx):
        def hook(module, inputs, output):
            r = output[0] if isinstance(output, (tuple, list)) else output
            if r.dim() == 3:
                buf[layer_idx] = r[:, token_pos, :].detach().cpu().float().squeeze(0)

        return hook

    handles = [inner.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    out = {l: [] for l in layers}
    try:
        for p in prompts:
            text = prompt_text(p, tokenizer)
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(device)
            buf.clear()
            with torch.no_grad():
                _ = model(**inputs)
            for l, r in buf.items():
                out[l].append(r)
    finally:
        for h in handles:
            h.remove()
    return out
