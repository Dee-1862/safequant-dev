"""Residual-stream extraction and model unwrapping (single canonical impl)."""

from __future__ import annotations

import torch


def get_inner(model):
    """
    Unwrap to the inner model that has .layers (handles GPTQ wrappers).

    Input:
    - model (torch.nn.Module): the model to unwrap

    Output:
    - inner (torch.nn.Module): the inner model
    """
    # Get the inner model
    inner = model.model if hasattr(model, "model") else model

    # Check if the inner model has a model and a layers attribute
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    return inner

def prompt_text(p, tok) -> str:
    """
    Format a prompt (str or dict) with the model chat template.

    Input:
    - p (str or dict): the prompt to format
    - tok (Tokenizer): the tokenizer to use

    Output:
    - text (str): the formatted prompt
    """
    # Get the text of the prompt
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


def default_token_pos(model_name: str) -> int:
    """
    Per-model heuristic for post-instruction residual token position.

    Input:
    - model_name (str): the name of the model

    Output:
    - token_pos (int): the position of the token to extract the residuals from
    """
    # Get the name of the model
    name = model_name.lower()

    # Check if the model is a llama model
    if "llama" in name:
        return -5
    if "phi" in name:
        return -3
    return -1


def extract_last_token_residuals(
    model, tokenizer, prompts, layers, device, token_pos: int = -1
):
    """
    Returns {layer: list of (hidden,) CPU tensors, one per prompt}.

    Input:
    - model (torch.nn.Module): the model to extract the residuals from
    - tokenizer (Tokenizer): the tokenizer to use
    - prompts (list): the prompts to extract the residuals from
    - layers (list): the layers to extract the residuals from
    - device (torch.device): the device to extract the residuals from
    - token_pos (int): the position of the token to extract the residuals from

    Output:
    - out (dict): a dictionary of the residuals for each layer and prompt
    """
    # Get the inner model
    inner = get_inner(model)
    buf = {}

    # Define a function to make a hook for the layer
    def make_hook(layer_idx):
        def hook(module, inputs, output):
            r = output[0] if isinstance(output, (tuple, list)) else output
            if r.dim() == 3:
                buf[layer_idx] = r[:, token_pos, :].detach().cpu().float().squeeze(0)

        return hook

    # Register the hooks for the layers
    handles = []
    for l in layers:
        handles.append(inner.layers[l].register_forward_hook(make_hook(l)))
    out = {l: [] for l in layers}
    try:
        # Extract the residuals for each prompt 
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
