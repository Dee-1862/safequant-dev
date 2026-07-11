"""Per-prompt operative-layer routing analysis (de-confounded null)."""
from __future__ import annotations

import numpy as np


def per_prompt_projections(residuals_by_layer, r_hat, normalize: bool = False):
    """
    Returns (projections ndarray [n_prompts, n_layers], layer list).
    normalize=True uses cosine projection (alignment) instead of raw dot product.
    """
    layers = sorted(residuals_by_layer.keys())
    n_prompts = len(residuals_by_layer[layers[0]])
    out = np.zeros((n_prompts, len(layers)), dtype=np.float64)
    for li, layer_idx in enumerate(layers):
        for pi, t in enumerate(residuals_by_layer[layer_idx]):
            v = float(t @ r_hat)
            if normalize:
                v = v / (float(t.norm()) + 1e-8)
            out[pi, li] = v
    return out, layers


def operative_layer_per_prompt(projections, layers):
    """Layer index per prompt where |projection| is maximal."""
    abs_proj = np.abs(projections)
    argmax_indices = abs_proj.argmax(axis=1)
    return [layers[i] for i in argmax_indices]
