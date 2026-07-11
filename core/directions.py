"""Refusal-direction geometry: r_hat, separation, cosine drift."""
from __future__ import annotations

import numpy as np
import torch


def compute_r_hat(harmful_lasts, harmless_lasts):
    h = torch.stack(harmful_lasts).mean(0)
    s = torch.stack(harmless_lasts).mean(0)
    r = h - s
    n = r.norm()
    return r / n if n > 1e-6 else r


def per_layer_separation(harmful_lasts, harmless_lasts, r_hat):
    """Return (separation_sigma, harmful_mean, harmless_mean)."""
    h_scores = np.array([float(t @ r_hat) for t in harmful_lasts], dtype=np.float64)
    s_scores = np.array([float(t @ r_hat) for t in harmless_lasts], dtype=np.float64)
    pooled = np.sqrt((h_scores.var() + s_scores.var()) / 2 + 1e-12)
    sep = (h_scores.mean() - s_scores.mean()) / pooled
    return float(sep), float(h_scores.mean()), float(s_scores.mean())


def cos_sim(a, b) -> float:
    a = a.float()
    b = b.float()
    na, nb = a.norm(), b.norm()
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))
