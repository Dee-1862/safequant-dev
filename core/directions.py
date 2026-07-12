from __future__ import annotations
import numpy as np
import torch


def compute_r_hat(harmful_lasts, harmless_lasts):
    """
    Computes the R-hat vector for the harmful and harmless last tokens.

    Input:
    - harmful_lasts (list): the harmful last tokens
    - harmless_lasts (list): the harmless last tokens

    Output:
    - r_hat (torch.Tensor): the R-hat vector
    """
    # Compute the mean of the harmful and harmless last tokens
    h = torch.stack(harmful_lasts).mean(0)
    s = torch.stack(harmless_lasts).mean(0)

    # Compute the R-hat vector
    r = h - s
    n = r.norm()
    if n > 1e-6:
        return r / n
    else:
        return r


def per_layer_separation(harmful_lasts, harmless_lasts, r_hat):
    """
    
    Return (separation_sigma, harmful_mean, harmless_mean).

    Input:
    - harmful_lasts (list): the harmful last tokens
    - harmless_lasts (list): the harmless last tokens
    - r_hat (torch.Tensor): the R-hat vector

    Output:
    - separation_sigma (float): the separation sigma
    - harmful_mean (float): the harmful mean
    - harmless_mean (float): the harmless mean

    """
    h_scores = []
    s_scores = []
    for t in harmful_lasts:
        h_scores.append(float(t @ r_hat))
    for t in harmless_lasts:
        s_scores.append(float(t @ r_hat))
    h_scores = np.array(h_scores, dtype=np.float64)
    s_scores = np.array(s_scores, dtype=np.float64)
    pooled = np.sqrt((h_scores.var() + s_scores.var()) / 2 + 1e-12)
    sep = (h_scores.mean() - s_scores.mean()) / pooled

    separation_sigma = float(sep)
    harmful_mean = float(np.mean(h_scores))
    harmless_mean = float(np.mean(s_scores))
    return separation_sigma, harmful_mean, harmless_mean


def cos_sim(a, b) -> float:
    """
    Computes the cosine similarity between two tensors.

    Input:
    - a (torch.Tensor): the first tensor
    - b (torch.Tensor): the second tensor

    Output:
    - cos_sim (float): the cosine similarity
    """
    # Convert the tensors to float
    a = a.float()
    b = b.float()

    # Compute the norms of the tensors
    na, nb = a.norm(), b.norm()

    # Check if the norms are too small
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float((a @ b) / (na * nb))
