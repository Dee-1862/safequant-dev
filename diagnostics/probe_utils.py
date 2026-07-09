"""probe_utils.py -- shared probe helpers.

Holds per_layer_probe_accuracy so that Test A (probe_causal_ablation.py, kept)
no longer imports from behavioral_outcome_probe.py (Test B, cut). Place this in
diagnostics/ alongside the other scripts.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.seeds import DEFAULT_SEED as SEED
except ModuleNotFoundError:
    SEED = 42


def per_layer_probe_accuracy(residuals_by_layer, labels):
    """5-fold CV accuracy per layer for a logistic-regression probe.

    residuals_by_layer: {layer: [CPU tensor per prompt]}
    labels:             list[int] aligned with the prompt order
    """
    layers = sorted(residuals_by_layer.keys())
    accs = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    y = np.asarray(labels, dtype=np.int64)

    for L in layers:
        X = np.stack([t.numpy() for t in residuals_by_layer[L]]).astype(np.float64)
        if len(np.unique(y)) < 2:            # degenerate single-class label
            accs[L] = float("nan")
            continue
        fold_accs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(C=1.0, max_iter=2000, n_jobs=1,
                                     random_state=SEED)
            clf.fit(X[train_idx], y[train_idx])
            fold_accs.append(clf.score(X[test_idx], y[test_idx]))
        accs[L] = float(np.mean(fold_accs))
    return accs