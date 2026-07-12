import os
import random as _python_random

import numpy as np
import torch

# THE SEED. Do not change. Every script in SafeQuant v2 uses this single value.
# If you ever need to vary the seed (example: to estimate variance), pass the variant
# explicitly via the script's --seed argument.
# Never change DEFAULT_SEED itself, as that would invalidate all reproductions.

DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    Input:
        seed: integer seed (default 42).
        deterministic: if True, also force torch.use_deterministic_algorithms
            and cudnn.deterministic. This eliminates most non-determinism but
            can disable some fast kernels. Default False (we accept tiny
            non-determinism from atomic adds in attention kernels in exchange
            for speed; reported numbers still reproduce to within +/-1 prompt
            on n_eval=1000).

    Output: the seed that was set.
    """
    _python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required before the first CUDA matmul when deterministic algorithms are on.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
