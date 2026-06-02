import os
import random as _python_random

import numpy as np
import torch


# THE SEED. Do not change. Every script in SafeQuant v2 uses this single
# value. If you ever need to vary the seed (e.g., to estimate variance),
# pass the variant explicitly via the script's --seed argument; never
# change DEFAULT_SEED itself, as that would invalidate all reproductions.
DEFAULT_SEED = 42


def set_global_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = False) -> int:
    """Seed Python, NumPy, PyTorch CPU, and PyTorch CUDA RNGs.

    Args:
        seed: integer seed (default 42).
        deterministic: if True, also force torch.use_deterministic_algorithms
            and cudnn.deterministic. This eliminates most non-determinism but
            can disable some fast kernels. Default False (we accept tiny
            non-determinism from atomic adds in attention kernels in exchange
            for speed; reported numbers still reproduce to within +/-1 prompt
            on n_eval=1000).

    Returns: the seed that was set.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    _python_random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def fixed_rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """Return an isolated NumPy Generator with fixed seed.

    Use this for any local randomness (random vectors, random sampling
    inside a function) instead of relying on the global RNG state. This
    way a script's random behavior is independent of how many random
    calls happened before it.
    """
    return np.random.default_rng(seed)


def torch_gen(seed: int = DEFAULT_SEED, device: str = "cpu") -> torch.Generator:
    """Return a fresh torch.Generator seeded deterministically.

    Use for torch.randn(..., generator=gen) where you need an isolated
    PyTorch RNG (e.g., random unit vectors as ablation controls).
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


__all__ = ["DEFAULT_SEED", "set_global_seed", "fixed_rng", "torch_gen"]
