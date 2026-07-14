# SafeQuant v2 -- Seed and Reproducibility Strategy

## The seed

Every random operation in SafeQuant v2 uses **seed = 42**.

This single value is defined in `core/seeds.py`:

```python
DEFAULT_SEED = 42
```

Do not change it. If you need to estimate variance across seeds for a specific experiment, pass `--seed <other>` on the command line; the global default stays 42.

## How randomness flows

There are five distinct random surfaces in this codebase. Each is seeded explicitly.

### 1. Dataset shuffling (`core/datasets.py`)

All dataset loaders shuffle their candidate prompts before truncating to `n_samples`. The shuffle uses a fresh `random.Random(42)` instance, *not* the global `random` module state:

```python
random.Random(42).shuffle(items)
if n_samples:
    items = items[:n_samples]
```

Implication: changing the global Python RNG state elsewhere in the codebase does not affect which prompts get loaded. The same first 1000 WildJailbreak prompts, the same first 1000 ARC questions, the same 32 JBB harmful prompts appear in every run.

### 2. Global RNG (`core.seeds.set_global_seed()`)

Every diagnostic script calls `set_global_seed(DEFAULT_SEED)` near the top. This sets:

- Python's `random` module
- NumPy's global RNG
- PyTorch CPU RNG
- PyTorch CUDA RNG on every visible device

```python
from core.seeds import set_global_seed, DEFAULT_SEED
SEED = set_global_seed(DEFAULT_SEED)
```

If you want strict bit-exact determinism (slower, may disable some kernels):

```python
set_global_seed(DEFAULT_SEED, deterministic=True)
```

This enables `torch.use_deterministic_algorithms(True, warn_only=True)` and `cudnn.deterministic = True`. Default is `deterministic=False` -- we accept the +/-1 prompt of CUDA atomic-add noise on n=1000 evaluations in exchange for ~2x speed on attention kernels.

### 3. Random unit-vector controls

Archived directional-ablation scripts use an isolated ``torch.Generator`` so the random direction is identical across runs regardless of prior global RNG calls:

```python
rng = torch.Generator().manual_seed(SEED)
rand_r = torch.randn(hidden, generator=rng)
rand_r = rand_r / rand_r.norm()
```

### 4. Calibration sampling in tuning loops

Codebook tuning samples one prompt per step from the calibration pool. The sampler uses `np.random.default_rng(SEED)`:

```python
rng = np.random.default_rng(SEED)
for step in range(num_steps):
    idx = int(rng.integers(0, len(sequences)))
```

Same 200 step indices on every run, drawn from the same 64 calibration sequences.

### 5. Quantization kernels (CUDA non-determinism)

The one surface we cannot fully control: certain CUDA kernels (notably fused attention and `F.embedding_bag` on GPU) use atomic adds in their backward or accumulation paths. These produce tiny non-deterministic differences in the last few bits of fp16 arithmetic.

On evaluation tasks with n=1000 prompts:
- ASR varies by <= +/-0.1pp (one prompt) across runs of the same code on the same hardware
- ARC varies by <= +/-0.1pp (one question)
- Per-prompt response text is identical for >= 999/1000 prompts; occasionally one prompt's generation diverges

These are below the noise floor of any conclusion we draw. If you need stricter reproducibility, set `deterministic=True` and accept ~2x slowdown on tuning runs.

## Verifying reproducibility

To check that your environment matches ours, run:

```bash
python -c "
import numpy as np
from core.seeds import set_global_seed
set_global_seed()
rng = np.random.default_rng(42)
print(rng.integers(0, 100, size=5).tolist())
"
# Expected output: [8, 77, 65, 43, 43]
```

If the integers differ, NumPy is not in a known state. Check that you're using the env at `/ocean/projects/cis240116p/bhoomire/conda_envs/sq_clean` (NumPy >= 1.21 has stable `Generator` output).

To verify that dataset loaders give the same 1000 prompts:

```bash
python -c "
from core.datasets import load_wildjailbreak_prompts
items = load_wildjailbreak_prompts(n_samples=5, config='eval')
print([it['vanilla'][:40] for it in items])
"
```

The first 5 prompts should be identical across runs and across machines.

## What changing the seed does not affect

These are deterministic regardless of seed:
- Model loading from HuggingFace (the weights are bit-identical across calls)
- Tokenizer output (deterministic given chat template)
- Greedy decoding (no sampling)
- HarmBench classifier output on a fixed input (greedy, temperature=0)

## What changing the seed *does* affect

- The specific 1000 WildJailbreak prompts (or 32 JBB calibration prompts) drawn for the run
- The random unit vector used as a control in directional ablation
- The step-by-step prompt order in the tuning loop (200 steps draws different prompts in a different order)
- Any place using `np.random.default_rng(SEED)` for stochastic decisions

If you want to run a different seed for a variance estimate:

```bash
# In each diagnostic script (already exposed):
python diagnostics/direction_drift.py --seed 123 ...
```

Report the alternate seed explicitly in any results derived from a non-default seed.
