# SafeQuant

Mechanistic study of how quantization affects LLM safety. Paper-facing code lives in `core/` + `experiments/`; Bridges-2 launchers in `scripts/`.

Randomness is pinned to seed **42** — see [docs/SEEDS.md](docs/SEEDS.md).

## Docs

| File | Contents |
|------|----------|
| [docs/RESULTS.md](docs/RESULTS.md) | Provenance table (script → command → output JSON) |
| [docs/PAPER_FIDELITY.md](docs/PAPER_FIDELITY.md) | Arditi / Q-Resafe / LR-QAT fidelity vs cloned repos |
| [docs/SEEDS.md](docs/SEEDS.md) | Seed and reproducibility strategy |
| [scripts/README.md](scripts/README.md) | Bridges-2 sbatch how-to |

## Layout

```
safequant/
  README.md
  config.py                 model aliases, HF_HOME, defaults
  docs/                     RESULTS, PAPER_FIDELITY, SEEDS
  core/                     shared library
  experiments/<claim>/      paper experiments
  scripts/<claim>/          sbatch launchers (mirrors claims)
  scripts/_env.sh           shared Bridges-2 paths
  outputs/<claim>/          result JSONs
  tests/                    regression anchors
```

## Bridges-2 quick start

```bash
cd /jet/home/bhoomire/projects/safequant_v2
bash scripts/setup_env.sh

sbatch scripts/headline/headline_eval_llama.sbatch
sbatch scripts/trained_repair/qresafe_pairs.sbatch
sbatch scripts/trained_repair/qresafe_dpo.sbatch read
```

Model aliases (e.g. `phi3`) and HuggingFace IDs are in [`config.py`](config.py). Caches go to Ocean via `HF_HOME` in `scripts/_env.sh`.

## Smoke test locally

```bash
pytest tests/test_anchors.py -v
python -m experiments.signal_geometry.direction_drift --model llama7b --quantizer aqlm2 --n_calib 20
```
