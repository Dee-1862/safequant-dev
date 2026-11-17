#!/bin/bash
# Create the two conda envs on Delta's project filesystem (run once, on a login node).
#
#   bash scripts/delta_make_envs.sh            # both envs
#   bash scripts/delta_make_envs.sh clean      # sq_clean only
#   bash scripts/delta_make_envs.sh autogptq   # sq_autogptq only
#
# Paths come from .env (prefix envs, so they live on the project/work filesystem):
#   CONDA_ENV  -> sq_clean     evaluation stack (everything except GPTQ generation)
#   AGENV      -> sq_autogptq  GPTQ generation only (scripts/quantize_gptq.py)
# Conda comes from CONDA_SH (path to conda.sh) or MODULES (e.g. an anaconda module),
# same as scripts/_env.sh. PIP_CACHE_DIR / TMPDIR default to the env's parent dir so
# the small $HOME quota is not used.
#
# Pins:
#   sq_clean     torch 2.9, transformers 4.57, bitsandbytes 0.49, optimum 2.2.0,
#                gptqmodel 7.1.0 are what the Bridges-2 env recorded (docs/LRQAT_REPAIR.md,
#                docs/MODELS_QUANTIZATION.md, core/model_loader.py load shim). The other
#                pins are minor-level best guesses (see requirements.txt). Replace them
#                with a Bridges `conda env export` if you have one.
#   sq_autogptq  auto-gptq 0.7.1, transformers 4.44.2, torch 2.2.2, numpy<2
#                (auto-gptq 0.7.1 does not support newer transformers/torch).
# Every pip install runs under a constraints file, so a dependency can never move torch,
# transformers or numpy. Each env ends with an import + version check that fails loudly,
# and `pip freeze` is saved to scripts/logs/env_<name>_freeze.txt.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source scripts/load_env.sh

WHICH="${1:-all}"
case "$WHICH" in all|clean|autogptq) ;; *) echo "usage: $0 [all|clean|autogptq]" >&2; exit 2 ;; esac

: "${CONDA_ENV:?set CONDA_ENV in .env (prefix path for sq_clean on the project filesystem)}"
if [[ "$WHICH" != "clean" ]]; then
  : "${AGENV:?set AGENV in .env (prefix path for sq_autogptq on the project filesystem)}"
fi
LOGDIR="${SQ_LOG_DIR:-${ROOT}/scripts/logs}"; mkdir -p "$LOGDIR"

# --- conda --------------------------------------------------------------------
if [[ -n "${MODULES:-}" ]] && command -v module >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  module load ${MODULES}
fi
if [[ -n "${CONDA_SH:-}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
fi
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found; set CONDA_SH or MODULES in .env" >&2; exit 1; }
# conda activate needs the shell hook even when conda came from a module
eval "$(conda shell.bash hook)"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$(dirname "${CONDA_ENV}")/.pip_cache}"
export TMPDIR="${TMPDIR:-$(dirname "${CONDA_ENV}")/.tmp}"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
export PIP_DISABLE_PIP_VERSION_CHECK=1

make_prefix() {  # make_prefix <prefix> <python-version>
  if [[ -d "$1/conda-meta" ]]; then
    echo "  env exists at $1 -> installing into it (pip is idempotent; nothing is deleted)"
  else
    conda create -y -p "$1" "python=$2" pip
  fi
}

freeze() {  # freeze <name>
  local out="${LOGDIR}/env_$1_freeze.txt"
  python -m pip freeze > "$out"
  echo "  pip freeze -> ${out}"
}

# --- sq_clean -------------------------------------------------------------------
make_clean() {
  echo "=== sq_clean -> ${CONDA_ENV} | $(date) ==="
  make_prefix "${CONDA_ENV}" 3.10
  conda activate "${CONDA_ENV}"
  local C="${TMPDIR}/sq_clean_constraints.txt"
  cat > "$C" <<'EOF'
torch==2.9.*
transformers==4.57.*
numpy==2.2.*
EOF
  # torch first (PyPI wheels bundle CUDA 12.x runtime; works with Delta's driver)
  python -m pip install -c "$C" "torch==2.9.*"
  # main stack from requirements.txt, constrained so nothing moves torch/transformers/numpy
  python -m pip install -c "$C" -r requirements.txt
  # GPTQ loading backend used by core/model_loader.py via transformers GPTQConfig ->
  # optimum -> gptqmodel. GPTQMODEL_FORCE_BACKEND=torch is set there, so the CUDA
  # extensions are not needed: skip compiling them.
  BUILD_CUDA_EXT=0 python -m pip install -c "$C" --no-build-isolation "gptqmodel==7.1.0"
  # (aqlm[gpu] for the 2-bit cells comes from requirements.txt above)
  # AutoAWQ is only needed for transformers' native AWQ loading; its own dependency pins
  # would downgrade transformers/torch, so install it without dependencies.
  python -m pip install --no-deps "autoawq==0.2.*"

  python - <<'PY'
import importlib, sys
want = {"torch": "2.9", "transformers": "4.57", "bitsandbytes": "0.49",
        "optimum": "2.2.0", "gptqmodel": "7.1.0"}
bad = []
for mod in ["torch", "transformers", "accelerate", "bitsandbytes", "datasets", "optimum",
            "gptqmodel", "aqlm", "awq", "sklearn", "scipy", "pandas", "numpy", "tqdm",
            "huggingface_hub", "sentencepiece"]:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", "?")
        print(f"  {mod:16s} {v}")
        if mod in want and not str(v).startswith(want[mod]):
            bad.append(f"{mod} {v} (want {want[mod]})")
    except Exception as e:
        bad.append(f"{mod} import failed: {type(e).__name__}: {e}")
import torch
print(f"  torch CUDA build: {torch.version.cuda}  (cuda available here: {torch.cuda.is_available()};"
      f" False is expected on a login node)")
if bad:
    print("SQ_CLEAN CHECK FAILED:\n  " + "\n  ".join(bad)); sys.exit(1)
print("sq_clean OK")
PY
  freeze sq_clean
  conda deactivate
}

# --- sq_autogptq ----------------------------------------------------------------
make_autogptq() {
  echo "=== sq_autogptq -> ${AGENV} | $(date) ==="
  make_prefix "${AGENV}" 3.10
  conda activate "${AGENV}"
  local C="${TMPDIR}/sq_autogptq_constraints.txt"
  cat > "$C" <<'EOF'
torch==2.2.2
transformers==4.44.2
numpy<2
EOF
  # auto-gptq 0.7.1's PyPI wheel is built against torch 2.2 + CUDA 12.1
  python -m pip install -c "$C" "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu121
  python -m pip install -c "$C" "transformers==4.44.2" "numpy<2" "auto-gptq==0.7.1" \
    "datasets" "pandas" "sentencepiece" "accelerate"

  python - <<'PY'
import sys
import torch, transformers, numpy, auto_gptq
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # noqa: F401
got = {"torch": torch.__version__, "transformers": transformers.__version__,
       "numpy": numpy.__version__, "auto_gptq": auto_gptq.__version__}
for k, v in got.items():
    print(f"  {k:14s} {v}")
bad = []
if not got["torch"].startswith("2.2.2"): bad.append("torch != 2.2.2")
if got["transformers"] != "4.44.2": bad.append("transformers != 4.44.2")
if int(got["numpy"].split(".")[0]) >= 2: bad.append("numpy >= 2")
if not got["auto_gptq"].startswith("0.7.1"): bad.append("auto-gptq != 0.7.1")
if bad:
    print("SQ_AUTOGPTQ CHECK FAILED: " + ", ".join(bad)); sys.exit(1)
print("sq_autogptq OK")
PY
  freeze sq_autogptq
  conda deactivate
}

[[ "$WHICH" == "all" || "$WHICH" == "clean" ]] && make_clean
[[ "$WHICH" == "all" || "$WHICH" == "autogptq" ]] && make_autogptq
echo "=== done | $(date) ==="
echo "CONDA_ENV=${CONDA_ENV} and AGENV=${AGENV:-<unset>} are what scripts/_env.sh and quantize_gptq_all.sh use."
