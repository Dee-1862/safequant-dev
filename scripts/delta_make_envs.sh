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
#
# CUDA: the version check runs on the login node, where torch.cuda.is_available() is
# False, so it cannot prove the GPU stack works. The smoke test
# (bash scripts/delta_queue.sh --smoke) is the real check. If smoke fails with a
# "driver too old" / CUDA driver error, reinstall torch for sq_clean from an older-CUDA
# wheel index:
#   TORCH_INDEX=https://download.pytorch.org/whl/cu126 bash scripts/delta_make_envs.sh clean
# TORCH_INDEX     wheel index for torch in sq_clean (default: PyPI, whose 2.9 wheels are
#                 CUDA 12.8). Setting it force-reinstalls torch from that index.
# TORCH_VERSION   torch pin for sq_clean (default 2.9.*). torch 2.9 wheels exist for cu126,
#                 cu128, cu130 only; the cu124 index stops at torch 2.6 (cu121 at 2.5), so
#                 TORCH_INDEX=.../cu124 also needs TORCH_VERSION=2.6.* (outside the
#                 recorded stack; check transformers 4.57 / gptqmodel still import).
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
  local TV="${TORCH_VERSION:-2.9.*}"
  local C="${TMPDIR}/sq_clean_constraints.txt"
  printf 'torch==%s\ntransformers==4.57.*\nnumpy==2.2.*\n' "$TV" > "$C"
  # torch first. Default: PyPI (2.9 wheels bundle the CUDA 12.8 runtime).
  # TORCH_INDEX: a download.pytorch.org/whl/cuXXX index, force-reinstalled so it replaces
  # an existing torch build (the driver-error fix; see header).
  if [[ -n "${TORCH_INDEX:-}" ]]; then
    echo "  torch==${TV} from ${TORCH_INDEX} (force-reinstall)"
    python -m pip install -c "$C" --force-reinstall "torch==${TV}" \
      --index-url "${TORCH_INDEX}" --extra-index-url https://pypi.org/simple
  else
    python -m pip install -c "$C" "torch==${TV}"
  fi
  # main stack from requirements.txt minus its torch line (torch is pinned above via $C),
  # constrained so nothing moves torch/transformers/numpy
  grep -v '^torch==' requirements.txt > "${TMPDIR}/sq_clean_requirements.txt"
  python -m pip install -c "$C" -r "${TMPDIR}/sq_clean_requirements.txt"
  # GPTQ loading backend used by core/model_loader.py via transformers GPTQConfig ->
  # optimum -> gptqmodel. GPTQMODEL_FORCE_BACKEND=torch is set there, so the CUDA
  # extensions are not needed: skip compiling them.
  BUILD_CUDA_EXT=0 python -m pip install -c "$C" --no-build-isolation "gptqmodel==7.1.0"
  # (aqlm[gpu] for the 2-bit cells comes from requirements.txt above)
  # AutoAWQ is only needed for transformers' native AWQ loading; its own dependency pins
  # would downgrade transformers/torch, so install it without dependencies.
  python -m pip install --no-deps "autoawq==0.2.*"

  SQ_TORCH_WANT="${TV%.\*}" python - <<'PY'
import importlib, os, sys
want = {"torch": os.environ["SQ_TORCH_WANT"], "transformers": "4.57", "bitsandbytes": "0.49",
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
print("sq_clean OK (imports + versions)")
PY
  cuda_note
  freeze sq_clean
  conda deactivate
}

cuda_note() {  # driver info if visible; CUDA itself is only proven by the smoke test
  echo "  --- NVIDIA driver ---"
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  /' || true
    nvidia-smi | grep -o "CUDA Version: [0-9.]*" | head -1 | sed 's/^/  driver supports /' || true
  else
    echo "  nvidia-smi not available here (normal on a login node)."
  fi
  echo "  NOTE: CUDA is NOT verified by this script (login nodes have no GPU). The smoke test"
  echo "        (bash scripts/delta_queue.sh --smoke) is the real check. If it fails with a"
  echo "        driver error, rerun:  TORCH_INDEX=https://download.pytorch.org/whl/cu126 \\"
  echo "                              bash scripts/delta_make_envs.sh clean"
  echo "        (cu124/cu121 indexes have no torch 2.9; they also need TORCH_VERSION=2.6.*)"
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
