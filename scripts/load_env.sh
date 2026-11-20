#!/bin/bash
# Load repo .env into the shell. Sourced by scripts/_env.sh, submit.sh and setup scripts.
# Variables already set in the environment WIN over .env (so
# `sbatch --export=ALL,CONDA_ENV=...` and one-off `VAR=x bash ...` overrides work),
# matching the Python loader in config.py.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${_ROOT}/.env" ]]; then
  declare -A _SQ_PRESET=()
  while IFS= read -r _k; do
    if [[ -n "${!_k+x}" ]]; then _SQ_PRESET["$_k"]="${!_k}"; fi
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "${_ROOT}/.env")
  set -a
  # shellcheck disable=SC1091
  source "${_ROOT}/.env"
  set +a
  for _k in "${!_SQ_PRESET[@]}"; do
    export "${_k}=${_SQ_PRESET[$_k]}"
  done
  unset _SQ_PRESET _k
fi
