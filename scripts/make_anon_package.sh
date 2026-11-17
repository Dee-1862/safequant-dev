#!/bin/bash
# Build a double-blind code package: copies the repo to ../safequant_anon/, strips
# identifying material, verifies, and zips to ../safequant_anon.zip.
#
#   bash scripts/make_anon_package.sh
#
# Excluded from the copy: .git, .gitattributes, .env (and .env.local), scripts/logs/,
# paper_refs/, __pycache__, *.out, *.err, local caches/trash, and the three packaging
# helpers themselves (they contain the forbidden strings they look for).
# JSON files under outputs/ are scrubbed IN THE COPY ONLY (absolute paths, hostnames,
# user/account fields); the originals are never modified.
# Exits non-zero (and does not zip) if any forbidden string remains.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$(cd "${ROOT}/.." && pwd)/safequant_anon"
ZIP="${DEST}.zip"

if [[ -e "$DEST" || -e "$ZIP" ]]; then
  echo "ERROR: $DEST or $ZIP already exists. Remove it yourself and re-run (this script never deletes)." >&2
  exit 1
fi

echo "== copy ${ROOT} -> ${DEST}"
mkdir -p "$DEST"
( cd "$ROOT" && tar cf - \
    --exclude='./.git' \
    --exclude='./.gitattributes' \
    --exclude='./.env' \
    --exclude='./.env.local' \
    --exclude='./scripts/logs' \
    --exclude='./paper_refs' \
    --exclude='./.gyst' --exclude='./.pytest_cache' --exclude='./.ruff_cache' \
    --exclude='./_trash' --exclude='./.trash' --exclude='./repos' \
    --exclude='__pycache__' \
    --exclude='*.out' --exclude='*.err' \
    --exclude='./scripts/make_anon_package.sh' \
    --exclude='./scripts/anon_check.py' \
    --exclude='./scripts/anon_scrub_json.py' \
    . ) | ( cd "$DEST" && tar xf - )
mkdir -p "$DEST/scripts/logs" && touch "$DEST/scripts/logs/.gitkeep"

echo "== scrub outputs/*.json in the copy"
python "${ROOT}/scripts/anon_scrub_json.py" "$DEST"

echo "== forbidden-string check (Dee-1862 bhoomi deekshith drb6015 psu.edu 'Penn State' Yuchen cis240116p /jet/home/ @gmail)"
if ! python "${ROOT}/scripts/anon_check.py" "$DEST"; then
  echo "FAILED: identifying strings remain in $DEST. Fix the source, then delete $DEST and re-run. No zip created." >&2
  exit 1
fi
echo "  clean"

echo "== zip"
if command -v zip >/dev/null 2>&1; then
  ( cd "$(dirname "$DEST")" && zip -qrX "$(basename "$ZIP")" "$(basename "$DEST")" )
else
  python - "$DEST" "$ZIP" <<'PY'
import os, sys, zipfile
d, z = sys.argv[1:3]
with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
    for dp, _, fns in os.walk(d):
        for fn in fns:
            f = os.path.join(dp, fn)
            zf.write(f, os.path.relpath(f, os.path.dirname(d)))
PY
fi
echo "DONE: $ZIP"
echo "Reminder: never link the GitHub repo in the paper; upload this zip only."
