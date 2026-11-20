#!/bin/bash
# Build a double-blind code package: copies the committed tree (HEAD) to ../safequant_anon/,
# strips identifying material, verifies, and zips to ../safequant_anon.zip.
#
#   bash scripts/make_anon_package.sh            # PYTHON=... to choose the interpreter
#
# The copy is `git archive HEAD`, so only tracked files go in: never .env, scripts/logs/,
# local archives (outputs/**/pieces/, outputs/**/superseded/), caches or uncommitted edits.
# Commit first; the script refuses a tree with uncommitted changes to tracked files.
# Also removed from the copy: .gitattributes, paper_refs/ and the four packaging helpers
# (they contain the forbidden strings they look for).
# In the copy only (originals are never modified):
#   - outputs/**/*.json scrubbed value by value (paths, host names, job / node / user /
#     commit fields): scripts/anon_scrub_json.py
#   - cluster names in text files replaced by neutral wording: scripts/anon_scrub_text.py
# Exits non-zero (and does not zip) if any forbidden string remains.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$(cd "${ROOT}/.." && pwd)/safequant_anon"
ZIP="${DEST}.zip"

if [[ -e "$DEST" || -e "$ZIP" ]]; then
  echo "ERROR: $DEST or $ZIP already exists. Remove it yourself and re-run (this script never deletes)." >&2
  exit 1
fi

if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "ERROR: uncommitted changes to tracked files; commit them first (the package is built from HEAD)." >&2
  exit 1
fi
PY="${PYTHON:-python3}"

echo "== copy HEAD ($(git -C "$ROOT" rev-parse --short HEAD)) -> ${DEST}"
mkdir -p "$DEST"
git -C "$ROOT" archive --format=tar HEAD | ( cd "$DEST" && tar xf - )
rm -rf "$DEST/.gitattributes" "$DEST/paper_refs" \
       "$DEST/scripts/make_anon_package.sh" "$DEST/scripts/anon_check.py" \
       "$DEST/scripts/anon_scrub_json.py" "$DEST/scripts/anon_scrub_text.py"
mkdir -p "$DEST/scripts/logs" && touch "$DEST/scripts/logs/.gitkeep"

echo "== scrub outputs/**/*.json in the copy"
"$PY" "${ROOT}/scripts/anon_scrub_json.py" "$DEST"

echo "== neutral cluster names in text files of the copy"
"$PY" "${ROOT}/scripts/anon_scrub_text.py" "$DEST"

echo "== forbidden-string check (patterns in scripts/anon_check.py)"
if ! "$PY" "${ROOT}/scripts/anon_check.py" "$DEST"; then
  echo "FAILED: identifying strings remain in $DEST. Fix the source, then delete $DEST and re-run. No zip created." >&2
  exit 1
fi
echo "  clean"

echo "== zip"
if command -v zip >/dev/null 2>&1; then
  ( cd "$(dirname "$DEST")" && zip -qrX "$(basename "$ZIP")" "$(basename "$DEST")" )
else
  "$PY" - "$DEST" "$ZIP" <<'PY'
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
