#!/usr/bin/env python
"""Scrub identifying values from outputs/**/*.json inside an anonymized COPY of the repo.
Never run this on the real repo.

Each file is parsed as JSON and its values are walked; the raw text is never edited with a
regex, so model outputs that merely look like paths are left alone. Changes:
  - values of identifying keys (job ids, partition, commit, host/node, user/account)
    become "<REDACTED>"
  - strings that are a path become "<PATH>/<basename>"; absolute Unix paths under a known
    root inside a longer string are replaced the same way
  - cluster host names inside strings become "<HOST>"
A file is rewritten only if something changed, with its original indentation.

    python scripts/anon_scrub_json.py <copy_root>
"""
import json
import os
import re
import sys

REDACT_KEYS = {
    # scheduler / provenance
    "job_id", "slurm_job_id", "computed_by_job", "partition", "git_sha",
    # host / node
    "host", "hostname", "node", "nodelist", "slurm_nodelist",
    # user / account
    "user", "username", "account", "slurm_account",
}
_ROOTS = r"(?:ocean|jet|home|Users|scratch|work|projects|mnt|tmp|u|data)"
PATH_FULL_RE = re.compile(rf"^(?:/{_ROOTS}/\S+|[A-Za-z]:[\\/][^\s]+)$")
PATH_IN_RE = re.compile(rf"(?<![\w.~-])/{_ROOTS}/[^\s\"',;:)\]]+")
HOST_RE = re.compile(
    r"\b(?:[A-Za-z0-9-]+\.)*(?:bridges2|delta|ncsa|psc|psu)[A-Za-z0-9.-]*\.(?:edu|org|net)\b", re.I)


def _base(p):
    """
    Replace a path by <PATH>/<basename>.

    Inputs:
        - p (str): Path

    Outputs:
        - s (str): Redacted path
    """
    return "<PATH>/" + p.replace("\\", "/").rstrip("/").split("/")[-1]


def _scrub_str(s):
    """
    Redact paths and cluster host names in one string value.

    Inputs:
        - s (str): Value

    Outputs:
        - s (str): Scrubbed value
    """
    if PATH_FULL_RE.match(s):
        return _base(s)
    s = PATH_IN_RE.sub(lambda m: _base(m.group(0)), s)
    return HOST_RE.sub("<HOST>", s)


def scrub(obj):
    """
    Walk a parsed JSON value and redact identifying keys and strings.

    Inputs:
        - obj: Parsed JSON (dict, list, str, number, bool or None)

    Outputs:
        - obj: Scrubbed copy
    """
    if isinstance(obj, dict):
        # only scalar values are redacted; a container under such a key (e.g. "host": {...})
        # is walked instead
        return {k: ("<REDACTED>" if k.lower() in REDACT_KEYS and isinstance(v, (str, int, float))
                    and not isinstance(v, bool) else scrub(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_str(obj)
    return obj


def _indent(raw):
    """
    Indentation of a json.dumps file (None for single-line files).

    Inputs:
        - raw (str): File text

    Outputs:
        - indent (int or None): Spaces before the first nested key
    """
    lines = raw.split("\n", 2)
    if len(lines) < 2:
        return None
    return len(lines[1]) - len(lines[1].lstrip(" ")) or None


def main(dest):
    changed = bad = 0
    for dp, _, fns in os.walk(os.path.join(dest, "outputs")):
        for fn in sorted(fns):
            if not fn.endswith(".json"):
                continue
            f = os.path.join(dp, fn)
            with open(f, encoding = "utf-8") as fh:
                raw = fh.read()
            try:
                data = json.loads(raw)
            except ValueError as e:
                print("  !! not valid JSON, left as is in copy:", f, e)
                bad += 1
                continue
            new = scrub(data)
            if new == data:
                continue
            with open(f, "w", encoding = "utf-8") as fh:
                fh.write(json.dumps(new, indent = _indent(raw)))
            changed += 1
            print("  scrubbed", os.path.relpath(f, dest))
    print(f"  {changed} JSON file(s) scrubbed in copy, {bad} not parsed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
