#!/usr/bin/env python
"""Scrub absolute paths / hostnames / user fields from outputs/*.json inside an
anonymized COPY of the repo. Never run this on the real repo.

    python scripts/anon_scrub_json.py <copy_root>
"""
import json
import os
import re
import sys

PATH_RE = re.compile(
    r'(?:(?:/ocean|/jet|/home|/Users|/scratch|/work|/projects|/mnt|/tmp|/u|/data)/[^"\s,\\\]]+)'
    r'|(?:[A-Za-z]:(?:\\\\|\\|/)[^"\s,\]]+)'
)
HOST_RE = re.compile(
    r'\b(?:[A-Za-z0-9-]+\.)*(?:bridges2|delta|ncsa|psc|psu)[A-Za-z0-9.-]*\.(?:edu|org|net)\b', re.I
)
NODE_KEY_RE = re.compile(
    r'("(?:host|hostname|node|nodelist|slurm_nodelist|machine)"\s*:\s*)"[^"]*"', re.I
)
USER_KEY_RE = re.compile(
    r'("(?:user|username|account|slurm_account)"\s*:\s*)"[^"]*"', re.I
)


def _base(m):
    p = m.group(0).replace("\\\\", "/").replace("\\", "/")
    return "<PATH>/" + p.rstrip("/").split("/")[-1]


def main(dest):
    changed = bad = 0
    for dp, _, fns in os.walk(os.path.join(dest, "outputs")):
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            f = os.path.join(dp, fn)
            with open(f, encoding="utf-8", newline="") as fh:
                raw = fh.read()
            new = PATH_RE.sub(_base, raw)
            new = HOST_RE.sub("<HOST>", new)
            new = NODE_KEY_RE.sub(r'\1"<REDACTED>"', new)
            new = USER_KEY_RE.sub(r'\1"<REDACTED>"', new)
            if new == raw:
                continue
            try:
                json.loads(new)
            except Exception as e:  # keep original text rather than corrupt the JSON
                print("  !! scrubbed JSON no longer parses, left as-is in copy:", f, e)
                bad += 1
                continue
            with open(f, "w", encoding="utf-8", newline="") as fh:
                fh.write(new)
            changed += 1
            print("  scrubbed", os.path.relpath(f, dest))
    print(f"  {changed} JSON file(s) scrubbed in copy, {bad} left unscrubbed (parse failure)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
