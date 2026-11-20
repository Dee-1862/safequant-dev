#!/usr/bin/env python
"""Fail (exit 1) if any identifying string appears in the anonymized copy.
Case-insensitive substring match on file names and file contents (text as UTF-8,
binary as latin-1 and UTF-16-LE). Also prints advisory notes about concrete
cluster paths / hostnames.

    python scripts/anon_check.py <copy_root>
"""
import os
import re
import sys

PATTERNS = [
    "Dee-1862", "bhoomi", "deekshith", "drb6015", "psu.edu", "Penn State",
    "Yuchen", "cis240116p", "/jet/home/", "@gmail",
    # GH200 cluster: user paths, partitions, allocations, host names
    "/work/", "ghx4", "bjwh", "bjwg", "dtai-gh", "ncsa.edu", "hsn.cm",
    # the package describes one cluster (H100 and GH200 nodes); never name or count clusters
    "GH200 cluster", "H100 cluster", "DeltaAI", "Bridges",
]
CLUSTER_RE = re.compile(
    r"/ocean/projects/[A-Za-z0-9_]+/[A-Za-z0-9_]+|/jet/home/[A-Za-z0-9_]+"
    r"|[A-Za-z0-9.-]+\.(?:psc|ncsa)\.[a-z]+"
)
PLACEHOLDER_RE = re.compile(r"YOUR_|youruser|\$USER|<")


def main(dest):
    low = [p.lower() for p in PATTERNS]
    hits = 0
    for dp, dns, fns in os.walk(dest):
        for name in fns + dns:
            for p, l in zip(PATTERNS, low):
                if l in name.lower():
                    print(f"  !! name matches '{p}': {os.path.join(dp, name)}")
                    hits += 1
        for fn in fns:
            f = os.path.join(dp, fn)
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                texts = [data.decode("latin-1").lower(), data.decode("utf-16le", "ignore").lower()]
            else:
                texts = [data.decode("utf-8", "ignore").lower()]
            for p, l in zip(PATTERNS, low):
                for t in texts:
                    i = t.find(l)
                    if i >= 0:
                        line = t.count("\n", 0, i) + 1
                        print(f"  !! '{p}' in {os.path.relpath(f, dest)} (line ~{line})")
                        hits += 1
                        break
            # advisory only
            try:
                txt = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for m in CLUSTER_RE.finditer(txt):
                if not PLACEHOLDER_RE.search(m.group(0)):
                    print(f"  note: possible cluster path/host in {os.path.relpath(f, dest)} -> {m.group(0)}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
