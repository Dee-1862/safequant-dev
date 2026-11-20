#!/usr/bin/env python
"""Replace cluster names with node-type wording in the text files of an anonymized COPY of
the repo (README, docs, scripts, code comments). Never run this on the real repo.
The copy then reads as one cluster with two node types; it never names or counts clusters.

    DeltaAI / NCSA Delta / Delta  -> the GH200 nodes  (GH200-node before a noun)
    Bridges-2 / Bridges           -> the H100 nodes   (H100-node before a noun)

"Delta" meaning a difference (Delta correction, Delta ASR, Delta vs, Delta-method) is kept,
as are lower-case identifiers such as delta_queue.sh or the "delta" result tag
(_BRIDGES_* shell variables become _H100_*). JSON and
binary files are skipped (scripts/anon_scrub_json.py handles outputs/).

    python scripts/anon_scrub_text.py <copy_root>
"""
import os
import re
import sys

TEXT_EXT = {".md", ".py", ".sh", ".sbatch", ".txt", ".example", ".cfg", ".toml", ".yaml", ".yml"}
NAMES = re.compile(
    r"(?P<the>\b[Tt]he\s+)?"
    r"(?P<name>NCSA\s+DeltaAI|NCSA\s+Delta|DeltaAI|Delta(?!\s*(?:correction|ASR|vs\b|-method))|Bridges-2|Bridges)"
    r"(?!\w)(?P<poss>'s)?"
)
NO_ARTICLE = re.compile(r"(?:'s|\ba|One-time|original|old|passing|its|their)\s+$")
# tidy-ups after replacement ("Delta / DeltaAI", "DeltaAI GH200")
TIDY = ((re.compile(r"\b([Tt]he) GH200 nodes / the GH200 nodes\b"), r"\1 GH200 nodes"),
        (re.compile(r"\bGH200 nodes GH200\b"), "GH200 nodes"),
        (re.compile(r"\bGH200 nodes \(GH200\)"), "GH200 nodes"),
        (re.compile(r"\ba H100\b"), "an H100"),
        (re.compile(r"\b(GH200|H100) nodes-"), r"\1-node-"),
        (re.compile(r"\b(GH200|H100) nodes (?=(?:user|environment|env|install|stack|block|GPU|CPU|smoke|"
                    r"probe|defaults?|headers?|checkpoints?|note|run|rebuilds?|repeats?|correction)\b)"),
         r"\1-node "),
        (re.compile(r"\b_BRIDGES_"), "_H100_"))      # shell variable names in scripts/_env.sh


def _repl(m, text):
    """
    Neutral replacement for one cluster-name match.

    Inputs:
        - m (re.Match): Match of NAMES
        - text (str): Whole text (to look at what precedes the match)

    Outputs:
        - s (str): Replacement
    """
    kind = "H100" if m.group("name").startswith("Bridges") else "GH200"
    target = f"{kind} nodes" + ("'" if m.group("poss") else "")
    if m.group("the"):
        return m.group("the") + target
    before = text[max(0, m.start() - 20):m.start()]
    if NO_ARTICLE.search(before):
        return f"{kind}-node" + ("'s" if m.group("poss") else "")   # adjective: "its GH200-node PPL"
    start = not before.strip() or re.search(r"(?:^|\n)\s*$|[.!?]\s+$|^#\s*$|\n#\s+$", before)
    return ("The " if start else "the ") + target


def scrub_text(text):
    """
    Replace every cluster name in one text.

    Inputs:
        - text (str): File content

    Outputs:
        - text (str): Scrubbed content
    """
    text = NAMES.sub(lambda m: _repl(m, text), text)
    for pat, rep in TIDY:
        text = pat.sub(rep, text)
    return text


def main(dest):
    changed = 0
    for dp, _, fns in os.walk(dest):
        for fn in sorted(fns):
            ext = os.path.splitext(fn)[1] or fn
            if ext not in TEXT_EXT:
                continue
            f = os.path.join(dp, fn)
            with open(f, encoding = "utf-8") as fh:
                raw = fh.read()
            new = scrub_text(raw)
            if new != raw:
                with open(f, "w", encoding = "utf-8") as fh:
                    fh.write(new)
                changed += 1
    print(f"  cluster names replaced in {changed} text file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
