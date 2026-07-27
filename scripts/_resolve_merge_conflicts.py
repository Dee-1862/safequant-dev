"""Resolve git_sha merge conflicts in outputs/*.json (keep HEAD: no git_sha)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "outputs"
PATTERN = re.compile(
    r"<<<<<<< HEAD\n=======\n  \"git_sha\": \"REDACTED\",\n>>>>>>> [0-9a-f]+\n"
)


def main() -> None:
    fixed = 0
    unmatched = []
    for path in ROOT.rglob("*.json"):
        text = path.read_text(encoding = "utf-8")
        if "<<<<<<< HEAD" not in text:
            continue
        new = PATTERN.sub("", text)
        if "<<<<<<< HEAD" in new:
            unmatched.append(path)
            continue
        path.write_text(new, encoding = "utf-8")
        fixed += 1
    print(f"fixed {fixed} files")
    if unmatched:
        print(f"UNMATCHED ({len(unmatched)}):")
        for p in unmatched[:10]:
            print(f"  {p.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
