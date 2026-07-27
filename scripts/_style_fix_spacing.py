"""Fix keyword-argument spacing: foo=bar -> foo = bar (skip ==, !=, <=, >=)."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {"__pycache__", ".git", "venv", ".venv", "outputs", "node_modules", "paper_refs"}
SKIP_PREFIX = "_style_"

# Identifier = value, not == != <= >=, not already spaced before =
KWARG = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\s*=(?!=)")


def fix_line(line: str) -> str:
    # Leave type hints in def lines that use Name=Default without breaking
    return KWARG.sub(r"\1 = ", line)


def fix_file(path: Path) -> int:
    text = path.read_text(encoding = "utf-8")
    lines = text.splitlines(keepends = True)
    changed = 0
    out = []
    for line in lines:
        new = fix_line(line)
        if new != line:
            changed += 1
        out.append(new)
    if changed:
        path.write_text("".join(out), encoding = "utf-8")
    return changed


def main() -> None:
    total_files = 0
    total_lines = 0
    targets = []
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name.startswith(SKIP_PREFIX):
            continue
        targets.append(path)

    for path in sorted(targets):
        n = fix_file(path)
        if n:
            total_files += 1
            total_lines += n
            print(f"{path.relative_to(ROOT)}: {n} lines")

    print(f"done: {total_lines} lines in {total_files} files")


if __name__ == "__main__":
    main()
