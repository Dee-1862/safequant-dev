"""Collapse accidental double spaces after = from the spacing fixer."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".git", "venv", ".venv", "outputs", "node_modules", "paper_refs"}
FIX = re.compile(r" =  +")


def main() -> None:
    n = 0
    for path in ROOT.rglob("*.py"):
        if any(p in SKIP for p in path.parts):
            continue
        if path.name.startswith("_style_"):
            continue
        text = path.read_text(encoding = "utf-8")
        new = FIX.sub(" = ", text)
        if new != text:
            path.write_text(new, encoding = "utf-8")
            n += 1
    print(f"fixed {n} files")


if __name__ == "__main__":
    main()
