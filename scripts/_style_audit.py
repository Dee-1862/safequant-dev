"""Quick audit for docstring and equals-spacing style rules."""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".git", "venv", ".venv", "outputs", "node_modules"}


def check_file(path: Path) -> dict:
    src = path.read_text(encoding = "utf-8", errors = "replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"syntax": str(exc)}

    issues = {"no_doc": [], "bad_doc": [], "eq_space": []}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        name = node.name
        if name.startswith("_") and name not in ("__init__",):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            issues["no_doc"].append(name)
        elif "Inputs:" not in doc or "Outputs:" not in doc:
            issues["bad_doc"].append(name)

    for i, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.search(r"(?<![<>!=])=(?!=)", line):
            issues["eq_space"].append(i)
    return issues


def main() -> None:
    files = []
    for path in ROOT.rglob("*.py"):
        if any(part in SKIP for part in path.parts):
            continue
        if path.name.startswith("_style_"):
            continue
        files.append(path)

    totals = {"no_doc": 0, "bad_doc": 0, "eq_space": 0}
    flagged = []
    for path in sorted(files):
        issues = check_file(path)
        if "syntax" in issues:
            flagged.append((path, issues))
            continue
        nd = len(issues["no_doc"])
        bd = len(issues["bad_doc"])
        es = len(issues["eq_space"])
        if nd or bd or es:
            totals["no_doc"] += nd
            totals["bad_doc"] += bd
            totals["eq_space"] += es
            flagged.append((path, issues))

    print(
        f"TOTAL no_doc={totals['no_doc']} bad_doc={totals['bad_doc']} "
        f"eq_space_lines={totals['eq_space']} files={len(flagged)}"
    )
    for path, issues in flagged:
        rel = path.relative_to(ROOT)
        if "syntax" in issues:
            print(f"SYNTAX {rel}: {issues['syntax']}")
            continue
        print(rel)
        if issues["no_doc"]:
            print(f"  no_doc: {issues['no_doc']}")
        if issues["bad_doc"]:
            print(f"  bad_doc: {issues['bad_doc']}")
        if issues["eq_space"]:
            sample = issues["eq_space"][:6]
            extra = " ..." if len(issues["eq_space"]) > 6 else ""
            print(f"  eq_space: {sample}{extra}")


if __name__ == "__main__":
    main()
