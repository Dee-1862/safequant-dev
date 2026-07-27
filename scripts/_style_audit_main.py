"""Audit style rules on main SafeQuant code (exclude paper_refs)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".git", "venv", ".venv", "outputs", "node_modules", "paper_refs"}


def check(path: Path) -> dict:
    src = path.read_text(encoding = "utf-8", errors = "replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"syntax": str(exc)}
    issues = {"no_doc": [], "bad_doc": []}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and node.name not in ("__init__",):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            issues["no_doc"].append(node.name)
        elif "Inputs:" not in doc or "Outputs:" not in doc:
            issues["bad_doc"].append(node.name)
    return issues


def main() -> None:
    flagged = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if any(p in SKIP for p in path.parts):
            continue
        if path.name.startswith("_style_"):
            continue
        if rel.parts[0] not in ("core", "experiments", "scripts") and rel.name not in (
            "run_pipeline.py", "config.py",
        ):
            continue
        issues = check(path)
        if "syntax" in issues or issues["no_doc"] or issues["bad_doc"]:
            flagged.append((rel, issues))

    nd = sum(len(i.get("no_doc", [])) for _, i in flagged)
    bd = sum(len(i.get("bad_doc", [])) for _, i in flagged)
    print(f"files={len(flagged)} no_doc={nd} bad_doc={bd}")
    for rel, issues in flagged:
        if "syntax" in issues:
            print(f"SYNTAX {rel}: {issues['syntax']}")
            continue
        print(rel)
        if issues["no_doc"]:
            print(f"  no_doc: {issues['no_doc']}")
        if issues["bad_doc"]:
            print(f"  bad_doc: {issues['bad_doc']}")


if __name__ == "__main__":
    main()
