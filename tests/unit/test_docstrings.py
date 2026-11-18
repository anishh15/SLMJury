"""Every docstring in the repository's Python code is four lines or fewer."""

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FOLDERS = ("slmjury", "scripts", "experiments", "figures", "tests")
MAX_LINES = 4

FILES = sorted(p for folder in FOLDERS for p in (REPO / folder).rglob("*.py") if "__pycache__" not in p.parts)


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_docstrings_are_short(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    long = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            if doc and len(doc.splitlines()) > MAX_LINES:
                long.append(f"{getattr(node, 'name', '<module>')} ({len(doc.splitlines())} lines)")
    assert not long, f"docstrings over {MAX_LINES} lines: {', '.join(long)}"
