#!/usr/bin/env python3
"""Validate the generated notebook before it reaches a GPU session.

Catches the failures that have already cost real time:
  * a Python syntax error in a cell;
  * a backslash line-continuation inside an IPython `!` shell line, which the magic
    does not honour, so the cell dies with SyntaxError;
  * a `{VAR}` interpolated into a shell line before any cell defines it.

Shell and magic lines are blanked to `pass` *at their original indentation* before
parsing, because `!cmd` is legal inside an `if` block and must stay one.
"""
import ast
import io
import json
import re
import sys

BACKSLASH = chr(92)
IS_SHELL = re.compile(r"^\s*[!%]")


def blank_magics(src: str) -> str:
    out = []
    for line in src.splitlines():
        if IS_SHELL.match(line):
            out.append(" " * (len(line) - len(line.lstrip())) + "pass")
        else:
            out.append(line)
    return "\n".join(out)


def assigned_names(src: str) -> set[str]:
    """Every name a cell binds: assignment (including chained), for, with-as, import."""
    names: set[str] = set()
    try:
        tree = ast.parse(blank_magics(src))
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.comprehension)):
            targets = [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            targets = [node.optional_vars]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((al.asname or al.name).split(".")[0] for al in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def main(path="sprint/kaggle_sweep.ipynb") -> int:
    nb = json.load(io.open(path, encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    print(f"{path}: {len(nb['cells'])} cells, {len(code_cells)} code")

    problems = 0
    defined: set[str] = set()
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        try:
            ast.parse(blank_magics(src))
        except SyntaxError as e:
            problems += 1
            print(f"  cell {i}: SyntaxError line {e.lineno}: {e.msg}")

        here = assigned_names(src)
        for line in src.splitlines():
            if IS_SHELL.match(line):
                if line.rstrip().endswith(BACKSLASH):
                    problems += 1
                    print(f"  cell {i}: shell line continuation (IPython ignores it): {line.strip()[:70]}")
                for name in re.findall(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", line):
                    if name not in defined and name not in here:
                        problems += 1
                        print(f"  cell {i}: shell line uses {{{name}}} before any cell defines it")
        defined |= here

    print("OK" if not problems else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
