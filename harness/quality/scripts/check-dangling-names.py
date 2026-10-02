#!/usr/bin/env python3
"""Find names a notebook uses but never binds, without executing it.

Swapping the generated media for committed assets removed several module-level
names (`png`, `teal`, `bands`, `make_png`, ...). One notebook still referenced a
removed name three cells later and only failed on execution, twenty minutes into a
run. This catches that class of break statically, in a second.

Each notebook is treated as one program: cells are concatenated in order, so a name
bound in cell 3 counts as defined in cell 9, which is how a notebook actually runs.

    python3 check-dangling-names.py [notebook ...]
"""
import ast
import builtins
import glob
import json
import os
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# Names the kernel or an earlier `!pip`/magic line provides.
PRESUPPLIED = {"get_ipython", "display", "In", "Out", "exit", "quit"}


class Scope(ast.NodeVisitor):
    """Collect bound names and loaded names at module level."""

    def __init__(self) -> None:
        self.bound: set[str] = set()
        self.loaded: list[tuple[str, int]] = []

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.bound.add(node.id)
        else:
            self.loaded.append((node.id, node.lineno))

    def visit_arg(self, node: ast.arg) -> None:
        self.bound.add(node.arg)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bound.add(node.name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.bound.add(alias.asname or alias.name)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.bound.add(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.bound.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.bound.update(node.names)


def module_scope_names(program: str) -> tuple[set[str], list[tuple[str, int]]]:
    """Names bound at MODULE level, and names loaded at module level.

    The permissive pass below unions every binding anywhere, which is deliberate: it
    keeps false positives near zero. But it cannot see scope, and that let a real bug
    through -- `import re` existed only inside a method, so a module-level
    `re.compile(...)` added later looked bound and would have raised NameError.
    """
    tree = ast.parse(program)
    bound: set[str] = set()
    loaded: list[tuple[str, int]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
            continue  # bodies run later, in their own scope
        else:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    bound.add(sub.id)
                elif isinstance(sub, ast.alias):
                    bound.add(sub.asname or sub.name.split(".")[0])
                elif isinstance(sub, ast.ExceptHandler) and sub.name:
                    # `except X as exc` binds a name that is not an ast.Name node.
                    # Missing this reported `exc` 76 times across the repo.
                    bound.add(sub.name)
        # Loads that happen while the module body executes. A lambda body is NOT
        # one of those: it runs later with its own parameters in scope, so
        # `sorted(x, key=lambda kv: -kv[1])` must not report `kv`. Descend manually
        # rather than with ast.walk so those subtrees can be pruned.
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stack = [node]
            while stack:
                cur = stack.pop()
                if isinstance(cur, ast.Name) and isinstance(cur.ctx, ast.Load):
                    loaded.append((cur.id, cur.lineno))
                for child in ast.iter_child_nodes(cur):
                    if isinstance(
                        child,
                        (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef,
                         ast.ClassDef),
                    ):
                        continue
                    stack.append(child)
    return bound, loaded


def check(path: str) -> list[str]:
    nb = json.load(open(path))
    chunks, offsets = [], []
    line = 1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith(("!", "%")):
            continue
        try:
            ast.parse(src)
        except SyntaxError:
            return [f"cell {i}: does not parse"]
        chunks.append(src)
        offsets.append((line, i))
        line += src.count("\n") + 2

    program = "\n\n".join(chunks)
    scope = Scope()
    scope.visit(ast.parse(program))

    known = scope.bound | set(dir(builtins)) | PRESUPPLIED
    problems = []
    for name, lineno in scope.loaded:
        if name in known:
            continue
        cell = next((c for start, c in reversed(offsets) if start <= lineno), "?")
        problems.append(f"cell {cell}: undefined name {name!r}")

    # Shadowing pass: a module-level assignment that rebinds an IMPORTED name kills
    # the helper for every later cell. `ok = sum(...)` in one cell made cell 21's
    # `ok(code_, first)` raise "'int' object is not callable" two cells later, and
    # nothing static caught it. Only imported names are reported, because notebooks
    # legitimately reassign their own variables.
    imported: set[str] = set()
    for node in ast.walk(ast.parse(program)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])
    for node in ast.parse(program).body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            targets = [i.optional_vars for i in node.items if i.optional_vars]
        # `mod = importlib.reload(mod)` is the documented way to pick up a module
        # rewritten earlier in the same session. It rebinds the name on purpose.
        reload_self = False
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if getattr(func, "attr", None) == "reload":
                reload_self = True
        if reload_self:
            continue
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id in imported:
                    cell = next(
                        (c for start, c in reversed(offsets) if start <= sub.lineno),
                        "?",
                    )
                    problems.append(
                        f"cell {cell}: assignment to {sub.id!r} shadows the imported "
                        f"name for every later cell"
                    )

    # Scope-aware pass, deliberately narrow. Reporting every module-level load that
    # is not module-level bound gave 76 findings across the repo, nearly all from
    # binding forms this pass does not model (`except X as exc`, `with ... as f`,
    # comprehension targets). So report ONLY the shape that actually misleads: a name
    # that IS bound somewhere -- so the permissive pass stays quiet -- but only inside
    # a function or class, while the module body uses it. That is the `import re`
    # inside a method with a module-level `re.compile()` case, and nothing else.
    mod_bound, mod_loaded = module_scope_names(program)
    nested_only = (scope.bound - mod_bound) & {n for n, _ in mod_loaded}
    for name, lineno in mod_loaded:
        if name not in nested_only or name in set(dir(builtins)) | PRESUPPLIED:
            continue
        cell = next((c for start, c in reversed(offsets) if start <= lineno), "?")
        problems.append(
            f"cell {cell}: {name!r} is used while the module body runs but is only "
            f"bound inside a function or class"
        )
    return sorted(set(problems))


def main() -> None:
    targets = sys.argv[1:] or sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb")))
    targets = [t if os.path.isabs(t) else os.path.join(REPO, t) for t in targets]
    total = 0
    for path in targets:
        found = check(path)
        if found:
            total += len(found)
            print(f"\n{os.path.relpath(path, REPO)}")
            for line in found:
                print(f"  {line}")
    print(f"\n{total} dangling name(s) across {len(targets)} notebook(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
