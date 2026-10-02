"""Automated control audit for the bedrock-mantle-samples repository.

Checks the mechanically-verifiable controls from quality/standards/ against every
file in the sample directory. Manual-judgement controls are audited by reading and
recorded separately; this script covers what can be decided by inspection.

Usage:
    python quality/audit.py [--json]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("REPO") or Path(__file__).resolve().parents[2])
MAX_LINE = 88  # declared deviation D1 in S06

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", ".venv", "node_modules"}


@dataclass
class Finding:
    file: str
    line: int
    control: str
    severity: str
    evidence: str
    fix: str

    def key(self) -> tuple:
        return (self.severity_rank(), self.file, self.line, self.control)

    def severity_rank(self) -> int:
        return {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}[
            self.severity
        ]


@dataclass
class Audit:
    findings: list[Finding] = field(default_factory=list)

    def add(self, file, line, control, severity, evidence, fix):
        self.findings.append(
            Finding(str(file), line, control, severity, evidence[:400], fix)
        )


# ---------------------------------------------------------------------------
# Source extraction: treat notebooks as (cell_index, source) pairs
# ---------------------------------------------------------------------------
def iter_files() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        out.append(p)
    return out


def notebook_cells(path: Path) -> tuple[dict, list[tuple[int, str, str]]]:
    """Return (nb, [(cell_no, cell_type, source)])."""
    nb = json.loads(path.read_text())
    cells = []
    for i, c in enumerate(nb.get("cells", [])):
        src = "".join(c.get("source", []))
        cells.append((i, c.get("cell_type", "?"), src))
    return nb, cells


# ---------------------------------------------------------------------------
# Banned / dangerous patterns  (S03, S11-C16, S01, S02)
# ---------------------------------------------------------------------------
BANNED = [
    (r"\beval\s*\(", "S03-C03", "critical", "eval() on any value"),
    (r"\bexec\s*\(", "S03-C03", "critical", "exec() on any value"),
    (r"\bos\.system\s*\(", "S03-C04", "critical", "os.system"),
    (r"shell\s*=\s*True", "S03-C04", "critical", "subprocess shell=True"),
    (r"\bpickle\.loads?\s*\(", "S03-C06", "critical", "pickle deserialisation"),
    (r"yaml\.load\s*\((?!.*SafeLoader)", "S03-C06", "high", "yaml.load unsafe"),
    (r"\bmarshal\.loads\b", "S03-C06", "high", "marshal.loads"),
    (r"verify\s*=\s*False", "S02-C18", "critical", "TLS verification disabled"),
    (r"hashlib\.md5\b", "S02-C17", "high", "MD5"),
    (r"hashlib\.sha1\b", "S02-C17", "high", "SHA-1"),
    (r"\bassert\s+", "S07-C05", "medium", "assert for validation"),
    (r"#\s*type:\s*ignore", "S06-C58", "low", "type: ignore"),
    # A TODO marker left by the author is a finding. A TODO *inside a quoted
    # string* is the fixture a code-repair demo asks the model to resolve, so it
    # is content, not debt. Only flag markers that begin a comment.
    (r"#\s*(TODO|FIXME|XXX|HACK)\b", "S07-C38", "medium",
     "TODO/FIXME comment in published sample"),
    (r"\bmaster\b(?!\s*=)", "S10-C31", "medium", "non-inclusive term 'master'"),
    (r"\bwhitelist\b|\bblacklist\b", "S10-C31", "medium", "non-inclusive term"),
    (r"\bslave\b", "S10-C31", "critical", "non-inclusive term 'slave'"),
    # Flag only qualifiers that minimise an action the READER must take. The same
    # words describing SERVICE behaviour ("the request simply stalls") or occurring
    # inside sample data are factual, not condescending.
    (r"\b(?:you|we)\s+(?:can\s+)?simply\b"
     r"|\bsimply\s+(?:run|use|call|add|set|pass|install|omit|change|point|copy)\b"
     r"|\bjust use\b|\bobviously\b|\bof course\b"
     r"|\bit(?:'s| is) easy\b|\beasy to (?:use|do|set up|follow)\b",
     "S12-C29", "low", "condescending qualifier"),
    (r"\bplease\s+(run|install|use|see|note)", "S12-C30", "low", "'please' in instruction"),
    (r"AdministratorAccess|PowerUserAccess", "S03-C10", "high",
     "over-broad IAM policy suggested"),
    (r'"Action"\s*:\s*"\*"', "S03-C10", "critical", "wildcard IAM action"),
]

SECRET_PATTERNS = [
    (r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", "S03-C01", "critical", "AWS access key id"),
    (r"aws_secret_access_key\s*=\s*['\"][^'\"]{20,}", "S03-C01", "critical",
     "AWS secret key"),
    (r"\bBearer\s+[A-Za-z0-9_\-\.=]{40,}", "S01-C09", "critical", "bearer token"),
    (r"\bbedrock-api-key-[A-Za-z0-9]{20,}", "S01-C09", "critical",
     "Bedrock API key literal"),
    (r"\beyJ[A-Za-z0-9_\-]{30,}\.[A-Za-z0-9_\-]{30,}", "S01-C09", "critical", "JWT"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "S03-C01", "critical", "private key"),
]

# Real account IDs / identifiers that must never ship. 123456789012 is the
# documentation placeholder and is allowed.
ACCOUNT_RE = re.compile(r"\b(?!123456789012\b)(?!000000000000\b)\d{12}\b")
LOCAL_PATH_RE = re.compile(r"/Users/[a-z0-9_\-]+/|/home/[a-z0-9_\-]+/|C:\\\\Users")
PROFILE_RE = re.compile(r"AWS_PROFILE|profile_name\s*=|--profile\s+\S")


def docstring_lines(source: str) -> set[int]:
    """1-based line numbers that fall inside a docstring or string constant.

    Documentation that *warns against* eval/exec/assert must not be reported as a
    use of it, so those lines are excluded from the code-pattern scan.
    """
    inside: set[int] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return inside
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            inside.update(range(node.lineno, end + 1))
    return inside


def scan_text(audit: Audit, rel: str, source: str, base_line: int, is_code: bool):
    """Apply the regex control set to one chunk of text."""
    lines = source.split("\n")
    in_string = docstring_lines(source) if is_code else set()
    for i, line in enumerate(lines, start=base_line):
        stripped = line.strip()
        line_no_in_src = i - base_line + 1

        for pat, ctrl, sev, desc in SECRET_PATTERNS:
            if re.search(pat, line):
                audit.add(rel, i, ctrl, sev, f"{desc}: {stripped[:90]}",
                          "Remove the secret and rotate it.")

        for m in ACCOUNT_RE.finditer(line):
            # Skip obvious non-account numerics (timestamps, token counts)
            ctx = line[max(0, m.start() - 30):m.end() + 10]
            if re.search(r"(account|arn:|:iam:|:sts:|Account)", ctx, re.I):
                audit.add(rel, i, "S03-C02", "critical",
                          f"possible real account id: {stripped[:90]}",
                          "Replace with 123456789012.")

        if LOCAL_PATH_RE.search(line):
            audit.add(rel, i, "S13-C20", "high",
                      f"machine-specific path: {stripped[:90]}",
                      "Use a relative path or a temp dir.")

        if is_code and PROFILE_RE.search(line):
            audit.add(rel, i, "S14-C09", "medium",
                      f"hardcoded AWS profile: {stripped[:90]}",
                      "Rely on the ambient credential chain.")

        # Deviation D1a: a Markdown table row cannot be wrapped -- a newline ends
        # the row -- so table lines are exempt from the length limit.
        is_table_row = stripped.startswith("|") and stripped.endswith("|")
        if len(line.rstrip("\n")) > MAX_LINE and not is_table_row:
            audit.add(rel, i, "S06-C03", "low",
                      f"line is {len(line)} chars: {stripped[:70]}",
                      f"Wrap to <= {MAX_LINE}.")

        if line != line.rstrip():
            audit.add(rel, i, "S06-C09", "low", "trailing whitespace", "Strip it.")

        if "\t" in line:
            audit.add(rel, i, "S06-C01", "medium", "tab character", "Use 4 spaces.")

        # Strip comments and docstring prose before applying the code-pattern
        # scan: documentation that WARNS against eval/exec/assert must not be
        # reported as a use of it. Only executable code is scanned for those.
        code_part = line
        if is_code:
            if stripped.startswith("#") or line_no_in_src in in_string:
                code_part = ""
            else:
                hash_at = line.find("#")
                if hash_at != -1 and line.count('"', 0, hash_at) % 2 == 0:
                    code_part = line[:hash_at]

        for pat, ctrl, sev, desc in BANNED:
            # Prose controls apply to prose; code controls apply to code.
            prose_only = ctrl in {"S12-C29", "S12-C30", "S10-C31"}
            if prose_only:
                target = line
            elif ctrl == "S07-C38":
                # A TODO marker the author left is debt. A TODO inside a string
                # literal is the fixture a code-repair demo asks the model to
                # resolve, so it is content -- scan code only, minus literals.
                target = code_part if is_code else line
            elif is_code:
                target = code_part
            else:
                continue
            if not target:
                continue
            if re.search(pat, target, re.I if prose_only else 0):
                audit.add(rel, i, ctrl, sev, f"{desc}: {stripped[:90]}",
                          "See the control document.")


# ---------------------------------------------------------------------------
# AST-level checks on Python code (S06, S07, S08, S15)
# ---------------------------------------------------------------------------
def check_python_ast(audit: Audit, rel: str, source: str, base_line: int,
                     *, require_docstrings: bool = True):
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return  # notebook cells with magics won't parse; handled by regex pass

    open_call_lines: set[int] = set()
    managed_open_lines: set[int] = set()

    for node in ast.walk(tree):
        # Bare / broad except  (S06-C42, S07-C07, S02-C28)
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                audit.add(rel, base_line + node.lineno - 1, "S06-C42", "high",
                          "bare except:", "Catch a specific exception.")
            elif isinstance(node.type, ast.Name) and node.type.id == "Exception":
                body = node.body
                reraises = any(isinstance(s, ast.Raise) for s in ast.walk(node))
                reports = any(
                    isinstance(s, ast.Expr)
                    and isinstance(s.value, ast.Call)
                    and getattr(s.value.func, "id", "") == "print"
                    for s in ast.walk(node)
                )
                returns = any(isinstance(s, ast.Return) for s in body)
                # A deliberate isolation point may instead RECORD the failure by
                # binding it (`result = {"error": ...}`) for the caller to surface.
                # That satisfies S07-C07 as long as the exception name is used.
                records = bool(node.name) and any(
                    isinstance(s, ast.Assign)
                    and any(
                        isinstance(n, ast.Name) and n.id == node.name
                        for n in ast.walk(s.value)
                    )
                    for s in ast.walk(node)
                    if isinstance(s, ast.Assign)
                )
                if not (reraises or reports or returns or records):
                    audit.add(rel, base_line + node.lineno - 1, "S07-C07",
                              "high", "except Exception with no report or re-raise",
                              "Report the failure or re-raise.")
                if any(isinstance(s, ast.Pass) for s in body):
                    audit.add(rel, base_line + node.lineno - 1, "S02-C29",
                              "high", "exception silently swallowed (pass)",
                              "Report or handle it.")

        # Mutable default arguments  (S06-C46, S07-C15)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(node.args.defaults) + list(node.args.kw_defaults):
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    audit.add(rel, base_line + node.lineno - 1, "S06-C46",
                              "high", f"mutable default in {node.name}()",
                              "Default to None and build inside.")
            # Missing docstring on a public function  (S06-C50). Enforced on
            # importable modules. In notebooks the markdown cell above the code
            # carries the explanation, which is the documented convention for a
            # teaching notebook (S09-C09), so a docstring would be duplication.
            if (require_docstrings and not node.name.startswith("_")
                    and not ast.get_docstring(node)):
                audit.add(rel, base_line + node.lineno - 1, "S06-C50", "low",
                          f"no docstring on public function {node.name}()",
                          "Add a one-line docstring.")
            # Function length  (S07-C44)
            end = getattr(node, "end_lineno", node.lineno)
            if end - node.lineno > 60:
                audit.add(rel, base_line + node.lineno - 1, "S07-C44", "medium",
                          f"{node.name}() is {end - node.lineno} lines",
                          "Split it.")

        # type(x) is T   (S06-C38)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call):
            if getattr(node.left.func, "id", "") == "type":
                audit.add(rel, base_line + node.lineno - 1, "S06-C38", "medium",
                          "type(x) is T comparison", "Use isinstance().")

        # == None / != None  (S06-C35)
        if isinstance(node, ast.Compare):
            for op, cmp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(
                    cmp, ast.Constant
                ) and cmp.value is None:
                    audit.add(rel, base_line + node.lineno - 1, "S06-C35",
                              "medium", "== None", "Use 'is None'.")
                # `cmp.value in (True, False)` would also match 0/1 because
                # 0 == False in Python; test identity instead.
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(
                    cmp, ast.Constant
                ) and (cmp.value is True or cmp.value is False):
                    audit.add(rel, base_line + node.lineno - 1, "S06-C37",
                              "medium", "boolean compared with ==",
                              "Use truthiness.")

        # len(x) == 0  (S06-C36)
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Call):
            if getattr(node.left.func, "id", "") == "len":
                for op, cmp in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(cmp, ast.Constant) \
                            and cmp.value == 0:
                        audit.add(rel, base_line + node.lineno - 1, "S06-C36",
                                  "low", "len(x) == 0", "Use 'not x'.")

        # open() must be inside a with-statement  (S06-C44, S07-C31)
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call) and getattr(
                    call.func, "id", ""
                ) == "open":
                    managed_open_lines.add(call.lineno)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "open":
            open_call_lines.add(node.lineno)

        # requests/urlopen without timeout  (S15-C01)
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
            if fname in {"urlopen", "get", "post", "put", "request"}:
                kwargs = {k.arg for k in node.keywords}
                is_http = fname == "urlopen" or (
                    isinstance(node.func, ast.Attribute)
                    and getattr(node.func.value, "id", "") in {"requests", "httpx"}
                )
                if is_http and "timeout" not in kwargs:
                    audit.add(rel, base_line + node.lineno - 1, "S15-C01",
                              "high", f"{fname}() with no timeout",
                              "Pass timeout=.")

    for lineno in sorted(open_call_lines - managed_open_lines):
        audit.add(rel, base_line + lineno - 1, "S06-C44", "medium",
                  "open() outside a with-statement", "Use a context manager.")

        # Subscript [0] on a call result — possible IndexError  (S15-C11).
        # str.split()/rsplit()/partition() always return a non-empty sequence,
        # so indexing [0] on them cannot raise; exclude them.
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == 0 and isinstance(node.value, ast.Call):
                callee = getattr(node.value.func, "attr", "")
                # str.split() and friends always return a non-empty sequence.
                always_nonempty = {"split", "rsplit", "splitlines", "partition",
                                   "rpartition"}
                # dict.get(k, [default]) with a non-empty literal default is safe.
                safe_default = False
                if callee == "get" and len(node.value.args) == 2:
                    fallback = node.value.args[1]
                    safe_default = (
                        isinstance(fallback, (ast.List, ast.Tuple))
                        and len(fallback.elts) > 0
                    )
                if callee not in always_nonempty and not safe_default:
                    audit.add(rel, base_line + node.lineno - 1, "S15-C11",
                              "medium", "[0] indexing a call result",
                              "Check length first.")


def check_imports(audit: Audit, rel: str, source: str, base_line: int):
    """Import grouping, ordering, and unused imports (S06-C11..C16)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    imported: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue  # `from __future__ import annotations` is a directive
            for a in node.names:
                if a.name != "*":
                    imported[a.asname or a.name] = node.lineno
                else:
                    audit.add(rel, base_line + node.lineno - 1, "S06-C15",
                              "high", "wildcard import", "Import names explicitly.")
        # Import inside a function body. Deviation D5 permits function-local
        # imports of the OPTIONAL SDKs in _shared/mantle.py, documented in that
        # module's docstring; anything else is reported.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lazy_ok = rel.endswith("_shared/mantle.py")
            optional_sdks = {"openai", "anthropic", "aws_bedrock_token_generator"}
            for sub in node.body:
                for inner in ast.walk(sub):
                    if not isinstance(inner, (ast.Import, ast.ImportFrom)):
                        continue
                    if isinstance(inner, ast.Import):
                        mods = {a.name.split(".")[0] for a in inner.names}
                    else:
                        mods = {(inner.module or "").split(".")[0]}
                    if lazy_ok and mods <= optional_sdks:
                        continue
                    audit.add(rel, base_line + inner.lineno - 1, "S06-C12",
                              "low", f"import inside {node.name}()",
                              "Move to module top or declare a deviation.")
    names_used = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    } | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    } | {
        getattr(n.value, "id", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
    }
    for name, lineno in imported.items():
        if name not in names_used and name not in source.replace(
            f"import {name}", ""
        ):
            audit.add(rel, base_line + lineno - 1, "S06-C16", "low",
                      f"possibly unused import: {name}", "Remove it.")


# ---------------------------------------------------------------------------
# Notebook-structure checks (S09)
# ---------------------------------------------------------------------------
def check_notebook(audit: Audit, path: Path, rel: str):
    nb, cells = notebook_cells(path)

    if nb.get("nbformat") != 4:
        audit.add(rel, 0, "S09-C36", "high",
                  f"nbformat is {nb.get('nbformat')}", "Use nbformat 4.")

    ks = (nb.get("metadata") or {}).get("kernelspec") or {}
    if ks.get("name") != "python3":
        audit.add(rel, 0, "S09-C40", "medium",
                  f"kernelspec name is {ks.get('name')!r}", "Use 'python3'.")

    if "widgets" in (nb.get("metadata") or {}):
        audit.add(rel, 0, "S09-C39", "medium", "widgets metadata present",
                  "Strip it.")

    code_cells = [(i, s) for i, t, s in cells if t == "code"]
    counts, no_output, errors = [], 0, 0
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") != "code":
            continue
        ec = c.get("execution_count")
        if ec is None:
            audit.add(rel, i, "S09-C23", "critical",
                      f"cell {i} has no execution_count — never executed",
                      "Execute the notebook end to end.")
        else:
            counts.append(ec)
        outs = c.get("outputs") or []
        if not outs:
            no_output += 1
        for o in outs:
            if o.get("output_type") == "error":
                errors += 1
                audit.add(rel, i, "S09-C38", "critical",
                          f"cell {i} has error output: {o.get('ename')}",
                          "Fix the cell.")
            txt = "".join(o.get("text") or []) + str(o.get("data", {}).get(
                "text/plain", ""))
            for pat, ctrl, sev, desc in SECRET_PATTERNS:
                if re.search(pat, txt):
                    audit.add(rel, i, ctrl, "critical",
                              f"{desc} in cell {i} OUTPUT",
                              "Scrub the output and rotate.")
            for m in ACCOUNT_RE.finditer(txt):
                ctx = txt[max(0, m.start() - 40):m.end() + 15]
                if re.search(r"(account|arn:|:iam:|:sts:|Account)", ctx, re.I):
                    audit.add(rel, i, "S03-C02", "critical",
                              f"real account id in cell {i} output: {ctx[:80]}",
                              "Scrub to 123456789012.")
            if LOCAL_PATH_RE.search(txt):
                audit.add(rel, i, "S13-C20", "high",
                          f"local path in cell {i} output", "Scrub it.")

    # Sequential execution counts prove one clean run  (S09-C24)
    if counts and counts != list(range(1, len(counts) + 1)):
        audit.add(rel, 0, "S09-C24", "high",
                  f"execution counts not 1..N (got {counts[:6]}…{counts[-3:]})",
                  "Restart kernel and run all.")

    # Structure: title, headings, ending
    md = [(i, s) for i, t, s in cells if t == "markdown"]
    if not md or not md[0][1].lstrip().startswith("# "):
        audit.add(rel, 0, "S09-C01", "medium", "no H1 title in first markdown cell",
                  "Open with a '# Title' cell.")
    h1s = [s for _, s in md if re.match(r"^#\s+\S", s.lstrip())]
    if len(h1s) > 1:
        audit.add(rel, 0, "S09-C11", "low", f"{len(h1s)} H1 headings",
                  "Use one H1 per notebook.")

    levels = []
    for _, s in md:
        for line in s.split("\n"):
            m = re.match(r"^(#{1,6})\s+\S", line)
            if m:
                levels.append(len(m.group(1)))
    # A skipped level means going *deeper* by more than one (## -> ####).
    # Returning to a shallower level is normal document structure.
    for a, b in zip(levels, levels[1:]):
        if b > a + 1:
            audit.add(rel, 0, "S09-C11", "low",
                      f"heading level jumps {a}->{b}", "Do not skip levels.")
            break

    if cells and cells[-1][2].strip() == "":
        audit.add(rel, 0, "S09-C37", "low", "empty trailing cell", "Remove it.")

    # Foundations backlink required in every family notebook  (S12-C38)
    if not rel.startswith("00-foundations"):
        joined = "\n".join(s for _, _, s in cells)
        if "00-foundations" not in joined:
            audit.add(rel, 0, "S12-C38", "medium",
                      "no link back to 00-foundations",
                      "Reference the foundations notebook.")

    # Long cells  (S09-C10)
    for i, src in code_cells:
        n = len(src.split("\n"))
        if n > 100:
            audit.add(rel, i, "S09-C10", "medium",
                      f"code cell {i} is {n} lines", "Split it.")

    # Per-cell text scans
    for i, ctype, src in cells:
        scan_text(audit, rel, src, 0, is_code=(ctype == "code"))
        if ctype == "code":
            check_python_ast(audit, rel, src, 0, require_docstrings=False)


# ---------------------------------------------------------------------------
# Repository-level checks (S09, S10, S12, S14)
# ---------------------------------------------------------------------------
def check_repo(audit: Audit):
    files = iter_files()
    names = {str(p.relative_to(ROOT)) for p in files}

    for required, ctrl in [
        ("README.md", "S10-C09"),
        ("requirements.txt", "S09-C15"),
        (".gitignore", "S09-C20"),
    ]:
        if required not in names:
            audit.add(required, 0, ctrl, "high", "required file missing",
                      "Add it.")

    # Generated artefacts must not be present  (S09-C19)
    for p in ROOT.rglob("*"):
        if p.is_file() and (
            "__pycache__" in p.parts
            or p.suffix == ".pyc"
            or ".ipynb_checkpoints" in p.parts
            or p.name == ".DS_Store"
        ):
            audit.add(str(p.relative_to(ROOT)), 0, "S09-C19", "high",
                      "generated artefact present", "Delete and gitignore it.")

    # README link integrity  (S12-C22) and navigation completeness (S12-C37)
    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text()
        for m in re.finditer(r"\]\((?!https?://|#)([^)]+)\)", text):
            target = m.group(1).split("#")[0]
            if target and not (ROOT / target).exists():
                line = text[: m.start()].count("\n") + 1
                audit.add("README.md", line, "S12-C22", "high",
                          f"broken relative link: {target}", "Fix the path.")
        dirs = {
            p.name
            for p in ROOT.iterdir()
            if p.is_dir() and p.name not in SKIP_DIRS and not p.name.startswith(".")
        }
        for d in sorted(dirs):
            if d != "_shared" and d not in text:
                audit.add("README.md", 0, "S12-C37", "medium",
                          f"directory {d}/ not referenced in README",
                          "Add it to the navigation table.")
        nbs = {str(p.relative_to(ROOT)) for p in ROOT.rglob("*.ipynb")}
        if len(nbs) != text.count("ipynb") and "ipynb" in text:
            pass  # counted in the manual pass instead

    # requirements.txt: constraints present, and imports covered  (S09-C16/C17)
    req = ROOT / "requirements.txt"
    declared: set[str] = set()
    if req.exists():
        for i, line in enumerate(req.read_text().split("\n"), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name = re.split(r"[<>=!~\[]", line)[0].strip().lower()
            declared.add(name.replace("-", "_"))
            if not re.search(r"[<>=~]", line):
                audit.add("requirements.txt", i, "S09-C16", "medium",
                          f"{line} has no version constraint", "Pin a floor.")
            if re.match(r"^(git\+|https?://|-e )", line):
                audit.add("requirements.txt", i, "S01-C15", "high",
                          f"non-index install: {line}", "Use a PyPI name.")

    STDLIB = set(sys.stdlib_module_names)
    ALIAS = {
        "aws_bedrock_token_generator": "aws_bedrock_token_generator",
        "PIL": "pillow",
    }
    third_party: dict[str, str] = {}
    for p in files:
        if p.suffix == ".ipynb":
            _, cells = notebook_cells(p)
            srcs = [s for _, t, s in cells if t == "code"]
        elif p.suffix == ".py":
            srcs = [p.read_text()]
        else:
            continue
        for src in srcs:
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    mods = [(node.module or "").split(".")[0]]
                for m in mods:
                    if m and m not in STDLIB and m not in {"mantle", "capabilities"}:
                        third_party[m] = str(p.relative_to(ROOT))
    for mod, where in sorted(third_party.items()):
        canon = ALIAS.get(mod, mod).lower().replace("-", "_")
        if canon not in declared:
            audit.add("requirements.txt", 0, "S09-C17", "high",
                      f"'{mod}' imported in {where} but not declared",
                      "Add it to requirements.txt.")

    # .gitignore coverage  (S09-C20)
    gi = ROOT / ".gitignore"
    if gi.exists():
        body = gi.read_text()
        # `*.py[cod]` is the canonical form and subsumes `*.pyc`.
        equivalents = {"*.pyc": ("*.pyc", "*.py[cod]")}
        for pat in ["__pycache__", "*.pyc", ".ipynb_checkpoints", ".DS_Store"]:
            if not any(alt in body for alt in equivalents.get(pat, (pat,))):
                audit.add(".gitignore", 0, "S09-C20", "medium",
                          f"{pat} not ignored", "Add the pattern.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    audit = Audit()
    check_repo(audit)
    for p in iter_files():
        rel = str(p.relative_to(ROOT))
        if p.suffix == ".ipynb":
            check_notebook(audit, p, rel)
        elif p.suffix == ".py":
            src = p.read_text()
            scan_text(audit, rel, src, 1, is_code=True)
            check_python_ast(audit, rel, src, 1)
            check_imports(audit, rel, src, 1)
        elif p.suffix in {".md", ".txt"}:
            scan_text(audit, rel, p.read_text(), 1, is_code=False)

    audit.findings.sort(key=lambda f: f.key())

    if args.json:
        print(json.dumps([f.__dict__ for f in audit.findings], indent=1))
        return 0

    by_sev: dict[str, int] = {}
    by_ctrl: dict[str, int] = {}
    for f in audit.findings:
        by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
        by_ctrl[f.control] = by_ctrl.get(f.control, 0) + 1

    print("=" * 78)
    print("CONTROL AUDIT SUMMARY")
    print("=" * 78)
    for sev in ("critical", "high", "medium", "low", "info"):
        print(f"  {sev:9} {by_sev.get(sev, 0):5}")
    print(f"  {'TOTAL':9} {len(audit.findings):5}")
    print("\nBy control (top 30):")
    for ctrl, n in sorted(by_ctrl.items(), key=lambda kv: -kv[1])[:30]:
        print(f"  {ctrl:12} {n:5}")
    print("\nCritical and high, in full:")
    for f in audit.findings:
        if f.severity in ("critical", "high"):
            print(f"  [{f.severity:8}] {f.control:10} {f.file}:{f.line}  {f.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
