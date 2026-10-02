# S06 — PEP 8, PEP 257, PEP 484, PEP 20

Sources: `peps.python.org/pep-0008/` (style), `pep-0257` (docstrings),
`pep-0484` (type hints), `pep-0020` (Zen of Python).

Notebook-specific deviations from PEP 8 are declared at the end of this document.
A deviation that is declared and consistent is compliant; an undeclared one is a
finding.

---

## PEP 8 — layout

- **S06-C01** 4 spaces per indent level. Never tabs.
- **S06-C02** Continuation lines either align with the opening delimiter or use a
  hanging indent with nothing after the opening bracket.
- **S06-C03** Line length ≤ 79 for code. *(See deviation D1.)*
- **S06-C04** Docstrings and comments ≤ 72 characters. *(See deviation D1.)*
- **S06-C05** No backslash continuation where implicit continuation inside
  brackets is possible.
- **S06-C06** Break before binary operators (Knuth style) for new code.
- **S06-C07** Two blank lines around top-level functions and classes; one around
  methods.
- **S06-C08** UTF-8 source, no encoding declaration.
- **S06-C09** No trailing whitespace anywhere.
- **S06-C10** No form feeds or stray control characters.

## PEP 8 — imports

- **S06-C11** One module per `import` line; `from x import a, b` is acceptable.
- **S06-C12** Imports at the top of the file (or the top of the notebook's setup
  cell), after the docstring, before globals.
- **S06-C13** Grouped standard library → third party → local, blank line between
  groups.
- **S06-C14** Sorted within each group.
- **S06-C15** Absolute imports preferred; no wildcard imports.
- **S06-C16** No unused imports.
- **S06-C17** Module-level dunders (`__all__`) after the docstring, before imports.

## PEP 8 — whitespace

- **S06-C18** No whitespace immediately inside brackets, or before a comma,
  semicolon, or colon.
- **S06-C19** No space before a call's opening paren or an index's opening bracket.
- **S06-C20** One space either side of assignment, comparison, and boolean
  operators.
- **S06-C21** No spaces around `=` for keyword arguments and unannotated defaults;
  **do** use spaces when the parameter is annotated.
- **S06-C22** Slice colons spaced symmetrically, treated as the lowest-priority
  operator.
- **S06-C23** No alignment padding — never multiple spaces to line up `=` or `#`.
- **S06-C24** No semicolons terminating or joining statements.
- **S06-C25** Parentheses used sparingly; `(foo,)` for single-element tuples.

## PEP 8 — comments and naming

- **S06-C26** Comments are complete sentences, capitalised, and explain *why* not
  *what*.
- **S06-C27** A comment that contradicts the code is worse than no comment.
- **S06-C28** Inline comments separated by at least two spaces, `#` then one space.
- **S06-C29** `module_name`, `ClassName`, `function_name`, `GLOBAL_CONSTANT`,
  `local_var_name`, `ExceptionNameError`.
- **S06-C30** Never `l`, `O`, or `I` as a single-character name.
- **S06-C31** Single leading underscore for internal names.
- **S06-C32** Constants at module level, `UPPER_CASE_WITH_UNDERSCORES`.
- **S06-C33** Descriptive names; no unexplained abbreviations.
- **S06-C34** No type embedded in the name (`paths_list`, `id_to_name_dict`).

## PEP 8 — programming recommendations

- **S06-C35** Compare to `None` with `is` / `is not`.
- **S06-C36** `if not seq:` not `if len(seq) == 0:`.
- **S06-C37** Never compare booleans with `==` or `is True`.
- **S06-C38** `isinstance(x, T)` not `type(x) is T`.
- **S06-C39** `str.startswith` / `endswith` rather than slice comparison.
- **S06-C40** Consistent returns: if any `return` has a value, all reachable exits
  return explicitly, including `return None`.
- **S06-C41** Derive exceptions from `Exception`, not `BaseException`; suffix with
  `Error`.
- **S06-C42** Catch specific exceptions. Bare `except:` only to log-and-re-raise or
  clean-up-and-re-raise.
- **S06-C43** Minimal `try` body; use `else` for the code that must not be guarded.
- **S06-C44** Use `with` for files, sockets, and other resources.
- **S06-C45** No `return`/`break`/`continue` in a `finally` that would swallow an
  in-flight exception.
- **S06-C46** No mutable default arguments.
- **S06-C47** `def f(): ...` rather than binding a lambda to a name.
- **S06-C48** No string accumulation with `+=` in a loop; build a list and `join`.
- **S06-C49** Do not rely on CPython-specific optimisations.

## PEP 257 — docstrings

- **S06-C50** Public modules, functions, classes, and methods have docstrings.
- **S06-C51** Triple double quotes always.
- **S06-C52** One-line summary, imperative or descriptive, ending in a period.
- **S06-C53** Multi-line docstrings: summary, blank line, body, closing `"""` on
  its own line.
- **S06-C54** Document arguments, return value, raised exceptions, and side effects
  where non-obvious.

## PEP 484 — type hints

- **S06-C55** Public functions in shared modules are annotated.
- **S06-C56** No implicit optional (`x: str = None`); write `str | None`.
- **S06-C57** Parameterise generics (`dict[str, int]`, not bare `dict`).
- **S06-C58** Annotations must be accurate — a wrong annotation is worse than none.

## PEP 20 — Zen, as reviewable criteria

- **S06-C59** Explicit over implicit.
- **S06-C60** Simple over complex; flat over nested.
- **S06-C61** Readability counts — in a teaching artefact it outranks cleverness.
- **S06-C62** Errors never pass silently, unless explicitly silenced *and* reported.
- **S06-C63** One obvious way to do it — the same task uses the same idiom
  throughout the codebase.

---

## Declared deviations

| ID | Deviation | Rationale |
|---|---|---|
| **D1** | Code lines up to 88 characters; markdown prose up to 88 | Notebook cells are read in a wide viewer; 79 forces awkward breaks in API request bodies that hurt readability. Applied consistently across all files. |
| **D1a** | Markdown **table rows** exempt from the line limit | A GitHub-flavoured Markdown table row cannot be wrapped — a newline ends the row. The alternative (dropping the tables) would cost far more readability than the long line does. Applies only to lines matching `^\|` or `^# \|`; prose and code are still held to D1. |
| **D2** | Imports appear in the notebook's first code cell rather than a file header | Structural property of notebooks. Grouping and sort order (C13, C14) still enforced. |
| **D3** | Module-level executable code in notebook cells | The notebook *is* the script. `if __name__ == '__main__':` is not applicable to `.ipynb`; it **is** required in `.py` helper modules. |
| **D4** | Short names permitted for loop variables in display code | `f`, `r`, `mid` in tight table-printing loops, where the surrounding two lines make the meaning unambiguous. |
| **D5** | Function-local SDK imports in `_shared/mantle.py` (`token`, `client`, `anthropic_client`) | The module is imported by all 26 notebooks, several of which never use the OpenAI or Anthropic SDK. A top-level import would make `import mantle` fail when only a subset of optional SDKs is installed. Documented in the module docstring. Stdlib imports remain at the top. |
