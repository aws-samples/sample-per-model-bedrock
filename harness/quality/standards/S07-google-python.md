# S07 — Google Python Style Guide

Source: `google.github.io/styleguide/pyguide.html`, Copyright Google, licensed CC BY 3.0. The controls
below paraphrase it.

PEP 8 covers formatting; this guide covers engineering judgement — what to avoid,
how to handle exceptions and global state, when a function is too long. Controls
here are the ones PEP 8 does *not* already state.

---

## Imports

- **S07-C01** `import x` for modules/packages; `from x import y` where `y` is a
  module. Importing individual functions is discouraged except for `typing`,
  `collections.abc`, and `dataclasses`.
- **S07-C02** No relative imports.
- **S07-C03** `as` aliases only for collisions or standard abbreviations.

## Exceptions

- **S07-C04** Reuse built-in exception types where they fit.
- **S07-C05** Never use `assert` for validating preconditions or for control flow —
  the code must still be correct with asserts stripped.
- **S07-C06** Custom exceptions inherit from an existing exception and end in `Error`.
- **S07-C07** Never a catch-all `except:`. `except Exception:` only at a deliberate
  isolation point, which records the failure.
- **S07-C08** Keep `try` bodies minimal; `finally` for cleanup.

## Mutable global state

- **S07-C09** Avoid mutable module-level state. Where unavoidable, prefix with `_`,
  expose through functions, and document why.
- **S07-C10** Module-level constants are encouraged and named in caps.

## Comprehensions, iterators, lambdas

- **S07-C11** Comprehensions: at most one `for` and one filter. Beyond that, use a
  loop. "Optimize for readability, not conciseness."
- **S07-C12** Use default iterators (`for k, v in d.items()`); never mutate a
  container while iterating it.
- **S07-C13** Lambdas are one-liners only; prefer a generator expression or
  `operator.*` over a lambda passed to `map`/`filter`.
- **S07-C14** Conditional expressions only when each of the three parts fits on its
  own line.

## Default arguments and properties

- **S07-C15** No mutable defaults; no defaults evaluated at import time.
- **S07-C16** Properties only for cheap, unsurprising access.

## Truth testing

- **S07-C17** Prefer implicit falsiness, but always `is None` for `None`.
- **S07-C18** Comparing integers explicitly against `0` is correct and preferred
  where `None` and `0` must be distinguished.

## Decorators, threading, power features

- **S07-C19** Avoid `staticmethod`; limit `classmethod` to named constructors.
- **S07-C20** No external dependencies (files, sockets, network) inside a decorator.
- **S07-C21** Do not rely on the atomicity of built-in types across threads; use
  `queue.Queue` or explicit `threading` primitives.
- **S07-C22** Avoid power features: custom metaclasses, bytecode manipulation,
  dynamic inheritance, import hacks, reflective `getattr` tricks, `__del__` cleanup.

## Type annotations

- **S07-C23** Annotate public APIs; use `X | None` explicitly, never implicit optional.
- **S07-C24** Prefer abstract containers in signatures (`Sequence`, `Mapping`) and
  built-in generics (`tuple`) over `typing.Tuple`.
- **S07-C25** Circular imports caused by typing are a code smell, not a thing to
  work around.

## Strings

- **S07-C26** Format with f-strings, `%`, or `.format()`; never build a formatted
  string with `+`.
- **S07-C27** Never accumulate strings with `+=` in a loop.
- **S07-C28** One quote style per file, switched only to avoid escaping.
- **S07-C29** Logging takes a pattern plus arguments, never a pre-rendered f-string.
- **S07-C30** Error messages must state the actual failure condition, mark
  interpolated values clearly, and be greppable.

## Resources

- **S07-C31** Explicitly close files and sockets; prefer `with`.
- **S07-C32** Never rely on `__del__` or object lifetime for cleanup.

## Comments and docstrings

- **S07-C33** Module docstring describes contents and usage.
- **S07-C34** Function docstrings give enough to call the function without reading
  the body, and document argument-mutating side effects.
- **S07-C35** `Args:` / `Returns:` / `Yields:` / `Raises:` sections with consistent
  hanging indent.
- **S07-C36** Pick descriptive or imperative docstring mood and stay consistent
  within a file.
- **S07-C37** Comments "never describe the code" — they explain intent.

## TODO comments

- **S07-C38** Format `# TODO: <reference> - <explanation>`. A sample repository
  should ship with **zero** TODOs; an unresolved TODO in published sample code is
  a finding.

## Naming and structure

- **S07-C39** `.py` extension always; never dashes in module names.
- **S07-C40** Avoid single-character names except counters, `e` in `except`, and
  file handles.
- **S07-C41** Prefer one leading underscore over `__` name mangling.

## Main and function length

- **S07-C42** Executable behaviour in `main()`, guarded by
  `if __name__ == '__main__':` — for `.py` files.
- **S07-C43** Avoid top-level work in an importable module that should not run at
  import time.
- **S07-C44** "Prefer small and focused functions." Past ~40 lines, justify or split.
