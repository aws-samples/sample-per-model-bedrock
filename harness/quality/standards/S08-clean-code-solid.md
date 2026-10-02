# S08 — Clean Code, SOLID, and design discipline

Sources: Martin, *Clean Code* (2008) and *Clean Architecture* (2017); Martin's
SOLID formulation; Hunt & Thomas, *The Pragmatic Programmer* (DRY, orthogonality);
Fowler, *Refactoring* (code smells catalogue).

These are judgement standards, so each control is written to be **falsifiable** —
a reviewer must be able to point at a line and say yes or no.

---

## Naming

- **S08-C01** A name states intent. If a comment is needed to explain what a name
  means, rename it.
- **S08-C02** No disinformation: `path` holds a path, `models` holds models, a
  function named `get_*` does not mutate.
- **S08-C03** Searchable names for anything referenced more than once. Magic numbers
  become named constants.
- **S08-C04** Consistent vocabulary: one concept, one word, across the whole
  codebase. Not `fetch`, `get`, and `retrieve` for the same operation.
- **S08-C05** Names are pronounceable and unabbreviated.

## Functions

- **S08-C06** Do one thing. A function whose name needs "and" does two things.
- **S08-C07** One level of abstraction per function.
- **S08-C08** Few arguments; prefer keyword-only for optional behaviour flags.
- **S08-C09** No boolean flag arguments that select between two behaviours — that is
  two functions.
- **S08-C10** No side effects the name does not imply.
- **S08-C11** Command-query separation: a function either does something or answers
  something, not both.
- **S08-C12** Prefer exceptions to error return codes; but in a batch/probe context,
  a documented `(status, payload)` result tuple is a legitimate, explicit choice.
- **S08-C13** Extract until you cannot extract further — but not past the point where
  the reader must chase indirection. In teaching code, one level of helper is often
  the right depth.

## Comments

- **S08-C14** Comments compensate for failure to express intent in code. Prefer
  fixing the code.
- **S08-C15** Legitimate comments: intent, clarification of a non-obvious API
  behaviour, warning of consequences, and — central to this codebase — *why the
  endpoint behaves unexpectedly*.
- **S08-C16** No commented-out code. Delete it.
- **S08-C17** No redundant, misleading, or mandated-boilerplate comments.
- **S08-C18** No journal or attribution comments; version control holds that.

## Formatting and structure

- **S08-C19** Related code sits together; vertical distance tracks conceptual
  distance.
- **S08-C20** Dependent functions are declared before use where the language allows,
  and **always** defined before first call in notebook cell order.
- **S08-C21** Consistent structure across sibling files — a reader who has read one
  family notebook can navigate any other.

## Error handling

- **S08-C22** Errors are handled where the handler has enough context to act.
- **S08-C23** Never return `None` to signal an error where the caller will index into
  the result.
- **S08-C24** Never pass `None` as a value where a real object is expected.
- **S08-C25** Error handling is one thing; a function that does work *and* elaborate
  error translation should be split.

## DRY, KISS, YAGNI, orthogonality

- **S08-C26** No duplicated logic within a file. Duplication *across* family
  notebooks is deliberate and required (each must stand alone) — but the duplicated
  block must be **identical**, so a fix applies uniformly.
- **S08-C27** Simplest thing that demonstrates the lesson.
- **S08-C28** No speculative generality: no parameter, branch, or abstraction that
  exists for a case the sample does not show.
- **S08-C29** No dead code, unreachable branches, or unused variables.
- **S08-C30** Orthogonality: changing one notebook must not require changing another.

## SOLID, as applicable to a helper module

- **S08-C31** Single responsibility — `mantle.py` does transport and parsing; it does
  not embed lesson-specific logic.
- **S08-C32** Open/closed — adding a model family requires no edit to the shared
  helper's control flow.
- **S08-C33** Interface segregation — a notebook imports only the helpers it uses.
- **S08-C34** Dependency inversion — notebooks depend on the helper's documented
  contract, not on its internals.

## Refactoring smells to check for explicitly

- **S08-C35** Long function, long parameter list, large class.
- **S08-C36** Duplicated code within a scope.
- **S08-C37** Feature envy, inappropriate intimacy.
- **S08-C38** Primitive obsession where a small structure would read better.
- **S08-C39** Shotgun surgery — one conceptual change requiring many scattered edits.
- **S08-C40** Speculative generality, dead code, comments-as-deodorant.
