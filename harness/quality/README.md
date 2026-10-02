# quality/

Checkers, scanners and their standards. Start with [`../README.md`](../README.md).

Every checker carries a `--self-test` that plants the defect it claims to catch. A
checker that has never been falsified is not trusted here: two were silently broken when
first written, and one was wrong in a way that would have made ten notebooks look
defective.

`scan-all.py` fails if any scanner did not actually run. That is not decoration:
`detect-secrets` once reported "0 potential secrets" for months while exiting 127,
because it was invoked by bare name outside its virtual environment and the empty output
parsed as a clean result.

## Gotcha: the virtual environment is not relocatable

Moving `harness/` breaks every console script in `.venv/bin`, because their shebangs
carry the absolute interpreter path. The scanner runner then sees `exit 126: bad
interpreter`. Recreate the environment after a move; nothing depends on its contents.

## The lesson the review rounds kept teaching

An expectation derived from the thing it checks is not an expectation. A retry check
that read the same literal set the code reads, a timeout check that read the parameter
rather than the socket: in each case the evidence and the claim shared a source, so they
could never disagree, which is the only thing evidence is for.
