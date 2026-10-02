---
name: gap-analysis
description: How to compare a discovered change against main and every open PR before writing code, so you never duplicate work or contradict an existing note. Use in step 3.
---

# Gap analysis

## Three questions, in order

**1. Does main already cover this?** `Grep` for the model ID, parameter name or API path.
Absence of the string is good evidence of a gap. Presence is not evidence of coverage:
the repository may name a model in a table while saying something about it that is no
longer true.

**2. Does an open PR cover it?** `github_list_open_prs` returns changed files per PR. If a
PR touches the files you were going to touch, read it. Either it covers the change (say so
and stop) or it conflicts (say so and stop; a maintainer resolves that, not you).

**3. Is what the repository says still true?** This is the one that gets missed. A new
model rarely just needs adding; it often makes an existing note false. When a new Claude
version appeared, the real finding was that a note saying the family "cannot be probed"
was wrong, because the 400 behind it happened on only one of the two endpoints.

So for every change, ask: **which note, table row or cell does this make wrong?** Search
for the behaviour, not just the model.

## Where facts live

A fact usually appears in more than one place. Change one and check the others:

- The family notebook itself: its model table (IDs per endpoint), the note beside the
  cell the fact affects, and its **Things that differ** table at the end.
- `README.md`, whose family table names the models, endpoints and APIs per folder.
- `99-cross-cutting/01-choosing-a-model-and-api.ipynb`, whose decision guide and live
  matrices compare families.
- `00-foundations/01-endpoints-auth-and-the-three-paths.ipynb` for endpoints, paths and
  IDs, and `00-foundations/04-bedrock-runtime-converse-and-profiles.ipynb` for Converse
  and inference profiles.
- `_shared/bedrock.py`, whose helper docstrings describe service behaviour.

## Prefer correcting to adding

A repository that adds every new model but never revisits its notes decays into a
confident set of stale statements. If you can do only one thing in a cycle, correct the
wrong note rather than adding the new model, and say that choice in the PR.

## Deciding the shape of the change

For each item you will act on, decide which of these it is, before writing anything:

- **A working cell** — the change alters the code a reader writes (a new model with its
  own request shape, a parameter that must now be sent, a route that replaced another).
  Show the working way; quote the error the old way now produces in a note.
- **A note or a Things that differ row** — the change is a fact a reader needs but that
  does not change the code (a Region, a refusal, a limit). One or two sentences, dated,
  with the error text quoted.
- **A model-table row** — a new model in a family the notebook already covers, following
  its neighbours.

Write down the files, the note or cell, and the measurement that will support it. The
measurement is part of the work; a change without one is not finished.
