---
name: self-review
description: How to review your own diff adversarially before opening a PR, using the defect shapes this repository has produced before. Use in step 6.
---

# Self-review

Read your own diff as if you were looking for a reason to reject it. You are the last
reviewer before a public repository, and you wrote the change, which makes you the worst
placed to see its faults, so use a checklist rather than judgement.

## The question that finds the most

**For every note or check I added: what would make it come out wrong, and did I try it?**

If the answer is "nothing could", it is not a check. Real examples from this repository:

- A retry policy judged sound from `529 in TRANSIENT`, a literal set containing 529. True
  for any code, including code with the retries deleted.
- A note written from a probe run once in a shell. The fact held on one endpoint and not
  the other, and the note said neither.
- A policy that denied two actions under one service's condition key. It read correctly
  and blocked nothing on the other endpoint.

The common shape: **the evidence and the claim came from the same place, so they could
never disagree.** Write expected values out longhand instead of computing them from the
code under test, and check a fact on every endpoint and Region the note speaks for.

## Checklist

- Is every change a working cell, a dated note or a Things that differ row? Remove any
  probe, sweep or proof you added.
- Does every answer in committed output come from the model, not from the budget? Check
  `finish_reason`, `status` or `stopReason`.
- Is every checkable answer correct, and does the cell compute its verdict?
- Does any prose contradict the output directly beneath it?
- Is each dated note dated, and each error message quoted verbatim?
- Did I add a rule inferred from one example? Test it against every case you can reach.
- Does every cross-reference point at a file and heading that exist?
- Would a reader who copies this cell get a working request, with the right model ID for
  the endpoint it targets? A profile ID looked up in the Region, not a `us.` prefix
  assumed?
- Does every Responses API call set `store`, and does the cell delete what it stored?
- Does every tool loop validate argument names and types, and bound its rounds?
- Does any shell or curl example put a credential in argv, including through a `$VAR`
  the shell expands into an argument?
- Is any printed reply cut mid-word by a slice?
- Did I leave a TODO, a debug print, or a hardcoded path?
- Is anything committed that should not be public: a token, an account ID, a presigned
  URL?
- Is the change the smallest one that keeps the notebook a working sample?

## Fixing what you find

Any fix here sends you back to step 4, then forward through 5 and 6. Do not fix a finding
and go straight to the PR; that is how a "fix" ships broken.

## If you find nothing

Say so explicitly in the PR body, and name what you checked. "Reviewed" is not a claim a
reviewer can weigh; "re-ran both notebooks, every answer matched its known value, and
every finish_reason was stop" is.
