---
name: implement-and-verify
description: How to make a change to the samples and verify it works - running notebooks, using the EC2 test host, and checking that each output shows what its prose says. Use in step 4.
---

# Implement and verify

## Make the smallest change that keeps the notebook a working sample

A good change to this repository is usually one of: a new or corrected working cell, a
dated note beside the cell it affects, or a row in the Things that differ table. A bad
change is a probe cell or a sweep added to prove a point, or three paragraphs asserting a
new fact. Measure in your own verification run and put the evidence in the PR body; the
notebook carries the conclusion.

If you are adding a model to an existing family notebook, follow the neighbouring model's
shape exactly, the same cells in the same order, so a reader who knows one knows all.

## Verify against the live service, from a clean machine

Your own container is not a customer's machine. Use the test host:

```
testhost_launch
testhost_run  — clone the repo, pip install -r requirements.txt, run the notebooks you changed
testhost_terminate
```

Keep each `testhost_run` to one logical step so a failure names itself. A useful sequence:

1. `git clone` the branch you are working on (it must be pushed for this, or copy the
   files in with a heredoc).
2. `pip install -r requirements.txt`
3. Run each notebook you changed from top to bottom, from its own directory, for
   example with `jupyter nbconvert --execute --to notebook --inplace`.
4. Print the outputs you care about and read them.

**Terminate the host as soon as you are done.** Not at the end of the cycle — as soon as
this step is finished.

## The check that matters: does each output show what its prose says?

Running is the low bar. Cells have run perfectly and shown nothing useful:

- A vision answer scored 1/3 because it was cut off at the token cap. The number looked
  like a capability measurement; it measured `max_tokens`.
- A tool loop printed an empty answer because the model called the tool again after
  every result until the loop's bound ran out, and nothing said the loop had given up.
- A structured-output cell printed valid JSON with a wrong total, and the prose above it
  said extraction worked.

So for every cell you touch, ask in order:

1. Did it run without an error output?
2. **Is the output the model's answer, or the budget, the cap or a truncation?** Print
   `finish_reason`, `status` or `stopReason`. If it says the budget ran out, raise it.
3. Is the answer correct? The collection's prompts have known answers (Canberra, 1 to
   10, teal, $0.05, 42 units, the invoice fields, the slide's title and callouts), and
   the cells check them. A wrong value is a finding, not a pass.
4. Does a printed conclusion follow from the printed data? It must be computed in the
   cell, never written in advance.
5. Does the prose above the cell claim anything the cell does not show?

A cell whose answer depends on the model's mood (a small model that misreads a field one
run in four) needs a note saying how often it happens, measured over several runs.

## Committed outputs

Outputs are committed in this repository. That means:

- A cell you change must be re-run and its real output committed. Never hand-write an
  output.
- Check the output for anything that must not be public: bearer tokens, real account IDs
  (the documented placeholder is `123456789012`), presigned URLs, internal hostnames.
- An error output in a committed notebook is a defect.

## There is no verification harness in the clone

The checks built for this repository live in a separate private repository, and you do
not have them. Do not go looking for a verification script; the clone does not contain
one. What those checks encode is written into these skills: the budget rule above, the
known-answer check, and the "could this ever come out differently" question in
`self-review`. Work through them by hand.

You can run `python tests/test_repo_references.py <clone-path>` from your own working
directory to check that the paths these skills name still exist in the repository. If it
reports a mismatch, the skill is stale and that is worth saying in your notification.
