---
name: security-gate
description: The Bandit, Semgrep, Checkov and ASH gate - how to run it, how to read a finding, and what you may and may not do to make it pass. Use in step 5.
---

# Security gate

## Run it

`scan_all` with the clone path. Read `gate_passed`. It is computed as: every scanner RAN,
and none reported a high or critical finding.

**`gate_passed: false` with `did_not_run` non-empty is not a security problem — it is a
broken scanner, and it must not be reported as a pass.** A zero from a tool that never
started is not evidence. This repository has already been burned by exactly that: a
secrets scanner reported "0 potential secrets" for months while exiting 127, because it
was invoked by a name that was not on the PATH.

If a scanner cannot run, say so plainly in the PR body and in the notification. Do not
paper over it.

## Reading a finding

Notebook code cells are extracted to `.py` files before scanning, and each file starts
with a comment naming the notebook and cell. Map the finding back to the real location
before you touch anything.

Most findings in this repository are one of three shapes:

- **A deliberate demonstration.** A cell that shows an insecure pattern in order to warn
  against it. Suppress narrowly, with a comment saying why, on that line only.
- **A real issue in helper code.** Fix it properly.
- **A false positive from a test fixture.** AWS's own documented example key
  `AKIAIOSFODNN7EXAMPLE` is used deliberately to prove a leak detector fires. Suppress it
  narrowly, on that line, with a comment saying what it is, and say so in the PR.

## What you may do

- Fix the underlying issue. Always the first choice.
- Add a narrow, commented suppression on a single line, when the finding is genuinely not
  applicable and you can say why in one sentence. There is no baseline file in this
  repository, so do not add an entry to one — introducing a baseline is a change a
  maintainer should decide on, not a way to make a number go to zero.

## What you may not do

- Widen a suppression to a file or directory to make a number go to zero.
- Delete or weaken a check, a test, or a probe so it stops finding things.
- Report the gate as passed when a scanner did not run.
- Remove a cell that a scanner dislikes, when that cell is the sample's point.

**If you change code here, go back to step 4 and re-verify behaviour before returning.**
A security fix that breaks a sample is worse than the finding — the finding was
theoretical, the break is real.

## Iterating

Expect two or three laps. Each lap: scan, fix the smallest thing, re-verify behaviour,
scan again. Record in the PR body how many laps it took and what changed on each — a
reviewer wants to know whether a fix was mechanical or a judgement call.
