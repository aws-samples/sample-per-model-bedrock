---
name: raise-pr
description: How to write the branch, commit and pull request so a human reviewer can trust the change quickly - what evidence to include and what to admit. Use in step 8.
---

# Raising the pull request

The reviewer is a human who did not watch you work. They need to decide, in a couple of
minutes, whether to trust a machine-authored change to a public repository. Write for that.

## Sequence

```
github_create_branch   slug like "claude-fable-5-2" or "nova-3-region-expansion"
github_commit          real message, subject under 72 chars
github_push
github_open_pr         draft by default
```

## The commit message

Subject: what changed, under 72 characters, no full stop.
Body: why, and what evidence. Name the announcement URL and the measurement. A reader
running `git log` a year from now should understand the change without the PR.

Never "update notebooks", "fix issues", or "add model".

## The PR body

Cover these, in this order. Be specific; a reviewer can tell the difference between
evidence and reassurance.

**What changed and why.** One paragraph. Lead with the thing a reader would have got
wrong before this change.

**Evidence.** The announcement URL, and the measurement that confirms it, pasted. If the
service and the announcement disagreed, say so — that is the most valuable thing in the
PR. Include the Region and endpoint every measurement came from.

**Verification.** What you ran and what it showed. Notebook execution result, whether
committed outputs were regenerated, and the `finish_reason` check on any number you
published. If you used the EC2 test host, say so and confirm it was terminated.

**Security gate.** The `scan_all` verdict, per scanner. If any scanner did not run, say
which and why — do not omit it. Say how many laps steps 4–7 took.

**Messaging review.** If a messaging-review skill is installed, confirm you checked against it, and note
anything you deliberately kept.

**What I did not do.** The most useful section, and the one a machine is tempted to skip.
Anything you could not verify, chose not to change, or left for a human. If you were
uncertain about something, this is where it goes — an admitted gap is cheap, a silent one
is expensive.

## Draft by default

Open as a draft unless the change is a small, fully-verified correction to something
already measured. A draft signals "ready for a look" rather than "ready to merge", which
is the honest state of anything you produced without a human in the loop.

## What you must not do

- Do not merge. You have no tool for it, and that is deliberate.
- Do not open a second PR for a change an open PR already covers.
- Do not open a PR at all if the gate did not pass. Report that instead.
- Do not pad the body to look thorough. A short PR with real evidence beats a long one
  with headings and no measurements.
