---
name: daily-cycle
description: The ten-step daily procedure, its ordering rules, and how to decide the cycle is finished. Read this first in every cycle, before any other skill.
---

# The daily cycle

## Orient before anything else

1. `checkpoint_read`. It gives you the previous cycle's outcome plus the last ten runs.
   Read the history, not just the latest entry: three `error` rows in a row means
   something is broken in the environment and your first job is to say so clearly in the
   notification, not to attempt the same thing a fourth time.
2. `github_clone`, then `github_recent_changes` with `since` set to the previous
   checkpoint's date. Humans change this repository too. If a maintainer already fixed
   the thing you were going to fix, you are done with that item.
3. `github_list_open_prs`. It returns two things and you need both.
   - **Open pull requests.** One may already carry your change, including one you opened
     yesterday that has not been merged. **Adding a second PR for the same change is the
     single most annoying thing you can do to a maintainer.**
   - **Bot branches pushed with no pull request.** This is the signature of a previous
     cycle that did the work, verified it, and then failed on the last step — the PR API
     was down, or a credential was missing. Read that branch before you start anything.
     If it carries the change you were going to make, check it out, re-verify that its
     claims still hold against the live service, and open the pull request. Redoing the
     work on a fresh branch wastes a cycle and leaves litter behind.
4. `testhost_list_orphans`. If a host leaked from a previous cycle, terminate it now.

## Ordering rules that are not negotiable

- 4 → 5 → 6 → 7 is a loop, not a pipeline. Any change made in 5, 6 or 7 sends you back
  to 4 to re-verify behaviour, then forward again. Two or three laps is normal.
- The security gate (5) must show `gate_passed: true` from `scan_all` before you open a
  PR. Not "I fixed the findings" — the computed boolean.
- Notify (9) and checkpoint (10) happen **exactly once**, at the end, on every path
  including failure. If you crash before them, tomorrow's run is blind.

## Deciding you are finished

You are finished when one of these is true, and you should say which:

- **no-change** — nothing found that needs a sample. Common. Record what you checked and
  the newest announcement you saw, so tomorrow does not re-read it.
- **success** — a PR is open, the gate passed, and the body carries the evidence.
- **partial** — you made progress but could not finish, e.g. a scanner would not run or
  the token is missing. Say exactly where you stopped and what is needed.
- **error** — something broke. Name it precisely; "failed" helps nobody.

## Budget discipline

You have bounded turns and dollars. Spend them on verification, not on reading.

- Do not read all 34 notebooks. Use `Grep` to find the two or three places a fact lives.
- `README.md` and `99-cross-cutting/01-choosing-a-model-and-api.ipynb` are where
  cross-cutting facts are recorded; per-family facts live in that family's directory.
- If you are running low, stop and finish cleanly with `partial` rather than being
  interrupted mid-change. An interrupted cycle that leaves an unpushed branch and no
  checkpoint is worse than one that did less.
