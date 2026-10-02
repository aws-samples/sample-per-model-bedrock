# S03 — CWE Top 25 Most Dangerous Software Weaknesses (2024)

Source: `cwe.mitre.org/top25/archive/2024/2024_top25_list.html` (MITRE), derived
from 31,770 CVE records.

ASVS says *what to verify*; CWE says *what the defect is called*. Including both
means findings can be expressed in the vocabulary scanners and reviewers use —
Bandit and Semgrep both report CWE IDs.

## The 25, with applicability to this codebase

| Rank | CWE | Weakness | Applies? |
|---|---|---|---|
| 1 | CWE-79 | Cross-site Scripting | Only if notebook output is rendered as HTML — `IPython.display.HTML` is the risk surface |
| 2 | CWE-787 | Out-of-bounds Write | No — memory-safe language |
| 3 | CWE-89 | SQL Injection | Only if a sample builds SQL |
| 4 | CWE-352 | CSRF | No |
| 5 | CWE-22 | Path Traversal | **Yes** — file-writing cells, model-produced paths |
| 6 | CWE-125 | Out-of-bounds Read | No |
| 7 | CWE-78 | OS Command Injection | **Yes** — `subprocess`, shell magics |
| 8 | CWE-416 | Use After Free | No |
| 9 | CWE-862 | Missing Authorization | No — client side |
| 10 | CWE-434 | Unrestricted Upload | Partly — Files API upload cells |
| 11 | CWE-94 | Code Injection | **Yes** — `eval`/`exec` on model output |
| 12 | CWE-20 | Improper Input Validation | **Yes** |
| 13 | CWE-77 | Command Injection | **Yes** |
| 14 | CWE-287 | Improper Authentication | Partly — credential handling |
| 15 | CWE-269 | Improper Privilege Management | **Yes** — IAM guidance |
| 16 | CWE-502 | Deserialization of Untrusted Data | **Yes** — `pickle`, `yaml.load` |
| 17 | CWE-200 | Exposure of Sensitive Information | **Yes** — committed outputs |
| 18 | CWE-863 | Incorrect Authorization | No |
| 19 | CWE-918 | SSRF | Partly — any URL taken from data |
| 20 | CWE-119 | Buffer bounds | No |
| 21 | CWE-476 | NULL Pointer Dereference | Analogue: `None` dereference — **yes** |
| 22 | CWE-798 | Use of Hard-coded Credentials | **Yes** |
| 23 | CWE-190 | Integer Overflow | No |
| 24 | CWE-400 | Uncontrolled Resource Consumption | **Yes** |
| 25 | CWE-306 | Missing Authentication for Critical Function | No |

---

## Controls

- **S03-C01** (CWE-798) Zero hardcoded credentials. Includes example values that
  look real enough to be copied — use obvious placeholders.
- **S03-C02** (CWE-200) Zero sensitive information in committed notebook outputs:
  account IDs, usernames, ARNs of real resources, tokens, internal endpoints.
- **S03-C03** (CWE-94) No `eval`, `exec`, `compile`, or `__import__` on any value
  that originated outside the notebook source.
- **S03-C04** (CWE-78/77) No `os.system`; no `subprocess` with `shell=True` and
  interpolated content. Prefer no subprocess at all in a sample.
- **S03-C05** (CWE-22) Any path constructed from data is resolved and checked to be
  within an intended root before use.
- **S03-C06** (CWE-502) No `pickle.loads`, `yaml.load` without `SafeLoader`, or
  `marshal` on external data.
- **S03-C07** (CWE-20) Validate structure and type of every parsed response before
  indexing into it.
- **S03-C08** (CWE-476 analogue) Never index `[0]` into a list that an API may
  return empty; never attribute-access a value that may be `None`.
- **S03-C09** (CWE-400) Bounded loops, bounded retries, explicit timeouts,
  bounded concurrency.
- **S03-C10** (CWE-269) IAM guidance names least-privilege policies; no
  `AdministratorAccess`, no wildcard actions in any example policy.
- **S03-C11** (CWE-79) If notebook output is rendered as HTML, escape any
  model-produced content first.
- **S03-C12** (CWE-918) No request to a URL taken from model output or response data.
- **S03-C13** (CWE-434) Uploaded sample files are created by the notebook itself
  with known content, never read from an arbitrary user path.
