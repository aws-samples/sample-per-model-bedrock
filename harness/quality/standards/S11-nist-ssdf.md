# S11 — NIST SSDF (SP 800-218 v1.1)

Source: NIST Special Publication 800-218, February 2022,
`nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-218.pdf`.

Four practice groups: **PO** Prepare the Organization, **PS** Protect the Software,
**PW** Produce Well-Secured Software, **RV** Respond to Vulnerabilities.

For a sample repository, PO and RV are largely organisational. PW is the group that
bites, and PS.1 (protect from unauthorised access/tampering) covers the secrets
question.

## Practices, with applicability

| ID | Practice | Applies? |
|---|---|---|
| PO.1 | Define security requirements for software development | Yes — this standards set *is* that definition |
| PO.2 | Implement roles and responsibilities | N/A — single-author sample |
| PO.3 | Implement supporting toolchains | Yes — scanners, build harness |
| PO.4 | Define and use criteria for software security checks | Yes — the pass/fail gate |
| PO.5 | Implement and maintain secure environments for development | Partly |
| PS.1 | Protect all forms of code from unauthorized access and tampering | Yes — no secrets committed |
| PS.2 | Provide a mechanism for verifying software release integrity | N/A — not a distributed artefact |
| PS.3 | Archive and protect each software release | N/A |
| PW.1 | Design software to meet security requirements and mitigate risks | Yes |
| PW.2 | Review the software design to verify compliance | Yes — the audit |
| PW.3 | Reuse existing, well-secured software rather than duplicating | Yes — use the SDKs |
| PW.4 | Reuse well-secured components | Yes |
| PW.5 | Create source code adhering to secure coding practices | **Yes, central** |
| PW.6 | Configure the compilation, interpreter, and build processes | Partly |
| PW.7 | Review and/or analyze human-readable code | **Yes** — the file-by-file audit |
| PW.8 | Test executable code to identify vulnerabilities | **Yes** — full notebook execution |
| PW.9 | Configure software to have secure settings by default | **Yes** |
| RV.1 | Identify and confirm vulnerabilities on an ongoing basis | Partly |
| RV.2 | Assess, prioritize, and remediate vulnerabilities | Yes |
| RV.3 | Analyze vulnerabilities to identify root causes | Yes |

---

## Controls

### PO.1 / PO.4 — requirements and check criteria

- **S11-C01** Security requirements are written down before the audit, not derived
  from whatever the scanners happened to find.
- **S11-C02** Pass/fail criteria are explicit and measurable (zero high/critical,
  all cells executed).
- **S11-C03** The criteria are applied to *every* file, with N/A recorded explicitly.

### PO.3 — toolchain

- **S11-C04** Static analysis is automated and rerunnable by a third party.
- **S11-C05** Tool versions are recorded with the results.
- **S11-C06** Suppressions are inline, specific, and carry a written justification.

### PS.1 — protect code from tampering / secrets

- **S11-C07** No credential material in the working tree.
- **S11-C08** No credential material in committed notebook outputs.
- **S11-C09** `.gitignore` prevents accidental inclusion of local credential and
  cache files.

### PW.1 / PW.2 — design

- **S11-C10** Attack surface is minimal: the sample makes outbound HTTPS calls and
  writes files it owns. Nothing listens, nothing executes untrusted input.
- **S11-C11** The design review is documented.

### PW.3 / PW.4 — reuse well-secured components

- **S11-C12** Use the official SDKs for signing and transport rather than
  hand-rolling.
- **S11-C13** Dependencies are widely-used, maintained packages.
- **S11-C14** No vendored copies of third-party code.

### PW.5 — secure coding practices

- **S11-C15** Validate all input at trust boundaries — for this codebase, the API
  response boundary.
- **S11-C16** No use of banned functions: `eval`, `exec`, `os.system`,
  `subprocess(shell=True)`, `pickle.loads`, `yaml.load` without SafeLoader,
  `random` for security purposes, MD5/SHA-1 for cryptographic purposes.
- **S11-C17** Error handling does not leak sensitive information.
- **S11-C18** Resource use is bounded.
- **S11-C19** No TLS verification disabled.

### PW.6 — build configuration

- **S11-C20** The build (jupytext → nbconvert execute) is reproducible and scripted.
- **S11-C21** The build fails on any cell error; `allow_errors` is false.

### PW.7 — code review

- **S11-C22** Every file is reviewed against every applicable control.
- **S11-C23** Automated analysis complements, not replaces, the manual review.

### PW.8 — test executable code

- **S11-C24** **Every cell of every notebook is executed** against the live service.
- **S11-C25** Execution is the gate: a notebook that has not run clean does not ship.
- **S11-C26** Re-execution after any change; a fix invalidates the prior pass.

### PW.9 — secure defaults

- **S11-C27** The default path shown to the reader is the secure one (short-term
  credentials, TLS, least privilege, bounded budgets).
- **S11-C28** Where an insecure-looking option exists, the sample shows the secure
  default first and explains the trade-off.

### RV.2 / RV.3 — remediate and root-cause

- **S11-C29** Findings are remediated at the root cause, not suppressed.
- **S11-C30** Where a class of defect was found once, the whole codebase is swept
  for other instances of the same class.
