# S02 — OWASP ASVS 5.0

Source: `github.com/OWASP/ASVS`, release 5.0.0, machine-readable CSV
(`OWASP_Application_Security_Verification_Standard_5.0.0_en.csv`) — 345
requirements across 17 chapters, of which 70 are Level 1.

OWASP LLM05 explicitly names ASVS as the reference for output validation and
encoding, which is why it is included as a distinct standard rather than folded
into S01.

## Chapters and applicability

| Ch | Title | Applies? |
|---|---|---|
| V1 | Encoding and Sanitization | **Yes** |
| V2 | Validation and Business Logic | **Yes** |
| V3 | Web Frontend Security | No — no web frontend |
| V4 | API and Web Service | Partly — as a *client*, not a server |
| V5 | File Handling | **Yes** — notebooks write files |
| V6 | Authentication | **Yes** — credential handling |
| V7 | Session Management | No — no sessions |
| V8 | Authorization | **Yes** — IAM least privilege |
| V9 | Self-contained Tokens | **Yes** — Bedrock API keys are presigned bearer tokens |
| V10 | OAuth and OIDC | No |
| V11 | Cryptography | Partly — SigV4 signing only, no custom crypto |
| V12 | Secure Communication | **Yes** — TLS to the endpoint |
| V13 | Configuration | **Yes** |
| V14 | Data Protection | **Yes** |
| V15 | Secure Coding and Architecture | **Yes** |
| V16 | Security Logging and Error Handling | **Yes** |
| V17 | WebRTC | No |

Target level: **L1 in full**, plus L2 requirements that are cheap and relevant
to sample code. L3 is out of scope (it targets high-assurance applications).

---

## Controls (mapped to ASVS requirement IDs)

### V1 Encoding and Sanitization

- **S02-C01** (V1.3.2) Avoid `eval()` and any dynamic code execution. Where code
  generation is the subject being taught, never execute the generated code.
- **S02-C02** (V1.2.4) Parameterised queries only — no string-built SQL.
- **S02-C03** (V1.2.5) No shell interpolation; use argument lists, never `shell=True`
  with interpolated content.
- **S02-C04** (V1.2.3) Encode when building JSON/JS content dynamically; use
  `json.dumps`, never string concatenation.
- **S02-C05** (V1.1.1) Decode/unescape exactly once, before validation, not after.

### V2 Validation and Business Logic

- **S02-C06** (V2.2.1) Positive validation (allowlist) over negative (denylist).
- **S02-C07** (V2.2.2) Validate at a trusted layer — in a client, that means
  validating what comes *back*, not trusting that what you sent constrained it.
- **S02-C08** (V2.1.1) Document the validation rules that apply to each input.

### V5 File Handling

- **S02-C09** Write only to paths the code controls; no path built from untrusted
  or model-produced data.
- **S02-C10** Clean up files a notebook creates, or write them to a temporary
  directory.
- **S02-C11** Use `pathlib` / `os.path` joins, never string concatenation for paths.

### V6 / V8 / V9 Authentication, Authorization, Tokens

- **S02-C12** No long-lived credentials in code, config, or output.
- **S02-C13** Prefer short-lived, automatically-minted credentials; state their TTL.
- **S02-C14** Recommend least-privilege IAM policy, naming the specific managed
  policy or actions required — never `*`.
- **S02-C15** Treat a bearer token as a secret in every code path, including logs,
  error messages, and exception text.
- **S02-C16** Never persist a minted token to disk in a sample.

### V11 / V12 Cryptography and Secure Communication

- **S02-C17** (V11.4.1) No MD5 or SHA-1 for any cryptographic purpose. If a hash is
  used for a non-cryptographic purpose (cache key), say so explicitly and prefer
  SHA-256 anyway.
- **S02-C18** (V12.2.1) TLS for all transport; never disable certificate
  verification, not even in a comment or a commented-out line.
- **S02-C19** Do not implement signing by hand where the SDK provides it.

### V13 Configuration

- **S02-C20** (V13.4.1) No source-control metadata or local paths in shipped files.
- **S02-C21** Configuration comes from environment or explicit parameters, with
  documented defaults — not from hardcoded machine-specific values.
- **S02-C22** No dead configuration: every declared knob must be used.

### V14 Data Protection

- **S02-C23** (V14.2.1) Secrets travel in headers or bodies, never in URLs.
- **S02-C24** (V15.3.1) Print only the fields needed to make the point; do not dump
  whole API responses containing account-identifying metadata.

### V15 Secure Coding and Architecture

- **S02-C25** (V15.1.1/V15.2.1) Dependencies carry version constraints and are
  current at publication; state the remediation expectation for a sample
  (dependencies are pinned to a floor, readers should update).
- **S02-C26** Fail closed: on an unexpected condition, stop or report — never
  silently continue with a wrong value.

### V16 Security Logging and Error Handling

- **S02-C27** Error messages must not leak credentials, tokens, or internal
  identifiers.
- **S02-C28** Catch specific exceptions. A bare `except:` or a broad
  `except Exception:` must either re-raise or be a documented isolation point.
- **S02-C29** Never swallow an exception silently; if suppression is intentional,
  report it.
- **S02-C30** Log/print enough context to diagnose a failure without exposing
  secrets — status code and error type, not the Authorization header.
