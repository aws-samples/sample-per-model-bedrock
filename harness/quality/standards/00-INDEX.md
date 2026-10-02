# Quality standards for `bedrock-mantle-samples`

The fifteen frameworks this codebase is held to, why each was selected, and what
angle of software quality it covers. Each has its own control document in this
folder; the audit in `../findings/` cites control IDs from these documents.

## Selection rationale

The codebase is a **public AWS sample-code repository of executable Jupyter
notebooks that call a GenAI inference API**. That shape dictates the choice:

- It is *published by AWS*, so AWS's own open-source and Well-Architected bars apply.
- It is *Python*, so PEP 8 / PEP 257 / PEP 484 and the Google Python Style Guide apply.
- It is *notebooks*, so notebook-specific reproducibility rules apply — this is
  the angle most sample repositories miss entirely.
- It *calls an LLM and handles the response*, so the OWASP LLM Top 10 applies —
  particularly output handling, which is the control most notebook samples violate.
- It is *teaching material*, so documentation-style and information-architecture
  standards carry the same weight as the code standards.
- It *ships credentials handling*, so secrets-management and supply-chain controls apply.

Frameworks deliberately **excluded**, with reasons, so the omissions are auditable:

| Excluded | Why not applicable |
|---|---|
| PCI-DSS, HIPAA, SOC 2 | No cardholder, health, or customer data is processed |
| WCAG 2.2 (as a conformance target) | No web UI is shipped; its *documentation* accessibility rules are folded into S12 |
| ISO 27001 | Organisational ISMS controls, not code controls |
| CIS Benchmarks | Host/OS hardening; no infrastructure is deployed |

## The fifteen

| # | Standard | Angle covered | Control doc |
|---|---|---|---|
| S01 | OWASP Top 10 for LLM Applications 2025 | GenAI-specific security | [S01](S01-owasp-llm-top10.md) |
| S02 | OWASP ASVS 5.0 | Application security verification | [S02](S02-owasp-asvs.md) |
| S03 | CWE Top 25 (2024) | Concrete weakness classes | [S03](S03-cwe-top25.md) |
| S04 | AWS Well-Architected — Security Pillar | Cloud security architecture | [S04](S04-aws-well-architected.md) |
| S05 | AWS Well-Architected — Reliability, Cost, Ops, Performance | Cloud non-security pillars | [S05](S05-aws-wa-other-pillars.md) |
| S06 | PEP 8 / PEP 257 / PEP 484 / PEP 20 | Python language conventions | [S06](S06-python-peps.md) |
| S07 | Google Python Style Guide | Python engineering practice beyond PEP 8 | [S07](S07-google-python.md) |
| S08 | Clean Code / SOLID / DRY-KISS-YAGNI | Design, naming, function structure | [S08](S08-clean-code-solid.md) |
| S09 | Ten Simple Rules for Jupyter Notebooks | Notebook reproducibility | [S09](S09-jupyter-reproducibility.md) |
| S10 | Open-source sample-code publication controls | Publication compliance | Internal policy summary, not published |
| S11 | NIST SSDF SP 800-218 | Secure development lifecycle | [S11](S11-nist-ssdf.md) |
| S12 | Google Developer Documentation Style Guide | Documentation quality | [S12](S12-doc-style.md) |
| S13 | ISO/IEC 25010 | Product-quality model, incl. maintainability | [S13](S13-iso-25010.md) |
| S14 | Twelve-Factor App (applicable factors) | Config, dependencies, logs | [S14](S14-twelve-factor.md) |
| S15 | Semantic HTTP / API client correctness | Idempotency, retries, timeouts, error taxonomy | [S15](S15-api-client-correctness.md) |

## How the audit uses these

Every control has a stable ID (`S06-C14`, `S01-C05`, …). The findings files record
`(file, line, control ID, severity, evidence, fix)`. A control that does not apply
to this codebase is recorded as **N/A with a reason** rather than silently dropped —
absence of a finding must mean "checked and clean", never "not looked at".

## Severity scale

Aligned to the scanners so counts are comparable across sources:

| Severity | Meaning here |
|---|---|
| **Critical** | Exploitable security defect, or the sample teaches a dangerous pattern as correct |
| **High** | Security-relevant defect, incorrect behaviour, or a claim contradicted by evidence |
| **Medium** | Correctness risk under plausible conditions; standards violation a reviewer would block on |
| **Low** | Style, consistency, polish |
| **Info** | Observation, no action required |
