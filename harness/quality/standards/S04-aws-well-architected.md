# S04 — AWS Well-Architected Framework, Security Pillar

Source: `docs.aws.amazon.com/wellarchitected/latest/security-pillar/`, publication
date 6 November 2024.

## Design principles

1. Implement a strong identity foundation — least privilege, centralised identity,
   eliminate long-term static credentials.
2. Maintain traceability — monitor, alert, audit in real time.
3. Apply security at all layers.
4. Automate security best practices.
5. Protect data in transit and at rest.
6. Keep people away from data.
7. Prepare for security events.

## Best-practice areas and question IDs

| Area | Questions |
|---|---|
| Security foundations | SEC 1 — how do you securely operate your workload? |
| Identity and access management | SEC 2 — identities for people and machines; SEC 3 — permissions |
| Detection | SEC 4 — detect and investigate security events |
| Infrastructure protection | SEC 5 — network resources; SEC 6 — compute resources |
| Data protection | SEC 7 — classification; SEC 8 — at rest; SEC 9 — in transit |
| Incident response | SEC 10 — anticipate, respond, recover |
| Application security | SEC 11 — how do you incorporate and validate the security properties of applications? |

---

## Controls applicable to this codebase

### SEC 2 — Identity and access management

- **S04-C01** Use temporary credentials. The sample must demonstrate short-term
  credential minting as the default path, not a long-lived key.
- **S04-C02** No static credentials stored in the repository, in notebook source,
  or in committed output.
- **S04-C03** Rely on the ambient credential provider chain (profile, role,
  instance metadata) rather than instructing readers to paste keys.
- **S04-C04** State credential lifetime and non-refreshability where relevant, so a
  reader building on the sample designs for expiry.

### SEC 3 — Permissions management

- **S04-C05** Least privilege: recommend the narrowest managed policy that works,
  and name the extra policy needed only for the optional feature that needs it.
- **S04-C06** Never suggest `AdministratorAccess` or `PowerUserAccess`.
- **S04-C07** Where a feature requires a permission the common managed policy
  lacks, say so and name the action — a 403 that a reader cannot diagnose is a
  defect in the sample.

### SEC 4 — Detection

- **S04-C08** Show the correct observability surface: the right CloudWatch namespace,
  the metrics that exist, and explicitly which ones do not.
- **S04-C09** Demonstrate request-level attribution (request IDs) so a reader can
  file a support case from a failed call.

### SEC 7/8/9 — Data protection

- **S04-C10** Show data-retention controls and their defaults; the reader must learn
  that the default is retention, not deletion, if that is the case.
- **S04-C11** Demonstrate the zero-data-retention path where the service offers it.
- **S04-C12** TLS in transit for every call; never disable verification.
- **S04-C13** Do not send data to a Region the reader did not choose; Region is
  explicit and visible, never implicit.

### SEC 11 — Application security

- **S04-C14** Automate the security checks: the repository states which scanners
  were run and at what result.
- **S04-C15** Validate inputs and outputs at boundaries — the API boundary is the
  security boundary in this codebase.
- **S04-C16** Dependencies are declared and reviewable.

### Shared-responsibility clarity

- **S04-C17** Be explicit about which controls the service provides and which the
  reader must implement. A sample that blurs this teaches a false sense of safety.
- **S04-C18** Where a control is *not* demonstrated (e.g. guardrails deliberately
  out of scope), say so rather than implying coverage.
