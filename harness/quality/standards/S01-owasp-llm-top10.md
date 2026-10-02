# S01 — OWASP Top 10 for LLM Applications 2025

Source: OWASP GenAI Security Project, `https://genai.owasp.org/llm-top-10/`
(risk pages `/llmrisk/llmNN2025-…/`). Licence CC BY-SA 4.0.

This is the single most relevant security framework for this codebase: the whole
repository is LLM client code. Applicability is judged per risk — several risks
address model *training* or *hosting*, which this sample does neither of.

| ID | Risk | Applies? |
|---|---|---|
| LLM01 | Prompt Injection | Partly — the tool-use loop feeds model output back as input |
| LLM02 | Sensitive Information Disclosure | **Yes** — credentials, account IDs, committed outputs |
| LLM03 | Supply Chain | **Yes** — pip dependencies |
| LLM04 | Data and Model Poisoning | No — no training or fine-tuning data is authored |
| LLM05 | Improper Output Handling | **Yes, primary** — every notebook parses model output |
| LLM06 | Excessive Agency | **Yes** — the agentic and computer-use notebooks |
| LLM07 | System Prompt Leakage | Partly — system prompts are printed by design (they are the lesson) |
| LLM08 | Vector and Embedding Weaknesses | No — no vector store or RAG pipeline |
| LLM09 | Misinformation | **Yes** — the notebooks make factual claims about API behaviour |
| LLM10 | Unbounded Consumption | **Yes** — token budgets, timeouts, retry amplification |

---

## Controls

### LLM05 Improper Output Handling — the dominant risk here

- **S01-C01** Treat model output as untrusted input. Zero-trust: validate before any
  downstream consumer touches it.
- **S01-C02** Never pass model output to `eval()`, `exec()`, a shell, or any dynamic
  code-execution path. *(Directly maps to CWE-94, CWE-78.)*
- **S01-C03** Never interpolate model output into a SQL string; use parameterised
  queries only.
- **S01-C04** Never build a filesystem path from model output without validating it
  against an allowlist or a resolved-root check. *(CWE-22.)*
- **S01-C05** Never `json.loads()` model output without handling the failure case.
  Structured-output modes are best-effort, not guarantees.
- **S01-C06** Apply context-aware encoding when rendering model output (HTML vs
  Markdown vs terminal).
- **S01-C07** Validate the *shape* of parsed output (required keys, types) not just
  that parsing succeeded.
- **S01-C08** When echoing model-produced tool arguments back to the API, validate
  or repair them first — a malformed echo is both a correctness and an injection risk.

### LLM02 Sensitive Information Disclosure

- **S01-C09** No credentials, tokens, API keys, or bearer values in source or in
  committed notebook output.
- **S01-C10** No real AWS account IDs, IAM principal names, ARNs identifying real
  resources, or internal hostnames in committed output.
- **S01-C11** Truncate any value that is partly sensitive before printing (e.g. show
  a token's length and prefix, never its body).
- **S01-C12** Teach retention controls explicitly: `store`, data-retention mode, and
  what the default is.
- **S01-C13** Do not send sensitive data in URLs or query strings. *(ASVS V14.2.1.)*

### LLM03 Supply Chain

- **S01-C14** Declare every dependency explicitly with a version constraint.
- **S01-C15** Pull only from the canonical index; no VCS or URL installs in a sample.
- **S01-C16** Prefer the minimum dependency set; each added package is an added risk
  a reader inherits.

### LLM06 Excessive Agency

- **S01-C17** Grant the narrowest tool set that demonstrates the lesson.
- **S01-C18** Bound agent loops with an explicit maximum iteration count.
- **S01-C19** Keep tool implementations side-effect-free in samples, or confine
  effects to a temporary directory the notebook creates and removes.
- **S01-C20** Where a demonstrated capability is genuinely dangerous in production
  (computer use, shell tools), say so in prose next to the code.

### LLM09 Misinformation

- **S01-C21** Every behavioural claim must be demonstrated by executed output in the
  same notebook, or explicitly labelled as unverified.
- **S01-C22** Do not state a limitation as permanent when it was observed at one
  point in time; date it or attribute it to the model card.
- **S01-C23** Where the model can be wrong, show the validation step rather than
  presenting output as authoritative.

### LLM10 Unbounded Consumption

- **S01-C24** Every request sets an explicit output-token budget.
- **S01-C25** Every network call sets a client-side timeout. A hung request must not
  hang a notebook.
- **S01-C26** Retries are bounded, use exponential backoff with jitter, and only
  retry retryable classes (429, 5xx, connection errors) — never a 400.
- **S01-C27** Concurrency is explicitly bounded when parallelising probes.
- **S01-C28** Document the cost profile so a reader knows what a run will spend.
- **S01-C29** Degrade gracefully: a failed probe reports and continues; it does not
  abort the lesson.
