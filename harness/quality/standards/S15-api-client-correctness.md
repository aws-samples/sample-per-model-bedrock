# S15 — HTTP and API client correctness

Sources: RFC 9110 (HTTP semantics), RFC 6585 (429 / Retry-After), RFC 9457 (problem
details), AWS Builders' Library — "Timeouts, retries, and backoff with jitter"
(Marc Brooker), AWS SDK retry-behaviour documentation, Google API Design Guide
(AIP) error-model guidance.

The other fourteen standards cover security, style, and structure. None of them
catch "this client retries a 400" or "this code assumes HTTP 200 means the answer
is complete". Those are the defects that actually break a reader's application, so
they get their own standard.

---

## Request construction

- **S15-C01** Every request sets an explicit timeout. No unbounded wait, ever.
- **S15-C02** Timeout values are chosen for the operation — a reasoning model needs a
  longer budget than a token count.
- **S15-C03** `Content-Type` is set correctly for the body being sent.
- **S15-C04** Required protocol headers are always present (e.g. the Anthropic
  `anthropic-version` header on the Messages path).
- **S15-C05** Body is serialised with a JSON encoder, never assembled as a string.
- **S15-C06** Only parameters the target model accepts are sent; a parameter that
  400s is not sent hopefully.
- **S15-C07** No secret in the query string (see S02-C23).
- **S15-C08** The URL path is chosen by an explicit, documented rule — not by
  guessing from the provider name.

## Response handling

- **S15-C09** Check the status code before parsing the body.
- **S15-C10** Never assume a JSON body: an error response may be text or empty.
- **S15-C11** Never index `[0]` into a returned list without checking length.
- **S15-C12** Never assume the first content block is the one you want — filter by
  type. Reasoning models emit a thinking block first.
- **S15-C13** **HTTP 200 does not mean complete.** Check `status`,
  `finish_reason`, or `stop_reason` before using the payload.
- **S15-C14** Distinguish "empty because truncated" from "empty because the model had
  nothing to say", and report which.
- **S15-C15** Validate the parsed structure against what the code will access.
- **S15-C16** Surface the provider's request ID on failure so a reader can escalate.

## Errors

- **S15-C17** Classify errors correctly:
  - `400` malformed / unsupported parameter — **never retry**
  - `401` expired or invalid credential — re-mint, do not blind-retry
  - `403` authorization — a permissions fix, not a retry
  - `404` wrong path or unknown resource — a code fix
  - `422` semantic rejection — a code fix
  - `429` throttling — **retry with backoff**, honour `Retry-After`
  - `500/502/503/504` — **retry with backoff**
  - connection reset / read timeout — **retry with backoff**
- **S15-C18** Error handling never masks the cause; the reported message includes the
  status and the provider's error text (minus secrets).
- **S15-C19** A failed call in a survey reports and continues; it does not abort.

## Retries

- **S15-C20** Exponential backoff with **jitter**. Fixed-interval retry is a defect.
- **S15-C21** Bounded attempt count.
- **S15-C22** Bounded total elapsed time, so retries cannot exceed the cell budget.
- **S15-C23** Only retryable classes are retried.
- **S15-C24** Retries are visible — a reader must be able to tell a retry happened.
- **S15-C25** SDK clients configure retries explicitly rather than relying on a
  default that may be zero.

## Idempotency and state

- **S15-C26** Retrying a non-idempotent operation is either safe by construction or
  guarded.
- **S15-C27** Server-side state created by the notebook (stored responses, projects,
  uploaded files) is cleaned up or explicitly left with a note.
- **S15-C28** Re-running the notebook does not accumulate duplicate server-side
  state.
- **S15-C29** Where state chaining is demonstrated, the prerequisite is stated
  (e.g. `previous_response_id` requires `store=True`).

## Streaming

- **S15-C30** SSE parsing handles the terminal sentinel and ignores unknown event
  types rather than crashing on them.
- **S15-C31** Streaming connections are closed deterministically.
- **S15-C32** A stalled stream is bounded by a timeout.

## Concurrency

- **S15-C33** Worker counts are explicit and modest, to avoid self-throttling.
- **S15-C34** Shared mutable state across workers is avoided or protected.
- **S15-C35** Per-task failure is isolated — one worker's exception does not lose the
  other results.

## Measurement honesty

- **S15-C36** A latency figure states what was measured and how many samples.
- **S15-C37** Single-sample timings are labelled as such, never presented as a
  benchmark.
- **S15-C38** Cache-hit and token-count claims come from the response's own usage
  fields, not from inference.
