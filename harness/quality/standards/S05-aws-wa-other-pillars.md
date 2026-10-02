# S05 — AWS Well-Architected: Operational Excellence, Reliability, Performance, Cost

Source: `docs.aws.amazon.com/wellarchitected/latest/framework/`. The Security
pillar is S04; this document covers the other four as they apply to client code
and to teaching material.

## Operational Excellence

Design principles: perform operations as code; make frequent, small, reversible
changes; refine operations procedures frequently; anticipate failure; learn from
all operational failures.

- **S05-C01** Operations as code — the build and test path is scripted and
  repeatable, not a sequence of manual steps.
- **S05-C02** Every failure mode the sample encountered in development is either
  fixed or documented as a known behaviour with a workaround.
- **S05-C03** Observability is demonstrated, not just described.
- **S05-C04** Runbook quality: a reader hitting the documented error can resolve it
  from what the sample says.

## Reliability

Design principles: automatically recover from failure; test recovery procedures;
scale horizontally; stop guessing capacity; manage change through automation.

- **S05-C05** Retry with exponential backoff **and jitter** on retryable errors.
- **S05-C06** Distinguish retryable (429, 500, 502, 503, 504, connection reset)
  from non-retryable (400, 401, 403, 404, 422) — retrying a 400 is a defect.
- **S05-C07** Bound total retry attempts and total elapsed time.
- **S05-C08** Set a client timeout on every request. Absence of a timeout is a
  reliability defect even when the endpoint is usually fast.
- **S05-C09** Handle partial success: a response can be HTTP 200 and still be
  incomplete. Check status/finish-reason fields, not just the status code.
- **S05-C10** Graceful degradation: one failed probe must not abort the notebook.
- **S05-C11** Quotas and limits are stated, with the correct process for raising them.
- **S05-C12** Idempotency: a re-run of the notebook must not accumulate state
  (duplicate projects, orphaned files).

## Performance Efficiency

- **S05-C13** Measure, don't assert — latency claims are backed by timed calls.
- **S05-C14** Report the measurement method (what is being timed, how many samples)
  so a single-sample number is not read as a benchmark.
- **S05-C15** Use streaming where time-to-first-token is the metric that matters.
- **S05-C16** Parallelise independent work, with a bounded worker count.
- **S05-C17** Choose the right service surface for the job and explain the trade-off
  (e.g. batch inference belongs on a different endpoint).

## Cost Optimization

- **S05-C18** State the cost profile of running the sample.
- **S05-C19** Flag the cells that cost materially more than the rest.
- **S05-C20** Use the smallest token budget that demonstrates the point — but never
  so small that the lesson breaks (an under-budgeted reasoning model returns empty
  output, which teaches the wrong thing).
- **S05-C21** Demonstrate the cost levers the service offers: caching, service
  tiers, model-size ladders.
- **S05-C22** Clean up billable resources the notebook creates.
