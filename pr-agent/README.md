# Bedrock samples PR agent

An autonomous agent that watches Amazon Bedrock for changes a code sample should reflect,
makes the change, proves it against the live service, and opens a pull request against
[aws-samples/sample-per-model-bedrock](https://github.com/aws-samples/sample-per-model-bedrock).
It wakes once a day. Most days it should decide there is nothing to do.

Extracted from a private working copy for publication: internal values are replaced by
settings (see *Configuring*), and the internal messaging-review skill is not included.

---

## Why it is built this way

### The agent gets tools and judgement, not a script

The daily cycle could have been ten Python functions with a model call in each. It is one
`query()` against a system prompt and eight skills instead, because the interesting part
of the job is judgement: whether an announcement needs a sample at all, whether a
notebook's committed output really demonstrates what its prose claims. Python can sequence
steps. It cannot make that call.

### Every side effect goes through a narrow tool

The agent runs with `bypassPermissions`, so it could shell out to `git` and `aws` and the
CLI would let it. It gets 21 purpose-built tools anyway, for three reasons:

1. **Standardisation.** A branch is always named the same way. A test host is always
   tagged. A notification always carries the same fields. Freehand shell would get that
   right most days.
2. **Auditability.** A tool call is a named span in AgentCore Observability. A `bash` call
   with a 400-character command line is not.
3. **Refusal.** `_assert_safe_branch` rejects `main`. There is no merge tool and no force
   push. That is enforced in code, not asked for in a prompt.

### It can open a pull request and nothing more

No merge, no push to `main`, no force push, draft by default. The repository is public and
the reviewer is human by design.

---

## The one rule the whole design serves

**Every claim must be derived from something that can disagree with it.**

The samples repository was reviewed three times and produced 78 defects. They were nearly
all one shape: the evidence and the claim came from the same place, so they could never
disagree. A pre-launch checklist scored 6/6 from expressions like `529 in TRANSIENT` —
true for any code at all, including code with the retry policy deleted. A secrets scanner
reported zero findings for months while exiting 127, because it was invoked by a name that
was not on the `PATH`.

That rule shaped the code here more than anything else:

| Where | What it prevents |
|---|---|
| `compute_gate()` derives `gate_passed` from parsed findings, and treats "did not run" as disqualifying — including an empty report | A zero from a scanner that never started reading as clean |
| `verify_image.py` imports the required command list from `scan_tools`, the module that invokes them, never from `uv tool list` | "The scanner we need is absent" being invisible |
| `sources_feed` reports `window_covered` | An empty feed result reading as "nothing was announced" when the feed simply does not reach back that far |
| `sources_fetch` reports `truncated` | A page cut off at a byte limit being treated as a page that was read |
| `step_observability()` reads the account setting back instead of trusting its own API call | A benign "already set" error reported as a failure, or a failed call reported as success |
| `tests/test_repo_references.py` resolves every path the skills name against the clone | An instruction telling the agent to run a file that does not exist |
| `scan_run`/`scan_all` report `files_scanned`, and zero files is `ran: false` | "0 findings" and "0 files scanned" being the same JSON |
| `infra/verify_policy.py` pairs every IAM allow with a matching deny | A policy that permits everything passing an allow-only test |

Each has a test that plants the specific defect it claims to catch. The suite is checked
by sabotage: break a guard, confirm the tests go red. Fourteen sabotages, fourteen
detected — and the last two only after that run exposed two gaps, one a claim made in a
docstring and tested by nothing, the other a test that *crashed* rather than reporting a
failure, so an automated check reading for the word "failure" scored it as undetected.

---

## Layout

```
container/
  app.py                      AgentCore entrypoint; builds ClaudeAgentOptions, runs one query()
  verify_image.py             build-time proof the image can do its job
  Dockerfile                  ARM64; runtime deps hash-pinned, scanners isolated per-tool
  requirements.txt            what the agent process imports
  requirements.lock.txt       113 packages, every one hash-pinned
  requirements-scanners.txt   uv tool install specs; ASH pinned by commit SHA
  agent/
    system_prompt.md          standing orders, scope, the ten steps, discovery sources
    tools/                    21 tools across 6 in-process MCP servers
  .claude/skills/             8 skills, one per step that needs detail
  tests/
    test_guards.py            8 sabotage-checked groups over the safety-critical logic
    test_repo_references.py   the skills' repo paths, resolved against a clone
infra/
  deploy.py                   idempotent boto3 provisioning, one step per resource group
  invoke.py                   run it by hand, tail its logs, read its run ledger
  verify_policy.py            IAM allow/deny pairs via the policy simulator
  teardown.py                 removes only what deploy.py created
  test_runtime_lookup.py      the runtime is found on any page of ListAgentRuntimes
```

### The six tool servers

| Server | Tools | Notes |
|---|---|---|
| `repo` | clone, recent_changes, list_open_prs, create_branch, commit, push, open_pr | `bot/bedrock-update-*` branches only; no merge, no force push |
| `bedrock` | list_models, list_runtime_models, invoke | Both endpoints; `invoke` validates which one rather than defaulting |
| `sources` | feed, fetch | Refuses hosts in `INTERNAL_HOST_SUFFIXES` and plaintext HTTP |
| `testhost` | launch, run, terminate, list_orphans | Three independent cleanup layers |
| `quality` | scan_run, scan_all | Notebook cells extracted with provenance comments |
| `ops` | checkpoint_read, checkpoint_write, notify_email | DynamoDB ledger, SES sandbox |

### Why the scanners are installed separately

`semgrep` pins `mcp==1.16.0` exactly; `claude-agent-sdk` needs `mcp>=1.23.0`. One
environment holding both is unsatisfiable. Each scanner gets its own virtual environment
via `uv tool install`, which also means a scanner upgrade can never break the agent's
runtime.

### ASH is not on PyPI

The name `automated-security-helper` on PyPI is a squat: an empty placeholder package,
four files, no code, `Summary: test-package-placeholder`, published by an unrelated
account. The real tool is at `github.com/awslabs/automated-security-helper`, pinned here to
the commit that `v3.7.1` points at rather than to the tag, because a tag can be moved.

---

## Guardrails

**Cannot push to `main`.** `_assert_safe_branch` refuses anything outside
`bot/bedrock-update-`, plus `main`, `master`, and path traversal. Twelve hostile branch
names are in the test suite.

**Cannot terminate an instance it did not create.** The IAM policy conditions
`ec2:TerminateInstances` on `ec2:ResourceTag/ManagedBy`, and `ec2:RunInstances` on two
small instance types. Three cleanup layers: tag-based discovery, terminate-leftovers-first
on launch, and a `shutdown -P +60` dead man's switch in user-data that fires even if the
agent, the session and the container all disappear.

**Cannot email anyone else.** SES stays in sandbox mode and the policy is scoped to the
verified identity ARNs in `SES_SENDERS`. An autonomous agent with unrestricted outbound
email is a worse idea than one that can reach only addresses its owner confirmed.

**Cannot read internal sources.** `source_tools` refuses every host suffix listed in
`INTERNAL_HOST_SUFFIXES`, and refuses plaintext HTTP.
It writes to a public repository, so an internally-documented change is not its to publish.

**Bounded spend.** `max_turns=400`, `max_budget_usd=40`, 30-minute idle session timeout,
8-hour ceiling.

---

## Configuring

Settings come from the environment, or from `infra/agent.env` (gitignored): copy
`infra/agent.env.example` and fill it in. The account is read from STS at run time.

| Variable | What it does |
|---|---|
| `AWS_PROFILE`, `AWS_REGION` | Which credentials and Region the infra scripts use (default `us-east-1`) |
| `SES_SENDERS` | Verified SES identities, comma separated; the first is the sender. Empty turns email off |
| `INTERNAL_HOST_SUFFIXES` | Host suffixes the agent must never read, because it publishes publicly |

Optional: a messaging-review skill with your organization's content rules, installed next
to the others in `container/.claude/skills/`. The agent applies it when present.

## Deploying

Needs AWS credentials for the target account from the default chain (set `AWS_PROFILE`
to pick a profile), and `finch` or Docker Buildx able to build `linux/arm64` images,
because AgentCore Runtime runs arm64 only.

```bash
cd infra
python3 deploy.py --step all           # iam, state, ses, observability, image, runtime, schedule
python3 deploy.py --step image         # rebuild and push after a code change
python3 deploy.py --step teardown      # removes only what this created; leaves secrets
```

Two things a human must do once, because the agent cannot create its own credentials:

1. **Add the deploy key** to the repository with write access (Settings → Deploy keys).
2. **Create a fine-grained PAT** with Pull requests: read/write and Contents: read/write,
   and store it as the `bedrock-samples-pr-agent/github-token` secret. A deploy key moves
   git objects but cannot open a pull request, which is why both exist.

Until the token exists the agent runs a read-only cycle and reports that it could not open
a PR — deliberately, so the first deployment can be verified without waiting on a human.

### Running it by hand

```bash
# prove the environment before trusting a real cycle
aws bedrock-agentcore invoke-agent-runtime --region us-east-1 \
  --agent-runtime-arn "$ARN" --runtime-session-id "$(uuidgen)$(uuidgen)" \
  --payload "$(printf '{"mode":"selftest"}' | base64)" /dev/stdout
```

`selftest` lists models, reads the checkpoint, clones the repo and runs one scanner, then
reports whether the environment is fit for a real cycle. It changes nothing.

### Observability

Traces, logs and metrics go to CloudWatch. Transaction Search must be active for spans to
appear on the GenAI observability pages — `--step observability` sets it and then reads the
state back, because both of its API calls reject a no-op with `InvalidRequestException`,
so whether the call succeeded says nothing about whether the account is configured.

`python3 invoke.py observability` reads all three pillars back from the service. It exists
because checking that the OTEL variables are set proves nothing: they were set correctly
for several cycles while no span was ever emitted, because the container launched
`python app.py` instead of `opentelemetry-instrument python app.py`.

- **Logs**: `/aws/bedrock-agentcore/runtimes/<runtime_id>-DEFAULT`. Note that
  `runtime-logs` is the log *stream* prefix inside that group, not part of the group name.
- **Metrics**: `AWS/Bedrock-AgentCore` — with the hyphen. Invocations, Duration, Latency,
  Sessions, Errors, Throttles, CPUUsed-vCPUHours, MemoryUsed-GBHours. A namespace called
  `bedrock-agentcore` also exists and is always empty, which reads as "no metrics" rather
  than "wrong namespace". The OTEL distro separately publishes Latency/Error/Fault/Throttle
  to `ApplicationSignals`.
- **Traces**: `aws/spans` plus X-Ray, with `service.name = bedrock_samples_pr_agent`. That
  log group is account-wide and large, so `filter_log_events` gives up on a wide window
  before it finds a match — a 30-minute scan returns spans where a 24-hour scan returns
  none. Widening the window makes the check less reliable, not more.
- **Run ledger**: the `bedrock-samples-pr-agent-state` DynamoDB table, `pk=run`, newest
  first. `invoke.py runs` reads it.

---

## Testing after a change

```bash
cd container
python tests/test_guards.py                          # no network, no AWS
python tests/test_repo_references.py <clone-path>     # needs a samples clone
python verify_image.py                                # meaningful inside the image
cd ../infra && python3 verify_policy.py               # IAM allow/deny pairs
python3 test_runtime_lookup.py                        # no network, no AWS
```

`deploy.py --step image` runs `verify_image.py` and `test_guards.py` inside the build, so a
broken guard fails the build rather than being discovered by its consequences.

## What the first three cycles found

Recorded because each was a defect that only running the thing could surface, and because
the pattern is worth knowing before changing any of this.

**Mine, found by the agent's own self-test:**

- `bedrock-mantle` is a separate IAM service namespace. Granting `bedrock:*` leaves the
  mantle catalogue answering 403 on `bedrock-mantle:ListModels`.
- `extract_scannable` globbed `*/*.ipynb` and `_shared/*.py`, so root-level notebooks,
  deeper notebooks and every `.py` outside `_shared/` were never scanned — including
  `99-cross-cutting/capabilities.py`. Bandit reported zero because it never looked.
- `GH_TOKEN` was passed only through `ClaudeAgentOptions.env`, which configures the CLI
  *subprocess*. The in-process MCP tools read this process's environ, so the agent could
  commit and push a whole change and then fail on the pull-request call.
- `ec2:RunInstances` conditioned on `ec2:InstanceType` against `Resource: "*"` denies the
  volume and network-interface parts of the launch, because that key only exists on the
  instance resource.
- `ses:SendEmail` alone is not enough to filter unverified recipients; without
  `ses:GetIdentityVerificationAttributes` the check is denied and one unconfirmed address
  silences the whole notification.
- The EventBridge universal target needs PascalCase field names and passes `Payload`
  through verbatim — base64 arrives as literal text and fails to parse before the
  entrypoint is reached.
- The session id comes from the runtime request context, not an environment variable, so
  the ledger recorded an invented id that correlated with no trace.
- `aws-opentelemetry-distro` only activates under `opentelemetry-instrument`. With a bare
  `python app.py` the OTEL variables are read by nobody and no span is ever emitted.
- Two of my own verification tools reported false negatives: the log-group name was wrong
  (`runtime-logs` is a stream prefix, not part of the group), and the span check scanned a
  24-hour window of an account-wide log group, which returns nothing where 30 minutes
  returns spans.

**In my own instructions to the agent, found by the agent:**

- Three skills told it to keep `capabilities.py` byte-identical with a literal in a
  notebook. That file is gitignored and generated by a cell when the notebook runs, so no
  clone contains it — and `tests/test_repo_references.py` had confirmed the instruction was
  fine, because it resolved paths against a working tree that had run the notebooks. It now
  resolves against `git ls-files`.
- Two skills pointed at a `verify-all.sh` that lives in the separate private harness repo,
  not in the clone.

**In the samples repository, found by the agent:**

- `amazon.nova-2-lite-v1:0` was live on `bedrock-runtime` and absent from the repository —
  and, more sharply, is `INFERENCE_PROFILE`-only while the other three Nova models accept
  a bare ID, so copying the working call and swapping the ID returns a 400.
- A cell printed `=> bedrock-mantle is False: Nova is a runtime-only family.`
  unconditionally, directly beneath its own helper's warning that the catalogue could not
  be read. A verdict asserted over missing data.
- `requirements.txt` needs Python 3.10 or later and nothing says so; under 3.9 the install
  fails in a way that reads like a bad pin.
- `02-anthropic-claude/02-thinking-tools-and-caching.ipynb` asserted in six places, one of
  them a Gotchas-table rule reading "never readable on mantle", that a Claude thinking
  block's text is empty on that endpoint. A 4-forms x 4-models probe showed readability
  tracks the thinking *form*, not the endpoint: `haiku-4-5` with `enabled` + `budget_tokens`
  returns a readable trace on mantle. One of the six was a hardcoded `<- always empty`
  verdict that printed regardless of the value. This is PR #2.
