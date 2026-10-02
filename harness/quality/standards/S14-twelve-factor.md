# S14 — Twelve-Factor App (applicable factors)

Source: `12factor.net`, Adam Wiggins.

Written for deployed services, so most factors do not apply to notebooks. Included
because three factors — **Dependencies**, **Config**, and **Logs** — are precisely
where sample code most often teaches bad habits, and because a reader will take this
code into a deployed application.

## Applicability

| # | Factor | Applies? |
|---|---|---|
| I | Codebase | Partly — one codebase, version controlled |
| II | Dependencies | **Yes** |
| III | Config | **Yes** |
| IV | Backing services | Partly — the endpoint is an attached resource |
| V | Build, release, run | Partly — the build harness separates them |
| VI | Processes | N/A |
| VII | Port binding | N/A |
| VIII | Concurrency | N/A |
| IX | Disposability | Partly — fast startup, clean teardown |
| X | Dev/prod parity | Partly |
| XI | Logs | **Yes** |
| XII | Admin processes | N/A |

---

## Controls

### II — Explicitly declare and isolate dependencies

- **S14-C01** All dependencies declared in a manifest; nothing relies on a
  system-wide or ambient install.
- **S14-C02** Isolation is instructed — the setup creates a virtual environment.
- **S14-C03** No implicit dependency on a tool that happens to be on the author's
  PATH.
- **S14-C04** Version constraints present, so a reader's install resolves to
  something compatible.

### III — Store config in the environment

- **S14-C05** Credentials come from the environment / ambient provider chain, never
  from source.
- **S14-C06** Region and model IDs are named constants at the top of the notebook —
  the notebook equivalent of config — not scattered literals.
- **S14-C07** No secret in a config file that could be committed.
- **S14-C08** Config that varies between readers (profile name, Region) is either
  defaulted sensibly or read from the environment; it is never a value only the
  author has.
- **S14-C09** No AWS profile name hardcoded in shipped code. A named profile is the
  author's local setup, not the reader's.

### IV — Treat backing services as attached resources

- **S14-C10** The endpoint is addressed by a URL derived from Region, not a hardcoded
  host.
- **S14-C11** Swapping Region requires changing one value.
- **S14-C12** No assumption that a specific model exists — check availability and
  degrade with a clear message.

### V — Strictly separate build and run

- **S14-C13** Source (`.py` jupytext) and executed artefact (`.ipynb`) are distinct;
  the executed artefact is generated, never hand-edited.
- **S14-C14** The build is scripted and repeatable.

### IX — Disposability

- **S14-C15** Fast startup: no long setup before the first useful call.
- **S14-C16** Graceful shutdown: resources the notebook creates are cleaned up, and
  an interrupted run leaves nothing billable behind.

### X — Dev/prod parity

- **S14-C17** The code shown is the code you would run in production, or the gap is
  stated explicitly.
- **S14-C18** No "this is fine for a demo" pattern presented without that caveat.

### XI — Treat logs as event streams

- **S14-C19** Output goes to stdout; no log files written.
- **S14-C20** Output is structured enough to be parsed — aligned tables, consistent
  field order.
- **S14-C21** No secrets in any printed line.
- **S14-C22** Failures are visible in the output stream, not swallowed.
