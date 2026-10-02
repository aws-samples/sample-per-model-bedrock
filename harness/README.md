# Verification harness

The checks behind the notebooks in this repository: static checkers that each prove they
can fail, live probes of the Bedrock endpoints the notebooks call, a scanner runner, and
the generator that builds notebooks from the sources in `build/src/`.

Extracted from a private working copy for publication. Paths are resolved from each
script's location, and personal paths, account details and internal integrations were
removed. The private copy's review records and its internal tooling are not included.

## Layout

```
build/               notebook sources (jupytext percent format) and the scripts that build and run them
quality/scripts/     checkers, verify-all.sh, scan-all.py, and one-off migration scripts
quality/standards/   the frameworks the checks map to (S10 is an internal summary, not included)
quality/findings/    output folder for check results
research/            the endpoint capability probes the notebooks rely on
```

## Configuring

Nothing is hardcoded. Each script finds the repository from its own location, and these
environment variables override the defaults:

| Variable | Default | Used by |
|---|---|---|
| `REPO` | the parent of `harness/` | every checker, `build/run_*.sh` |
| `PY` | `harness/.venv/bin/python3` (`python3` for the build scripts) | `verify-all.sh`, `build/run_*.sh` |
| `SRC`, `LOG` | `build/src`, `$TMPDIR/notebook-run.log` | `build/run_*.sh` |
| `AWS_PROFILE`, `AWS_REGION` | the default credential chain | live probes and notebook runs |

```bash
python3 -m venv harness/.venv && harness/.venv/bin/pip install nbformat jupytext nbconvert boto3 openai
harness/quality/scripts/verify-all.sh          # static checks, seconds
harness/quality/scripts/verify-all.sh --live   # plus live service probes, minutes, costs money
python3 harness/quality/scripts/scan-all.py    # Bandit, Semgrep, Checkov, detect-secrets, pip-audit, ASH
```

`verify-all.sh` refuses to start if it cannot find at least 34 notebooks under `REPO`: a
suite that scans nothing reports success.

## Notes

- The `fix-*.py` and `add-*.py` scripts are one-off migrations, already applied to the
  notebooks. They are kept as a record of how outputs were produced; do not rerun them.
- A few scripts read an earlier result from `quality/findings/`. Regenerate it with the
  script that writes it before running them.
- Internal integrations of the private copy, such as an internal content scanner and a
  messaging digest, are not part of this extract. Nothing here needs them.
