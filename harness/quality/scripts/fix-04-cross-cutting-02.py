#!/usr/bin/env python3
"""Audit fix 4 — 99-cross-cutting/02, the migration notebook.

Defects fixed:
  B3  the compatibility shim failed on 3 of 6 models while the next cell claimed
      "no 400s, because the shim resolved the differences"
  B4  NO_CHAT_COMPLETIONS = ("openai.gpt-5.6",) — dead code AND factually wrong;
      the same false claim in three prose sites
  B5  two probes always POSTed /responses, including for a Chat-Completions-only
      model, so every qwen row was a false negative and the prose contradicted
      its own table
  E9  "five models" for a six-model loop

The shim is rebuilt around the fact this collection teaches everywhere else: when
the service refuses a parameter it *names* it. So instead of carrying a static
table of per-model restrictions that ages badly — which is exactly how it broke —
it drops the named parameter and retries. That works today and keeps working.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PATH = f"{REPO}/99-cross-cutting/02-migrating-from-openai.ipynb"

SHIM = '''class MantleCompat:
    """Model-agnostic wrapper over the three bedrock-mantle API surfaces.

    The design point worth copying is the error-driven retry in `complete()`.
    An earlier version of this class carried a static table of which models reject
    which parameters. That table was correct when written and wrong within weeks --
    Gemma 4 and Grok both changed twice -- and the class failed on half the models
    it claimed to handle. Since the service *names* the offending parameter in its
    400 (see section 4), the robust move is to drop that parameter and retry rather
    than to remember a rule.
    """

    # Families with no Responses API: they answer on Chat Completions only.
    CHAT_ONLY = (
        "qwen.", "deepseek.", "zai.", "minimax.", "moonshotai.", "mistral.",
        "nvidia.", "writer.",
        # The safeguard variants are Chat-Completions-only even though the base
        # gpt-oss models serve both. Provider prefix is not enough.
        "openai.gpt-oss-safeguard",
    )
    # Chat Completions takes max_completion_tokens rather than max_tokens here.
    COMPLETION_TOKENS = ("openai.gpt-5.6",)
    # Reasoning-first models spend the budget on thinking before any answer text,
    # so a small cap returns HTTP 200 and an empty string.
    REASONS_FIRST = ("xai.", "openai.gpt-5.6", "moonshotai.kimi-k2-thinking")

    def __init__(self, region=REGION, project=None):
        self.region, self.project = region, project

    # ---- resolution -----------------------------------------------------
    def prefix(self, model):
        if model.startswith("anthropic."):
            return "/anthropic/v1"
        if model.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
            return "/openai/v1"
        return "/v1"

    def surface(self, model):
        """Which API to use for this model."""
        if model.startswith("anthropic."):
            return "messages"
        if model.startswith(self.CHAT_ONLY):
            return "chat"
        return "responses"

    def _budget_field(self, model, surface):
        if surface == "responses":
            return "max_output_tokens"
        if surface == "chat" and model.startswith(self.COMPLETION_TOKENS):
            return "max_completion_tokens"
        return "max_tokens"

    def _floor(self, model, max_tokens):
        """Responses enforces a minimum of 16; reasoning-first models need far more."""
        floor = 1000 if model.startswith(self.REASONS_FIRST) else 16
        return max(floor, max_tokens)

    # ---- one call, any model -------------------------------------------
    def _headers(self, surface):
        """Per-surface headers: project attribution, and Anthropic's version pin."""
        headers = {}
        if self.project:
            key = "anthropic-workspace" if surface == "messages" else "OpenAI-Project"
            headers[key] = self.project
        if surface == "messages":
            headers["anthropic-version"] = "2023-06-01"
        return headers

    def _build(self, model, prompt, surface, prefix, system, max_tokens,
               temperature, top_p, tier):
        """Return (path, body) in whichever wire format this surface expects."""
        budget = {self._budget_field(model, surface): self._floor(model, max_tokens)}
        sampling = {}
        if temperature is not None:
            sampling["temperature"] = temperature
        if top_p is not None:
            sampling["top_p"] = top_p

        if surface == "messages":
            body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                    **budget, **sampling}
            if system:
                body["system"] = system  # Messages takes system as a top-level field
            return f"{prefix}/messages", body

        turns = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        if surface == "chat":
            return f"{prefix}/chat/completions", {
                "model": model, "messages": turns, "service_tier": tier,
                **budget, **sampling,
            }
        return f"{prefix}/responses", {
            "model": model, "input": turns, "service_tier": tier, "store": False,
            **budget, **sampling,
        }

    @staticmethod
    def _offending_key(message):
        """The parameter the service just refused, if it named one.

        Mantle is consistent about this: it quotes the field in the message. Three
        shapes seen in practice, all covered by the quoted-token scan below:
          Unsupported parameter: 'top_p' is not supported with this model.
          `temperature` is deprecated for this model.
          `temperature` and `top_p` cannot both be specified for this model.
        """
        import re

        named = re.findall(r"[\\'`\\"]([a-z_]+)[\\'`\\"]", message or "")
        tunable = ("temperature", "top_p", "service_tier")
        hits = [n for n in named if n in tunable]
        return hits[0] if hits else None

    def complete(self, model, prompt, *, system=None, max_tokens=400,
                 temperature=None, top_p=None, tier="default", timeout=120,
                 attempts=2, max_drops=3):
        """One call surface over all three APIs, self-healing on refused parameters.

        `timeout` and `attempts` are bounded on purpose. Reasoning-first models can
        take many minutes for a single call under load, and post()'s defaults
        (240s x 5) would turn one slow model into a 20-minute stall for the loop.
        """
        prefix, surface = self.prefix(model), self.surface(model)
        headers = self._headers(surface)
        optional = {"temperature": temperature, "top_p": top_p, "tier": tier}
        dropped = []

        for _ in range(max_drops + 1):
            path, body = self._build(
                model, prompt, surface, prefix, system, max_tokens,
                optional["temperature"], optional["top_p"], optional["tier"],
            )
            if surface == "messages":
                body.pop("service_tier", None)  # not a Messages parameter
            code, data = post(path, body, region=self.region,
                              headers=headers or None, timeout=timeout,
                              attempts=attempts)
            if code == 200:
                if dropped:
                    print(f"      [{model}: dropped {', '.join(dropped)} and retried]")
                return self._extract(surface, data)

            # Read the message, drop what it named, try again. Anything else is
            # a real failure and must not be retried.
            key = self._offending_key(err(data))
            key = "tier" if key == "service_tier" else key
            if key is None or optional.get(key) in (None, "default"):
                raise RuntimeError(f"{model}: HTTP {code}: {err(data)}")
            optional[key] = None if key != "tier" else "default"
            dropped.append(key)
        raise RuntimeError(f"{model}: still refused after dropping {dropped}")

    @staticmethod
    def _extract(surface, data):
        if surface == "messages":
            return "".join(
                b.get("text", "")
                for b in data.get("content", [])
                if b.get("type") == "text"
            )
        if surface == "chat":
            return (data.get("choices") or [{}])[0].get("message", {}).get(
                "content"
            ) or ""
        return response_text(data)


compat = MantleCompat()
PROMPT = "Name one benefit of a managed inference endpoint. One sentence."'''

PARAM_PROBE = '''# Probe each model on the API it ACTUALLY serves. The earlier version of this cell
# always POSTed /responses, so every row for a Chat-Completions-only model was a
# 400 about the missing API -- and the paragraph below it then contradicted its own
# table. Blaming the parameter for a missing API is the same mistake in reverse.
CASES = [
    {"temperature": 0.7, "top_p": 0.95},
    {"temperature": 0.7},
    {"temperature": 1.0},
    {"top_p": 0.95},
]
LABELS = ["0.7+top_p", "temp 0.7", "temp 1.0", "top_p"]

print(f"{'model':26} {'surface':10}" + "".join(f"{h:>11}" for h in LABELS))
print("-" * 78)
param_support = {}
for mid in SURVEY:
    surface = compat.surface(mid)
    row = []
    for extra in CASES:
        path, body = compat._build(
            mid, "Hi", surface, compat.prefix(mid), None, 16, None, None, "default"
        )
        body.update(extra)
        if surface == "messages":
            body.pop("service_tier", None)
        code, _ = post(
            path, body, region=REGION, attempts=1, timeout=60,
            headers=compat._headers(surface) or None,
        )
        row.append("ok" if code == 200 else str(code))
    param_support[mid] = row
    print(f"{mid:26} {surface:10}" + "".join(f"{v:>11}" for v in row))'''

TIER_PROBE = '''print(f"{'model':26} {'surface':10} {'flex':>8} {'priority':>10}")
print("-" * 58)
for mid in SURVEY:
    surface = compat.surface(mid)
    if surface == "messages":
        # service_tier is not a Messages parameter at all, so there is nothing to
        # test here. Saying "400" would imply the tier was refused on its merits.
        print(f"{mid:26} {surface:10} {'n/a':>8} {'n/a':>10}")
        continue
    row = []
    for tier in ("flex", "priority"):
        path, body = compat._build(
            mid, "Hi", surface, compat.prefix(mid), None, 16, None, None, tier
        )
        code, _ = post(path, body, region=REGION, attempts=1, timeout=60)
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {surface:10} {row[0]:>8} {row[1]:>10}")'''

API_PROBE = '''# Try BOTH budget parameter names before concluding an API is missing. A 400 from
# `max_tokens` on a model that wants `max_completion_tokens` looks identical to a
# 400 that means "this API does not exist here" -- and reading it the wrong way is
# how "gpt-5.6 has no Chat Completions" got into three places in this notebook.
print(f"{'model':26} {'Responses':>10} {'ChatCompl':>10} {'CC budget field':>22}")
print("-" * 72)
for mid in SURVEY:
    if mid.startswith("anthropic."):
        print(f"{mid:26} {'n/a':>10} {'n/a':>10} {'Messages only':>22}")
        continue
    prefix = compat.prefix(mid)
    code_r, _ = post(
        f"{prefix}/responses",
        {"model": mid, "input": "Hi", "max_output_tokens": 16},
        region=REGION, attempts=1, timeout=60,
    )
    field, code_c = None, None
    for candidate in ("max_tokens", "max_completion_tokens"):
        code_c, _ = post(
            f"{prefix}/chat/completions",
            {"model": mid, "messages": [{"role": "user", "content": "Hi"}],
             candidate: 16},
            region=REGION, attempts=1, timeout=60,
        )
        if code_c == 200:
            field = candidate
            break
    print(f"{mid:26} {code_r:>10} {code_c:>10} {(field or '-'):>22}")'''

EXERCISE = '''print(f"{'model':30} {'surface':10}  answer")
print("-" * 96)
for mid in SURVEY:
    try:
        # Deliberately pass BOTH sampling params and a flex tier. The shim sends
        # them, reads any 400 that names one, drops it and retries -- so watch for
        # the "[dropped ...]" notes.
        answer = compat.complete(
            mid, PROMPT, temperature=0.7, top_p=0.95, tier="flex", max_tokens=200
        )
        shown = " ".join(answer.split())[:52] or "(empty - raise max_tokens)"
        print(f"{mid:30} {compat.surface(mid):10}  {shown!r}")
    except RuntimeError as exc:
        print(f"{mid:30} {compat.surface(mid):10}  FAILED: {str(exc)[:52]}")'''


def main() -> None:
    nb = Notebook(PATH)

    # A single model list, used by every probe, so the surfaces stay consistent.
    nb.sub('REGION = "us-east-1"\nprint("region:", REGION)',
           '''REGION = "us-east-1"

# One survey list, shared by every probe below. Six models across three wire
# protocols and all three mantle path prefixes.
SURVEY = [
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
]
print("region:", REGION, "|", len(SURVEY), "models surveyed")''')

    # §4 sampling probe, on the right surface per model.
    nb.sub('''CANDIDATES = [
    ("openai.gpt-5.6-sol", "/openai/v1"),
    ("google.gemma-4-31b", "/openai/v1"),
    ("xai.grok-4.3", "/openai/v1"),
    ("openai.gpt-oss-120b", "/v1"),
    ("qwen.qwen3-32b", "/v1"),
]
print(f"{'model':26} {'0.7+top_p':>11} {'temp 0.7':>9} {'temp 1.0':>9} {'top_p':>7}")
print("-" * 66)
for mid, prefix in CANDIDATES:
    row = []
    for extra in (
        {"temperature": 0.7, "top_p": 0.95},
        {"temperature": 0.7},
        {"temperature": 1.0},
        {"top_p": 0.95},
    ):
        code, _ = post(
            f"{prefix}/responses",
            {"model": mid, "input": "Hi", "max_output_tokens": 16, **extra},
            region=REGION,
            attempts=1,
            timeout=45,
        )
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {row[0]:>11} {row[1]:>9} {row[2]:>9} {row[3]:>7}")''',
           PARAM_PROBE)

    nb.sub("""Three different behaviours in one table:

- **Gemma 4 / gpt-5.6** reject `top_p`, and accept `temperature` **only at its
  default `1.0`** — so a ported `temperature=0.7` breaks.
- **Grok** rejects `temperature` at every value but accepts `top_p`.
- **gpt-oss / qwen** accept both, at any value.

The safest port is to **send neither parameter** on the Responses API unless you
have a specific reason, then add each back per model after testing. If you truly
need sampling control, Chat Completions is more permissive.""",
           """Read the table, not a remembered rule — which models refuse what has changed
twice during this collection's life. What is durable:

- **There is no single sampling config that works everywhere.** At the time of
  writing the GPT-5.5 and GPT-5.6 families accept `temperature` only at its default
  `1.0` and refuse `top_p` outright; newer Claude models reject both as
  *deprecated*; `claude-haiku-4-5` accepts either one but **not both together**.
  Gemma 4 and Grok have been in and out of that set.
- **The safest port is to send neither parameter**, then add each back per model
  after testing.
- **The service names the parameter it refused.** That is what makes the shim in §7
  able to recover automatically instead of carrying a table that ages.""")

    # §5 tier probe.
    nb.sub('''print(f"{'model':26} {'flex':>8} {'priority':>10}")
print("-" * 48)
for mid, prefix in CANDIDATES:
    row = []
    for tier in ("flex", "priority"):
        code, _ = post(
            f"{prefix}/responses",
            {
                "model": mid,
                "input": "Hi",
                "max_output_tokens": 16,
                "service_tier": tier,
            },
            region=REGION,
            attempts=1,
            timeout=45,
        )
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {row[0]:>8} {row[1]:>10}")''',
           TIER_PROBE)

    # §6 API probe: try both budget fields.
    nb.sub("""If your codebase is built on Chat Completions, note that **gpt-5.6 does not serve
it**. If it is built on Responses, note that most open-weight families do not.""",
           """If your codebase is built on Responses, note that most open-weight families do not
serve it. The reverse trap is subtler: **gpt-5.6 *does* serve Chat Completions**,
but it refuses `max_tokens` in favour of `max_completion_tokens`, so a naive probe
gets a 400 and concludes the API is missing. An earlier version of this notebook
made exactly that mistake and recorded it as fact in three places.

The rule: send both candidate budget fields before deciding an API is absent, and
read *which* thing the error names — the path, or the parameter.""")
    nb.sub('''print(f"{'model':26} {'Responses':>10} {'ChatCompl':>10}")
print("-" * 50)
for mid, prefix in CANDIDATES:
    code_r, _ = post(
        f"{prefix}/responses",
        {"model": mid, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    code_c, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": mid,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        },
        region=REGION,
        attempts=1,
        timeout=45,
    )
    print(f"{mid:26} {code_r:>10} {code_c:>10}")''',
           API_PROBE)

    # §7 the shim itself.
    old_shim_start = 'class MantleCompat:\n    """Model-agnostic wrapper over the three bedrock-mantle API surfaces."""'
    idx = nb.find(old_shim_start)
    assert len(idx) == 1, idx
    nb.set_source(idx[0], SHIM)

    # §7 exercise + the claim that followed it.
    nb.sub("""Exercise the shim across every path family — one call site, five models, three
different wire protocols underneath.""",
           """Exercise the shim across every path family — one call site, six models, three
different wire protocols underneath. Watch for `[dropped ...]` notes: that is the
shim reading a 400, removing the parameter the service named, and retrying.""")
    old_ex = nb.find('        # Deliberately pass BOTH sampling params and a flex tier — the shim drops')
    assert len(old_ex) == 1, old_ex
    nb.set_source(old_ex[0], EXERCISE)
    nb.sub("""Same call signature, six models, three different API surfaces — and no 400s,
because the shim resolved the differences.""",
           """Same call signature, six models, three different API surfaces. Where a parameter
was refused the shim dropped it and retried rather than failing, which is why the
answers come back even though the call site passed `temperature`, `top_p` and
`flex` to every model indiscriminately.

That recovery is the part to copy. A shim built on a table of per-model
restrictions works until the table is stale; one that reads the error the service
actually returned keeps working.""")

    # §10 gains/losses table.
    nb.sub("| OpenAI API | IAM auth, Projects, ZDR (zero data retention), AWS-hosted Web Search, service tiers | OpenAI model IDs; some params; Chat Completions on gpt-5.6 |",
           "| OpenAI API | IAM auth, Projects, ZDR (zero data retention), AWS-hosted Web Search, service tiers | OpenAI model IDs; some sampling params; per-model budget-field names |")
    nb.sub("| Anthropic API | IAM auth, Workspaces, `count_tokens` | `output_config.format` (use forced tools) |",
           "| Anthropic API | IAM auth, Workspaces, `count_tokens` on mantle | `output_config.format` (use forced tools); `temperature` on newer models |")

    # Checklist + gotcha table.
    nb.sub("5. **Gate `service_tier`** — gpt-5.x is `default`-only.",
           "5. **Gate `service_tier`** — gpt-5.x is `default`-only, and it is not a\n"
           "   Messages parameter at all.")
    nb.sub("| Chat Completions | Absent on gpt-5.6; Responses absent on most open-weight families |",
           "| Budget field name | Chat Completions wants `max_completion_tokens` on gpt-5.6, `max_tokens` elsewhere. A 400 here reads like a missing API |\n"
           "| Responses coverage | Absent on most open-weight families, and on the gpt-oss **safeguard** variants even though base gpt-oss serves it |")
    nb.sub("| Shared sampling defaults | On Responses, `temperature` must be `1.0` for Gemma 4 / gpt-5.6, and is rejected outright by Grok |",
           "| Shared sampling defaults | No universal config. gpt-5.5/5.6 want `temperature=1.0` and refuse `top_p`; newer Claude rejects both; `haiku-4-5` takes either but not both |")
    nb.save()
    print(f"99-cross-cutting/02: {nb.changes} edits")


if __name__ == "__main__":
    main()
