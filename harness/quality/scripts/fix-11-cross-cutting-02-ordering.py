#!/usr/bin/env python3
"""Audit fix 11 — 99-cross-cutting/02 used `compat` before §7 defined it.

Fix 4 rewrote the §4/§5/§6 probes to hit each model's real API surface, but reached
for the `MantleCompat` instance to do the routing — and that class is not defined
until §7. Three cells raised NameError.

Routing is also the wrong thing to learn from the shim: the shim's contribution is
the error-driven retry, not the prefix table. So the primitives move into §3, where
`api_prefix` already lives, the probes use them directly, and §7's class composes
them instead of restating them.
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

ROUTING = '''# The four routing questions, answered per model. Everything below in this notebook
# uses these, and section 7's compatibility shim composes them rather than
# restating them.

# Families with no Responses API: Chat Completions only. Note gpt-oss-safeguard is
# here while base gpt-oss is not -- the provider prefix is not enough to decide.
CHAT_ONLY = (
    "qwen.", "deepseek.", "zai.", "minimax.", "moonshotai.", "mistral.",
    "nvidia.", "writer.", "openai.gpt-oss-safeguard", "google.gemma-3",
)
# Chat Completions wants max_completion_tokens rather than max_tokens for these.
COMPLETION_TOKENS = ("openai.gpt-5.6",)
# Reasoning-first models spend the budget thinking before any answer text, so a
# small cap returns HTTP 200 and an empty string.
REASONS_FIRST = ("xai.", "openai.gpt-5.6", "moonshotai.kimi-k2-thinking")


def api_prefix(model_id: str) -> str:
    """Which of the three URL prefixes serves this model."""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(("google.gemma-4", "openai.gpt-5", "xai.")):
        return "/openai/v1"
    return "/v1"


def surface(model_id: str) -> str:
    """Which API this model actually serves: messages / chat / responses."""
    if model_id.startswith("anthropic."):
        return "messages"
    if model_id.startswith(CHAT_ONLY):
        return "chat"
    return "responses"


def budget_field(model_id: str) -> str:
    """The token-budget parameter this model+API expects.

    Getting this wrong returns a 400 that reads exactly like "this API does not
    exist here", which is the trap section 6 is about.
    """
    which = surface(model_id)
    if which == "responses":
        return "max_output_tokens"          # minimum 16
    if which == "chat" and model_id.startswith(COMPLETION_TOKENS):
        return "max_completion_tokens"
    return "max_tokens"                     # Messages: required, no default


def build_request(model_id, prompt="Hi", *, budget=16, system=None,
                  sampling=None, tier=None):
    """Return (path, body, headers) in whichever wire format this model wants."""
    which, prefix = surface(model_id), api_prefix(model_id)
    floor = 1000 if model_id.startswith(REASONS_FIRST) else 16
    body = {"model": model_id, budget_field(model_id): max(floor, budget)}
    body.update(sampling or {})
    headers = {}

    if which == "messages":
        # service_tier is not a Messages parameter at all, so it is never sent.
        headers["anthropic-version"] = "2023-06-01"
        body["messages"] = [{"role": "user", "content": prompt}]
        if system:
            body["system"] = system
        return f"{prefix}/messages", body, headers

    if tier:
        body["service_tier"] = tier
    turns = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    if which == "chat":
        body["messages"] = turns
        return f"{prefix}/chat/completions", body, headers
    body["input"] = turns
    body["store"] = False
    return f"{prefix}/responses", body, headers


print(f"{'model':30} {'prefix':16} {'surface':10} budget field")
print("-" * 74)
for mid in SURVEY:
    print(f"{mid:30} {api_prefix(mid):16} {surface(mid):10} {budget_field(mid)}")'''

PARAM_PROBE = '''# Probe each model on the API it ACTUALLY serves, using the routing from section 3.
# The earlier version of this cell always POSTed /responses, so every row for a
# Chat-Completions-only model was a 400 about the missing API -- and the paragraph
# below it then contradicted its own table. Blaming the parameter for a missing API
# is the same mistake in reverse.
CASES = [
    ("0.7+top_p", {"temperature": 0.7, "top_p": 0.95}),
    ("temp 0.7", {"temperature": 0.7}),
    ("temp 1.0", {"temperature": 1.0}),
    ("top_p", {"top_p": 0.95}),
]

print(f"{'model':26} {'surface':10}" + "".join(f"{h:>11}" for h, _ in CASES))
print("-" * 78)
param_support = {}
for mid in SURVEY:
    row = []
    for _, extra in CASES:
        path, body, headers = build_request(mid, sampling=extra)
        code, _ = post(path, body, region=REGION, attempts=1, timeout=60,
                       headers=headers or None)
        row.append("ok" if code == 200 else str(code))
    param_support[mid] = row
    print(f"{mid:26} {surface(mid):10}" + "".join(f"{v:>11}" for v in row))'''

TIER_PROBE = '''print(f"{'model':26} {'surface':10} {'flex':>8} {'priority':>10}")
print("-" * 58)
for mid in SURVEY:
    if surface(mid) == "messages":
        # service_tier is not a Messages parameter, so there is nothing to test.
        # Printing "400" here would imply the tier was refused on its merits.
        print(f"{mid:26} {'messages':10} {'n/a':>8} {'n/a':>10}")
        continue
    row = []
    for tier in ("flex", "priority"):
        path, body, headers = build_request(mid, tier=tier)
        code, _ = post(path, body, region=REGION, attempts=1, timeout=60,
                       headers=headers or None)
        row.append("ok" if code == 200 else str(code))
    print(f"{mid:26} {surface(mid):10} {row[0]:>8} {row[1]:>10}")'''

API_PROBE = '''# Try BOTH budget parameter names before concluding an API is missing. gpt-5.6
# serves Chat Completions but refuses `max_tokens`, and reading that 400 as "no
# Chat Completions" is how a false claim reached three places in this notebook.
print(f"{'model':26} {'Responses':>10} {'ChatCompl':>10} {'CC budget field':>22}")
print("-" * 72)
for mid in SURVEY:
    if mid.startswith("anthropic."):
        print(f"{mid:26} {'n/a':>10} {'n/a':>10} {'Messages only':>22}")
        continue
    prefix = api_prefix(mid)
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

SHIM = '''class MantleCompat:
    """Model-agnostic wrapper over the three bedrock-mantle API surfaces.

    Routing comes from section 3 -- `surface()`, `budget_field()`,
    `build_request()`. What this class adds is the **error-driven retry**, and that
    is the part worth copying.

    An earlier version carried a static table of which models reject which
    parameters. That table was correct when written and wrong within weeks -- Gemma 4
    and Grok both changed twice -- and the class failed on half the models it
    claimed to handle. Since the service *names* the offending parameter in its 400
    (section 4), the robust move is to drop that parameter and retry rather than to
    remember a rule.
    """

    TUNABLE = ("temperature", "top_p", "service_tier")

    def __init__(self, region=REGION, project=None):
        self.region, self.project = region, project

    def prefix(self, model):
        return api_prefix(model)

    def surface(self, model):
        return surface(model)

    def _project_header(self, which):
        if not self.project:
            return {}
        key = "anthropic-workspace" if which == "messages" else "OpenAI-Project"
        return {key: self.project}

    @staticmethod
    def _offending_key(message):
        """The parameter the service just refused, if it named one.

        Mantle is consistent about naming it. Three shapes seen in practice:
          Unsupported parameter: 'top_p' is not supported with this model.
          `temperature` is deprecated for this model.
          unsupported service_tier 'flex'                  <- field unquoted
        """
        import re

        text = message or ""
        quoted = re.findall(r"[\\'`\\"]([a-z_]+)[\\'`\\"]", text)
        for name in quoted:
            if name in MantleCompat.TUNABLE:
                return name
        # A bare mention is only safe to act on when the message is not about a
        # path: "does not support the '/v1/responses' API" names no parameter.
        if "API" not in text:
            for name in MantleCompat.TUNABLE:
                if name in text:
                    return name
        return None

    def complete(self, model, prompt, *, system=None, max_tokens=400,
                 temperature=None, top_p=None, tier="default", timeout=120,
                 attempts=2):
        """One call surface over all three APIs, self-healing on refused parameters.

        `timeout` and `attempts` are bounded on purpose. Reasoning-first models can
        take many minutes for one call under load, and post()'s defaults (240s x 5)
        would turn one slow model into a 20-minute stall for the whole loop.
        """
        sampling = {}
        if temperature is not None:
            sampling["temperature"] = temperature
        if top_p is not None:
            sampling["top_p"] = top_p
        current_tier = tier
        dropped = []

        for _ in range(len(self.TUNABLE) + 1):
            path, body, headers = build_request(
                model, prompt, budget=max_tokens, system=system,
                sampling=sampling, tier=current_tier,
            )
            headers.update(self._project_header(surface(model)))
            code, data = post(path, body, region=self.region,
                              headers=headers or None, timeout=timeout,
                              attempts=attempts)
            if code == 200:
                if dropped:
                    print(f"      [{model}: dropped {', '.join(dropped)} and retried]")
                return self._extract(surface(model), data)

            refused = self._offending_key(err(data))
            if refused == "service_tier" and current_tier != "default":
                current_tier, _ = "default", dropped.append("service_tier")
                continue
            if refused in sampling:
                sampling.pop(refused)
                dropped.append(refused)
                continue
            raise RuntimeError(f"{model}: HTTP {code}: {err(data)}")
        raise RuntimeError(f"{model}: still refused after dropping {dropped}")

    @staticmethod
    def _extract(which, data):
        if which == "messages":
            return "".join(
                b.get("text", "")
                for b in data.get("content", [])
                if b.get("type") == "text"
            )
        if which == "chat":
            return (data.get("choices") or [{}])[0].get("message", {}).get(
                "content"
            ) or ""
        return response_text(data)


compat = MantleCompat()
PROMPT = "Name one benefit of a managed inference endpoint. One sentence."'''


def main() -> None:
    nb = Notebook(PATH)

    # §3: the routing primitives, which everything downstream now uses.
    idx = nb.find('def api_prefix(model_id: str) -> str:\n    if model_id.startswith("anthropic."):')
    assert len(idx) == 1, idx
    nb.set_source(idx[0], ROUTING)
    nb.sub("""## 3. Breakage 1 — the path prefix depends on the model

There is no single base URL that serves every model. Three prefixes exist, and the
OpenAI family is split across two of them.""",
           """## 3. Breakage 1 — routing: prefix, API surface, and budget field

There is no single base URL that serves every model. Three prefixes exist, and the
OpenAI family is split across two of them. Two more routing decisions ride along
with the prefix and are just as easy to get wrong:

- **Which API the model actually serves.** Most open-weight families have no
  Responses API, and the gpt-oss *safeguard* variants lack it even though base
  gpt-oss has it.
- **What the token budget is called.** `max_output_tokens` on Responses,
  `max_tokens` on Messages and most of Chat Completions — but
  `max_completion_tokens` for gpt-5.6 on Chat Completions.

The helpers below answer all three, and every probe in this notebook uses them. §7's
compatibility shim composes them too, so there is one place to correct.""")

    # §4, §5, §6 probes now use build_request rather than the not-yet-defined shim.
    idx = nb.find("CASES = [\n    {\"temperature\": 0.7, \"top_p\": 0.95},")
    assert len(idx) == 1, idx
    nb.set_source(idx[0], PARAM_PROBE)

    idx = nb.find('        # service_tier is not a Messages parameter at all, so there is nothing to')
    assert len(idx) == 1, idx
    nb.set_source(idx[0], TIER_PROBE)

    idx = nb.find("# Try BOTH budget parameter names before concluding an API is missing.")
    assert len(idx) == 1, idx
    nb.set_source(idx[0], API_PROBE)

    # §7 composes the primitives instead of restating them.
    idx = nb.find("class MantleCompat:")
    assert len(idx) == 1, idx
    nb.set_source(idx[0], SHIM)

    # The old §3 demo cell listed prefixes only; the new §3 cell covers it.
    dupes = nb.find('    print(f"  {mid:30} -> {api_prefix(mid)}")')
    for i in reversed(dupes):
        nb.delete(i)

    nb.save()
    print(f"99-cross-cutting/02: {nb.changes} edits")


if __name__ == "__main__":
    main()
