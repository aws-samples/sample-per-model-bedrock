#!/usr/bin/env python3
"""Audit fix 5 — 99-cross-cutting/01, the capability survey.

Defects fixed:
  * §3's API probe sent `max_tokens` to Chat Completions, so gpt-5.6 (which wants
    `max_completion_tokens`) read as "no Chat Completions".
  * §4's prose asserted "Gemma 4 takes `temperature`, rejects `top_p`" directly
    under a table showing `temperature` refused — three-way inconsistency.
  * §6's `capabilities.py` — explicitly offered for pasting into a project —
    routed every non-Anthropic model to `/responses`, and had no rule for Claude's
    "one of temperature or top_p, not both". §6's verification cell then said
    "a resolver that is wrong is worse than none" and showed it failing on 2 of 5.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PATH = f"{REPO}/99-cross-cutting/01-choosing-a-model-and-api.ipynb"

CAPABILITIES = r'''
"""Resolve bedrock-mantle request details per model. Generated from live probes.

Read the caveat before you rely on this. The constants below are a *snapshot*:
which models refuse which parameters has changed twice during this collection's
life. A resolver built on a static table is right until it is not, and it fails
closed in the worst way -- a 400 in production on a model you never re-tested.

The durable design is the one in `02-migrating-from-openai.ipynb` section 7: send
the request, and when the service returns a 400 that *names* a parameter, drop that
parameter and retry. Mantle is consistent about naming it. Use this module for
routing (which path, which API, which budget field) and let error handling deal
with sampling.
"""

OPENAI_PREFIX_FAMILIES = ("google.gemma-4", "openai.gpt-5", "xai.")

# Families served by Chat Completions only -- the Responses API 400s for these.
# Note gpt-oss-safeguard is here while base gpt-oss is not: the provider prefix
# is not enough to decide.
CHAT_ONLY = ("qwen.", "deepseek.", "zai.", "minimax.", "moonshotai.", "mistral.",
             "nvidia.", "writer.", "openai.gpt-oss-safeguard", "google.gemma-3")

# Chat Completions wants max_completion_tokens rather than max_tokens for these.
COMPLETION_TOKENS_FAMILIES = ("openai.gpt-5.6",)

# Models that reject `temperature` and `top_p` outright ("deprecated").
NO_SAMPLING = ("anthropic.claude-opus-5", "anthropic.claude-sonnet-5",
               "anthropic.claude-opus-4-8", "anthropic.claude-opus-4-7")
# Models that accept EITHER temperature or top_p but not both in one request.
ONE_SAMPLING_PARAM = ("anthropic.claude-haiku-4-5",)
# Models that accept `temperature` ONLY at its default 1.0, and reject `top_p`.
TEMPERATURE_DEFAULT_ONLY = ("openai.gpt-5.5", "openai.gpt-5.6")
# Models that accept flex/priority service tiers. Not a Messages parameter at all.
TIERED = ("openai.gpt-oss", "google.gemma-", "xai.", "qwen.", "deepseek.",
          "zai.", "minimax.", "moonshotai.", "mistral.", "nvidia.", "writer.")


def api_prefix(model_id):
    """Return the URL prefix that serves this model's inference APIs."""
    if model_id.startswith("anthropic."):
        return "/anthropic/v1"
    if model_id.startswith(OPENAI_PREFIX_FAMILIES):
        return "/openai/v1"
    return "/v1"


def surface(model_id):
    """Which API this model actually serves: messages / chat / responses."""
    if model_id.startswith("anthropic."):
        return "messages"
    if model_id.startswith(CHAT_ONLY):
        return "chat"
    return "responses"


def inference_path(model_id, api="auto"):
    """Full inference path, e.g. "/v1/chat/completions".

    `api="auto"` picks the surface the model actually serves. Defaulting to
    /responses for everything non-Anthropic -- which an earlier version of this
    module did -- 400s for every Chat-Completions-only family.
    """
    prefix = api_prefix(model_id)
    chosen = surface(model_id) if api == "auto" else api
    if chosen == "messages":
        return prefix + "/messages"
    if chosen == "chat":
        return prefix + "/chat/completions"
    return prefix + "/responses"


def token_limit_field(model_id, api="auto"):
    """The budget parameter this model+API expects.

    Getting this wrong yields a 400 that reads exactly like "this API does not
    exist here", which is a trap worth avoiding by construction.
    """
    chosen = surface(model_id) if api == "auto" else api
    if chosen == "responses":
        return "max_output_tokens"          # minimum 16
    if chosen == "chat" and model_id.startswith(COMPLETION_TOKENS_FAMILIES):
        return "max_completion_tokens"
    return "max_tokens"                     # Messages: required, no default


def sampling(model_id, temperature=None, top_p=None):
    """Drop or clamp parameters this model rejects, instead of earning a 400."""
    if model_id.startswith(NO_SAMPLING):
        return {}
    out = {}
    if temperature is not None:
        if model_id.startswith(TEMPERATURE_DEFAULT_ONLY):
            if abs(float(temperature) - 1.0) < 1e-9:
                out["temperature"] = 1.0
            # else omit entirely rather than 400
        else:
            out["temperature"] = temperature
    if top_p is not None and not model_id.startswith(TEMPERATURE_DEFAULT_ONLY):
        out["top_p"] = top_p
    # "`temperature` and `top_p` cannot both be specified for this model."
    if model_id.startswith(ONE_SAMPLING_PARAM) and len(out) == 2:
        out.pop("top_p")
    return out


def service_tier(model_id, tier="default"):
    """Downgrade to 'default' where flex/priority are unsupported.

    Returns None for Messages, where service_tier is not a parameter at all.
    """
    if surface(model_id) == "messages":
        return None
    if tier == "default" or model_id.startswith(TIERED):
        return tier
    return "default"
'''

VERIFY = '''# Verify the resolver against the live endpoint. A resolver that is wrong is worse
# than none -- so this cell exists to fail loudly if the constants above have aged.
print(f"{'model':40} {'path':30} {'resolved call':>14}")
print("-" * 88)
failures = []
for mid in (
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "openai.gpt-oss-safeguard-20b",
    "qwen.qwen3-32b",
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-sonnet-5",
):
    path = capabilities.inference_path(mid)
    limit_field = capabilities.token_limit_field(mid)
    body = {
        "model": mid,
        limit_field: 16,
        **capabilities.sampling(mid, temperature=0.7, top_p=0.95),
    }
    tier = capabilities.service_tier(mid, "flex")
    if tier:
        body["service_tier"] = tier
    if path.endswith("/messages"):
        body["messages"] = [{"role": "user", "content": "Hi"}]
        headers = AV
    elif path.endswith("/responses"):
        body["input"] = "Hi"
        headers = None
    else:
        body["messages"] = [{"role": "user", "content": "Hi"}]
        headers = None
    code, data = post(
        path, body, region=REGION, headers=headers, attempts=1, timeout=60
    )
    verdict = "200 OK" if code == 200 else str(code)
    print(f"{mid:40} {path:30} {verdict:>14}")
    if code != 200:
        failures.append((mid, err(data)[:100]))

print()
if failures:
    print(f"{len(failures)} of the calls above failed — the constants in")
    print("capabilities.py have aged. What the service said:")
    for mid, message in failures:
        print(f"   {mid}: {message}")
    print("\\nThis is the expected failure mode of a static table, and the reason")
    print("02-migrating-from-openai.ipynb §7 recovers from the error instead.")
else:
    print("All resolved calls succeeded: routing, budget field, sampling and tier")
    print("are correct for every model tried, today.")'''

API_PROBE = '''    # Try BOTH budget field names before concluding an API is missing. gpt-5.6
    # serves Chat Completions but refuses `max_tokens`, and reading that 400 as
    # "no Chat Completions" is how a false claim reached three other notebooks.
    code, _ = post(
        f"{prefix}/responses",
        {"model": model_id, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    out["responses"] = code
    for field in ("max_tokens", "max_completion_tokens"):
        code, _ = post(
            f"{prefix}/chat/completions",
            {
                "model": model_id,
                "messages": [{"role": "user", "content": "Hi"}],
                field: 16,
            },
            region=REGION,
            attempts=1,
            timeout=45,
        )
        if code == 200:
            out["chat_field"] = field
            break
    out["chat"] = code
    out.setdefault("chat_field", "-")
    out["messages"] = "-"
    return out'''


def main() -> None:
    nb = Notebook(PATH)

    # §3 probe: both budget fields.
    nb.sub('''    code, _ = post(
        f"{prefix}/responses",
        {"model": model_id, "input": "Hi", "max_output_tokens": 16},
        region=REGION,
        attempts=1,
        timeout=45,
    )
    out["responses"] = code
    code, _ = post(
        f"{prefix}/chat/completions",
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
        },
        region=REGION,
        attempts=1,
        timeout=45,
    )
    out["chat"] = code
    out["messages"] = "-"
    return out''', API_PROBE)
    nb.sub('''    if prefix == "/anthropic/v1":
        code, _ = post(
            f"{prefix}/messages",
            {
                "model": model_id,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            region=REGION,
            headers=AV,
            attempts=1,
            timeout=45,
        )
        out["messages"] = code
        out["responses"] = out["chat"] = "-"
        return out''',
           '''    if prefix == "/anthropic/v1":
        code, _ = post(
            f"{prefix}/messages",
            {
                "model": model_id,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            region=REGION,
            headers=AV,
            attempts=1,
            timeout=45,
        )
        out["messages"] = code
        out["responses"] = out["chat"] = "-"
        out["chat_field"] = "-"
        return out''')
    nb.sub('''print(f"{'model':40} {'prefix':16} {'Resp':>6} {'Chat':>6} {'Msg':>6}")
print("-" * 80)
for row in api_results:
    print(
        f"{row['model']:40} {row['prefix']:16} {str(row['responses']):>6} "
        f"{str(row['chat']):>6} {str(row['messages']):>6}"
    )''',
           '''print(f"{'model':40} {'prefix':16} {'Resp':>6} {'Chat':>6} {'Msg':>6} {'CC budget':>22}")
print("-" * 104)
for row in api_results:
    print(
        f"{row['model']:40} {row['prefix']:16} {str(row['responses']):>6} "
        f"{str(row['chat']):>6} {str(row['messages']):>6} "
        f"{str(row.get('chat_field', '-')):>22}"
    )''')

    # §4 prose that contradicted its own table.
    nb.sub("""Read the inverses carefully:

- **Gemma 4** takes `temperature`, rejects `top_p`.
- **Grok** rejects `temperature`, takes `top_p`.
- **Frontier Claude** rejects `temperature` (deprecated); `haiku-4-5` accepts it.
- **gpt-5.x** rejects the `flex` and `priority` tiers; gpt-oss accepts them.

So sampling and tier settings must be resolved per model, not per provider.""",
           """Read the table, not this paragraph — the paragraph is the part that goes stale.
When this notebook was first written it claimed Gemma 4 took `temperature` and
refused `top_p`, directly beneath a table showing `temperature` refused. Both the
table and the prose have since been wrong in different directions.

What the survey reliably shows:

- **There is no universal sampling config.** The GPT-5.5 and GPT-5.6 families
  accept `temperature` only at its default `1.0` and refuse `top_p`; newer Claude
  models refuse both as *deprecated*; `haiku-4-5` accepts either but not both in
  one request. Everything else is permissive.
- **`gpt-5.x` and Claude reject `flex`/`priority`.** For Claude, `service_tier` is
  not a Messages parameter at all, so `n/a` rather than a refusal.
- **A `-1` means the request stalled**, not that anything was rejected. That is why
  every probe here sets a timeout.

Resolve sampling and tier per model, and re-probe: three of these rows have
changed since the notebook was written.""")

    # §6 the resolver.
    old = nb.find("OPENAI_PREFIX_FAMILIES = (\"google.gemma-4\", \"openai.gpt-5\", \"xai.\")")
    assert len(old) == 1, old
    body = nb.source(old[0])
    head, _, tail = body.partition("CAPABILITIES_PY = '''")
    assert tail, "CAPABILITIES_PY assignment not found"
    after = tail.split("'''", 1)[1]
    nb.set_source(old[0], head + "CAPABILITIES_PY = '''" + CAPABILITIES + "'''" + after)

    # Show surface + budget field in the resolver demo table.
    nb.sub('''print(f"{'model':40} {'path':34} {'sampling':>34}")
print("-" * 112)
for mid in (
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "anthropic.claude-sonnet-5",
    "qwen.qwen3-32b",
):
    path = capabilities.inference_path(mid)
    sampling = capabilities.sampling(mid, temperature=0.7, top_p=0.95)
    print(f"{mid:40} {path:34} {json.dumps(sampling):>34}")''',
           '''print(f"{'model':32} {'path':30} {'budget field':22} sampling")
print("-" * 118)
for mid in (
    "google.gemma-4-31b",
    "xai.grok-4.3",
    "openai.gpt-5.6-sol",
    "openai.gpt-oss-120b",
    "openai.gpt-oss-safeguard-20b",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5",
    "qwen.qwen3-32b",
):
    print(
        f"{mid:32} {capabilities.inference_path(mid):30} "
        f"{capabilities.token_limit_field(mid):22} "
        f"{json.dumps(capabilities.sampling(mid, temperature=0.7, top_p=0.95))}"
    )''')

    # §6 verification.
    old_v = nb.find("# Verify the resolver against the live endpoint")
    assert len(old_v) == 1, old_v
    nb.set_source(old_v[0], VERIFY)

    # Decision guide + gotchas.
    nb.sub("| Reasoning traces you can read | Responses API: gemma-4, gpt-5.x, gpt-oss, grok |",
           "| Reasoning traces you can read | Responses API: gemma-4 (at `effort=\"high\"`), gpt-5.x, gpt-oss. Grok's is **encrypted**. On Chat Completions several `/v1` families put it in the non-standard `message.reasoning` |")
    nb.sub("| No universal sampling config | On Responses, `temperature` must be `1.0` for Gemma 4 / gpt-5.6; Grok rejects it entirely |",
           "| No universal sampling config | gpt-5.5/5.6 want `temperature=1.0` and refuse `top_p`; newer Claude refuses both; `haiku-4-5` takes either but not both |")
    nb.sub("| Responses API coverage | Only a minority of families; Chat Completions is universal |",
           "| Responses API coverage | Only a minority of families — and the gpt-oss **safeguard** variants lack it while base gpt-oss has it |\n"
           "| Budget field name | Chat Completions wants `max_completion_tokens` on gpt-5.6. That 400 reads like a missing API |")
    nb.save()
    print(f"99-cross-cutting/01: {nb.changes} edits")


if __name__ == "__main__":
    main()
