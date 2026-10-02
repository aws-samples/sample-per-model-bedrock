#!/usr/bin/env python3
"""Audit fix 1 — two systemic defects.

A. `output_text` read from a raw HTTP body. The Responses API has no top-level
   `output_text`; it is an OpenAI SDK convenience. Four sites printed `""` or
   dumped raw JSON. `response_text()` in _shared/bedrock.py already walks
   output[].content[].text correctly.

B. The claim "Gemma 4 rejects top_p, and Grok rejects temperature", which the
   service no longer does. Rather than substituting today's answer — which would
   age exactly the same way — the prose now states the durable rule and points at
   the probe in the same section.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))


def fix_foundations_01() -> None:
    nb = Notebook(f"{REPO}/00-foundations/01-endpoints-auth-and-the-three-paths.ipynb")

    # Cell 2 imports safe_print only; response_text is needed below.
    nb.sub("from bedrock import safe_print",
           "from bedrock import response_text, safe_print")

    # §3 SigV4 probe printed json.dumps(data.get("output_text", "")) -> always "".
    nb.sub(
        '''    parsed = json.loads(resp.text) if resp.text.strip() else {}
    return resp.status_code, parsed''',
        '''    parsed = json.loads(resp.text) if resp.text.strip() else {}
    return resp.status_code, parsed''')
    nb.sub(
        'print("SigV4 ->", status, "|", json.dumps(data.get("output_text", ""))[:60])',
        '# The wire format has NO top-level "output_text" -- that is a convenience the\n'
        '# OpenAI SDK computes. On the raw body the answer is at\n'
        '# output[].content[].text, which is what response_text() walks.\n'
        'print("SigV4 ->", status, "|", repr(response_text(data)[:60]))\n'
        'print("top-level keys:", sorted(data)[:6], "...")\n'
        'print("output_text present on the wire?", "output_text" in data)')

    # §5 curl probe fell through to dumping raw JSON for the same reason.
    nb.sub(
        '''    body = json.loads(completed.stdout)
    print("live result:", body.get("output_text", completed.stdout[:120]))''',
        '''    body = json.loads(completed.stdout)
    # Same trap as section 3: read output[].content[].text, not "output_text".
    print("live result:", repr(response_text(body)[:80]))''')

    # The helper table should name the helper the cells now use.
    nb.sub(
        "| `safe_print` | `print()` with account IDs, IAM principals and opaque service IDs redacted |",
        "| `response_text` | assistant text from a Responses API payload — the raw body has **no** `output_text` field |\n"
        "| `safe_print` | `print()` with account IDs, IAM principals and opaque service IDs redacted |")

    # Gotcha table: record the trap.
    nb.sub(
        "| `/openai/v1/models` | 404. Inventory is only at `/v1/models` |",
        "| `/openai/v1/models` | 404. Inventory is only at `/v1/models` |\n"
        "| No `output_text` on the wire | It is an SDK convenience. On a raw body read `output[].content[].text` |")
    nb.save()
    print(f"foundations/01: {nb.changes} edits")


def fix_foundations_03() -> None:
    nb = Notebook(f"{REPO}/00-foundations/03-scaling-tiers-and-latency.ipynb")

    nb.sub("from bedrock import err, post, stream_lines, ttft",
           "from bedrock import err, post, response_text, stream_lines, ttft")

    nb.sub('print("result:", status, json.dumps(data.get("output_text", ""))[:40])',
           '# response_text(), not data["output_text"]: the raw Responses body has no\n'
           '# such field, so .get() would silently return "" on a perfectly good reply.\n'
           'print("result:", status, repr(response_text(data)[:40]))')

    # The hardened client dropped every answer.
    nb.sub('''        return {
            "text": data.get("output_text", ""),''',
           '''        return {
            # response_text() walks output[].content[].text. Reading
            # data["output_text"] returns "" for every successful call.
            "text": response_text(data),''')

    nb.sub(
        "| `ttft` | times a streaming call: time-to-first-token and output frames/sec |",
        "| `response_text` | assistant text from a Responses API payload — the raw body has **no** `output_text` field |\n"
        "| `ttft` | times a streaming call: time-to-first-token and output frames/sec |")

    nb.sub("| Spiking | Ramp over minutes; 0→peak invites 503s |",
           "| Spiking | Ramp over minutes; 0→peak invites 503s |\n"
           "| No `output_text` on the wire | An SDK convenience. On a raw body read `output[].content[].text` |")
    nb.save()
    print(f"foundations/03: {nb.changes} edits")


# --- B. the stale sampling claim ------------------------------------------

# The seven Chat-Completions family notebooks share this paragraph verbatim.
FAMILY_OLD = """This family accepts both `temperature` and `top_p`. That is *not* universal on
mantle — Gemma 4 rejects `top_p`, and Grok rejects `temperature` — so never share
one sampling config across families."""

FAMILY_NEW = """This family accepts both `temperature` and `top_p`. That is *not* universal on
mantle: the GPT-5.5 and GPT-5.6 families accept `temperature` only at its default
`1.0` and refuse `top_p` outright, and newer Claude models reject both as
deprecated. So never share one sampling config across families — probe each one,
as the cell below does.

Which models refuse what has already changed twice during this collection's life:
Gemma 4 and Grok both tightened in August 2026 and have since been relaxed again.
Read the probe output, not this paragraph."""

MISTRAL_OLD = """This family accepts both `temperature` and `top_p`. That is *not* universal on
mantle: Gemma 4 rejects `top_p`, Grok rejects `temperature`, and newer Claude
models reject `temperature` too. Never share one sampling config across families."""


def fix_family_notebooks() -> None:
    for rel in ("04-qwen/01-qwen3-core-and-tools.ipynb",
                "05-deepseek/01-deepseek-v3-reasoning.ipynb",
                "06-zai-glm/01-glm-family.ipynb",
                "08-moonshot-kimi/01-kimi-k2.ipynb",
                "09-minimax/01-minimax-m2.ipynb",
                "10-nvidia-nemotron/01-nemotron-nano-and-super.ipynb",
                "12-writer-palmyra/01-palmyra-vision.ipynb"):
        nb = Notebook(f"{REPO}/{rel}")
        nb.sub(FAMILY_OLD, FAMILY_NEW)
        nb.save()
        print(f"{rel}: {nb.changes} edits")

    nb = Notebook(f"{REPO}/07-mistral/01-mistral-text-and-sizes.ipynb")
    nb.sub(MISTRAL_OLD, FAMILY_NEW)
    nb.save()
    print(f"07-mistral/01: {nb.changes} edits")


def fix_readme() -> None:
    path = f"{REPO}/README.md"
    with open(path) as fh:
        text = fh.read()
    old = ("Bedrock model behaviour is not uniform, and the differences are the part that costs\n"
           "you time. Gemma 4 pins `temperature` to its default and rejects `top_p`. Grok takes\n"
           "`max_completion_tokens` where the `/v1` families take `max_tokens`. Palmyra Vision has\n"
           "no tool support at all. Most Claude models must be addressed through a cross-Region\n"
           "inference profile and are rejected by their bare model ID. None of that is discoverable\n"
           "from a generic example, so each family gets its own notebook rather than a shared one.")
    new = ("Bedrock model behaviour is not uniform, and the differences are the part that costs\n"
           "you time. GPT-5.6 pins `temperature` to its default, rejects `top_p`, and takes\n"
           "`max_completion_tokens` where the `/v1` families take `max_tokens`. Palmyra Vision has\n"
           "no usable tool support at all. Most Claude models must be addressed through a\n"
           "cross-Region inference profile and are rejected by their bare model ID. Newer Claude\n"
           "models reject `temperature` as deprecated. None of that is discoverable from a generic\n"
           "example, so each family gets its own notebook rather than a shared one.\n"
           "\n"
           "It also moves. Gemma 4's parameter surface tightened in August 2026 and was later\n"
           "relaxed again; Grok's did the same. Every notebook therefore **probes** the endpoint in\n"
           "front of you rather than reprinting a table from the day it was written, and where a\n"
           "table does appear it is labelled as a snapshot.")
    if old not in text:
        raise AssertionError("README paragraph not found")
    with open(path, "w") as fh:
        fh.write(text.replace(old, new))
    print("README.md: 1 edit")


if __name__ == "__main__":
    fix_foundations_01()
    fix_foundations_03()
    fix_family_notebooks()
    fix_readme()
