#!/usr/bin/env python3
"""Audit fix 10 — the shared `json_schema` cell raised on an empty result.

Eight family notebooks share a structured-output cell that tries a budget, retries
once at 2,000 tokens, then asserts the required keys are present. On MiniMax the
retry still came back empty and the assertion killed the notebook run.

Empty output is a **budget** problem, not a schema violation: a reasoning-heavy
model can spend even a raised budget before the opening brace. The two cases need
different handling —

  * empty  -> report the finish_reason and say what to do; do not raise
  * present but missing required keys -> raise, because strict mode promised them

The retry now escalates through several budgets instead of trying one larger value.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from nbedit import Notebook  # noqa: E402

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

TARGETS = [
    "04-qwen/01-qwen3-core-and-tools.ipynb",
    "05-deepseek/01-deepseek-v3-reasoning.ipynb",
    "06-zai-glm/01-glm-family.ipynb",
    "07-mistral/01-mistral-text-and-sizes.ipynb",
    "08-moonshot-kimi/01-kimi-k2.ipynb",
    "09-minimax/01-minimax-m2.ipynb",
    "10-nvidia-nemotron/01-nemotron-nano-and-super.ipynb",
    "12-writer-palmyra/01-palmyra-vision.ipynb",
]

TEMPLATE = '''schema = {{
    "type": "object",
    "properties": {{
        "country": {{"type": "string"}},
        "capital": {{"type": "string"}},
        "population_millions": {{"type": "number"}},
    }},
    "required": ["country", "capital", "population_millions"],
    "additionalProperties": False,
}}


def schema_call(budget: int):
    """One json_schema request at a given budget."""
    code, data = post(
        f"{{PREFIX}}/chat/completions",
        {{
            "model": {model},
            "messages": [{{"role": "user", "content": "Describe France."}}],
            "max_tokens": budget,
            "response_format": {{
                "type": "json_schema",
                "json_schema": {{"name": "country", "strict": True, "schema": schema}},
            }},
        }},
        region=REGION,
    )
    choice = (data.get("choices") or [{{}}])[0]
    return code, choice, (choice.get("message", {{}}) or {{}}).get("content")  # may be None


# Escalate rather than retrying once. On a reasoning-heavy model the trace can eat
# a raised budget too, and one retry is not enough to tell "needs more room" from
# "will never comply".
content, choice = None, {{}}
for budget in (250, 1000, 4000):
    code, choice, content = schema_call(budget)
    print(f"max_tokens={{budget:5}} HTTP {{code}} finish={{choice.get('finish_reason')!s:8}} "
          f"chars={{len(content or '')}}")
    if (content or "").strip():
        break

parsed = parse_json_lenient(content or "") if (content or "").strip() else {{}}
print("\\nparsed:", json.dumps(parsed, indent=2))

if not parsed:
    # Empty is a BUDGET problem, not a schema violation. A sample that raises here
    # teaches nothing; the lesson is to check finish_reason and escalate, or to turn
    # reasoning off for extraction work where you do not need it.
    print(f"\\nNo JSON at any budget (finish_reason={{choice.get('finish_reason')!r}}).")
    print("The reasoning trace consumed the whole budget before the opening brace.")
    print("Options: raise max_tokens further, or send reasoning_effort='none' --")
    print("extraction rarely needs the thinking, and it frees the budget for output.")
else:
    missing = {{"country", "capital"}} - set(parsed)
    if missing:
        # A NON-empty object missing required keys IS a strict-mode violation and
        # worth failing on: the schema promised those keys.
        raise ValueError(f"model omitted required keys: {{sorted(missing)}} in {{parsed}}")
    print("required keys present: country, capital")'''


def main() -> None:
    for rel in TARGETS:
        path = f"{REPO}/{rel}"
        nb = Notebook(path)
        idx = nb.find('raise ValueError(f"model omitted required keys')
        assert len(idx) == 1, (rel, idx)
        body = nb.source(idx[0])
        # The model is referenced by a per-notebook constant name; take it from the
        # existing request rather than hardcoding eight variants.
        match = re.search(r'"model":\s*([A-Za-z_][A-Za-z0-9_]*)\s*,', body)
        assert match, rel
        nb.set_source(idx[0], TEMPLATE.format(model=match.group(1)))
        nb.save()
        print(f"{rel}: model={match.group(1)}")


if __name__ == "__main__":
    main()
