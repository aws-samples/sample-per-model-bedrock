#!/usr/bin/env python3
"""Retitle the family notebooks and give each a live endpoint/model-ID section.

Eighteen notebooks were titled "... on Amazon Bedrock Mantle". For most of those
families that is now misleading: since August 2026 the models are on
bedrock-runtime as well, which is the endpoint AWS recommends. The titles said
mantle-only, so a reader would conclude mantle-only.

Two changes per notebook:
  1. Title: drop "Mantle" where the family is on both endpoints. Keep it, and say
     why, where the model or the feature really is mantle-only.
  2. A new "Which endpoint, and the model ID for each" section right after the
     imports, which PROBES both catalogues instead of tabulating them.
"""
import json
import re
import sys
import os
REPO_ROOT = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT + "/_shared")
from bedrock import list_models, runtime_id_for  # noqa: E402
from nbedit import Notebook  # noqa: E402

REPO = REPO_ROOT + "/"
REGION = "us-east-1"
MANTLE = set(list_models(REGION))
PROV = (r"(?:anthropic|google|openai|xai|qwen|deepseek|zai|mistral|moonshotai"
        r"|minimax|nvidia|writer|amazon|meta)\.")

# Title rewrites. Where a notebook stays "Mantle" the reason is in the value.
TITLES = {
    "01-openai-gpt/01-responses-api-core.ipynb":
        "# OpenAI GPT on Amazon Bedrock — Responses API core",
    "01-openai-gpt/03-tools-and-structured-output.ipynb":
        "# Tools and structured output — OpenAI GPT on Amazon Bedrock",
    "01-openai-gpt/04-prompt-caching-and-cost.ipynb":
        "# Prompt caching on Amazon Bedrock — OpenAI GPT models",
    "02-anthropic-claude/01-messages-api-core.ipynb":
        "# Anthropic Claude on Amazon Bedrock — Messages API core",
    "02-anthropic-claude/02-thinking-tools-and-caching.ipynb":
        "# Claude on Amazon Bedrock — thinking, tools, and prompt caching",
    "02-anthropic-claude/03-agentic-computer-use-and-memory.ipynb":
        "# Claude agentic tools on Amazon Bedrock — computer use, memory, compaction",
    "04-qwen/01-qwen3-core-and-tools.ipynb":
        "# Qwen3 on Amazon Bedrock",
    "04-qwen/02-qwen3-coder-and-vision.ipynb":
        "# Qwen3 Coder and Vision on Amazon Bedrock",
    "05-deepseek/01-deepseek-v3-reasoning.ipynb":
        "# DeepSeek V3 on Amazon Bedrock",
    "06-zai-glm/01-glm-family.ipynb":
        "# Z.AI GLM on Amazon Bedrock",
    "07-mistral/01-mistral-text-and-sizes.ipynb":
        "# Mistral AI on Amazon Bedrock — text models and the size ladder",
    "07-mistral/02-devstral-and-voxtral.ipynb":
        "# Devstral and Voxtral on Amazon Bedrock — Mistral's specialists",
    "08-moonshot-kimi/01-kimi-k2.ipynb":
        "# Moonshot AI Kimi K2 on Amazon Bedrock",
    "09-minimax/01-minimax-m2.ipynb":
        "# MiniMax M2 on Amazon Bedrock",
    "10-nvidia-nemotron/01-nemotron-nano-and-super.ipynb":
        "# NVIDIA Nemotron on Amazon Bedrock",
    "12-writer-palmyra/01-palmyra-vision.ipynb":
        "# Writer Palmyra Vision on Amazon Bedrock",
    "99-cross-cutting/01-choosing-a-model-and-api.ipynb":
        "# Choosing a model, an endpoint, and an API on Amazon Bedrock",
    "99-cross-cutting/02-migrating-from-openai.ipynb":
        "# Migrating to Amazon Bedrock from OpenAI or Anthropic",
    "99-cross-cutting/03-production-hardening.ipynb":
        "# Production hardening for Amazon Bedrock",
    # These two stay "Mantle" because the FEATURE is mantle-only, not just the
    # model. The retitle would be the misleading change here.
    "01-openai-gpt/02-web-search-and-grounding.ipynb":
        "# Web Search on Amazon Bedrock Mantle — grounding OpenAI GPT models",
    "01-openai-gpt/05-server-side-tools-and-fine-tuning.ipynb":
        "# Server-side tools, batch, and fine-tuning — OpenAI GPT on Bedrock Mantle",
    # Genuinely mantle-only models.
    "03-google-gemma/01-gemma4-end-to-end.ipynb":
        "# Gemma 4 on Amazon Bedrock Mantle — end to end",
    "11-xai-grok/01-grok-4-3.ipynb":
        "# xAI Grok 4.3 on Amazon Bedrock Mantle",
}

SECTION_MD = """### Which endpoint, and the model ID for each

AWS recommends `bedrock-runtime` for new applications, and since August 2026 it
serves the OpenAI- and Anthropic-compatible APIs as well as Converse. So before
the first call, the question is which endpoint you want — and that has a
complication worth knowing about:

**the same model often carries a different ID on each endpoint.** Send a
`bedrock-mantle` ID to `bedrock-runtime` and you get *"The provided model
identifier is invalid"*, which reads like a missing model rather than a missing
translation.

The cell below asks both catalogues rather than stating an answer that will age.
`runtime_id_for()` returns `None` when a model is genuinely not on
`bedrock-runtime`, which is the honest signal for "you need mantle for this one"."""


def section_code(models: list[str]) -> str:
    listed = "\n".join(f'    "{m}",' for m in models)
    return f'''from bedrock import endpoints_for, runtime_id_for

COVERED = [
{listed}
]

print(f"{{'model (as named on mantle)':38}} {{'on runtime as':40}} endpoints")
print("-" * 96)
mantle_only = []
for model_id in COVERED:
    runtime_id = runtime_id_for(model_id, REGION)
    where = endpoints_for(model_id, REGION)
    label = ", ".join(name for name, present in where.items() if present) or "neither"
    if runtime_id is None:
        mantle_only.append(model_id)
    print(f"{{model_id:38}} {{(runtime_id or '-- not on runtime --'):40}} {{label}}")

renamed = [
    m for m in COVERED
    if (r := runtime_id_for(m, REGION)) is not None and r != m
]
print()
print(f"=> {{len(COVERED) - len(mantle_only)}}/{{len(COVERED)}} of these are on "
      f"bedrock-runtime; {{len(renamed)}} under a different id.")
if mantle_only:
    print(f"   bedrock-mantle only: {{mantle_only}}")
    print("   For those, this notebook's endpoint is the only one that serves them.")
else:
    print("   Every model here is on both endpoints. This notebook shows the")
    print("   bedrock-mantle calls; the ids above are what you send to switch.")
print("   Region matters too: a model absent here can be present elsewhere, so")
print("   re-run this in the Region you intend to deploy in.")'''


def models_in(path: str) -> list[str]:
    nb = json.load(open(REPO + path))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    found = {m for m in re.findall(PROV + r"[A-Za-z0-9._:\-]+", src) if m in MANTLE}
    return sorted(found)


changed = 0
for rel, new_title in sorted(TITLES.items()):
    nb = Notebook(REPO + rel)
    old_title = nb.source(0).split("\n")[0]
    if old_title != new_title:
        nb.sub_cell(0, old_title, new_title)

    models = models_in(rel)
    if not models:
        nb.save()
        print(f"{rel:58} title only (no model ids found)")
        changed += 1
        continue

    # Insert after the imports cell: the first code cell in the notebook.
    imports_at = next(
        i for i, c in enumerate(nb.nb["cells"]) if c["cell_type"] == "code"
    )
    if "### Which endpoint, and the model ID for each" in "".join(
        "".join(c["source"]) for c in nb.nb["cells"]
    ):
        nb.save()
        print(f"{rel:58} already has the endpoint section")
        continue

    i = nb.insert_after(imports_at, "markdown", SECTION_MD)
    nb.insert_after(i, "code", section_code(models))
    nb.save()
    print(f"{rel:58} retitled + endpoint section ({len(models)} models)")
    changed += 1

print(f"\n{changed} notebook(s) updated")
