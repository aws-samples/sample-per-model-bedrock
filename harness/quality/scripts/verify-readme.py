#!/usr/bin/env python3
"""Check every factual claim in README.md against the live service and the repo.

Written because two false claims reached that file this week, both introduced while
*shortening* prose. Reading it again is not a check; this is.

    python3 verify-readme.py
"""
import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO + "/_shared")
from bedrock import (  # noqa: E402
    api_prefix, endpoints_for, list_models, runtime_id_for, runtime_models,
)

REGION = "us-east-1"
README = open(REPO + "/README.md").read()
ok_count = 0
fails = []


def check(label, condition, detail=""):
    global ok_count
    if condition:
        ok_count += 1
        print(f"  ok    {label}")
    else:
        fails.append(f"{label} — {detail}")
        print(f"  FAIL  {label}  {detail}")


print("=== files and links referenced ===")
for target in sorted(set(re.findall(r"\]\((?!http)([^)#]+)", README))):
    check(f"path exists: {target}", os.path.exists(os.path.join(REPO, target)))
for inline in sorted(set(re.findall(r"`([0-9]{2}-[a-z0-9-]+/[0-9]{2}-[a-z0-9-]+\.ipynb)`",
                                    README))):
    check(f"inline path: {inline}", os.path.exists(os.path.join(REPO, inline)))

print("\n=== helper symbols the README names ===")
import bedrock as B  # noqa: E402
for name in ("post", "converse", "safe_print", "ok", "runtime_id_for",
             "endpoints_for", "api_prefix"):
    check(f"_shared/bedrock.py exposes {name}()", hasattr(B, name))

print("\n=== requirements.txt pins the SDKs the snippets import ===")
reqs = open(REPO + "/requirements.txt").read()
for pkg in ("openai", "boto3", "aws-bedrock-token-generator"):
    check(f"requirements pins {pkg}", re.search(rf"^{re.escape(pkg)}==", reqs, re.M)
          is not None)

print("\n=== the per-endpoint model ID claim ===")
check("openai.gpt-oss-20b -> openai.gpt-oss-20b-1:0 on runtime",
      runtime_id_for("openai.gpt-oss-20b", REGION) == "openai.gpt-oss-20b-1:0",
      f"got {runtime_id_for('openai.gpt-oss-20b', REGION)!r}")
for m in ("anthropic.claude-opus-5", "openai.gpt-5.6-sol", "xai.grok-4.6"):
    rid = runtime_id_for(m, REGION)
    check(f"{m} needs a us./global. prefix on runtime",
          rid is not None and rid.startswith(("us.", "global.")), f"got {rid!r}")

print("\n=== the per-endpoint path claim ===")
runtime_ids = [r for m in list_models(REGION)
               if (r := runtime_id_for(m, REGION)) is not None]
prefixes = {api_prefix(r, "runtime") for r in runtime_ids}
check("runtime serves no /v1 inference path", "/v1" not in prefixes,
      f"prefixes seen: {sorted(prefixes)}")
non_anthropic = [r for r in runtime_ids if "anthropic." not in r]
check("every non-Claude runtime model is on /openai/v1",
      all(api_prefix(r, "runtime") == "/openai/v1" for r in non_anthropic))
mantle_prefixes = {api_prefix(m) for m in list_models(REGION)}
check("mantle prefix varies by family", len(mantle_prefixes) > 1,
      f"got {sorted(mantle_prefixes)}")

print("\n=== the family table, row by row ===")
# (folder, model-id substrings the row names, endpoint claim per model)
ROWS = {
    "01-openai-gpt": {"openai.gpt-5.6-sol": "both", "openai.gpt-5.6-terra": "both",
                      "openai.gpt-5.6-luna": "both", "openai.gpt-5.5": "mantle",
                      "openai.gpt-5.4": "mantle", "openai.gpt-oss-20b": "both",
                      "openai.gpt-oss-120b": "both"},
    "02-anthropic-claude": {"anthropic.claude-opus-5": "both",
                            "anthropic.claude-sonnet-5": "both",
                            "anthropic.claude-opus-4-8": "both",
                            "anthropic.claude-opus-4-7": "both",
                            "anthropic.claude-haiku-4-5": "both",
                            "anthropic.claude-fable-5": "both"},
    "03-google-gemma": {"google.gemma-4-31b": "mantle", "google.gemma-4-26b-a4b": "mantle",
                        "google.gemma-4-e2b": "mantle", "google.gemma-3-4b-it": "both",
                        "google.gemma-3-12b-it": "both", "google.gemma-3-27b-it": "both"},
    "04-qwen": {"qwen.qwen3-32b": "both", "qwen.qwen3-235b-a22b-2507": "mantle",
                "qwen.qwen3-next-80b-a3b-instruct": "both",
                "qwen.qwen3-coder-30b-a3b-instruct": "both",
                "qwen.qwen3-coder-480b-a35b-instruct": "mantle",
                "qwen.qwen3-coder-next": "both",
                "qwen.qwen3-vl-235b-a22b-instruct": "both"},
    "05-deepseek": {"deepseek.v3.2": "both", "deepseek.v3.1": "mantle"},
    "06-zai-glm": {"zai.glm-5": "both", "zai.glm-4.7": "both",
                   "zai.glm-4.7-flash": "both", "zai.glm-4.6": "mantle"},
    "07-mistral": {"mistral.mistral-large-3-675b-instruct": "both",
                   "mistral.ministral-3-3b-instruct": "both",
                   "mistral.ministral-3-8b-instruct": "both",
                   "mistral.ministral-3-14b-instruct": "both",
                   "mistral.magistral-small-2509": "both",
                   "mistral.devstral-2-123b": "both",
                   "mistral.voxtral-small-24b-2507": "both"},
    "08-moonshot-kimi": {"moonshotai.kimi-k2.5": "both",
                         "moonshotai.kimi-k2-thinking": "both"},
    "09-minimax": {"minimax.minimax-m2.5": "both", "minimax.minimax-m2.1": "both",
                   "minimax.minimax-m2": "both"},
    "10-nvidia-nemotron": {"nvidia.nemotron-super-3-120b": "both",
                           "nvidia.nemotron-nano-9b-v2": "both",
                           "nvidia.nemotron-nano-12b-v2": "both",
                           "nvidia.nemotron-nano-3-30b": "both"},
    "11-xai-grok": {"xai.grok-4.3": "mantle"},   # 4.6 checked separately
    "12-writer-palmyra": {"writer.palmyra-vision-7b": "both"},
}
for folder, models in ROWS.items():
    for mid, claim in models.items():
        where = endpoints_for(mid, REGION)
        actual = ("both" if where["mantle"] and where["runtime"]
                  else "mantle" if where["mantle"]
                  else "runtime" if where["runtime"] else "neither")
        check(f"{folder}: {mid} claimed {claim}", actual == claim,
              f"live says {actual} ({where})")

print("\n=== runtime-only families ===")
rt_cat = runtime_models(REGION)
for mid in ("amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0",
            "meta.llama4-scout-17b-instruct-v1:0",
            "meta.llama4-maverick-17b-instruct-v1:0",
            "writer.palmyra-x4-v1:0", "writer.palmyra-x5-v1:0"):
    key = mid.split(":")[0]
    check(f"runtime catalogue has {mid}", key in rt_cat)
    check(f"{mid} is NOT on mantle", not endpoints_for(mid, REGION)["mantle"])

print("\n=== gpt-oss-safeguard is on both, as the 14- row claims ===")
for mid in ("openai.gpt-oss-safeguard-20b", "openai.gpt-oss-safeguard-120b"):
    w = endpoints_for(mid, REGION)
    check(f"{mid} on both", w["mantle"] and w["runtime"], str(w))

print(f"\n{ok_count} checks passed, {len(fails)} failed")
if fails:
    print("\nFAILURES:")
    for f in fails:
        print(f"  - {f}")
sys.exit(1 if fails else 0)
