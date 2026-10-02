#!/usr/bin/env python3
"""Enumerate every falsifiable assertion the notebooks make about the service.

Step 2 of the quality goal reads "for every feature-model-cell, find evidence". 1,262
cells is not a workable unit: most are prose scaffolding that asserts nothing, and a
single cell can carry six independent claims. The workable unit is the CLAIM -- a
sentence a reader would act on, that the service could falsify.

This finds them by shape rather than by meaning, so it cannot quietly skip one:

  ID        a model identifier, with the family prefix the service uses
  PATH      an API path or URL prefix
  REGION    a Region name asserted as available or not
  SUPPORT   "supports" / "does not support" / "only" / "refuses" / "ignores"
  STATUS    an HTTP status asserted for a named call
  FIELD     a request or response field path
  NUMBER    a quantity in prose that an adjacent output must establish

Every claim is emitted with the file, cell, sentence and category, so evidence can be
attached to each one and the gaps counted. A claim with no evidence is the finding.

    python3 extract-claims.py                 # table to stdout
    python3 extract-claims.py --json out.json # machine-readable inventory
    python3 extract-claims.py --self-test     # prove each rule fires and stays quiet
"""
import argparse
import glob
import json
import os
import re
import sys

# Resolved from this script's own location, so moving the tree costs nothing.
# Override with REPO=... to point at a different clone.
REPO = os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

# Families as the service spells them, so a typo in a notebook shows up as a
# non-matching ID rather than silently passing.
FAMILIES = (
    "openai", "anthropic", "google", "qwen", "deepseek", "zai", "mistral",
    "moonshotai", "minimax", "nvidia", "xai", "writer", "amazon", "meta",
)
GEOS = ("us", "eu", "apac", "in", "global")

# A model ID's second component always carries a digit or a hyphen (`gpt-5.6`,
# `nova-lite-v1`, `claude-opus-5`, `gemma-4-31b`). Requiring one is what stops
# `docs.aws.amazon.com` being harvested as an Amazon model, which it was: 11 of the
# first 45 findings were the string `amazon.com` lifted out of a documentation link.
# The lookbehind rejects a match that is part of a longer dotted hostname or a path.
ID_RE = re.compile(
    r"(?<![./\w-])(?:(?:" + "|".join(GEOS) + r")\.)?"
    r"(?:" + "|".join(FAMILIES) + r")\."
    r"(?=[a-z0-9.\-]*[0-9\-])"
    r"[a-z0-9][a-z0-9.\-]*[a-z0-9](?::[0-9]+)?\b"
)
PATH_RE = re.compile(r"(?:https://[a-z0-9.\-]+\.api\.aws)?/(?:openai/|anthropic/)?v1"
                     r"(?:/[a-z0-9\-/]*)?")
REGION_RE = re.compile(r"\b(?:us|eu|ap|ca|sa|me|af|il)-[a-z]+-[0-9]\b")
STATUS_RE = re.compile(r"\b(?:HTTP\s*)?(?:200|400|403|404|409|422|424|429|5[0-9]{2})\b")
# A dotted name in backticks is a request or response field -- unless it is a filename.
# `requirements.txt` appeared in all 34 notebooks' preamble and was the single largest
# false-positive source.
NOT_A_FIELD = re.compile(
    r"\.(?:txt|md|py|ipynb|json|yaml|yml|cfg|toml|lock|env|sh|png|jpg|mp3|wav)$")
FIELD_RE = re.compile(
    r"`(?:[a-z_][a-z0-9_]*)(?:\[[0-9\"'a-z_]+\]|\.[a-z_][a-z0-9_]*)+`|"
    r"`(?:max_tokens|max_completion_tokens|max_output_tokens|reasoning_effort|"
    r"reasoning_budget|temperature|top_p|top_k|tool_choice|service_tier|"
    r"input_audio|audio_url|image_url|json_schema|inputSchema|toolUseId|"
    r"toolResult|thinking|anthropic_version|chat_template_kwargs)`"
)
# "supports", "does not support", "only", "refuses", "rejects", "ignores", "accepts"
SUPPORT_RE = re.compile(
    r"\b(?:does not |doesn't |never |cannot |can't |won't )?"
    r"(?:support|accept|honour|honor|allow|serve|expose|return|refuse|reject|ignore)"
    r"(?:s|ed|ing)?\b", re.I)
ONLY_RE = re.compile(r"\b(?:only|solely|exclusively|no other|nothing else)\b", re.I)
NUMBER_RE = re.compile(r"\b[0-9]+(?:\.[0-9]+)?\s*(?:of|/)\s*[0-9]+\b|"
                       r"\b[0-9]+\s*(?:models?|rows?|families|tokens?|ms|"
                       r"seconds?|levels?|calls?|replies|samples?|variants?)\b", re.I)

# Prose that talks ABOUT the code rather than about the service makes no claim the
# service can falsify. These are the shapes that produced false positives.
HEDGED = re.compile(
    r"\b(?:may|might|could|probably|likely|appears|seems|expect|typically|usually|"
    r"generally|often|sometimes|in principle|as of|at the time|your own|yours|"
    r"depends on|varies|drift|check|verify|re-run|rerun|probe|measure)\b", re.I)


def sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking on `gpt-5.6` or `us-east-1`."""
    # Protect decimals inside identifiers and versions before splitting on ". ".
    guarded = re.sub(r"([a-z0-9])\.([0-9])", "\\1\x00\\2", text)
    parts = re.split(r"(?<=[.!?])\s+|\n\s*[-*]\s+|\n\s*\|", guarded)
    return [p.replace("\x00", ".").strip() for p in parts if p.strip()]


def fields(sentence: str) -> list[str]:
    """Field paths in a sentence, with filenames removed."""
    return [m.strip("`") for m in FIELD_RE.findall(sentence)
            if not NOT_A_FIELD.search(m.strip("`"))]


def classify(sentence: str) -> list[str]:
    """Which claim categories this sentence carries. Empty means it asserts nothing."""
    cats = []
    if ID_RE.search(sentence):
        cats.append("ID")
    if PATH_RE.search(sentence):
        cats.append("PATH")
    if REGION_RE.search(sentence):
        cats.append("REGION")
    if STATUS_RE.search(sentence):
        cats.append("STATUS")
    if fields(sentence):
        cats.append("FIELD")
    if SUPPORT_RE.search(sentence) or ONLY_RE.search(sentence):
        cats.append("SUPPORT")
    if NUMBER_RE.search(sentence):
        cats.append("NUMBER")
    return cats


def harvest(path: str) -> list[dict]:
    """Every claim in a notebook's markdown, with where it sits."""
    nb = json.load(open(path))
    out = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        # Fenced blocks are code being shown, not prose asserting anything.
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        for sentence in sentences(text):
            if sentence.startswith("#") or len(sentence) < 12:
                continue
            cats = classify(sentence)
            if not cats:
                continue
            out.append({
                "file": os.path.relpath(path, REPO),
                "cell": i,
                "sentence": " ".join(sentence.split()),
                "categories": cats,
                "hedged": bool(HEDGED.search(sentence)),
                "ids": sorted(set(ID_RE.findall(sentence))),
            })
    return out


SELF_TEST = [
    # (markdown, must-contain-category, must-NOT-contain-category)
    ("The model `us.xai.grok-4.6` is profile-only on bedrock-runtime.", "ID", None),
    ("Requests go to `/openai/v1/responses` on this family.", "PATH", None),
    ("Gemma 4 is available in us-east-1 and us-west-2.", "REGION", None),
    ("Passing `audio_url` returns 400 while `input_audio` returns 200.",
     "STATUS", None),
    ("The trace arrives in `choices[0].message.reasoning`.", "FIELD", None),
    ("gpt-oss supports `tool_choice: \"auto\"` only.", "SUPPORT", None),
    ("The sweep covers 5 levels across 2 prompts.", "NUMBER", None),
    # Shapes that must NOT be read as service claims.
    ("This section is organised in three parts.", None, "ID"),
    ("We import the shared helper at the top.", None, "PATH"),
    ("Read the committed output rather than trusting this sentence.", None, "REGION"),
    # Both of these WERE harvested as claims and were the top two false-positive
    # sources, 11 and 34 occurrences. A regex that has never been falsified is not
    # a regex, so they stay here as cases.
    ("See the [endpoints page](https://docs.aws.amazon.com/bedrock/index.html).",
     None, "ID"),
    ("`requirements.txt` pins the versions this was tested against.", None, "FIELD"),
]


def self_test() -> int:
    bad = 0
    for text, want, forbid in SELF_TEST:
        cats = classify(sentences(text)[0])
        if want and want not in cats:
            print(f"  MISS  {want!r} not detected in: {text}")
            bad += 1
        if forbid and forbid in cats:
            print(f"  FALSE {forbid!r} wrongly detected in: {text}")
            bad += 1
    # A rule that never fires is not a rule: prove each category appears somewhere.
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb"))):
        for claim in harvest(path):
            seen.update(claim["categories"])
    for cat in ("ID", "PATH", "REGION", "STATUS", "FIELD", "SUPPORT", "NUMBER"):
        if cat not in seen:
            print(f"  DEAD  category {cat!r} matched nothing in the repo")
            bad += 1
    print(f"  {len(SELF_TEST)} cases, {bad} problem(s)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", metavar="PATH")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--category")
    ap.add_argument("--unhedged", action="store_true",
                    help="only claims stated flatly, which are the ones that can be "
                         "wrong without a caveat to fall back on")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    claims = []
    for path in sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb"))):
        claims.extend(harvest(path))
    if args.category:
        claims = [c for c in claims if args.category in c["categories"]]
    if args.unhedged:
        claims = [c for c in claims if not c["hedged"]]

    if args.json:
        json.dump(claims, open(args.json, "w"), indent=1)
        print(f"{len(claims)} claim(s) -> {args.json}")
        return

    by_file: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for c in claims:
        by_file[c["file"]] = by_file.get(c["file"], 0) + 1
        for cat in c["categories"]:
            by_cat[cat] = by_cat.get(cat, 0) + 1
    print(f"{len(claims)} claim(s) across {len(by_file)} notebook(s)\n")
    print("by category")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<8} {n:>4}")
    print("\nby notebook")
    for f, n in sorted(by_file.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {f}")


if __name__ == "__main__":
    main()
