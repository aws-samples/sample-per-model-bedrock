#!/usr/bin/env python3
"""Catch a model ID spelled for the wrong endpoint.

The two endpoints do not share model IDs. On bedrock-mantle the model is
`openai.gpt-oss-20b`; on bedrock-runtime the same model is `openai.gpt-oss-20b-1:0`,
and most Claude and Grok models on runtime need a geo profile prefix. Mixing them
gives a 404 or "Model not found", and in a code sample it gives a reader who blames
Bedrock.

This found a real one: `01-openai-gpt/05` §5 built an OpenAI client against
`bedrock-runtime` and then passed `X-Amzn-Bedrock-ModelId: openai.gpt-oss-20b`, the
mantle spelling, in a block a reader is meant to copy. AWS's own batch documentation
passes `openai.gpt-oss-20b-1:0` there. Nothing caught it, because the cell is
illustrative markdown and never executes.

## Why this is deliberately narrow

The first version of this file flagged 21 sites and **20 were false**. It looked for
any quoted ID after any endpoint marker, which is hopeless in a collection whose whole
subject is the difference between the two endpoints:

  - `00-foundations/01` §2 is a TABLE comparing the two, by design.
  - `00-foundations/01` §2b defines `api_prefix(model_id, endpoint)`, so its example
    IDs belong to both endpoints at once.
  - `00-foundations/04` §9 passes bare mantle IDs to `runtime_id_for()` on purpose --
    translating them is the point of the cell.
  - `00-foundations/01` §12 calls `bedrock.post` (mantle) in a cell whose COMMENT
    mentions `runtime_post`, and the marker matched the comment.
  - `08-moonshot-kimi/01` §7 spells out both namespaces in one sentence to teach the
    difference.

So the rule now fires only on the unambiguous shape: an ID that is the value of a
model-designating key, in a cell that names exactly one endpoint and does not resolve
IDs. Everything discursive is out of scope, which is correct -- prose comparing the two
spellings is the repository doing its job.

    python3 check-endpoint-id-match.py            # static, cached catalogue
    python3 check-endpoint-id-match.py --live     # refresh the catalogue first
    python3 check-endpoint-id-match.py --self-test
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
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     ".endpoint-catalogue.json")

# An ID only counts when it is designated as THE model for a call.
MODEL_VALUE = re.compile(
    # Longest key first: `X-Amzn-Bedrock-ModelId` must not be consumed as `model`.
    # The optional quote on each side of the key is what makes `"key": "value"` and
    # `key="value"` both match -- omitting it silently matched neither, and the
    # self-test caught that.
    r"""["']?\b(?:X-Amzn-Bedrock-ModelId|modelId|model)["']?\s*[:=]\s*["']"""
    r"""(((?:us|eu|apac|in|global)\.)?"""
    r"""(?:openai|anthropic|google|qwen|deepseek|zai|mistral|moonshotai|moonshot|"""
    r"""minimax|nvidia|xai|writer|amazon|meta)\.[a-z0-9][a-z0-9.\-]*[a-z0-9]"""
    r"""(?::[0-9]+)?)["']"""
)
# Real calls, not the word appearing in prose or a comment. The host middle must
# admit `{REGION}` and `${...}`: every base URL in this collection is an f-string, so
# a `[\w.-]*` middle matched none of them. The self-test used a literal Region and
# passed, which is why this went unnoticed until the checker was run against the
# actual defect it was written for.
RUNTIME_CALL = re.compile(
    r"bedrock-runtime[\w.${}-]*\.amazonaws\.com|RUNTIME_HOST|"
    r"boto3\.client\(\s*[\"']bedrock-runtime|"
    r"\bruntime_post\s*\(|\bruntime_client\s*\(|\bruntime_openai_client\s*\(|"
    r"\bruntime_anthropic_client\s*\(|\bruntime_base_url\s*\(")
# Deliberately NOT `\bclient\s*\(` or `\bbase_url\s*\(`. Those matched
# `boto3.client("bedrock-runtime", ...)` and `control_client(...)`, so a cell holding
# both hosts was classified as mantle-only and its correct runtime IDs were reported
# as wrong. A marker that matches the other endpoint's own constructor is not a marker.
MANTLE_CALL = re.compile(
    r"bedrock-mantle[\w.${}-]*\.api\.aws|MANTLE_HOST|"
    r"(?<!runtime_)(?<!\.)\bpost\s*\(|\banthropic_client\s*\(")
# A cell that translates IDs is passing the other endpoint's spelling on purpose.
RESOLVES = re.compile(r"runtime_id_for|resolve_runtime_id|endpoints_for|api_prefix|"
                      r"runtime_models")


def load_catalogue(live: bool) -> dict:
    if not live and os.path.exists(CACHE):
        return json.load(open(CACHE))
    sys.path.insert(0, os.path.join(REPO, "_shared"))
    import bedrock  # noqa: PLC0415

    cat: dict = {"mantle": {}, "runtime": {}}
    for region in ("us-east-1", "us-west-2", "eu-central-1"):
        try:
            cat["mantle"][region] = bedrock.list_models(region=region)
        except Exception as exc:  # noqa: BLE001
            print(f"  mantle {region}: {type(exc).__name__}: {exc}", file=sys.stderr)
            cat["mantle"][region] = []
        try:
            ids = set()
            for entry in bedrock.runtime_models(region=region).values():
                ids.add(entry.get("id", ""))
                for v in entry.get("variants") or []:
                    ids.add(v.get("id", ""))
            try:
                ids.update(bedrock.inference_profiles(region=region) or [])
            except Exception:  # noqa: BLE001
                pass
            cat["runtime"][region] = sorted(i for i in ids if i)
        except Exception as exc:  # noqa: BLE001
            print(f"  runtime {region}: {type(exc).__name__}: {exc}", file=sys.stderr)
            cat["runtime"][region] = []
    json.dump(cat, open(CACHE, "w"), indent=1)
    return cat


def flat(cat: dict, endpoint: str) -> set[str]:
    out: set[str] = set()
    for ids in cat.get(endpoint, {}).values():
        out.update(ids)
    return out


def scan_text(text: str, cat: dict) -> list[tuple[str, str, str]]:
    """(id, endpoint, reason) for IDs designated for the wrong endpoint."""
    runtime_ids, mantle_ids = flat(cat, "runtime"), flat(cat, "mantle")
    if not runtime_ids or not mantle_ids:
        return []
    if RESOLVES.search(text):
        return []
    is_runtime = bool(RUNTIME_CALL.search(text))
    is_mantle = bool(MANTLE_CALL.search(text))
    if is_runtime == is_mantle:  # both, or neither -- nothing can be concluded
        return []
    endpoint = "runtime" if is_runtime else "mantle"

    out = []
    for match in MODEL_VALUE.finditer(text):
        mid = match.group(1)
        if endpoint == "runtime" and mid not in runtime_ids and mid in mantle_ids:
            out.append((mid, "runtime", "mantle-only spelling designated on a "
                                       "bedrock-runtime call"))
        elif endpoint == "mantle" and mid not in mantle_ids and mid in runtime_ids:
            out.append((mid, "mantle", "runtime-only spelling designated on a "
                                       "bedrock-mantle call"))
    return out


def check(path: str, cat: dict) -> list[str]:
    nb = json.load(open(path))
    problems = []
    for i, cell in enumerate(nb["cells"]):
        for mid, _endpoint, why in scan_text("".join(cell["source"]), cat):
            problems.append(f"cell {i}: {mid!r} -- {why}")
    return sorted(set(problems))


SELF_TEST_CAT = {
    "mantle": {"us-west-2": ["openai.gpt-oss-20b", "qwen.qwen3-32b",
                             "moonshotai.kimi-k2-thinking", "google.gemma-4-31b"]},
    "runtime": {"us-west-2": ["openai.gpt-oss-20b-1:0", "qwen.qwen3-32b-1:0",
                              "moonshot.kimi-k2-thinking",
                              "us.anthropic.claude-opus-5-v1:0"]},
}
SELF_TEST = [
    # --- must fire: the defect this exists for, as it was before the fix ---
    ('client = OpenAI(base_url="https://bedrock-runtime.us-west-2.amazonaws.com'
     '/openai/v1")\njob = client.batches.create(\n'
     '    extra_headers={"X-Amzn-Bedrock-ModelId": "openai.gpt-oss-20b"},\n)', 1),
    ('code, data = runtime_post("/openai/v1/responses", {"model": "qwen.qwen3-32b"})',
     1),
    # The same defect with an f-string host, which is how every cell actually spells
    # it. The literal-Region case above passed while this one silently did not.
    ('client = OpenAI(\n'
     '    base_url=f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1",\n)\n'
     '    extra_headers={"X-Amzn-Bedrock-ModelId": "openai.gpt-oss-20b"},', 1),
    # --- must stay silent: every false positive the first version produced ---
    # 1. A table comparing the two endpoints.
    ('| `/v1/...` | `openai.gpt-oss*` on bedrock-mantle |\n'
     '| `/openai/v1/...` | everything on bedrock-runtime |', 0),
    # 2. A resolver being defined or used; bare IDs are its input.
    ('RUNTIME = f"https://bedrock-runtime.{REGION}.amazonaws.com"\n'
     'rid = runtime_id_for("qwen.qwen3-32b", REGION)', 0),
    # 3. A mantle call in a cell whose COMMENT mentions runtime_post.
    ('# had drifted to omit runtime_post, converse and endpoints_for\n'
     'code, data = bedrock.post("/openai/v1/responses",\n'
     '    {"model": "google.gemma-4-31b", "input": "Reply OK"})', 0),
    # 4. Prose teaching that the namespaces differ.
    ('this model is `moonshotai.kimi-k2-thinking` on `bedrock-mantle` and '
     '`moonshot.kimi-k2-thinking` on `bedrock-runtime`', 0),
    # 5. Correct spelling for the endpoint named.
    ('client = OpenAI(base_url="https://bedrock-runtime.us-west-2.amazonaws.com'
     '/openai/v1")\n'
     '    extra_headers={"X-Amzn-Bedrock-ModelId": "openai.gpt-oss-20b-1:0"},', 0),
    # 6. An ID mentioned with no call anywhere.
    ('The model is `openai.gpt-oss-20b`.', 0),
    # 7. A cell that probes BOTH endpoints, holding each one's correct IDs. This was
    #    reported as three mantle errors because `boto3.client(` matched the mantle
    #    marker `\bclient\s*\(`.
    ('RUNTIME_HOST = f"https://bedrock-runtime.{REGION}.amazonaws.com"\n'
     'MANTLE_HOST = f"https://bedrock-mantle.{REGION}.api.aws"\n'
     'SURFACES = [\n'
     '    ("runtime", RUNTIME_HOST, {"model": "openai.gpt-oss-20b-1:0"}),\n'
     '    ("mantle", MANTLE_HOST, {"model": "openai.gpt-oss-20b"}),\n'
     ']\n'
     'br = boto3.client("bedrock-runtime", region_name=REGION)', 0),
    # 8. A runtime-only cell that reaches runtime through boto3 rather than a URL.
    ('br = boto3.client("bedrock-runtime", region_name=REGION)\n'
     'br.invoke_model(modelId="openai.gpt-oss-20b")', 1),
]


def self_test() -> int:
    bad = 0
    for text, want in SELF_TEST:
        got = scan_text(text, SELF_TEST_CAT)
        if len(got) != want:
            print(f"  want {want}, got {len(got)}: {[g[0] for g in got]}")
            print(f"    in: {text[:100]!r}")
            bad += 1
    print(f"  {len(SELF_TEST)} cases, {bad} problem(s)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(1 if self_test() else 0)

    cat = load_catalogue(args.live)
    n_rt, n_mt = len(flat(cat, "runtime")), len(flat(cat, "mantle"))
    if not n_rt or not n_mt:
        print("catalogue incomplete -- cannot decide; run with --live")
        sys.exit(1)
    print(f"catalogue: {n_mt} mantle id(s), {n_rt} runtime id(s)")

    total = 0
    for path in sorted(glob.glob(os.path.join(REPO, "*", "*.ipynb"))):
        found = check(path, cat)
        if found:
            total += len(found)
            print(f"\n{os.path.relpath(path, REPO)}")
            for line in found:
                print(f"  {line}")
    print(f"\n{total} endpoint/ID mismatch(es)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
