"""Shared helpers for the per-model Amazon Bedrock sample notebooks.

Import from a notebook with:

    import sys; sys.path.insert(0, "../_shared")
    from bedrock import endpoints_for, parse_json_lenient

Amazon Bedrock serves models through two inference endpoints. AWS recommends
`bedrock-runtime` for new applications, and as of September 2026 it speaks all five
APIs:

    bedrock-runtime   InvokeModel / Converse via the AWS SDK, plus the
                      OpenAI-compatible Responses and Chat Completions APIs and
                      the Anthropic Messages API on its /openai/v1 and
                      /anthropic/v1 paths. SigV4 *or* a Bedrock API key.
                      Helpers: runtime_id_for(), runtime_models().
    bedrock-mantle    Responses / Chat Completions / Messages. Adds server-side
                      tool use, asynchronous inference (background=true), and
                      Projects and Workspaces. Helpers: post(), list_models().

Which endpoint serves a given model is a per-model fact, not a preference, and
so is the URL path and even the model ID — the same model can be
`openai.gpt-oss-20b` on mantle and `openai.gpt-oss-20b-1:0` on runtime.
`endpoints_for()` answers the first question, `api_prefix()` the second and
`runtime_id_for()` the third. Each family's notebook states the answer for its
own models.

Everything here is deliberately small and dependency-light: the notebooks are the
teaching material, this file only removes repetition.

See 00-foundations/ for the full explanation of auth, the three Mantle URL path
families, Converse and inference profiles, and model discovery.

Style note: the SDK imports in token() and control_client() are
function-local on purpose, against the usual
imports-at-top rule (PEP 8). This module is imported by every notebook,
including ones that never touch a given SDK, and a function-local import keeps
`import bedrock` working when only a subset of the optional SDKs is installed.
The stdlib imports below follow the normal convention.
"""

from __future__ import annotations

import ast
import base64
import json
import os
import random  # retry jitter only -- never for tokens, keys, or nonces
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence

DEFAULT_REGION = "us-east-1"

# ---------------------------------------------------------------------------
# URL paths, which are per-endpoint as well as per-model.
#
# bedrock-mantle has three families:
#   /openai/v1/*        google gemma-4, the hosted openai gpt models (gpt-5.x,
#                       gpt-6), xai
#   /v1/*               openai gpt-oss + every Chat-Completions-only family
#   /anthropic/v1/*     anthropic claude only
#
# bedrock-runtime has TWO, and the split falls in a different place:
#   /openai/v1/*        every OpenAI-compatible model, gpt-oss and qwen included
#   /anthropic/v1/*     anthropic claude only
#
# So the same model can live on different paths depending on the endpoint:
# `openai.gpt-oss-20b` is /v1 on mantle, and its runtime twin
# `openai.gpt-oss-20b-1:0` is /openai/v1. There is no /v1 inference surface on
# bedrock-runtime at all -- see unknown_op() for what asking for one looks like.
#
# Control-plane paths (models, files, projects, fine-tuning, data retention)
# live under mantle's /v1/*, never /openai/v1/*.
# ---------------------------------------------------------------------------
_OPENAI_PREFIX_FAMILIES = ("google.gemma-4", "xai.")

# The OpenAI family splits on mantle by line, not by generation: the hosted GPT
# models are served under /openai/v1, and the open-weight gpt-oss line
# (gpt-oss-safeguard included) under bare /v1. Keying on a generation such as
# "openai.gpt-5" would send the next generation to /v1, where mantle answers
# `model ... isn't supported on this route`.
_OPENAI_HOSTED_GPT = "openai.gpt-"
_OPENAI_OPEN_WEIGHT = "openai.gpt-oss"


def api_prefix(model_id: str, endpoint: str = "mantle") -> str:
    """Return the URL prefix serving this model's inference APIs.

    `endpoint` is "mantle" or "runtime" and it changes the answer:

        api_prefix("openai.gpt-oss-20b")                 -> "/v1"
        api_prefix("openai.gpt-oss-20b-1:0", "runtime")  -> "/openai/v1"
        api_prefix("openai.gpt-6-astra")                 -> "/openai/v1"

    Measured in us-east-1 and us-west-2 in September 2026. This is a lookup over
    measured behaviour, not a rule the service guarantees: probe the model you
    actually intend to call.
    """
    if endpoint not in ("mantle", "runtime"):
        raise ValueError(f"endpoint must be 'mantle' or 'runtime', got {endpoint!r}")
    # A geo/global inference-profile prefix is not part of the family name.
    bare = re.sub(r"^(us|eu|apac|global|in)\.", "", model_id)
    if bare.startswith("anthropic."):
        return "/anthropic/v1"
    if endpoint == "runtime":
        # Runtime serves every OpenAI-compatible model on /openai/v1.
        return "/openai/v1"
    if bare.startswith(_OPENAI_HOSTED_GPT):
        # gpt-oss is the open-weight line and sits on bare /v1; every other
        # openai.gpt-* is hosted and sits on /openai/v1.
        return "/v1" if bare.startswith(_OPENAI_OPEN_WEIGHT) else "/openai/v1"
    if any(bare.startswith(p) for p in _OPENAI_PREFIX_FAMILIES):
        return "/openai/v1"
    return "/v1"


def host(region: str = DEFAULT_REGION) -> str:
    """Return the regional bedrock-mantle endpoint origin (scheme + host)."""
    return f"https://bedrock-mantle.{region}.api.aws"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
def token(region: str = DEFAULT_REGION) -> str:
    """Short-term Bedrock API key minted from the ambient IAM credentials.

    Expires in <=12h and cannot be refreshed - mint a new one instead.
    00-foundations/01 covers the key lifetime and the SigV4 alternative.
    """
    from aws_bedrock_token_generator import provide_token

    return provide_token(region=region)


# ---------------------------------------------------------------------------
# bedrock-runtime: Converse and inference profiles
#
# Converse is the AWS-native, model-agnostic API. It takes SigV4 credentials
# through boto3 rather than a bearer token, and it normalises the request shape
# across providers - so the same call works for Nova, Claude and Llama.
#
# The catch is the model ID. Bedrock has two ways to address a model on runtime:
#
#   bare model ID          amazon.nova-lite-v1:0
#   inference profile ID   us.amazon.nova-lite-v1:0
#
# An inference profile routes the request across several Regions in a geography,
# which raises availability and effective throughput (this is Cross-Region
# Inference, CRIS). Models listed as ON_DEMAND accept EITHER form. Models listed
# as INFERENCE_PROFILE only - which today includes almost the whole Claude family
# - accept ONLY the prefixed form:
#
#   Converse(modelId="anthropic.claude-sonnet-5")     -> ValidationException
#   Converse(modelId="us.anthropic.claude-sonnet-5")  -> 200
#
# Verified against us-east-1. resolve_runtime_id() below hides the difference by
# asking the service which profiles exist rather than guessing at the prefix.
# ---------------------------------------------------------------------------
DEFAULT_GEO = "us"

# list_inference_profiles is a control-plane call; the answer changes only when
# AWS adds models, so cache it per (region, geo) rather than per notebook cell.
_PROFILE_CACHE: dict[tuple[str, str], set[str]] = {}


def control_client(region: str = DEFAULT_REGION):
    """A boto3 bedrock client (the control plane: model and profile catalogues)."""
    import boto3

    return boto3.client("bedrock", region_name=region)


def inference_profiles(region: str = DEFAULT_REGION) -> set[str]:
    """Every inference profile ID available in this Region, cached.

    Returns an empty set if the caller lacks bedrock:ListInferenceProfiles, so a
    notebook degrades to bare model IDs instead of failing outright.
    """
    key = (region, "all")
    if key in _PROFILE_CACHE:
        return _PROFILE_CACHE[key]
    ids: set[str] = set()
    try:
        paginator = control_client(region).get_paginator("list_inference_profiles")
        for page in paginator.paginate():
            for profile in page.get("inferenceProfileSummaries", []):
                ids.add(profile["inferenceProfileId"])
    except Exception as exc:
        # Missing permission or an older botocore: fall back to bare IDs. An empty
        # set is indistinguishable from "this Region has no profiles", so say so out
        # loud -- otherwise resolve_runtime_id() hands back a bare ID for an
        # INFERENCE_PROFILE-only model and the 400 blames the model.
        _warn_once(
            f"profiles:{region}",
            f"could not list inference profiles in {region} ({type(exc).__name__}). "
            "Falling back to bare model IDs, which INFERENCE_PROFILE-only models "
            "will refuse.",
        )
        ids = set()
    _PROFILE_CACHE[key] = ids
    return ids


def resolve_runtime_id(
    model_id: str, region: str = DEFAULT_REGION, geo: str = DEFAULT_GEO
) -> str:
    """Return the model ID that Converse will accept for this model.

    Prefers the geo-prefixed inference profile when one exists, because it is
    required for INFERENCE_PROFILE-only models and strictly better (cross-Region
    routing) for the rest. Where that geo has no profile here and the model is not
    offered ON_DEMAND, tries the Region's other profile geos and then `global.`,
    because the bare ID is refused outright for such a model. Otherwise falls back
    to the ID you passed in, which is right for a model whose bare ID is callable.

        amazon.nova-lite-v1:0      -> us.amazon.nova-lite-v1:0
        anthropic.claude-sonnet-5  -> us.anthropic.claude-sonnet-5
        moonshotai.kimi-k3         -> us.moonshotai.kimi-k3     (in us-east-1)
        moonshotai.kimi-k3         -> global.moonshotai.kimi-k3 (in eu-central-1,
                                      which carries no eu. profile for it)
        anthropic.claude-opus-5    -> eu.anthropic.claude-opus-5 (in eu-central-1)
        some-model-with-no-profile -> some-model-with-no-profile

    `global.` is tried last: a regional profile keeps traffic in the geography,
    and prompt-cache hits are more likely there than across every Region.
    """
    # Keep in step with the geo alternation used by api_prefix() and
    # _norm_model_key().
    if model_id.split(".", 1)[0] in {"us", "eu", "apac", "global", "in"}:
        return model_id  # already a profile ID
    profiles = inference_profiles(region)
    candidate = f"{geo}.{model_id}"
    if candidate in profiles:
        return candidate
    # Some catalogue entries omit the ":0" suffix that the profile carries.
    versioned = f"{candidate}:0"
    if versioned in profiles:
        return versioned
    try:
        entry = runtime_models(region).get(model_id.split(":")[0])
    except Exception:
        entry = None
    # No profile in the geo asked for. For a model the catalogue offers ON_DEMAND
    # the bare ID below is callable, so leave it alone. For one that it does not,
    # the bare ID is refused ("Invocation of model ID ... with on-demand throughput
    # isn't supported"), and another profile in this Region can work: in
    # eu-central-1, moonshotai.kimi-k3 has a global. profile and no eu. one.
    #
    # The other geos come from the Region's own profile list rather than a
    # region-to-geo table, so "apac" and "in" are found without being enumerated
    # here. global is tried last, for the reason in the docstring.
    if entry and "ON_DEMAND" not in (entry.get("id_infer") or entry.get("infer") or ()):
        others = sorted({p.split(".", 1)[0] for p in profiles} - {geo, "global"})
        for fallback_geo in others + ["global"]:
            for fallback in (f"{fallback_geo}.{model_id}", f"{fallback_geo}.{model_id}:0"):
                if fallback in profiles:
                    return fallback
    # No profile. Converse is strict about the version suffix where the catalogue
    # has one: `qwen.qwen3-32b-v1` is rejected as an invalid identifier while
    # `qwen.qwen3-32b-v1:0` succeeds. The catalogue key drops that suffix, so
    # recover the full ID rather than handing Converse a form it will refuse.
    if entry and entry.get("id"):
        return entry["id"]
    return model_id


# --------------------------------------------------------------------------
# Sample media.
#
# Vision and audio cells need an input with a known answer, so each cell can
# check what the model returned:
#
#     slide    ->  a labelled architecture diagram, for "read the title" and
#                  "quote the callouts"
#     clip     ->  seven seconds of speech, for "transcribe this"
#     invoice  ->  a rendered invoice holding the same fields as the text
#                  extraction examples, for "extract this image as JSON"
#
# The slide and the clip are excerpts from a public AWS talk, "AWS Summit Online
# ASEAN re:Cap 2020 | AI/ML: Cost-optimise Your Machine Learning Pipeline (L300)"
# (AWS Events, 17 Nov 2020), presented by the author of this repository:
#
#     https://www.youtube.com/watch?v=YjFI-n2YC7M
#
# The invoice is synthetic, rendered for this repository. All three files are
# small (75 KB together) and are committed rather than fetched, so the notebooks
# run offline and the input cannot change underneath a cell.
# --------------------------------------------------------------------------

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

SLIDE_PATH = os.path.join(_ASSETS, "aws-summit-slide.jpg")
CLIP_PATH = os.path.join(_ASSETS, "aws-summit-clip.mp3")
INVOICE_PATH = os.path.join(_ASSETS, "invoice.png")

# Ground truth for the slide, transcribed by hand from the frame itself.
SLIDE_TITLE = "Ingestion from database"
SLIDE_CALLOUTS = (
    "Pay according to job duration",
    "Lower compute cost, up to 90%",
    "Lower storage cost for rarely accessed data",
)

# Ground truth for the clip, transcribed by hand from the audio itself. The
# speaker says this at 4:19 in the recording above.
CLIP_TRANSCRIPT = (
    "you can save cost by leveraging the tiering mechanism in Amazon S3, "
    "in Amazon S3 you can store your infrequently accessed data"
)
# Score transcription on these rather than on an exact match. Speech recognition
# legitimately differs on word boundaries and filler words, and "tiering" is
# sometimes returned as "tearing", so an exact-match assertion would fail on a
# correct transcript.
CLIP_KEYWORDS = ("save cost", "leveraging", "Amazon S3")

# Ground truth for the invoice: the fields it was rendered from.
INVOICE_FIELDS = {"invoice_id": "INV-1042", "vendor": "Acme Pty Ltd", "total": 1280.5, "currency": "AUD"}


def slide_jpeg() -> bytes:
    """Raw JPEG bytes of the architecture slide. Converse takes these directly."""
    with open(SLIDE_PATH, "rb") as handle:
        return handle.read()


def slide_data_url() -> str:
    """The slide as a base64 data URL, the form the OpenAI-shaped APIs want."""
    return "data:image/jpeg;base64," + base64.b64encode(slide_jpeg()).decode()


def invoice_png() -> bytes:
    """Raw PNG bytes of the invoice. Converse takes these directly."""
    with open(INVOICE_PATH, "rb") as handle:
        return handle.read()


def invoice_data_url() -> str:
    """The invoice as a base64 data URL, the form the OpenAI-shaped APIs want."""
    return "data:image/png;base64," + base64.b64encode(invoice_png()).decode()


def clip_mp3() -> bytes:
    """Raw MP3 bytes of the speech excerpt: 7 s, mono, 16 kHz, 28 KB."""
    with open(CLIP_PATH, "rb") as handle:
        return handle.read()


def clip_mp3_b64() -> str:
    """The clip as bare base64, the form an `input_audio` block wants."""
    return base64.b64encode(clip_mp3()).decode()


def keyword_recall(answer: str, keywords: Sequence[str]) -> tuple[int, int]:
    """How many of `keywords` appear in `answer`, case-insensitively.

    Returned as (hits, total) so a cell can print a score it computed rather
    than assert a verdict written in advance.

    Whitespace is collapsed on both sides before matching. Without that, a model
    that emits a double space or a non-breaking space inside an otherwise
    correct phrase scores zero, while the same cell's `" ".join(answer.split())`
    display normalises it and looks correct. That mismatch reads as a broken
    model rather than a formatting difference.
    """
    lowered = " ".join((answer or "").lower().split())
    hits = sum(1 for word in keywords if " ".join(word.lower().split()) in lowered)
    return hits, len(keywords)


_RUNTIME_CATALOGUE_CACHE: dict[str, dict[str, dict]] = {}
# Remembers a FAILED catalogue call, so it is attempted once per Region rather than
# once per lookup.
_RUNTIME_CATALOGUE_FAILED: dict[str, str] = {}
# The same pair for the bedrock-mantle catalogue, used by list_models().
_MANTLE_CATALOGUE_CACHE: dict[str, list[str]] = {}
_MANTLE_CATALOGUE_FAILED: dict[str, str] = {}


def runtime_models(region: str = DEFAULT_REGION) -> dict[str, dict]:
    """Serverless bedrock-runtime catalogue, keyed by model ID without the version.

    Each value carries {"in", "out", "infer", "provider"}, plus "id" and "id_infer"
    for the callable variant and "variants" for all of them. Used by the
    endpoint-availability tables in the notebooks so the claims come from the
    service rather than from a hand-maintained list that ages.

    Cached per Region, like inference_profiles(). ListFoundationModels changes only
    when AWS adds a model, and runtime_id_for() calls this once per lookup, so an
    uncached version turns a 40-model table into 40 control-plane calls.
    """
    if region in _RUNTIME_CATALOGUE_CACHE:
        return _RUNTIME_CATALOGUE_CACHE[region]
    # Remember a failure too (for example, no bedrock:ListFoundationModels), so a
    # 40-row table makes one failing call rather than 40, each with SDK retries.
    if region in _RUNTIME_CATALOGUE_FAILED:
        raise RuntimeError(_RUNTIME_CATALOGUE_FAILED[region])
    out: dict[str, dict] = {}
    try:
        summaries = control_client(region).list_foundation_models()
        summaries = summaries.get("modelSummaries", [])
    except Exception as exc:
        message = f"list_foundation_models failed: {exc}"
        _RUNTIME_CATALOGUE_FAILED[region] = message
        raise RuntimeError(message) from exc
    for summary in summaries:
        key = summary["modelId"].split(":")[0]
        entry = out.setdefault(
            key,
            {
                "in": set(),
                "out": set(),
                "infer": set(),
                "id": summary["modelId"],
                "id_infer": set(),
                "variants": [],
                "provider": summary.get("providerName", "?"),
            },
        )
        entry["in"].update(summary.get("inputModalities", []))
        entry["out"].update(summary.get("outputModalities", []))
        types = set(summary.get("inferenceTypesSupported", []))
        entry["infer"].update(types)
        entry["variants"].append({"id": summary["modelId"], "infer": types})

    # One catalogue key can carry several model IDs, and they do NOT share
    # inference types: the `:0:24k` and `:0:300k` Nova entries are PROVISIONED-only
    # context-window SKUs while the plain `:0` entry is the on-demand one. Converse
    # answers "Model not found." for the PROVISIONED-only IDs.
    #
    # So choose the variant that is actually callable: on-demand first, then
    # profile-addressable, then whatever came first. Among equals prefer the
    # shortest ID, which is the one without a context-window segment.
    def _rank(variant: dict) -> tuple:
        infer = variant["infer"]
        return (
            0 if "ON_DEMAND" in infer else 1 if "INFERENCE_PROFILE" in infer else 2,
            variant["id"].count(":"),
            len(variant["id"]),
        )

    for entry in out.values():
        best = min(entry["variants"], key=_rank)
        entry["id"] = best["id"]
        entry["id_infer"] = best["infer"]
    _RUNTIME_CATALOGUE_CACHE[region] = out
    return out


def _norm_model_key(value: str) -> str:
    """Normalise a model ID so the two endpoints' catalogues can be compared.

    The same model is named differently on each endpoint, in four ways that all
    show up in us-east-1 today:

        version suffix     openai.gpt-oss-20b   vs  openai.gpt-oss-20b-1:0
        -v1:0 suffix       qwen.qwen3-32b       vs  qwen.qwen3-32b-v1:0
        provider prefix    moonshotai.kimi-...  vs  moonshot.kimi-...
        "-instruct" tail   qwen.qwen3-next-80b-a3b-instruct vs ...-a3b

    The delicate part is the trailing "-1" in `openai.gpt-oss-20b-1:0`, which is a
    version and must go. Stripping ANY trailing "-<digit>" would map
    `anthropic.claude-sonnet-5` onto `anthropic.claude-sonnet-4-...`, because both
    collapse to `anthropic.claude-sonnet`. Model generations live in those digits.

    So the trailing "-<digit>" is only removed when it looks like a *version*: the
    ID carried a ":<n>" suffix and no embedded release date.
    `claude-sonnet-4-20250514-v1:0` has the date, so its "-4" is kept; `gpt-oss-20b-1:0` has no date, so
    its "-1" goes.
    """
    value = re.sub(r"^(us|eu|apac|global|in)\.", "", value)
    had_version_suffix = ":" in value
    value = value.split(":")[0]
    # A "-vN" tail IS the version marker, so any digit before it belongs to the model
    # generation. Without this flag, `anthropic.claude-opus-4-7-v1:0` and
    # `...-4-8-v1:0` would both collapse to `anthropic.claude-opus-4`, and
    # `zai.glm-5-v1:0` to `zai.glm`.
    had_v_suffix = re.search(r"-v\d+$", value) is not None
    value = re.sub(r"-v\d+$", "", value)
    dated = re.search(r"-\d{8}$", value) is not None
    value = re.sub(r"-\d{8}$", "", value)
    value = value.replace("moonshotai.", "moonshot.").replace("-instruct", "")
    if had_version_suffix and not dated and not had_v_suffix:
        value = re.sub(r"-\d$", "", value)
    return value.lower()


_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    """Print a degradation notice once per process.

    Used when a catalogue call fails, for example without
    `bedrock:ListFoundationModels` or `bedrock:ListInferenceProfiles`. endpoints_for(),
    runtime_id_for() and inference_profiles() then fall back to a degraded answer,
    and the notice says so, so that answer is not read as "the model is absent".
    """
    if key in _WARNED:
        return
    _WARNED.add(key)
    print(f"  [bedrock helper] {message}")


def endpoints_for(model_id: str, region: str = DEFAULT_REGION) -> dict[str, bool]:
    """Which endpoints serve this model: {"mantle": bool, "runtime": bool}.

    Ask this before writing code against a model. The same model can carry
    DIFFERENT IDs on the two endpoints - `openai.gpt-oss-120b` on mantle is
    `openai.gpt-oss-120b-1:0` on runtime - so this compares on a normalised key.
    """
    target = _norm_model_key(model_id)
    try:
        on_mantle = any(_norm_model_key(m) == target for m in list_models(region))
    except Exception as exc:
        _warn_once(
            f"list_models:{region}",
            f"could not list bedrock-mantle models in {region} ({type(exc).__name__}). "
            "Every 'mantle' answer below is False because the catalogue is unavailable, "
            "NOT because the model is absent.",
        )
        on_mantle = False
    try:
        # Compare against entry["id"], NOT the dict key. runtime_models() keys off
        # modelId.split(":")[0], so the key for `openai.gpt-oss-20b-1:0` is
        # `openai.gpt-oss-20b-1`: the ":0" is gone, and _norm_model_key cannot tell
        # that the trailing "-1" is a version. runtime_id_for() compares the same way.
        on_runtime = any(
            _norm_model_key(entry["id"]) == target
            for entry in runtime_models(region).values()
        )
    except Exception as exc:
        # Same treatment as the mantle branch above.
        _warn_once(
            f"endpoints_for:{region}",
            f"could not list bedrock-runtime models in {region} "
            f"({type(exc).__name__}). Every 'runtime' answer below is False because "
            "the catalogue is unavailable, NOT because the model is absent.",
        )
        on_runtime = False
    return {"mantle": on_mantle, "runtime": on_runtime}


def runtime_id_for(model_id: str, region: str = DEFAULT_REGION) -> str | None:
    """The ID bedrock-runtime wants for the model you named, or None.

    Pass a mantle model ID and get back the runtime form, including the geo
    inference-profile prefix when the model requires one:

        openai.gpt-oss-20b         -> openai.gpt-oss-20b-1:0
        qwen.qwen3-32b             -> qwen.qwen3-32b-v1:0
        moonshotai.kimi-k2.5       -> moonshotai.kimi-k2.5
        anthropic.claude-opus-5    -> us.anthropic.claude-opus-5
        google.gemma-4-31b         -> None  (mantle only)

    Returns None when the model is not on runtime at all, so callers get an
    explicit "not there" rather than a guessed ID that 400s later.

    This answers for Converse, InvokeModel and the /openai/v1 paths. The
    /anthropic/v1/messages surface on bedrock-runtime serves a smaller set of Claude
    models, and no ID shape predicts which: as of September 2026,
    `us.anthropic.claude-sonnet-4-6` returns 404 on Messages while
    `us.anthropic.claude-haiku-4-5-20251001-v1:0` returns 200. Use Converse on
    runtime unless you need a Messages-only feature, and probe the model you intend
    to call.
    """
    try:
        catalogue = runtime_models(region)
    except Exception as exc:
        # None here is indistinguishable from "this model is mantle-only", and every
        # caller treats it that way. Say which it is.
        _warn_once(
            f"runtime_id_for:{region}",
            f"could not list bedrock-runtime models in {region} "
            f"({type(exc).__name__}). runtime_id_for() returns None for EVERY model "
            "while the catalogue is unavailable; that is not the same as mantle-only.",
        )
        return None

    def _addressable(entry: dict) -> str:
        # The chosen variant's OWN types, not the union across variants: the union
        # can include ON_DEMAND from a sibling while this ID is PROVISIONED-only.
        if "ON_DEMAND" in (entry.get("id_infer") or entry["infer"]):
            return entry["id"]
        # INFERENCE_PROFILE-only: the bare ID is refused outright.
        return resolve_runtime_id(entry["id"], region)

    bare = re.sub(r"^(us|eu|apac|global|in)\.", "", model_id)

    # Exact first. Normalisation is lossy by design, so an exact match must win.
    for entry in catalogue.values():
        if entry["id"] == bare or entry["id"].split(":")[0] == bare:
            return _addressable(entry)

    target = _norm_model_key(model_id)
    for entry in catalogue.values():
        if _norm_model_key(entry["id"]) == target:
            return _addressable(entry)
    return None


# ---------------------------------------------------------------------------
# Raw HTTP with retries - used where the SDKs don't reach (control plane,
# Anthropic beta headers, deliberately-invalid requests that must show a 400).
# ---------------------------------------------------------------------------
# 529 is Anthropic's "overloaded" status: transient by definition, and returned by
# the Messages API under load. It is outside the usual 5xx set, so a policy that
# only knows 500-504 gives up on a retryable blip.
_TRANSIENT = {429, 500, 502, 503, 504, 529}

# mantle sometimes reports a SERVER fault with a 4xx status and the body
# "Internal server error", and the same request succeeds on the next attempt.
# Status alone misclassifies it as a permanent client error.
#
# So: retry a 4xx ONLY when the body says the server failed. Never widen this to
# all 400s: a genuine "unsupported parameter" 400 must fail fast.
_SERVER_FAULT_TEXT = ("internal server error", "internal failure", "internal error")


def _is_retryable(status: int, payload: dict) -> bool:
    """True when this response is worth another attempt."""
    if status in _TRANSIENT:
        return True
    if 400 <= status < 500:
        # Match over the WHOLE serialised body, not err()'s extracted message.
        # err() reads error.message and truncates, which misses a marker in a
        # sibling field ({"message": "Bad request", "details": "internal server
        # error while validating"}) or past the truncation limit.
        #
        # json.dumps on an arbitrary payload can still fail (a set, bytes), and this
        # runs inside post()'s HTTPError handler where raising would turn the
        # documented "returns (code, body)" contract into an exception.
        try:
            text = json.dumps(payload, default=str).lower()
        except (TypeError, ValueError):
            text = str(payload).lower()
        return any(marker in text for marker in _SERVER_FAULT_TEXT)
    return False


def _open_https(req: urllib.request.Request, timeout: int):
    """urlopen restricted to HTTPS.

    urllib honours file://, ftp:// and other schemes, so a URL that ever comes
    from data rather than from code could read a local file. Every call here is
    built from host() + a literal path, but the guard is cheap and keeps the
    property locally checkable (CWE-22 / Bandit B310).
    """
    if req.full_url.split("://", 1)[0] != "https":
        raise ValueError(f"refusing non-HTTPS URL: {req.full_url[:60]}")
    # Scheme verified https above; urllib's other schemes cannot be reached.
    # nosemgrep: dynamic-urllib-use-detected - scheme verified https above
    return urllib.request.urlopen(req, timeout=timeout)  # nosec B310  # noqa: S310


def post(
    path: str,
    body: dict | None,
    *,
    region: str = DEFAULT_REGION,
    headers: dict | None = None,
    method: str = "POST",
    attempts: int = 5,
    timeout: int = 240,
) -> tuple[int, dict]:
    """Signed-by-bearer-token JSON call. Returns (status_code, parsed_body).

    Never raises on HTTP errors: 4xx/5xx come back as (code, error_body) so the
    notebooks can *show* the error rather than blowing up the kernel.

    Retries 429 and 5xx with exponential backoff + jitter, because mantle has no
    RPM quota and sheds load under regional pressure. Also retries a 4xx whose body
    reports an internal server error - see _is_retryable. A genuine client error
    (unsupported parameter, unknown model) still fails on the first attempt.
    """
    url = host(region) + path
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(attempts):
        hdrs = {
            "Authorization": f"Bearer {token(region)}",
            "Content-Type": "application/json",
        }
        if headers:
            hdrs.update(headers)
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with _open_https(req, timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
                return resp.status, (json.loads(raw) if raw.strip() else {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                parsed = {"raw": raw[:500]}
            if _is_retryable(e.code, parsed) and attempt < attempts - 1:
                # Retry jitter, not a security decision.
                time.sleep(
                    min(2**attempt, 16) + random.random()  # nosec B311  # noqa: S311
                )
                continue
            return e.code, parsed
        except Exception as e:  # timeouts, connection resets
            if attempt < attempts - 1:
                # Retry jitter, not a security decision.
                time.sleep(
                    min(2**attempt, 16) + random.random()  # nosec B311  # noqa: S311
                )
                continue
            return -1, {"error": {"message": f"{type(e).__name__}: {e}"}}
    return -1, {"error": {"message": "retries exhausted"}}


# ---------------------------------------------------------------------------
# The HTTP-200 trap on bedrock-runtime
#
# Ask bedrock-runtime for a path it does not serve and it answers 200 OK with a
# Coral fault in the body:
#
#   {"Output":{"__type":"com.amazon.coral.service#UnknownOperationException"},
#    "Version":"1.0"}
#
# So `if resp.status_code == 200:` reads a missing route as a success. This is not
# hypothetical: the Chat Completions user-guide page shows a runtime base URL of
# ".../v1" (rather than ".../openai/v1"), and that URL returns this body for
# every model ID. Check the body, not only the status.
# ---------------------------------------------------------------------------
def unknown_op(payload: dict) -> bool:
    """True when a body is a Coral UnknownOperationException, whatever the status."""
    return "UnknownOperation" in json.dumps(payload)[:400]


def ok(status: int, payload: dict) -> bool:
    """True for a real success: 200 AND not a Coral fault wearing a 200."""
    return status == 200 and not unknown_op(payload)


# Opaque service identifiers that appear in error text. They are not credentials,
# but they are long, high-entropy, and account-scoped: printing them in full adds
# nothing for a reader and trips secret scanners on committed notebook output.
_OPAQUE_ID = re.compile(r"\b((?:resp|req|msg|file|ft|proj|batch)[_-][A-Za-z0-9]{12,})\b")


def redact_ids(text: str, keep: int = 8) -> str:
    """Shorten opaque service IDs in a string, keeping enough to correlate a log.

    `resp_example00000000000000000000000000000000`
        -> `resp_example0...`
    """

    def _shorten(m: re.Match) -> str:
        """Keep the type prefix and the first `keep` characters of the body."""
        token = m.group(1)
        prefix, _, body = token.partition("_")
        if not body:
            prefix, _, body = token.partition("-")
        return f"{prefix}_{body[:keep]}..." if body else token

    return _OPAQUE_ID.sub(_shorten, text or "")


# A 12-digit AWS account ID. 123456789012 is the documentation placeholder.
_ACCOUNT_ID = re.compile(r"(?<!\d)(?!123456789012)\d{12}(?!\d)")
_IAM_PRINCIPAL = re.compile(r"(:(?:user|role|assumed-role)/)[^\s\"',]+")


def redact_account(text: str) -> str:
    """Replace real account IDs and IAM principal names with placeholders.

    Notebook output is committed to a public repository, so anything printed here
    is published. An account ID is not a secret, but it identifies a real AWS
    account to anyone reading the samples and it trips content scanners. Call this
    on any string that may carry an ARN or a caller identity.

        arn:aws:iam::<your-account-id>:user/alice
            -> arn:aws:iam::123456789012:user/sample-user
    """
    out = _ACCOUNT_ID.sub("123456789012", text or "")
    return _IAM_PRINCIPAL.sub(r"\1sample-user", out)


def safe_print(*parts: object) -> None:
    """print() with account IDs, IAM principals AND opaque service IDs redacted.

    Use it for anything derived from STS, an ARN, or a control-plane response.

    It applies both redact_account() and redact_ids(), so the call site does not
    have to remember either.
    """
    print(*(redact_account(redact_ids(str(p))) for p in parts))


def err(payload: dict, limit: int = 160) -> str:
    """Pull the human-readable message out of an error body.

    Service error text often echoes back the ARN or ID you sent, so this redacts
    account IDs, IAM principals, and opaque IDs before returning. Notebook output is
    committed to a public repository; anything printed there is published.

    Every container access is guarded because `post()` promises never to raise:
    `{"error": "Internal server error"}` (a string, not an object) and a JSON array
    body both come back as text.

    `limit` truncates for display. Callers that MATCH on the text must pass a limit
    large enough to contain what they look for, which can sit past character 160.
    """
    if not isinstance(payload, dict):
        return redact_account(redact_ids(json.dumps(payload)))[:limit]
    e = payload.get("error")
    if isinstance(e, dict):
        msg = e.get("message") or e.get("code")
    elif isinstance(e, str):
        msg = e
    else:
        msg = None
    msg = msg or payload.get("message") or payload.get("raw") or json.dumps(payload)
    return redact_account(redact_ids(str(msg)))[:limit]


# ---------------------------------------------------------------------------
# Small conveniences used across notebooks
# ---------------------------------------------------------------------------
def list_models(region: str = DEFAULT_REGION) -> list[str]:
    """Model inventory on bedrock-mantle, which is the endpoint that serves one.

    `GET /v1/models` here, and GET-only: a POST to it is 405. `/openai/v1/models`
    is 404 on mantle, and bedrock-runtime serves neither path, so this helper is
    mantle-only; discovery on runtime is ListFoundationModels and
    ListInferenceProfiles, wrapped by runtime_models() and inference_profiles()
    above.

    Pointed at runtime, the GET below gets 404 and raises. A POST with no body
    would instead get HTTP 200 and a Coral SerializationException, which reads as
    a Region with no models, so keep `method="GET"`.

    Cached per Region: a setup cell that calls endpoints_for() for eight models makes
    one catalogue call. A refusal such as 403 is cached too, so it is not repeated per
    model. Anything _is_retryable() calls transient is not cached, so the next call
    tries again -- including a 4xx whose body reports an internal server error.
    """
    if region in _MANTLE_CATALOGUE_FAILED:
        raise RuntimeError(_MANTLE_CATALOGUE_FAILED[region])
    if region not in _MANTLE_CATALOGUE_CACHE:
        code, payload = post("/v1/models", None, region=region, method="GET")
        if code != 200:
            message = f"list_models failed {code}: {err(payload)}"
            if not _is_retryable(code, payload if isinstance(payload, dict) else {}):
                _MANTLE_CATALOGUE_FAILED[region] = message
            raise RuntimeError(message)
        _MANTLE_CATALOGUE_CACHE[region] = sorted(m["id"] for m in payload.get("data", []))
    return list(_MANTLE_CATALOGUE_CACHE[region])


def response_text(payload: dict) -> str:
    """Extract assistant text from a Responses API payload.

    Prefers the top-level output_text, falls back to walking output[] - the
    Responses API returns reasoning/tool items alongside the message.
    """
    if isinstance(payload.get("output_text"), str) and payload["output_text"]:
        return payload["output_text"]
    parts = []
    for item in payload.get("output", []) or []:
        if item.get("type") == "message":
            for block in item.get("content", []) or []:
                if block.get("text"):
                    parts.append(block["text"])
    return "".join(parts)


def parse_json_lenient(text: str) -> dict:
    """Parse the first complete JSON object out of model output.

    Some models append trailing characters after a well-formed object even in
    "strict" structured-output mode (Gemma 4 does this intermittently). Plain
    json.loads() then raises even though the useful payload is intact. This walks braces to find the first balanced object and
    parses that.
    """
    text = (text or "").strip()
    if text.startswith("```"):  # strip markdown fences if present
        text = text.split("```")[1] if "```" in text[3:] else text.lstrip("`")
        text = text[4:] if text.startswith("json") else text
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object found in {len(text)} chars: {text[:300]!r}")
    depth, in_string, escaped = 0, False, False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    # Unclosed object - some models truncate tool-call arguments mid-object
    # (qwen3-coder does this reproducibly). Close the open braces and retry
    # once; that recovers the fields that did arrive.
    if depth > 0:
        patched = text[start:] + ('"' if in_string else "") + ("}" * depth)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"unbalanced JSON after {len(text)} chars: {text[:300]!r}"
        + ("..." if len(text) > 300 else "")
    )


def repair_tool_arguments(raw: str) -> str:
    """Return a JSON string that is safe to echo back to the API.

    Some models emit truncated tool-call arguments (e.g. `{"path": "x.py"` with no
    closing brace). Echoing that verbatim into the next request is rejected with a
    400. This re-serialises whatever parsed successfully.
    """
    try:
        return json.dumps(parse_json_lenient(raw or "{}"))
    except ValueError:
        return "{}"


# ---------------------------------------------------------------------------
# Inspecting model-generated code SAFELY
#
# A coding model returns source text. It is tempting to exec() it to prove it
# works - do not. Model output is untrusted input (OWASP LLM05), and a notebook
# kernel holds your live AWS credentials, so exec() there is arbitrary code
# execution against your own account. It is also unnecessary: everything worth
# checking about generated code can be checked statically.
#
# To actually RUN generated code you need real isolation - a container or
# microVM with no credentials, no network, and a CPU/memory cap. AWS Lambda in a
# dedicated account, or Bedrock AgentCore's code-interpreter tool, both give you
# that. Running it in this kernel does not.
# ---------------------------------------------------------------------------
_FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)$", re.M)


def _dedent_block(body: str) -> str:
    """Strip the common indent, ignoring blank lines.

    An indented fence gives every body line that indent, and Python cares.
    textwrap.dedent needs uniform leading whitespace, which a blank line inside the
    block breaks, so the common indent is computed over non-blank lines only.
    """
    lines = body.split("\n")
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if indents:
        cut = min(indents)
        body = "\n".join(ln[cut:] if ln.strip() else ln for ln in lines)
    return body.strip()


def _is_info_string(rest: str) -> bool:
    """Is the text after a fence marker an info string, or the rest of a sentence?

    extract_code_block depends on this. It must accept "python title=x" and an
    indented fence (what a model emits under a numbered list), and reject a prose
    line that merely starts with a marker, such as "``` is the fence marker. Here
    is the code:".

    A CommonMark info string is short and word-like. A sentence has sentence
    punctuation and more words. That is the discriminator.
    """
    rest = rest.strip()
    if not rest:
        return True
    if len(rest) > 40 or len(rest.split()) > 3:
        return False
    return not re.search(r"[.,:;!?](?:\s|$)", rest)


def extract_code_block(markdown: str) -> str:
    """Return the first fenced code block from a model response.

    Handles ``` and ~~~ fences, four-or-more markers, an indented fence, an info
    string after the marker, and a block whose closing fence is missing because the
    generation was truncated. Falls back to the whole string when the model answered
    without fences at all.
    """
    text = markdown or ""
    opens = [m for m in _FENCE_LINE.finditer(text) if _is_info_string(m.group(2))]
    if not opens:
        return text.strip()
    opener = opens[0]
    after = text[opener.end():].lstrip("\n")
    closer = _FENCE_LINE.search(after)
    if closer is not None:
        return _dedent_block(after[:closer.start()])
    # No closing fence: a truncated generation. Return what came after the opener,
    # so the ```python line does not reach inspect_code() as a syntax error.
    return _dedent_block(after)


def inspect_code(source: str) -> dict:
    """Statically analyse generated Python. Never executes it.

    Returns a dict describing what the code declares:

        parses     bool  - is it syntactically valid Python?
        error      str   - the SyntaxError message when it is not
        functions  dict  - {name: [parameter names]} for each TOP-LEVEL def
        methods    dict  - {"Class.name": [parameters]} for defs inside a class
        classes    list  - top-level class names
        imports    list  - modules the code would import
        raises     list  - exception type names in `raise` statements
        calls      list  - names of functions the code calls

    Use it to assert that the model met a specification - the right function
    name, the right parameters, the required guard clause - without ever
    handing control to the generated text.
    """
    out: dict = {
        "parses": False,
        "error": "",
        "functions": {},
        "methods": {},
        "classes": [],
        "imports": [],
        "raises": [],
        "calls": [],
    }
    try:
        tree = ast.parse(source or "")
    except SyntaxError as exc:
        out["error"] = f"line {exc.lineno}: {exc.msg}"
        return out

    out["parses"] = True
    # TOP-LEVEL defs and classes only, from tree.body rather than ast.walk, so a
    # method is not mistaken for an importable function. Methods are reported under
    # their own key, so a caller can tell the two apart.
    def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
        """Every parameter name, in call order.

        Includes positional-only parameters, `*rest` and `**kw`, so
        `def parse_config(path, /, strict=False)` reports ["path", "strict"].
        """
        a = fn.args
        names = [p.arg for p in (*a.posonlyargs, *a.args)]
        if a.vararg:
            names.append(f"*{a.vararg.arg}")
        names += [p.arg for p in a.kwonlyargs]
        if a.kwarg:
            names.append(f"**{a.kwarg.arg}")
        return names

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out["functions"][node.name] = _params(node)
        elif isinstance(node, ast.ClassDef):
            # Top-level classes only, for the same reason as the defs above.
            out["classes"].append(node.name)
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out["methods"][f"{node.name}.{sub.name}"] = _params(sub)
    # Imports, raises and calls are legitimately anywhere, so these keep walking.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out["imports"] += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out["imports"].append((node.module or "").split(".")[0])
        elif isinstance(node, ast.Raise):
            # Read `.attr` as well as `.id`, so `raise json.JSONDecodeError(...)` and
            # `raise exc.ValidationError(...)` are recorded.
            exc_node = node.exc
            called = getattr(exc_node, "func", None)
            name = (
                getattr(exc_node, "id", None)
                or getattr(exc_node, "attr", None)
                or getattr(called, "id", None)
                or getattr(called, "attr", None)
            )
            if name:
                out["raises"].append(name)
        elif isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name:
                out["calls"].append(name)
    return out


def check_spec(
    source: str,
    *,
    function: str,
    params: list[str] | None = None,
    raises: str | None = None,
) -> dict:
    """Score generated code against a specification, statically.

    Returns {"parses", "defines", "signature", "guard", "ok", "reason"}: each a bool
    except the reason string. `params` is the expected parameter-name list;
    `raises` an exception type the code must raise somewhere.
    """
    info = inspect_code(source)
    defines = function in info["functions"]
    # Compare the NAMED parameters, ignoring `*args` / `**kwargs`:
    # `def chunk_by_tokens(text, max_tokens, overlap=50, **kwargs)` meets a spec of
    # ["text", "max_tokens", "overlap"]. A spec names the parameters a caller
    # passes; extra catch-alls do not violate it.
    actual = [a for a in info["functions"].get(function, [])
              if not a.startswith("*")]
    signature = defines and (params is None or actual == list(params))
    guard = raises is None or raises in info["raises"]
    # Say WHICH check failed, so a notebook can print something the reader can act on.
    if info["error"]:
        reason = info["error"]
    elif not defines:
        found = ", ".join(sorted(info["functions"])) or "no top-level defs"
        method = next((m for m in info["methods"] if m.endswith(f".{function}")), None)
        reason = f"no top-level def {function} (found: {found})"
        if method:
            reason += f"; it is defined as the method {method}"
    elif not signature:
        reason = (f"{function} takes {actual}, expected {list(params)}"
                  + (f" (it also accepts "
                     f"{[a for a in info['functions'][function] if a.startswith('*')]}"
                     f", which does not violate the spec)"
                     if any(a.startswith("*") for a in info["functions"][function])
                     else ""))
    elif not guard:
        raised = ", ".join(sorted(set(info["raises"]))) or "nothing"
        reason = f"{function} raises {raised}, expected {raises}"
    else:
        reason = ""
    return {
        "parses": info["parses"],
        "defines": defines,
        "signature": signature,
        "guard": guard,
        "ok": info["parses"] and defines and signature and guard,
        "reason": reason,
    }


