"""Read the live Bedrock catalogues, and make one real call.

The samples repository exists because Bedrock's behaviour is not reliably derivable
from its documentation: model IDs differ per endpoint, an API can return HTTP 200 with
an error in the body, and a rule inferred from one model has twice turned out to be
wrong. So the agent is given the ability to ask the service directly, and the system
prompt tells it that the service outranks any announcement it has read.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Annotated, Any

import boto3
from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.bedrock")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def _text(payload: Any) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": str(payload)}]}


def _bearer(region: str) -> str:
    """A short-lived bearer token for the OpenAI-compatible paths."""
    from aws_bedrock_token_generator import provide_token  # noqa: PLC0415
    return provide_token(region=region)


@tool(
    "bedrock_list_models",
    "The production Amazon Bedrock model catalogue for a Region, read from GET /v1/models "
    "on the bedrock-mantle endpoint (bedrock-mantle.<region>.api.aws). To be explicit "
    "because the hostname does not look like it: this IS production Bedrock, not a "
    "staging surface. Bedrock exposes two runtime endpoints -- bedrock-runtime, the "
    "original AWS-shaped API, and bedrock-mantle, the OpenAI-compatible one -- and they "
    "serve different model ID forms, which is much of what this repository documents. "
    "This is the authoritative answer to 'does this model exist on the mantle endpoint', "
    "and it outranks any announcement. Pass a Region such as us-east-1.",
    {"region": Annotated[str, "AWS Region, e.g. us-east-1. Empty means us-east-1."]},
)
async def bedrock_list_models(args: dict[str, Any]) -> dict[str, Any]:
    region = (args.get("region") or REGION).strip()
    try:
        req = urllib.request.Request(
            f"https://bedrock-mantle.{region}.api.aws/v1/models",
            headers={"Authorization": f"Bearer {_bearer(region)}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=90) as resp:  # noqa: S310
            data = json.loads(resp.read())
        ids = sorted(m["id"] for m in data.get("data", []))
        return _text(f"{len(ids)} model(s) on bedrock-mantle in {region}:\n" + "\n".join(ids))
    except urllib.error.HTTPError as exc:
        return _text(f"ERROR HTTP {exc.code}: {exc.read().decode('utf-8','replace')[:400]}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "bedrock_list_runtime_models",
    "The production bedrock-runtime catalogue via ListFoundationModels, plus inference "
    "profiles. This is the other of Bedrock's two runtime endpoints, and its model IDs "
    "are NOT the same as the mantle endpoint's; profile-only models refuse a bare ID. "
    "Check here before writing a runtime sample, and compare the two catalogues before "
    "claiming a model is absent -- absent from one endpoint is not absent from Bedrock.",
    {"region": Annotated[str, "AWS Region, e.g. us-east-1. Empty means us-east-1."]},
)
async def bedrock_list_runtime_models(args: dict[str, Any]) -> dict[str, Any]:
    region = (args.get("region") or REGION).strip()
    try:
        control = boto3.client("bedrock", region_name=region)
        summaries = control.list_foundation_models().get("modelSummaries", [])
        rows = sorted(
            f"{s['modelId']:52} in={','.join(s.get('inputModalities', []))} "
            f"types={','.join(s.get('inferenceTypesSupported', []))}"
            for s in summaries
        )
        profiles: list[str] = []
        try:
            pages = control.get_paginator("list_inference_profiles").paginate()
            for page in pages:
                profiles += [p["inferenceProfileId"] for p in page.get("inferenceProfileSummaries", [])]
        except Exception as exc:  # noqa: BLE001
            profiles = [f"(profiles unavailable: {type(exc).__name__})"]
        return _text(
            f"{len(rows)} runtime model(s) in {region}:\n" + "\n".join(rows)
            + f"\n\n{len(profiles)} inference profile(s):\n" + "\n".join(sorted(profiles))
        )
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")


@tool(
    "bedrock_invoke",
    "Make ONE real inference call and return the status and body, so you can verify a "
    "claim instead of asserting it. endpoint is 'mantle' or 'runtime'; path is the URL "
    "path such as /v1/chat/completions or /openai/v1/responses or "
    "/anthropic/v1/messages; body is a JSON string. A non-200 response is a RESULT, not "
    "a failure -- record it. Note that bedrock-runtime answers an unknown path with 200 "
    "and an UnknownOperationException in the body, so read the body, not just the status.",
    {
        "endpoint": Annotated[str, "'mantle' or 'runtime'"],
        "path": Annotated[str, "URL path, e.g. /v1/chat/completions"],
        "body": Annotated[str, "Request body as a JSON string"],
        "region": Annotated[str, "AWS Region. Empty means us-east-1."],
        "extra_headers": Annotated[str, "Optional JSON object of extra headers."],
    },
)
async def bedrock_invoke(args: dict[str, Any]) -> dict[str, Any]:
    region = (args.get("region") or REGION).strip()
    endpoint = (args.get("endpoint") or "mantle").strip().lower()
    path = (args.get("path") or "").strip()
    if not path.startswith("/"):
        return _text("ERROR: path must start with '/'")
    # Validated rather than defaulted. The two endpoints answer differently for the same
    # model, which is most of what this repository documents, so quietly sending a
    # mistyped 'mantle' to runtime would produce a confident claim about the wrong
    # endpoint -- the single most expensive error this tool could make.
    hosts = {
        "mantle": f"https://bedrock-mantle.{region}.api.aws",
        "runtime": f"https://bedrock-runtime.{region}.amazonaws.com",
    }
    if endpoint not in hosts:
        return _text(
            f"ERROR: endpoint must be 'mantle' or 'runtime', got {endpoint!r}. These are "
            "different services with different model IDs and paths, so this is not "
            "guessed for you."
        )
    host = hosts[endpoint]
    try:
        payload = json.loads(args.get("body") or "{}")
    except json.JSONDecodeError as exc:
        return _text(f"ERROR body is not valid JSON: {exc}")
    headers = {
        "Authorization": f"Bearer {_bearer(region)}",
        "Content-Type": "application/json",
    }
    try:
        if args.get("extra_headers"):
            headers.update(json.loads(args["extra_headers"]))
    except json.JSONDecodeError as exc:
        return _text(f"ERROR extra_headers is not valid JSON: {exc}")
    req = urllib.request.Request(
        host + path, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=240) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", "replace")
            return _text(f"HTTP {resp.status}\n{body[:6000]}")
    except urllib.error.HTTPError as exc:
        return _text(f"HTTP {exc.code}\n{exc.read().decode('utf-8','replace')[:6000]}")
    except Exception as exc:  # noqa: BLE001
        return _text(f"ERROR {type(exc).__name__}: {exc}")
