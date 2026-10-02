#!/usr/bin/env python3
"""Probe the notebook claims that no committed output establishes.

`evidence-audit.py` reduced 1,173 claims to 48 that are stated flatly and that nothing
in the repository derives. Three were Region or fine-tuning availability, settled
separately by `verify-region-claims.py` and the AWS fine-tuning documentation. Most of
the rest are field names and API shapes, and those the service can answer directly.

Each check below prints what the service did, then the claim, then a verdict. Nothing
is hardcoded as a conclusion: the verdict is computed from the response, so a service
change turns this red rather than leaving a confident sentence in a notebook.

    python3 verify-asserted-claims.py            # all checks
    python3 verify-asserted-claims.py --only ttl # substring match on the check name
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
import bedrock  # noqa: E402

GPT = "openai.gpt-5.6-sol"
GEMMA = "google.gemma-4-31b"
GROK = "xai.grok-4.3"
CLAUDE = "anthropic.claude-haiku-4-5"
AV = {"anthropic-version": "2023-06-01"}

RESULTS: list[tuple[str, str, str, str]] = []  # (name, claim, observed, verdict)


def record(name: str, claim: str, observed: str, holds: bool | None) -> None:
    verdict = "HOLDS" if holds else ("DOES NOT HOLD" if holds is False else "UNSETTLED")
    RESULTS.append((name, claim, observed, verdict))
    print(f"  {verdict}: {observed}")


# --------------------------------------------------------------------------- checks
def check_minimal_effort() -> None:
    """01-openai-gpt/01 §gotchas and 03-google-gemma/01 §gotchas:
    reasoning.effort accepts none/low/medium/high; `minimal` is rejected."""
    print("\n[minimal-effort] reasoning.effort='minimal' is rejected")
    seen = {}
    for model in (GPT, GEMMA):
        for effort in ("minimal", "none", "low", "medium", "high"):
            code, data = bedrock.post(
                "/openai/v1/responses",
                {"model": model, "input": "Reply OK", "max_output_tokens": 2000,
                 "reasoning": {"effort": effort}}, attempts=1, timeout=120)
            seen[(model, effort)] = code
            print(f"    {model:<24} effort={effort:<8} -> {code}")
    rejected = all(seen[(m, "minimal")] >= 400 for m in (GPT, GEMMA))
    accepted = all(seen[(m, e)] == 200 for m in (GPT, GEMMA)
                   for e in ("none", "low", "medium", "high"))
    record("minimal-effort", "minimal rejected, the other four accepted",
           f"minimal rejected on both={rejected}, other four all 200={accepted}",
           rejected and accepted)


def check_cache_ttl() -> None:
    """01-openai-gpt/04 §5: prompt_cache_options.ttl takes 30m, and only 30m."""
    print("\n[cache-ttl] prompt_cache_options.ttl accepts only '30m'")
    prompt = "You are a careful assistant. " * 120
    codes = {}
    for ttl in ("30m", "5m", "1h", "60m", "24h"):
        code, data = bedrock.post(
            "/openai/v1/responses",
            {"model": GPT, "input": prompt + "Reply OK", "max_output_tokens": 2000,
             "prompt_cache_options": {"ttl": ttl}}, attempts=1, timeout=120)
        codes[ttl] = code
        detail = "" if code == 200 else f"  {bedrock.err(data, limit=110)}"
        print(f"    ttl={ttl:<5} -> {code}{detail}")
    only_30m = codes["30m"] == 200 and all(
        codes[t] >= 400 for t in ("5m", "1h", "60m", "24h"))
    record("cache-ttl", "30m accepted; 5m/1h/60m/24h refused",
           f"30m={codes['30m']}, others={ {k: v for k, v in codes.items() if k != '30m'} }",
           only_30m)


def check_response_lifecycle() -> None:
    """01-openai-gpt/01 §12: GET /openai/v1/responses/{id} retrieves a stored
    response and POST .../{id}/cancel cancels an in-flight one."""
    print("\n[response-lifecycle] GET /responses/{id} and POST /responses/{id}/cancel")
    code, data = bedrock.post(
        "/openai/v1/responses",
        {"model": GPT, "input": "Reply OK", "max_output_tokens": 2000, "store": True},
        attempts=1, timeout=120)
    rid = (data or {}).get("id") if isinstance(data, dict) else None
    print(f"    create store=True -> {code}, id={rid!r}")
    if not rid:
        record("response-lifecycle", "GET retrieves, POST cancel exists",
               f"could not create a stored response ({code})", None)
        return
    g_code, g_data = bedrock.post(f"/openai/v1/responses/{rid}", None,
                                  method="GET", attempts=1, timeout=90)
    print(f"    GET  /responses/{{id}}        -> {g_code}")
    c_code, c_data = bedrock.post(f"/openai/v1/responses/{rid}/cancel", {},
                                  attempts=1, timeout=90)
    # A completed response cannot be cancelled; the operation EXISTING is the claim,
    # so a 4xx that names the state is a pass and a 404-unknown-route is a fail.
    body = bedrock.err(c_data, limit=200) if c_code != 200 else "ok"
    print(f"    POST /responses/{{id}}/cancel -> {c_code}  {body}")
    route_exists = c_code == 200 or (
        c_code < 500 and "UnknownOperation" not in str(c_data)
        and "not found" not in body.lower())
    record("response-lifecycle", "GET retrieves a stored response; cancel route exists",
           f"GET={g_code}, cancel={c_code} ({body[:80]})",
           g_code == 200 and route_exists)


def check_cache_usage_fields() -> None:
    """01-openai-gpt/01 §14: usage.input_tokens_details reports cache_write_tokens on
    the first call and cached_tokens on a repeat."""
    print("\n[cache-usage] usage.input_tokens_details fields")
    prompt = "Cache probe. " * 400
    body = {"model": GPT, "input": prompt + "Reply OK", "max_output_tokens": 2000,
            "prompt_cache_options": {"ttl": "30m"}}
    first_fields: dict = {}
    second_fields: dict = {}
    for label, sink in (("first", first_fields), ("repeat", second_fields)):
        code, data = bedrock.post("/openai/v1/responses", body,
                                  attempts=1, timeout=120)
        usage = (data or {}).get("usage") or {} if isinstance(data, dict) else {}
        details = usage.get("input_tokens_details") or {}
        sink.update(details)
        print(f"    {label:<7} -> {code}  input_tokens_details={details}")
        if label == "first":
            time.sleep(3)
    has_write = "cache_write_tokens" in first_fields
    has_read = "cached_tokens" in second_fields
    record("cache-usage",
           "cache_write_tokens on the first call, cached_tokens on a repeat",
           f"first has cache_write_tokens={has_write}, "
           f"repeat has cached_tokens={has_read} "
           f"(repeat value={second_fields.get('cached_tokens')})",
           has_write and has_read)


def check_count_tokens() -> None:
    """02-anthropic-claude/01 §9: POST /anthropic/v1/messages/count_tokens."""
    print("\n[count-tokens] POST /anthropic/v1/messages/count_tokens on mantle")
    code, data = bedrock.post(
        "/anthropic/v1/messages/count_tokens",
        {"model": CLAUDE, "messages": [{"role": "user", "content": "Hello there"}]},
        headers=AV, attempts=1, timeout=90)
    n = (data or {}).get("input_tokens") if isinstance(data, dict) else None
    print(f"    -> {code}  input_tokens={n}")
    record("count-tokens", "the count_tokens path exists and returns input_tokens",
           f"status={code}, input_tokens={n}",
           code == 200 and isinstance(n, int) and n > 0)


def check_mistral_no_reasoning() -> None:
    """07-mistral/01 §gotchas: Mistral returns no message.reasoning, while qwen,
    deepseek, glm, kimi, minimax and nemotron-super do."""
    print("\n[mistral-reasoning] message.reasoning: absent for Mistral, present for six")
    absent_for = ["mistral.mistral-large-3-675b-instruct",
                  "mistral.magistral-small-2509"]
    present_for = ["qwen.qwen3-32b", "deepseek.v3.2", "zai.glm-4.7",
                   "moonshotai.kimi-k2-thinking", "minimax.minimax-m2",
                   "nvidia.nemotron-super-3-120b"]
    question = "A bat and ball cost 1.10 together. The bat costs 1.00 more than the "\
               "ball. How much is the ball? Think it through."
    got: dict[str, object] = {}
    for model in absent_for + present_for:
        code, data = bedrock.post(
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": question}],
             "max_tokens": 1200}, attempts=1, timeout=180)
        msg = {}
        if isinstance(data, dict) and data.get("choices"):
            msg = data["choices"][0].get("message") or {}
        trace = msg.get("reasoning")
        got[model] = None if trace is None else len(str(trace))
        print(f"    {model:<40} {code}  reasoning="
              f"{'absent' if trace is None else str(got[model]) + ' chars'}")
    mistral_absent = all(got[m] is None for m in absent_for)
    others_present = {m: got[m] is not None for m in present_for}
    record("mistral-reasoning",
           "Mistral absent; qwen/deepseek/glm/kimi/minimax/nemotron-super present",
           f"mistral absent={mistral_absent}; others present={others_present}",
           mistral_absent and all(others_present.values()))


def check_grok_responses_path() -> None:
    """11-xai-grok/01 §gotchas: bare /v1 chat/completions 400s fast, but
    /v1/responses stalls."""
    print("\n[grok-paths] bare /v1 for Grok: chat/completions vs responses")
    t0 = time.monotonic()
    c1, d1 = bedrock.post("/v1/chat/completions",
                          {"model": GROK, "messages": [{"role": "user",
                                                        "content": "Hi"}],
                           "max_tokens": 16}, attempts=1, timeout=60)
    fast = time.monotonic() - t0
    print(f"    /v1/chat/completions -> {c1} in {fast:.1f}s  "
          f"{bedrock.err(d1, limit=90) if c1 != 200 else ''}")
    t0 = time.monotonic()
    try:
        c2, d2 = bedrock.post("/v1/responses",
                              {"model": GROK, "input": "Hi",
                               "max_output_tokens": 2000}, attempts=1, timeout=45)
        slow = time.monotonic() - t0
        note = f"{c2} in {slow:.1f}s"
        stalls = slow > fast * 3
    except Exception as exc:  # noqa: BLE001 - a timeout IS the observation
        slow = time.monotonic() - t0
        note = f"{type(exc).__name__} after {slow:.1f}s"
        stalls = True
    print(f"    /v1/responses        -> {note}")
    record("grok-paths", "chat/completions fails fast; /v1/responses stalls",
           f"chat={c1} in {fast:.1f}s vs responses {note}",
           c1 >= 400 and stalls)


def check_grok_cache_field() -> None:
    """11-xai-grok/02 §9: cached tokens appear in
    usage.prompt_tokens_details.cached_tokens."""
    print("\n[grok-cache] usage.prompt_tokens_details.cached_tokens")
    prompt = "Grok cache probe. " * 400
    seen = []
    for _ in range(2):
        code, data = bedrock.post(
            "/openai/v1/chat/completions",
            {"model": GROK, "messages": [{"role": "user",
                                          "content": prompt + " Reply OK"}],
             "max_tokens": 32}, attempts=1, timeout=120)
        usage = (data or {}).get("usage") or {} if isinstance(data, dict) else {}
        details = usage.get("prompt_tokens_details")
        seen.append(details)
        print(f"    -> {code}  prompt_tokens_details={details}")
        time.sleep(2)
    present = any(isinstance(d, dict) and "cached_tokens" in d for d in seen)
    record("grok-cache", "the field exists on Grok usage",
           f"prompt_tokens_details across two calls: {seen}", present)


def check_retention_modes() -> None:
    """00-foundations/02 §3 and 99-cross-cutting/03 §4: data_retention.mode defaults to
    `inherit`, and the modes are default / none / provider_data_share / inherit."""
    print("\n[retention-modes] data_retention.mode values")
    code, data = bedrock.post("/v1/data_retention", None, method="GET",
                              attempts=1, timeout=90)
    print(f"    GET /v1/data_retention -> {code}  {str(data)[:200]}")
    # The authoritative check is which values the API accepts. Probing that would
    # WRITE account-level retention configuration, which is not something a
    # verification script should do, so this reports the readable state only.
    record("retention-modes", "default is inherit; four modes exist",
           f"GET returned {code}; the mode enumeration cannot be confirmed without "
           f"writing account retention settings, which this refuses to do", None)


CHECKS = {
    "minimal-effort": check_minimal_effort,
    "cache-ttl": check_cache_ttl,
    "response-lifecycle": check_response_lifecycle,
    "cache-usage": check_cache_usage_fields,
    "count-tokens": check_count_tokens,
    "mistral-reasoning": check_mistral_no_reasoning,
    "grok-paths": check_grok_responses_path,
    "grok-cache": check_grok_cache_field,
    "retention-modes": check_retention_modes,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    for name, fn in CHECKS.items():
        if args.only and args.only not in name:
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR in {name}: {type(exc).__name__}: {exc}")
            RESULTS.append((name, "-", f"{type(exc).__name__}: {exc}", "ERROR"))

    print("\n" + "=" * 78)
    print(f"{'check':<20} {'verdict':<15} claim")
    print("-" * 78)
    for name, claim, _observed, verdict in RESULTS:
        print(f"{name:<20} {verdict:<15} {claim[:42]}")
    bad = [r for r in RESULTS if r[3] in ("DOES NOT HOLD", "ERROR")]
    print(f"\n{len(RESULTS)} claim(s) probed, {len(bad)} not holding")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
