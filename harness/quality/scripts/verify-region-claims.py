#!/usr/bin/env python3
"""Probe the availability claims the notebooks state flatly.

`evidence-audit.py` isolated 48 unhedged claims that no committed output establishes.
Three of them are the exact shape that has bitten this repository before: a bare "only"
about Region or feature availability, written once and never re-derived.

    06-zai-glm/01   §gotchas  "Only glm-4.6 and glm-4.7-flash reach eu-central-1"
    07-mistral/01   §gotchas  "mistral-large-3 absent from eu-central-1; Ministrals present"
    04-qwen/01      §gotchas  "qwen3-32b is one of only two mantle-fine-tunable models"

The first two are answerable from the per-Region catalogue, which is cheap and
authoritative. This prints what the service actually lists, so the sentence can be
rewritten from the answer instead of from memory.

    python3 verify-region-claims.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.environ.get("REPO") or os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), "_shared"))
import bedrock  # noqa: E402

REGIONS = ("us-east-1", "us-west-2", "eu-central-1")


def catalogue(region: str) -> list[str]:
    """Model IDs bedrock-mantle lists in a Region, or [] if it will not say."""
    try:
        models = bedrock.list_models(region=region)
    except TypeError:
        # Older signature took no Region; fall back to the env var the helper reads.
        prev = os.environ.get("AWS_REGION")
        os.environ["AWS_REGION"] = region
        try:
            models = bedrock.list_models()
        finally:
            if prev is None:
                os.environ.pop("AWS_REGION", None)
            else:
                os.environ["AWS_REGION"] = prev
    ids = []
    for m in models or []:
        if isinstance(m, dict):
            ids.append(m.get("id") or m.get("modelId") or "")
        else:
            ids.append(str(m))
    return sorted(i for i in ids if i)


def main() -> None:
    per_region = {}
    for region in REGIONS:
        try:
            per_region[region] = catalogue(region)
            print(f"{region}: {len(per_region[region])} model(s) listed")
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            per_region[region] = []
            print(f"{region}: catalogue unavailable -- {type(exc).__name__}: {exc}")

    print("\n--- claim 1: 'Only glm-4.6 and glm-4.7-flash reach eu-central-1'")
    eu = per_region.get("eu-central-1") or []
    zai_eu = [m for m in eu if m.startswith("zai.")]
    print(f"  zai.* in eu-central-1: {zai_eu or 'none listed'}")
    claimed = {"zai.glm-4.6", "zai.glm-4.7-flash"}
    if eu:
        extra = sorted(set(zai_eu) - claimed)
        missing = sorted(claimed - set(zai_eu))
        print(f"  listed but not claimed: {extra or 'none'}")
        print(f"  claimed but not listed: {missing or 'none'}")
        print(f"  VERDICT: {'holds' if not extra and not missing else 'DOES NOT HOLD'}")

    print("\n--- claim 2: 'mistral-large-3 absent from eu-central-1; Ministrals present'")
    mistral_eu = [m for m in eu if m.startswith("mistral.")]
    print(f"  mistral.* in eu-central-1: {mistral_eu or 'none listed'}")
    if eu:
        large = [m for m in mistral_eu if "large-3" in m]
        mini = [m for m in mistral_eu if "ministral" in m.lower()]
        print(f"  large-3 present: {large or 'no'}   ministral present: {mini or 'no'}")
        holds = not large and bool(mini)
        print(f"  VERDICT: {'holds' if holds else 'DOES NOT HOLD'}")

    print("\n--- claim 3: 'qwen3-32b is one of only two mantle-fine-tunable models'")
    print("  Not answerable from the model catalogue: the list carries no")
    print("  fine-tunability flag. Needs the fine-tuning docs or a CreateModelCustom")
    print("  probe, and an 'only two' count cannot be established by either.")

    print("\n--- full zai / mistral / qwen catalogue per Region, for rewriting the tables")
    for region in REGIONS:
        rows = [m for m in (per_region.get(region) or [])
                if m.split(".")[0] in {"zai", "mistral", "qwen"}]
        print(f"\n  {region}:")
        for m in rows:
            print(f"    {m}")


if __name__ == "__main__":
    main()
