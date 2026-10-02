"""Reading the outside world, completely and with dates attached.

The CLI's own WebFetch tool asks a model to summarise a page. That is right for reading
a blog post and wrong for reading a feed: a summary can omit an entry, and an omission
in the discovery step is indistinguishable from "nothing was announced". So feeds are
parsed here, in code, and every entry within the window is returned.

Two things this module reports that a summariser cannot:

* **Whether the window is actually covered.** A feed holds a fixed number of entries.
  If its oldest entry is newer than the date asked for, part of the window is simply
  not in the feed, and "no entries" would be a false negative. `window_covered` says
  so explicitly.
* **Whether the response was truncated.** A page cut off at a byte limit is not a page
  that was read. `truncated` is a fact about the response, not a guess.

Both exist because of the same defect this project keeps finding: a measurement and the
claim about it drawn from the same place, so they can never disagree.
"""
from __future__ import annotations

import gzip
import logging
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Annotated, Any
from xml.etree import ElementTree  # noqa: S405

from claude_agent_sdk import tool

log = logging.getLogger("pr-agent.sources")

UA = "bedrock-samples-pr-agent (+https://github.com/aws-samples/sample-per-model-bedrock)"

# Two budgets, because the two jobs have different failure modes. A page is read for
# meaning, so truncating it costs little. A feed must parse as XML to yield any entries
# at all, so a partial read yields nothing -- and the ML blog feed embeds full post
# bodies, which is what pushed it over the page budget.
#
# Sizes on 2026-09-08, uncompressed: What's New 0.23 MB, ML blog 0.69 MB, News blog
# 0.27 MB. The feed budget is deliberately far above those: a feed that grows is a
# normal thing, and the failure mode of too small a budget is a silent gap in
# discovery, while the cost of too large a one is some memory for a few seconds.
PAGE_MAX_BYTES = 400_000
FEED_MAX_BYTES = 16_000_000

# This agent commits to a PUBLIC repository. Hosts internal to your organization must
# never be a source for it: an internal roadmap page or a wiki is not publishable, and the
# safest way to guarantee that is to be unable to read one. List them, comma separated, in
# INTERNAL_HOST_SUFFIXES (for example "corp.example.com,wiki.example.net"). Refusing by
# host is a coarse control, and coarse is what is wanted here.
def _internal_host_markers() -> tuple[str, ...]:
    raw = os.environ.get("INTERNAL_HOST_SUFFIXES", "")
    return tuple(m.strip().lower() for m in raw.split(",") if m.strip())


class RefusedSource(RuntimeError):
    """The URL is not one this agent may read."""


def _check_url(url: str) -> str:
    url = (url or "").strip()
    if not url.startswith("https://"):
        raise RefusedSource(
            f"refusing {url!r}: only https URLs are read, so a source cannot be "
            "tampered with in transit"
        )
    host = url.split("/", 3)[2].lower().split("@")[-1].split(":")[0]
    for marker in _internal_host_markers():
        if host == marker.lstrip(".") or host.endswith(marker):
            raise RefusedSource(
                f"refusing {url!r}: {host} is listed as internal in INTERNAL_HOST_SUFFIXES. This agent "
                "commits to a public repository, so it reads public sources only. If a "
                "change is only documented internally, it is not yours to publish."
            )
    return url


def _get(url: str, timeout: int = 60, limit: int = PAGE_MAX_BYTES
         ) -> tuple[str, bool, int]:
    """Returns (text, truncated, bytes_read). Never silently shortens anything."""
    req = urllib.request.Request(
        _check_url(url),
        headers={"User-Agent": UA, "Accept-Encoding": "gzip", "Accept": "*/*"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        raw = resp.read(limit + 1)
        if resp.headers.get("Content-Encoding") == "gzip":
            try:
                raw = gzip.decompress(raw)
            except (OSError, EOFError):
                # A truncated gzip stream cannot be decompressed; say so rather than
                # returning the compressed bytes as if they were text.
                raise RuntimeError(
                    "response was gzip-encoded and cut off at the byte limit, so it "
                    "cannot be decoded. Fetch a more specific URL."
                ) from None
    truncated = len(raw) > limit
    return raw[:limit].decode("utf-8", "replace"), truncated, len(raw)


def _entry_date(elem: ElementTree.Element) -> datetime | None:
    for path in ("pubDate", "{http://www.w3.org/2005/Atom}updated",
                 "{http://www.w3.org/2005/Atom}published",
                 "{http://purl.org/dc/elements/1.1/}date"):
        node = elem.find(path)
        if node is None or not (node.text or "").strip():
            continue
        text = node.text.strip()
        for parse in (parsedate_to_datetime, datetime.fromisoformat):
            try:
                dt = parse(text.replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None


def _text_of(elem: ElementTree.Element, *paths: str) -> str:
    for path in paths:
        node = elem.find(path)
        if node is not None:
            if (node.text or "").strip():
                return node.text.strip()
            href = node.attrib.get("href")
            if href:
                return href
    return ""


def feed_recent(url: str, since: str = "") -> dict[str, Any]:
    """Entries at or after `since`, plus whether the feed covers that window."""
    cutoff = None
    if since.strip():
        try:
            cutoff = datetime.fromisoformat(since.strip().replace("Z", "+00:00"))
            if not cutoff.tzinfo:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
        except ValueError:
            return {"error": f"since={since!r} is not an ISO date, e.g. 2026-09-01"}
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=14)

    try:
        body, truncated, nbytes = _get(url, timeout=120, limit=FEED_MAX_BYTES)
    except RefusedSource as exc:
        return {"error": str(exc)}
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "url": url}

    if truncated:
        # A truncated feed will not parse as XML, and a partial parse would silently
        # drop the entries that were cut off.
        return {
            "error": f"feed exceeded {FEED_MAX_BYTES} bytes and was cut off, so it cannot be "
                     "parsed completely. Use a narrower feed URL.",
            "url": url,
        }
    try:
        root = ElementTree.fromstring(body)  # noqa: S314
    except ElementTree.ParseError as exc:
        return {"error": f"not parseable as XML: {exc}", "url": url,
                "starts_with": body[:120]}

    # Parsing as XML is not the same as being a feed. An HTML error page -- a captive
    # portal, a 403 body, a "service unavailable" page -- is often well-formed enough to
    # parse, contains no <item> elements, and would otherwise be reported as "0 entries":
    # a false negative that looks exactly like a quiet day.
    root_tag = root.tag.rsplit("}", 1)[-1].lower()
    if root_tag not in ("rss", "feed", "rdf"):
        return {
            "error": f"response parsed as XML but its root element is <{root_tag}>, not a "
                     "feed. This is most likely an error page, so treat it as 'could not "
                     "read this source', NOT as 'nothing was announced'.",
            "url": url,
            "starts_with": body[:160],
        }

    items = root.findall(".//item") or root.findall(
        ".//{http://www.w3.org/2005/Atom}entry"
    )
    dated = [(_entry_date(i), i) for i in items]
    oldest = min((d for d, _ in dated if d), default=None)

    hits = []
    for dt, item in dated:
        if dt is not None and dt < cutoff:
            continue
        hits.append({
            "date": dt.date().isoformat() if dt else "undated",
            "title": _text_of(item, "title", "{http://www.w3.org/2005/Atom}title")[:300],
            "link": _text_of(item, "link", "{http://www.w3.org/2005/Atom}link")[:400],
        })

    covered = oldest is not None and oldest <= cutoff
    return {
        "url": url,
        "since": cutoff.date().isoformat(),
        "bytes": nbytes,
        "entries_in_feed": len(items),
        "oldest_entry_in_feed": oldest.date().isoformat() if oldest else None,
        # The distinction that matters: an empty result from a feed that does not reach
        # back to `since` is not evidence that nothing was announced.
        "window_covered": covered,
        "coverage_note": (
            "This feed reaches back to " + (oldest.date().isoformat() if oldest else "?")
            + f", which is after {cutoff.date().isoformat()}. Entries between those two "
            "dates are not in this feed, so an empty result here does NOT mean nothing "
            "was announced. Check a dated index page for the gap."
        ) if not covered else "",
        "matched": len(hits),
        "entries": hits[:80],
    }


@tool(
    "sources_feed",
    "Parse an RSS or Atom feed and return every entry published on or after a date. "
    "Prefer this over WebFetch for feeds: it returns all entries rather than a model's "
    "summary, so nothing is silently dropped. Check `window_covered` in the result -- "
    "when it is false the feed does not reach back far enough and an empty result is "
    "NOT evidence that nothing was announced.",
    {
        "url": Annotated[str, "https feed URL, e.g. AWS What's New RSS."],
        "since": Annotated[str, "ISO date, e.g. 2026-09-01. Empty means 14 days ago."],
    },
)
async def sources_feed(args: dict[str, Any]) -> dict[str, Any]:
    import json

    result = feed_recent(args.get("url") or "", args.get("since") or "")
    return {"content": [{"type": "text", "text": json.dumps(result, indent=1)}]}


@tool(
    "sources_fetch",
    "Fetch a public https page as plain text, with HTML tags stripped. Use for AWS "
    "documentation pages such as the Bedrock doc-history page, where you want the "
    "actual text rather than a summary. Reports `truncated` honestly -- a page that was "
    "cut off has not been read, so do not draw a negative conclusion from one.",
    {"url": Annotated[str, "https URL of a public page."]},
)
async def sources_fetch(args: dict[str, Any]) -> dict[str, Any]:
    url = args.get("url") or ""
    try:
        body, truncated, nbytes = _get(url)
    except RefusedSource as exc:
        return {"content": [{"type": "text", "text": f"REFUSED: {exc}"}]}
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        return {"content": [{"type": "text",
                             "text": f"ERROR {type(exc).__name__}: {exc}"}]}
    text = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", body)
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;?", " ", text)
    text = re.sub(r"&amp;?", "&", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text).strip()
    header = (
        f"url={url}\nbytes={nbytes}\ntruncated={truncated}\n"
        + ("NOTE: this page was cut off at the byte limit. It has not been read in "
           "full; do not conclude anything from its absence of content.\n"
           if truncated else "")
        + "---\n"
    )
    return {"content": [{"type": "text", "text": header + text}]}


sources_feed.raw = feed_recent  # type: ignore[attr-defined]
