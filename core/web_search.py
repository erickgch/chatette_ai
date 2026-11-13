"""
Web search through DuckDuckGo (ddgs library) — no API key, no account.

Two kinds of search:
  - current=True  → DuckDuckGo News over the past week, keeping only results from the last
                    CURRENT_DAYS days (DuckDuckGo's own filters only offer day/week/month/year).
                    If that leaves too little, a web search over the past day is added.
  - current=False → a normal web search, no time limit.

ddgs reads DuckDuckGo's public pages, not an official API: too many searches in a short time
get a "bot check" page back, and web search then returns nothing for a while (news search kept
working when this happened in testing). So web search tries DuckDuckGo first and falls back to
the other engines ddgs supports; when everything fails, search() raises SearchUnavailable and
Chatette says so instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

CURRENT_DAYS = 3
MAX_RESULTS = 6

# A region per answer language — ddgs's worldwide region ("wt-wt") breaks its web search.
_REGIONS = {"en": "us-en", "de": "de-de", "es": "es-es"}


class SearchUnavailable(Exception):
    """The search engines could not be reached or refused the request."""


@dataclass
class Result:
    title: str
    snippet: str
    url: str
    source: str          # short name shown to the user: the publisher, or the site's domain
    date: datetime | None = None


def site_name(url: str) -> str:
    """'https://www.reuters.com/world/…' → 'reuters.com'."""
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def search(query: str, current: bool, lang: str = "en") -> list[Result]:
    """Search the web. Raises SearchUnavailable when no engine answers at all."""
    from ddgs import DDGS

    region = _REGIONS.get(lang, "us-en")
    ddgs = DDGS(timeout=10)
    results: list[Result] = []
    errors: list[str] = []

    if current:
        try:
            results += _recent_news(ddgs, query, region)
        except Exception as e:
            errors.append(f"news: {e}")
        if len(results) < 3:
            results += _web(ddgs, query, region, timelimit="d", errors=errors)
    else:
        results += _web(ddgs, query, region, timelimit=None, errors=errors)

    if not results and errors and all(_is_failure(e) for e in errors):
        raise SearchUnavailable("; ".join(errors))
    return _dedupe(results)[:MAX_RESULTS]


def _recent_news(ddgs, query: str, region: str) -> list[Result]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=CURRENT_DAYS)
    out = []
    for r in _call(ddgs.news, query, region=region, timelimit="w", max_results=15):
        date = _parse_date(r.get("date"))
        if date is None or date < cutoff:
            continue
        # "USA TODAY · via Yahoo News" → "USA TODAY" (the link may point to the aggregator)
        source = str(r.get("source") or "").split(" · ")[0].strip() or site_name(r.get("url", ""))
        out.append(Result(title=str(r.get("title", "")), snippet=str(r.get("body", "")),
                          url=str(r.get("url", "")), source=source, date=date))
    return sorted(out, key=lambda x: x.date, reverse=True)


def _web(ddgs, query: str, region: str, timelimit: str | None, errors: list[str]) -> list[Result]:
    for backend in ("duckduckgo", "auto"):
        try:
            rows = _call(ddgs.text, query, region=region, timelimit=timelimit,
                         max_results=MAX_RESULTS, backend=backend)
        except Exception as e:
            errors.append(f"web/{backend}: {e}")
            continue
        if rows:
            return [Result(title=str(r.get("title", "")), snippet=str(r.get("body", "")),
                           url=str(r.get("href", "")), source=site_name(r.get("href", "")))
                    for r in rows]
    return []


def _call(fn, query: str, **kwargs) -> list[dict]:
    """ddgs raises 'No results found.' instead of returning [] — treat that as empty."""
    try:
        return fn(query, **kwargs) or []
    except Exception as e:
        if "no results found" in str(e).lower():
            return []
        raise


def _is_failure(error: str) -> bool:
    return "no results found" not in error.lower()


def _parse_date(value) -> datetime | None:
    try:
        date = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return date if date.tzinfo else date.replace(tzinfo=timezone.utc)


def _dedupe(results: list[Result]) -> list[Result]:
    seen, out = set(), []
    for r in results:
        if r.url and r.url not in seen:
            seen.add(r.url)
            out.append(r)
    return out
