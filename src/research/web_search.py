"""Web search for real-time fact verification of slide claims.

Uses Google News RSS (no API key) to fetch current data for specific claims,
so the fact-checker can compare slide content against up-to-date information.
"""

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser


def search_claim(query: str, max_results: int = 5) -> list[dict]:
    """Search for recent information about a specific claim.

    Uses Google News RSS to find recent articles matching the query.

    Args:
        query: The search query (e.g. "oil price today", "S&P 500 current level").
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, source, published, summary, link.
    """
    encoded_query = query.replace(" ", "+")
    # Search recent articles (last 7 days for broader coverage)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")
    url = (
        f"https://news.google.com/rss/search"
        f"?q={encoded_query}+after:{cutoff}&hl=en-US&gl=US&ceid=US:en"
    )

    results = []
    try:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_results]:
            summary = entry.get("summary", "")
            summary = (
                summary.replace("<b>", "")
                .replace("</b>", "")
                .replace("<br>", " ")
                .replace("&amp;", "&")
            )
            results.append({
                "title": entry.get("title", ""),
                "source": entry.get("source", {}).get("title", "Unknown"),
                "published": entry.get("published", ""),
                "summary": summary[:300],
                "link": entry.get("link", ""),
            })
    except Exception as e:
        print(f"[web_search] Failed to search for '{query}': {e}")

    return results


def search_claims_batch(claims: list[str], max_results_per: int = 3) -> dict[str, list[dict]]:
    """Search for multiple claims and return results keyed by claim.

    Args:
        claims: List of claim strings to verify.
        max_results_per: Max results per claim.

    Returns:
        Dict mapping each claim to its search results.
    """
    results = {}
    for claim in claims:
        results[claim] = search_claim(claim, max_results=max_results_per)
    return results


def format_search_results_for_prompt(claim: str, results: list[dict]) -> str:
    """Format search results for a single claim into a prompt-friendly string."""
    if not results:
        return f'Claim: "{claim}"\nSearch results: No recent articles found.\n'

    lines = [f'Claim: "{claim}"', "Recent search results:"]
    for i, r in enumerate(results, 1):
        lines.append(f"  {i}. [{r['source']}] {r['title']}")
        if r["summary"]:
            lines.append(f"     {r['summary'][:200]}")
        if r["published"]:
            lines.append(f"     Published: {r['published']}")
    lines.append("")
    return "\n".join(lines)
