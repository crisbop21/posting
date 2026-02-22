"""Fetch trending finance topics from Google News RSS and other free news sources."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import feedparser

from src.research.text_utils import sanitize_text


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    link: str
    summary: str


# Base search queries per topic (without date filter — added dynamically)
_TOPIC_QUERIES = {
    "stocks": "stock+market+today",
    "crypto": "cryptocurrency+bitcoin",
    "earnings": "earnings+report+quarterly",
    "market trends": "market+trends+finance",
    "economic indicators": "economic+indicators+GDP+inflation",
}

_RSS_BASE = "https://news.google.com/rss/search"
_RSS_PARAMS = "hl=en-US&gl=US&ceid=US:en"


def _build_feed_url(query: str) -> str:
    """Build a Google News RSS URL with an after: date filter for the last 2 days."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
    return f"{_RSS_BASE}?q={query}+after:{cutoff}&{_RSS_PARAMS}"


def _is_within_48_hours(published_str: str) -> bool:
    """Check if a published date string is within the last 48 hours."""
    if not published_str:
        return False
    try:
        pub_dt = parsedate_to_datetime(published_str)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = now - pub_dt
        return age.total_seconds() <= 48 * 3600
    except Exception:
        return False


def fetch_news_topics(
    topics: list[str],
    max_per_topic: int = 5,
    recent_hours: int = 48,
) -> list[NewsItem]:
    """Fetch recent news articles for the given finance topics.

    Args:
        topics: List of topic keywords to search for.
        max_per_topic: Maximum number of articles to fetch per topic.
        recent_hours: Only include articles published within this many hours.

    Returns:
        List of NewsItem objects with article details (filtered to last 48h).
    """
    items: list[NewsItem] = []

    for topic in topics:
        query = _TOPIC_QUERIES.get(topic)
        if not query:
            query = topic.replace(" ", "+") + "+finance"
        feed_url = _build_feed_url(query)

        try:
            feed = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_topic:
                    break
                published = entry.get("published", "")
                if not _is_within_48_hours(published):
                    continue
                items.append(
                    NewsItem(
                        title=sanitize_text(entry.get("title", "")),
                        source=sanitize_text(entry.get("source", {}).get("title", "Unknown")),
                        published=published,
                        link=entry.get("link", ""),
                        summary=sanitize_text(entry.get("summary", "")),
                    )
                )
                count += 1
        except Exception as e:
            print(f"[news] Failed to fetch feed for '{topic}': {e}")

    return items


def format_news_for_prompt(items: list[NewsItem]) -> str:
    """Format news items into a string suitable for an LLM prompt."""
    if not items:
        return "No news articles found."

    lines = [f"=== Trending Finance News ({datetime.now().strftime('%Y-%m-%d')}) ===\n"]
    for i, item in enumerate(items, 1):
        lines.append(f"{i}. [{item.source}] {item.title}")
        if item.summary:
            clean = item.summary.replace("<b>", "").replace("</b>", "")
            clean = clean.replace("<br>", " ").replace("&amp;", "&")
            lines.append(f"   Summary: {clean[:200]}")
        lines.append(f"   Published: {item.published}")
        lines.append("")

    return "\n".join(lines)
