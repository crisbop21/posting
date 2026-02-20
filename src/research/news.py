"""Fetch trending finance topics from Google News RSS and other free news sources."""

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    link: str
    summary: str


# Google News RSS feeds for finance topics
# "when:2d" restricts results to the last 2 days at the source level
NEWS_FEEDS = {
    "stocks": "https://news.google.com/rss/search?q=stock+market+today+when:2d&hl=en-US&gl=US&ceid=US:en",
    "crypto": "https://news.google.com/rss/search?q=cryptocurrency+bitcoin+when:2d&hl=en-US&gl=US&ceid=US:en",
    "earnings": "https://news.google.com/rss/search?q=earnings+report+quarterly+when:2d&hl=en-US&gl=US&ceid=US:en",
    "market trends": "https://news.google.com/rss/search?q=market+trends+finance+when:2d&hl=en-US&gl=US&ceid=US:en",
    "economic indicators": "https://news.google.com/rss/search?q=economic+indicators+GDP+inflation+when:2d&hl=en-US&gl=US&ceid=US:en",
}


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
        # If we can't parse the date, exclude the article
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
        feed_url = NEWS_FEEDS.get(topic)
        if not feed_url:
            query = topic.replace(" ", "+")
            feed_url = f"https://news.google.com/rss/search?q={query}+finance+when:2d&hl=en-US&gl=US&ceid=US:en"

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
                        title=entry.get("title", ""),
                        source=entry.get("source", {}).get("title", "Unknown"),
                        published=published,
                        link=entry.get("link", ""),
                        summary=entry.get("summary", ""),
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
