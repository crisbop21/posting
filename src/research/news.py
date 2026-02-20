"""Fetch trending finance topics from Google News RSS and other free news sources."""

from dataclasses import dataclass
from datetime import datetime

import feedparser


@dataclass
class NewsItem:
    title: str
    source: str
    published: str
    link: str
    summary: str


# Google News RSS feeds for finance topics
NEWS_FEEDS = {
    "stocks": "https://news.google.com/rss/search?q=stock+market+today&hl=en-US&gl=US&ceid=US:en",
    "crypto": "https://news.google.com/rss/search?q=cryptocurrency+bitcoin&hl=en-US&gl=US&ceid=US:en",
    "earnings": "https://news.google.com/rss/search?q=earnings+report+quarterly&hl=en-US&gl=US&ceid=US:en",
    "market trends": "https://news.google.com/rss/search?q=market+trends+finance&hl=en-US&gl=US&ceid=US:en",
    "economic indicators": "https://news.google.com/rss/search?q=economic+indicators+GDP+inflation&hl=en-US&gl=US&ceid=US:en",
}


def fetch_news_topics(topics: list[str], max_per_topic: int = 5) -> list[NewsItem]:
    """Fetch recent news articles for the given finance topics.

    Args:
        topics: List of topic keywords to search for.
        max_per_topic: Maximum number of articles to fetch per topic.

    Returns:
        List of NewsItem objects with article details.
    """
    items: list[NewsItem] = []

    for topic in topics:
        feed_url = NEWS_FEEDS.get(topic)
        if not feed_url:
            # Build a generic Google News RSS query for unknown topics
            query = topic.replace(" ", "+")
            feed_url = f"https://news.google.com/rss/search?q={query}+finance&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_per_topic]:
                items.append(
                    NewsItem(
                        title=entry.get("title", ""),
                        source=entry.get("source", {}).get("title", "Unknown"),
                        published=entry.get("published", ""),
                        link=entry.get("link", ""),
                        summary=entry.get("summary", ""),
                    )
                )
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
            # Strip HTML tags from summary
            clean = item.summary.replace("<b>", "").replace("</b>", "")
            clean = clean.replace("<br>", " ").replace("&amp;", "&")
            lines.append(f"   Summary: {clean[:200]}")
        lines.append(f"   Published: {item.published}")
        lines.append("")

    return "\n".join(lines)
