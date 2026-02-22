"""Fetch trending finance discussions from Reddit."""

import os
from dataclasses import dataclass
from datetime import datetime

import praw

from src.research.text_utils import sanitize_text


@dataclass
class RedditPost:
    title: str
    subreddit: str
    score: int
    num_comments: int
    url: str
    selftext: str


def fetch_reddit_topics(
    subreddits: list[str], max_per_sub: int = 5
) -> list[RedditPost]:
    """Fetch top/hot posts from finance-related subreddits.

    Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT
    environment variables to be set.

    Args:
        subreddits: List of subreddit names to scan.
        max_per_sub: Maximum number of posts to fetch per subreddit.

    Returns:
        List of RedditPost objects.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "posting-bot/0.1")

    if not client_id or not client_secret:
        print("[reddit] REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET not set, skipping Reddit.")
        return []

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    posts: list[RedditPost] = []

    for sub_name in subreddits:
        try:
            subreddit = reddit.subreddit(sub_name)
            for post in subreddit.hot(limit=max_per_sub):
                if post.stickied:
                    continue
                posts.append(
                    RedditPost(
                        title=sanitize_text(post.title),
                        subreddit=sub_name,
                        score=post.score,
                        num_comments=post.num_comments,
                        url=post.url,
                        selftext=sanitize_text(post.selftext[:500]) if post.selftext else "",
                    )
                )
        except Exception as e:
            print(f"[reddit] Failed to fetch r/{sub_name}: {e}")

    # Sort by engagement (score + comments)
    posts.sort(key=lambda p: p.score + p.num_comments, reverse=True)
    return posts


def format_reddit_for_prompt(posts: list[RedditPost]) -> str:
    """Format Reddit posts into a string suitable for an LLM prompt."""
    if not posts:
        return "No Reddit posts found."

    lines = [
        f"=== Trending Reddit Finance Discussions ({datetime.now().strftime('%Y-%m-%d')}) ===\n"
    ]
    for i, post in enumerate(posts, 1):
        lines.append(f"{i}. [r/{post.subreddit}] {post.title}")
        lines.append(f"   Score: {post.score} | Comments: {post.num_comments}")
        if post.selftext:
            lines.append(f"   Preview: {post.selftext[:150]}...")
        lines.append("")

    return "\n".join(lines)
