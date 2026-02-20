"""Main orchestrator: research → generate → review → build slides."""

import argparse
import os
import sys

import yaml

from src.research.news import fetch_news_topics, format_news_for_prompt
from src.research.reddit import fetch_reddit_topics, format_reddit_for_prompt
from src.content.generator import (
    suggest_topics,
    generate_hooks,
    generate_slide_content,
    fact_check_slides,
    generate_tiktok_metadata,
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)

    slides_cfg = config.get("slides", {})
    research_cfg = config.get("research", {})
    content_cfg = config.get("content", {})
    output_cfg = config.get("output", {})

    slide_count = slides_cfg.get("count", 5)
    tone = slides_cfg.get("tone", "bold")
    audience = slides_cfg.get("audience", "retail investors")
    colors = slides_cfg.get("colors", {})
    aspect_ratio = slides_cfg.get("aspect_ratio", "9:16")

    sources = research_cfg.get("sources", ["news"])
    topics = research_cfg.get("topics", ["stocks"])
    subreddits = research_cfg.get("subreddits", ["stocks"])

    review_iterations = content_cfg.get("review_iterations", 2)
    style_notes = content_cfg.get("style_notes", "")

    output_dir = output_cfg.get("directory", "./output")

    # ── Step 1: Research ──────────────────────────────────────────────
    print("Step 1: Researching trending finance topics...")
    research_parts = []

    if "news" in sources:
        print("  Fetching news feeds...")
        news_items = fetch_news_topics(topics)
        print(f"  Found {len(news_items)} news articles.")
        research_parts.append(format_news_for_prompt(news_items))

    if "reddit" in sources:
        print("  Fetching Reddit discussions...")
        reddit_posts = fetch_reddit_topics(subreddits)
        print(f"  Found {len(reddit_posts)} Reddit posts.")
        research_parts.append(format_reddit_for_prompt(reddit_posts))

    research_text = "\n\n".join(research_parts)

    empty_markers = {"No news articles found.", "No Reddit posts found."}
    if not research_parts or all(p.strip() in empty_markers for p in research_parts):
        print("No research data found. Check your network connection and config.")
        sys.exit(1)

    # ── Step 2: Suggest topics ────────────────────────────────────────
    print("\nStep 2: Suggesting topics...")
    topic_options = suggest_topics(research_text, audience)
    for i, t in enumerate(topic_options, 1):
        print(f"  {i}. {t['title']} — {t['description']}")

    # Auto-pick the first topic in CLI mode
    chosen_topic = topic_options[0]
    print(f"\n  Auto-selected: {chosen_topic['title']}")

    # Use a generic angle for CLI mode
    angle = "Provide actionable insights with real data points"

    # ── Step 3: Generate hooks ─────────────────────────────────────────
    print("\nStep 3: Generating hooks...")
    hook_options = generate_hooks(
        topic=chosen_topic["title"],
        angle=angle,
        tone=tone,
        audience=audience,
    )
    for i, h in enumerate(hook_options, 1):
        print(f"  {i}. [{h['style']}] {h['hook']}")

    # Auto-pick the first hook
    chosen_hook = hook_options[0]
    print(f"\n  Auto-selected: {chosen_hook['hook']}")

    # ── Step 4: Generate slide content ─────────────────────────────────
    print(f"\nStep 4: Generating {slide_count} slides with Claude...")
    slides = generate_slide_content(
        topic=chosen_topic["title"],
        angle=angle,
        hook=chosen_hook["hook"],
        slide_count=slide_count,
        tone=tone,
        audience=audience,
        style_notes=style_notes,
    )
    print(f"  Generated {len(slides)} slides.")

    # ── Step 5: Review and improve engagement ─────────────────────────
    print(f"\nStep 5: Reviewing engagement ({review_iterations} iterations)...")
    slides = review_and_improve(
        slides=slides,
        tone=tone,
        audience=audience,
        iterations=review_iterations,
    )
    print("  Review complete.")

    # ── Step 6: Fact-check ─────────────────────────────────────────────
    print("\nStep 6: Fact-checking all claims...")
    fc_result = fact_check_slides(slides, chosen_topic["title"], angle)
    for item in fc_result.get("fact_check_report", []):
        status = item.get("status", "unknown")
        notes = item.get("notes", "")
        print(f"  Slide {item.get('slide', '?')}: [{status}] {notes}")
    slides = fc_result.get("corrected_slides", slides)

    # ── Step 7: Generate TikTok metadata ───────────────────────────────
    print("\nStep 7: Generating TikTok title & description...")
    metadata = generate_tiktok_metadata(
        slides=slides,
        topic=chosen_topic["title"],
        angle=angle,
        hook=chosen_hook["hook"],
    )
    print(f"  Title: {metadata.get('title', '')}")
    print(f"  Description ({len(metadata.get('description', ''))} chars):")
    print(f"  {metadata.get('description', '')}")

    # ── Step 8: Build PPTX ─────────────────────────────────────────────
    print("\nStep 8: Building PPTX...")
    filepath = build_pptx(
        slides=slides,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
    )
    print(f"  Saved to: {filepath}")

    # Print slide preview
    print("\n── Slide Preview ──")
    for i, slide in enumerate(slides, 1):
        print(f"\n  Slide {i}:")
        print(f"    Title:  {slide.get('title', '')}")
        print(f"    Body:   {slide.get('body', '')}")
        print(f"    Footer: {slide.get('footer', '')}")

    print(f"\nDone! Open {filepath} to review your slides.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily finance slides for social media"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()
    run(config_path=args.config)


if __name__ == "__main__":
    main()
