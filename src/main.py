"""Main orchestrator for the research, generation, review, and slide building pipeline."""

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
    extract_news_facts,
    layered_fact_check,
    validate_conclusion,
    check_narrative_coherence,
    strip_claim_tags,
    add_value_pass,
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx
from src.slides.video_builder import build_video_from_slides


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

    # ── Layer 1: Extract structured facts ──────────────────────────────
    print("\nLayer 1: Extracting structured facts from research...")
    try:
        research_facts = extract_news_facts(research_text)
        print(f"  Extracted {len(research_facts)} verifiable facts.")
        for fact in research_facts[:5]:
            print(f"    [{fact.get('type')}] {fact.get('fact')} ({fact.get('source')})")
        if len(research_facts) > 5:
            print(f"    ... and {len(research_facts) - 5} more")
    except Exception as exc:
        print(f"  Facts extraction failed ({exc}), continuing without grounding")
        research_facts = []

    # ── Step 2: Suggest topics ────────────────────────────────────────
    print("\nStep 2: Suggesting topics...")
    topic_options = suggest_topics(research_text, audience)
    for i, t in enumerate(topic_options, 1):
        print(f"  {i}. {t['title']}: {t['description']}")

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

    # ── Layer 2: Generate fact-grounded slides ─────────────────────────
    print(f"\nLayer 2: Generating {slide_count} fact-grounded slides...")
    slides = generate_slide_content(
        topic=chosen_topic["title"],
        angle=angle,
        hook=chosen_hook["hook"],
        slide_count=slide_count,
        tone=tone,
        audience=audience,
        style_notes=style_notes,
        research_facts=research_facts or None,
    )
    print(f"  Generated {len(slides)} slides.")

    # ── Engagement review ──────────────────────────────────────────────
    print(f"\nReviewing engagement ({review_iterations} iterations)...")
    slides = review_and_improve(
        slides=slides,
        tone=tone,
        audience=audience,
        iterations=review_iterations,
        hook=chosen_hook["hook"],
    )
    print("  Review complete.")

    # ── Layer 3: Layered fact-check ────────────────────────────────────
    if research_facts:
        print("\nLayer 3: Layered fact-check (A: news-sourced, B: supporting data)...")
        max_fc_rounds = 3
        for fc_round in range(1, max_fc_rounds + 1):
            print(f"  Round {fc_round}/{max_fc_rounds}...")
            fc_result = layered_fact_check(
                slides, research_text, research_facts,
                chosen_topic["title"], angle,
            )
            layer_a = fc_result.get("layer_a_report", [])
            layer_b = fc_result.get("layer_b_report", [])
            slides = fc_result.get("corrected_slides", slides)

            print(f"  Layer A (news-sourced): {len(layer_a)} claims checked")
            for item in layer_a:
                print(f"    Slide {item.get('slide', '?')}: [{item.get('status')}] {item.get('notes', '')}")
            print(f"  Layer B (supporting data): {len(layer_b)} claims checked")
            for item in layer_b:
                print(f"    Slide {item.get('slide', '?')}: [{item.get('status')}] {item.get('notes', '')}")

            has_issues = any(
                item.get("status") in ("flagged", "unverifiable")
                for item in layer_a + layer_b
            )
            if not has_issues:
                print("  All claims verified.")
                break
    else:
        print("\nFact-checking all claims (fallback)...")
        fc_result = fact_check_slides(slides, chosen_topic["title"], angle)
        for item in fc_result.get("fact_check_report", []):
            status = item.get("status", "unknown")
            notes = item.get("notes", "")
            print(f"  Slide {item.get('slide', '?')}: [{status}] {notes}")
        slides = fc_result.get("corrected_slides", slides)

    # ── Layer 4: Conclusion validation ─────────────────────────────────
    print("\nLayer 4: Validating conclusion logic...")
    conclusion_result = validate_conclusion(
        slides, research_facts or [], chosen_topic["title"], angle,
    )
    logic_valid = conclusion_result.get("logic_valid", True)
    print(f"  Logic valid: {logic_valid}")
    if not logic_valid:
        for issue in conclusion_result.get("issues", []):
            print(f"    Issue: {issue}")
        print("  Conclusion corrected.")
    slides = conclusion_result.get("corrected_slides", slides)

    # ── Layer 5: Narrative coherence ──────────────────────────────────
    print("\nLayer 5: Checking narrative coherence...")
    coherence_result = check_narrative_coherence(
        slides, chosen_topic["title"], angle, chosen_hook["hook"],
    )
    score = coherence_result.get("coherence_score", "?")
    print(f"  Coherence score: {score}/10")
    arc = coherence_result.get("arc_analysis", "")
    if arc:
        print(f"  Arc: {arc}")
    for issue in coherence_result.get("issues", []):
        print(f"    Fixed: {issue}")
    slides = coherence_result.get("corrected_slides", slides)

    # ── Final combined value + cohesion pass ──────────────────────────
    print("\nFinal pass: value + cohesion polish...")
    value_result = add_value_pass(
        slides=slides,
        topic=chosen_topic["title"],
        angle=angle,
        audience=audience,
        hook=chosen_hook["hook"],
    )
    if isinstance(value_result, list):
        value_result = {"corrected_slides": value_result}
    slides = value_result.get("corrected_slides", slides)
    final_score = value_result.get("coherence_score", score)
    print(f"  Final coherence: {final_score}/10")
    for issue in value_result.get("issues", []):
        print(f"    Fixed: {issue}")

    # Strip claim tags for final output
    slides = strip_claim_tags(slides)

    # ── Generate TikTok metadata ───────────────────────────────────────
    print("\nGenerating TikTok title & description...")
    metadata = generate_tiktok_metadata(
        slides=slides,
        topic=chosen_topic["title"],
        angle=angle,
        hook=chosen_hook["hook"],
    )
    print(f"  Title: {metadata.get('title', '')}")
    print(f"  Description ({len(metadata.get('description', ''))} chars):")
    print(f"  {metadata.get('description', '')}")

    # ── Build PPTX ─────────────────────────────────────────────────────
    print("\nBuilding PPTX...")
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

    # ── Build narrated video (if ElevenLabs key is set) ──────────────────
    video_cfg = config.get("video", {})
    use_ai_images = video_cfg.get("ai_images", False) and os.environ.get("REPLICATE_API_TOKEN")
    if os.environ.get("ELEVENLABS_API_KEY"):
        ai_label = " with AI images" if use_ai_images else ""
        print(f"\nBuilding narrated video{ai_label}...")
        try:
            video_result = build_video_from_slides(
                slides=slides,
                colors=colors,
                topic=chosen_topic["title"],
                angle=angle,
                aspect_ratio=aspect_ratio,
                output_dir=output_dir,
                voice_id=video_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
                use_ai_images=use_ai_images,
            )
            print(f"  Video saved to: {video_result['video_path']}")
            print("\n── Voiceover Scripts ──")
            for i, script in enumerate(video_result["scripts"], 1):
                print(f"  Slide {i}: {script}")
            if video_result.get("image_prompts"):
                print("\n── Image Prompts ──")
                for i, prompt in enumerate(video_result["image_prompts"], 1):
                    print(f"  Slide {i}: {prompt}")
        except Exception as exc:
            print(f"  Video generation failed: {exc}")
            print("  (Slides were exported successfully as PPTX above)")
    else:
        print("\nTip: Set ELEVENLABS_API_KEY to auto-generate narrated video.")
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("Tip: Set REPLICATE_API_TOKEN for AI-generated slide images (Flux).")


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
