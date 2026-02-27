"""Gradio UI for the finance slide generator. Guided 6-step workflow.

Equivalent to streamlit_app.py but using Gradio Blocks.

Flow:
  1. Research & Pick a Topic (card-based selection)
  2. Consolidate Data (20+ verified bullet points)
  3. Provide Angle & Additional Data
  4. Choose a Hook (card-based selection)
  5. Generate & Verify Slides (verification dashboard)
  6. Studio: Edit + Export (tabbed — Edit | Slides | AI Images | Video)

Launch:
  python gradio_app.py          # or:  gradio gradio_app.py
"""

import html as html_mod
import io
import json
import os
import time
import traceback
import zipfile

import anthropic
import gradio as gr
import yaml

from src.research.news import fetch_news_topics, format_news_for_prompt
from src.research.reddit import fetch_reddit_topics, format_reddit_for_prompt
from src.research.web_search import search_claim
from src.content.generator import (
    suggest_topics,
    generate_hooks,
    generate_slide_content,
    add_value_pass,
    fact_check_news,
    generate_tiktok_metadata,
    extract_news_facts,
    consolidate_topic_data,
    research_angle,
    layered_fact_check,
    validate_conclusion,
    check_narrative_coherence,
    strip_claim_tags,
    web_search_fact_check,
    enforce_hook_and_count,
    generate_image_prompts,
    generate_video_script,
    generate_image_search_queries,
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx
from src.slides.png_builder import build_pngs, build_style_alternatives
from src.slides.video_builder import (
    build_video,
    build_video_with_ai_images,
    build_video_with_searched_images,
)
from src.slides.image_generator import generate_slide_images


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# ── Demo data ─────────────────────────────────────────────────────────────────

_DEMO_TOPICS = [
    {
        "title": "Tesla Stock Surges 15% After Record Q4 Deliveries",
        "description": "Tesla reported 495,570 vehicle deliveries in Q4 2024, beating Wall Street estimates of 483,000. The stock rallied 15% in two days. Analysts point to strong Model Y demand in China and Europe as key drivers.",
    },
    {
        "title": "Fed Holds Rates Steady — What It Means for Your Portfolio",
        "description": "The Federal Reserve kept the federal funds rate at 4.25-4.50% for the second consecutive meeting. Chair Powell signaled patience on cuts, citing sticky inflation at 2.7%. Bond yields fell while growth stocks rallied.",
    },
    {
        "title": "Bitcoin Breaks $100K — Is This Time Different?",
        "description": "Bitcoin surged past $100,000 for the first time, driven by spot ETF inflows exceeding $2B/week. BlackRock's IBIT is now the fastest-growing ETF in history.",
    },
    {
        "title": "NVIDIA Earnings Crush Estimates — AI Spending Boom Continues",
        "description": "NVIDIA reported Q4 revenue of $22.1B, up 265% YoY, with data center revenue at $18.4B. CEO Jensen Huang announced Blackwell GPU demand is 'insane.'",
    },
    {
        "title": "Oil Drops Below $70 — Winners and Losers",
        "description": "Crude oil fell to $66/barrel on weakening Chinese demand and rising US production at 13.3M barrels/day. Airline stocks surged while energy faced pressure.",
    },
]

_DEMO_BULLETS = [
    {"bullet": "Tesla delivered 495,570 vehicles in Q4 2024, beating consensus of 483,000", "value": "495,570 deliveries", "source": "Tesla IR", "confidence": "high"},
    {"bullet": "TSLA stock rallied 15% in two trading sessions following the report", "value": "15% rally", "source": "Yahoo Finance", "confidence": "high"},
    {"bullet": "Model Y was the best-selling car globally in 2024 with 1.2M units", "value": "1.2M Model Y", "source": "Reuters", "confidence": "high"},
    {"bullet": "Tesla's China sales grew 8.8% YoY in Q4, with 157,000 units delivered", "value": "157K China deliveries", "source": "CPCA data", "confidence": "high"},
    {"bullet": "Gross margins improved to 18.2%, up from 17.6% in Q3", "value": "18.2% gross margin", "source": "Tesla 10-Q", "confidence": "high"},
    {"bullet": "Cybertruck production exceeded 4,000 units/week by end of Q4", "value": "4K/week Cybertruck", "source": "Tesla earnings call", "confidence": "high"},
    {"bullet": "Tesla Energy revenue hit $1.4B in Q4, up 75% YoY", "value": "$1.4B energy revenue", "source": "Tesla IR", "confidence": "high"},
    {"bullet": "Free cash flow was $2.1B in Q4, above analyst expectations of $1.5B", "value": "$2.1B FCF", "source": "Tesla 10-Q", "confidence": "high"},
    {"bullet": "Elon Musk reaffirmed target of 20M vehicles/year by 2030", "value": "20M by 2030", "source": "Earnings call", "confidence": "medium"},
    {"bullet": "Average selling price declined 5% YoY due to price cuts in China and Europe", "value": "5% ASP decline", "source": "Bloomberg", "confidence": "high"},
    {"bullet": "Tesla Full Self-Driving v12.5 showed 60% fewer interventions in testing", "value": "60% fewer interventions", "source": "Tesla AI blog", "confidence": "medium"},
    {"bullet": "Short interest on TSLA dropped to 2.8%, lowest since 2020", "value": "2.8% short interest", "source": "S3 Partners", "confidence": "high"},
    {"bullet": "Institutional ownership increased to 44% from 41% in Q3", "value": "44% institutional", "source": "SEC 13-F filings", "confidence": "high"},
    {"bullet": "Tesla market cap reached $850B, 7th most valuable company globally", "value": "$850B market cap", "source": "Yahoo Finance", "confidence": "high"},
    {"bullet": "Q4 automotive revenue was $21.6B, up 8% YoY", "value": "$21.6B auto revenue", "source": "Tesla IR", "confidence": "high"},
    {"bullet": "Operating expenses grew only 3% YoY despite revenue growing 8%", "value": "3% OpEx growth", "source": "Tesla 10-Q", "confidence": "high"},
    {"bullet": "Tesla deployed 14.7 GWh of energy storage in 2024, up 125% from 2023", "value": "14.7 GWh storage", "source": "Tesla IR", "confidence": "high"},
    {"bullet": "Supercharger network grew to 60,000+ stalls globally", "value": "60K+ superchargers", "source": "Tesla website", "confidence": "high"},
    {"bullet": "Analyst consensus price target is $285, high of $380 (Morgan Stanley)", "value": "$285 consensus PT", "source": "TipRanks", "confidence": "medium"},
    {"bullet": "Tesla announced next-gen affordable vehicle starting under $25K in late 2025", "value": "Sub-$25K vehicle", "source": "Earnings call", "confidence": "medium"},
    {"bullet": "Options market implied 12% move ahead of earnings, actual was 15%", "value": "15% vs 12% implied", "source": "CBOE", "confidence": "high"},
    {"bullet": "Tesla Semi deliveries to PepsiCo exceeded 100 units with 500-mile range", "value": "100+ Semi units", "source": "PepsiCo report", "confidence": "medium"},
]

_DEMO_HOOKS = [
    {"hook": "Tesla just did something it hasn't done in 3 years — and Wall Street is losing it", "style": "curiosity gap", "data_used": "Record Q4 deliveries beating estimates, 15% stock rally", "fit_score": 9},
    {"hook": "495,570 vehicles. That's not a typo.", "style": "stat shock", "data_used": "Q4 delivery numbers, Wall Street consensus miss", "fit_score": 8},
    {"hook": "Everyone said Tesla was done. The numbers say otherwise.", "style": "contrarian", "data_used": "Short interest at 2.8% low, delivery beat, margin improvement", "fit_score": 8},
    {"hook": "I analyzed Tesla's Q4 numbers so you don't have to. Here's what matters.", "style": "authority / value promise", "data_used": "Full Q4 financial data, delivery numbers, FSD progress", "fit_score": 7},
    {"hook": "If you own TSLA, you need to see this before Monday", "style": "urgency", "data_used": "Price movement, analyst upgrades, options implied volatility", "fit_score": 7},
]

_DEMO_SLIDES = [
    {"title": "Tesla just did something it hasn't done in 3 years", "body": "Record Q4 deliveries. Record energy revenue. And the stock is finally responding. Here's what retail investors need to know.", "footer": ""},
    {"title": "495,570 Vehicles Delivered", "body": "Tesla crushed Q4 estimates by 12,570 units. Wall Street expected 483K. Model Y alone sold 1.2M globally in 2024 — the best-selling car on earth.", "footer": "source: Tesla IR, Reuters"},
    {"title": "China Is Back", "body": "157,000 units delivered in China — up 8.8% YoY. Price cuts worked. Market share is climbing while BYD's growth slows in the premium segment.", "footer": "source: CPCA data"},
    {"title": "Margins Are Healing", "body": "Gross margins hit 18.2%, up from 17.6% last quarter. The price war is ending. Tesla is proving it can grow volume AND protect margins.", "footer": "source: Tesla 10-Q"},
    {"title": "The Energy Business Nobody Talks About", "body": "Tesla Energy did $1.4B in Q4 revenue — up 75% YoY. They deployed 14.7 GWh of storage in 2024. This segment alone could be worth $100B.", "footer": "source: Tesla IR"},
    {"title": "Cash Machine", "body": "Free cash flow hit $2.1B — analysts expected $1.5B. Operating costs only grew 3% while revenue jumped 8%. The efficiency gains are real.", "footer": "source: Tesla 10-Q"},
    {"title": "What's Next: The $25K Car", "body": "Tesla confirmed a sub-$25K vehicle for late 2025. FSD v12.5 has 60% fewer interventions. And Cybertruck just hit 4,000 units/week.", "footer": "source: Tesla earnings call"},
    {"title": "The Verdict", "body": "Shorts are at a 4-year low (2.8%). Institutions are buying (44% ownership). Analyst target: $285. Tesla is executing — and the market is noticing.", "footer": "source: S3 Partners, TipRanks"},
]

_DEMO_FACT_REPORT = [
    {"slide": 2, "status": "verified", "notes": "[news-sourced] Q4 delivery figure matches Tesla IR press release"},
    {"slide": 3, "status": "verified", "notes": "[news-sourced] China figures match CPCA monthly data"},
    {"slide": 4, "status": "verified", "notes": "[supporting data] Margin figures match 10-Q filing"},
    {"slide": 5, "status": "verified", "notes": "[supporting data] Energy revenue confirmed in earnings report"},
    {"slide": 6, "status": "verified", "notes": "[web search] FCF and OpEx figures cross-checked with SEC filing"},
    {"slide": 7, "status": "verified", "notes": "[news-sourced] Sub-$25K vehicle confirmed in earnings call transcript"},
    {"slide": 8, "status": "verified", "notes": "[web search] Short interest data matches S3 Partners latest report"},
]

_DEMO_METADATA = {
    "title": "Tesla's Q4 Was INSANE — Here's What You Missed",
    "description": (
        "Tesla just reported record Q4 deliveries of 495,570 vehicles, crushing Wall Street estimates by 12,570 units. "
        "The stock surged 15% in two days.\n\nBut the real story isn't just cars — it's the $1.4B energy business growing "
        "75% YoY, margins healing to 18.2%, and $2.1B in free cash flow.\n\nPlus: the sub-$25K vehicle is coming in late "
        "2025, FSD is improving fast, and short sellers are at a 4-year low.\n\nFollow for more finance breakdowns.\n\n"
        "#Tesla #TSLA #Stocks #Investing #Finance #StockMarket #RetailInvestor #WallStreet #EV"
    ),
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_env_key(name: str) -> str:
    return os.environ.get(name, "")


def _set_api_key(key: str):
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key


def _safe_get_slides(result, fallback: list[dict]) -> list[dict]:
    if isinstance(result, list):
        return result if all(isinstance(s, dict) for s in result) else fallback
    if isinstance(result, dict):
        slides = result.get("corrected_slides", fallback)
        if isinstance(slides, list) and all(isinstance(s, dict) for s in slides):
            return slides
        return fallback
    return fallback


def _slide_preview_html(slide: dict, idx: int, total: int, colors: dict, handle_text: str) -> str:
    bg = colors.get("background", "#0D0D15")
    tc = colors.get("title", "#FFFFFF")
    bc = colors.get("body", "#C0C0D0")
    ac = colors.get("accent", "#F7B731")

    title = html_mod.escape(slide.get("title", "") or "")
    body = html_mod.escape(slide.get("body", "") or "").replace("\n", "<br>")
    footer = html_mod.escape(slide.get("footer", "") or "")
    h = html_mod.escape(handle_text)
    footer_line = f"{footer} &middot; {h}" if footer else h

    return f"""
    <div style="
        background: {bg};
        border-left: 4px solid {ac};
        border-radius: 12px;
        padding: 24px 20px 48px 20px;
        min-height: 260px;
        position: relative;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        overflow: hidden;
        margin-bottom: 8px;
    ">
        <div style="position:absolute;top:14px;right:18px;color:{ac};font-size:13px;opacity:0.7;">
            {idx + 1}/{total}
        </div>
        <div style="color:{tc};font-size:18px;font-weight:700;margin-top:14px;line-height:1.35;">
            {title}
        </div>
        <div style="border-top:2px solid {ac};width:25%;margin:14px 0;opacity:0.4;"></div>
        <div style="color:{bc};font-size:13px;line-height:1.6;">
            {body}
        </div>
        <div style="position:absolute;bottom:14px;left:0;right:0;text-align:center;
                    color:#4A5568;font-size:11px;">
            {footer_line}
        </div>
    </div>
    """


def _build_zip(paths: list[str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))
    return buf.getvalue()


def _topic_cards_html(topics: list[dict]) -> str:
    """Render topic options as HTML cards with index."""
    cards = []
    for i, t in enumerate(topics):
        cards.append(
            f'<div style="border:1px solid #333;border-radius:8px;padding:14px 16px;margin-bottom:8px;">'
            f'<strong>{i + 1}. {html_mod.escape(t["title"])}</strong><br>'
            f'<span style="color:#999;font-size:13px;">{html_mod.escape(t["description"])}</span>'
            f'</div>'
        )
    return "".join(cards)


def _hook_cards_html(hooks: list[dict]) -> str:
    """Render hook options as HTML cards with fit scores."""
    cards = []
    for i, h in enumerate(hooks):
        fit = h.get("fit_score", 0)
        fit_color = "#22c55e" if fit >= 8 else "#f59e0b" if fit >= 5 else "#ef4444"
        meta = f"Style: {h['style']}"
        if h.get("data_used"):
            meta += f" | Data: {h['data_used']}"
        cards.append(
            f'<div style="border:1px solid #333;border-radius:8px;padding:14px 16px;margin-bottom:8px;'
            f'display:flex;align-items:center;gap:14px;">'
            f'<div style="text-align:center;min-width:48px;">'
            f'<span style="font-size:24px;font-weight:700;color:{fit_color};">{fit}</span>'
            f'<br><span style="font-size:11px;color:#888;">/10</span></div>'
            f'<div>'
            f'<strong>{i + 1}. {html_mod.escape(h["hook"])}</strong><br>'
            f'<span style="color:#999;font-size:12px;">{html_mod.escape(meta)}</span>'
            f'</div></div>'
        )
    return "".join(cards)


def _bullets_markdown(bullets: list[dict]) -> str:
    """Format bullets as markdown grouped by confidence."""
    high = [b for b in bullets if b.get("confidence") == "high"]
    other = [b for b in bullets if b.get("confidence") != "high"]
    lines = []
    if high:
        lines.append(f"### High confidence ({len(high)})")
        for b in high:
            lines.append(f"- **{b['bullet']}** — *source: {b.get('source', 'unknown')}*")
    if other:
        lines.append(f"\n### Medium confidence ({len(other)})")
        for b in other:
            lines.append(f"- {b['bullet']} — *source: {b.get('source', 'unknown')}*")
    return "\n".join(lines)


def _fact_report_markdown(report: list[dict]) -> str:
    if not report:
        return "*No fact-check report available.*"
    lines = ["### Fact-Check Report"]
    for item in report:
        slide_num = item.get("slide", "?")
        status = item.get("status", "unknown")
        notes = item.get("notes", "")
        icon = {"verified": "✅", "corrected": "🟠", "flagged": "🔴"}.get(status, "❓")
        lines.append(f"- {icon} **Slide {slide_num}** ({status}): {notes}")
    return "\n".join(lines)


def _coherence_report_markdown(report: dict | None) -> str:
    if not report:
        return ""
    score = report.get("coherence_score", "?")
    arc = report.get("arc_analysis", "")
    lines = [f"**Coherence Score:** {score}/10"]
    if arc:
        lines.append(f"**Arc:** {arc}")
    return "\n".join(lines)


# ── Application State ─────────────────────────────────────────────────────────
# Gradio doesn't have built-in session state like Streamlit. We use gr.State
# to hold a single dict that carries all workflow data between callbacks.


def _default_state() -> dict:
    slides_cfg = CONFIG.get("slides", {})
    research_cfg = CONFIG.get("research", {})
    content_cfg = CONFIG.get("content", {})
    video_cfg = CONFIG.get("video", {})
    colors_cfg = slides_cfg.get("colors", {})

    return {
        "step": 1,
        # Research
        "research_text": "",
        "research_facts": [],
        "topic_options": [],
        "selected_topic": None,
        # Data
        "verified_bullets": [],
        # Angle
        "angle": "",
        "angle_verified": False,
        "user_facts": "",
        # Hook
        "hook_options": [],
        "selected_hook": None,
        # Slides
        "slides": [],
        "fact_check_report": [],
        "conclusion_report": None,
        "coherence_report": None,
        "tiktok_metadata": None,
        # Export
        "pptx_path": None,
        "png_paths": [],
        "mcp_alternatives": [],
        "video_path": None,
        "video_scripts": [],
        "video_search_queries": [],
        "video_search_results": {},
        "ai_image_paths": [],
        "ai_image_prompts": [],
        # Settings (from config, overridden by sidebar)
        "slide_count": min(max(slides_cfg.get("count", 7), 3), 15),
        "tone": slides_cfg.get("tone", "bold"),
        "audience": slides_cfg.get("audience", "retail investors"),
        "aspect_ratio": slides_cfg.get("aspect_ratio", "9:16"),
        "handle": slides_cfg.get("handle", "@cristian.bojaca"),
        "bg_color": colors_cfg.get("background", "#0D0D15"),
        "title_color": colors_cfg.get("title", "#FFFFFF"),
        "body_color": colors_cfg.get("body", "#C0C0D0"),
        "accent_color": colors_cfg.get("accent", "#F7B731"),
        "highlight_color": colors_cfg.get("highlight", "#FF5757"),
        "sources": research_cfg.get("sources", ["news"]),
        "topics": research_cfg.get("topics", ["stocks"]),
        "subreddits": research_cfg.get("subreddits", ["stocks"]),
        "review_iterations": content_cfg.get("review_iterations", 2),
        "style_notes": content_cfg.get("style_notes", ""),
        "voice_id": video_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
    }


def _colors_from_state(state: dict) -> dict:
    return {
        "background": state.get("bg_color", "#0D0D15"),
        "title": state.get("title_color", "#FFFFFF"),
        "body": state.get("body_color", "#C0C0D0"),
        "accent": state.get("accent_color", "#F7B731"),
        "highlight": state.get("highlight_color", "#FF5757"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP CALLBACKS
# ══════════════════════════════════════════════════════════════════════════════


def step1_research(api_key, research_mode, custom_topic, sources, topics, subreddits_str, audience, state, progress=gr.Progress()):
    """Research topics and return suggestions."""
    demo_mode = not api_key
    subreddits = [s.strip() for s in subreddits_str.split(",") if s.strip()]

    if demo_mode:
        progress(0.3, desc="Loading demo data...")
        time.sleep(0.5)
        state["research_text"] = "Demo research data for Tesla Q4 deliveries."
        state["research_facts"] = [{"fact": b["bullet"], "source": b["source"]} for b in _DEMO_BULLETS[:10]]
        state["topic_options"] = list(_DEMO_TOPICS)
        progress(1.0, desc="Done")
        return (
            state,
            _topic_cards_html(state["topic_options"]),
            gr.update(visible=True),    # topic selection area
            gr.update(visible=True),    # topic dropdown
            "Demo mode: loaded 5 sample topics",
        )

    _set_api_key(api_key)
    research_parts = []

    if research_mode == "Custom Topic":
        if not custom_topic.strip():
            return state, "", gr.update(visible=False), gr.update(visible=False), "Please enter a topic."

        progress(0.1, desc=f"Researching '{custom_topic}'...")
        custom_news = fetch_news_topics([custom_topic], max_per_topic=10)
        if custom_news:
            research_parts.append(format_news_for_prompt(custom_news))

        web_results = search_claim(custom_topic + " finance", max_results=10)
        if web_results:
            lines = [f"=== Web Search: {custom_topic} ===\n"]
            for idx, r in enumerate(web_results, 1):
                lines.append(f"{idx}. [{r['source']}] {r['title']}")
                if r["summary"]:
                    lines.append(f"   {r['summary'][:200]}")
                lines.append(f"   Published: {r['published']}")
                lines.append("")
            research_parts.append("\n".join(lines))

        if not research_parts:
            return state, "", gr.update(visible=False), gr.update(visible=False), f"No results found for '{custom_topic}'."
    else:
        progress(0.1, desc="Fetching latest trends...")
        src_list = [s.strip() for s in sources] if isinstance(sources, list) else [sources]
        topic_list = [t.strip() for t in topics] if isinstance(topics, list) else [topics]

        if "news" in src_list:
            news_items = fetch_news_topics(topic_list)
            if news_items:
                raw_news = format_news_for_prompt(news_items)
                try:
                    verdicts = fact_check_news(raw_news)
                    corrections = {v["index"]: v for v in verdicts if v.get("status") == "corrected"}
                    for v_idx, verdict in corrections.items():
                        if 1 <= v_idx <= len(news_items):
                            item = news_items[v_idx - 1]
                            item.title = verdict.get("corrected_title", item.title)
                            item.summary = verdict.get("corrected_summary", item.summary)
                except Exception:
                    pass
                research_parts.append(format_news_for_prompt(news_items))

        if "reddit" in src_list:
            reddit_posts = fetch_reddit_topics(subreddits)
            research_parts.append(format_reddit_for_prompt(reddit_posts))

    research_text = "\n\n".join(research_parts)
    empty_markers = {"No news articles found.", "No Reddit posts found."}
    if not research_parts or all(p.strip() in empty_markers for p in research_parts):
        return state, "", gr.update(visible=False), gr.update(visible=False), "No research data found. Check your config."

    state["research_text"] = research_text

    progress(0.5, desc="Extracting facts...")
    try:
        research_facts = extract_news_facts(research_text)
        state["research_facts"] = research_facts
    except Exception:
        state["research_facts"] = []

    progress(0.7, desc="Generating topic suggestions...")
    try:
        topic_options = suggest_topics(research_text, audience)
        state["topic_options"] = topic_options
    except anthropic.AuthenticationError:
        return state, "", gr.update(visible=False), gr.update(visible=False), "Invalid API key."
    except anthropic.APIError as exc:
        return state, "", gr.update(visible=False), gr.update(visible=False), f"API error: {exc}"

    progress(1.0, desc="Done")
    return (
        state,
        _topic_cards_html(state["topic_options"]),
        gr.update(visible=True),
        gr.update(visible=True),
        f"Found {len(state['topic_options'])} topic suggestions.",
    )


def step1_select_topic(topic_index, state):
    """User selects a topic by number and moves to step 2."""
    opts = state.get("topic_options", [])
    if not opts:
        return state, "No topics available. Research first."
    idx = max(0, min(int(topic_index) - 1, len(opts) - 1))
    state["selected_topic"] = opts[idx]
    state["step"] = 2
    return state, f"Selected: **{opts[idx]['title']}**"


def step2_consolidate(api_key, audience, state, progress=gr.Progress()):
    """Consolidate 20+ verified bullet points."""
    topic = state.get("selected_topic")
    if not topic:
        return state, "No topic selected.", ""

    demo_mode = not api_key

    if demo_mode:
        progress(0.5, desc="Loading demo data...")
        time.sleep(0.3)
        state["verified_bullets"] = list(_DEMO_BULLETS)
        return state, f"Found {len(_DEMO_BULLETS)} verified data points.", _bullets_markdown(_DEMO_BULLETS)

    _set_api_key(api_key)
    progress(0.2, desc="Consolidating and verifying data...")
    try:
        bullets = consolidate_topic_data(
            research_text=state["research_text"],
            research_facts=state["research_facts"],
            topic=topic,
            audience=audience,
        )
        state["verified_bullets"] = bullets
        progress(1.0, desc="Done")
        return state, f"Found {len(bullets)} verified data points.", _bullets_markdown(bullets)
    except anthropic.APIError as exc:
        return state, f"API error: {exc}", ""


def step3_research_angle(api_key, angle_text, audience, state, progress=gr.Progress()):
    """Research the user's angle and verify additional data."""
    topic = state["selected_topic"]
    bullets = state["verified_bullets"]
    demo_mode = not api_key

    if not angle_text.strip():
        return state, "Please enter an angle.", _bullets_markdown(bullets)

    state["angle"] = angle_text.strip()

    if demo_mode:
        state["angle_verified"] = True
        return state, "Demo mode: angle accepted.", _bullets_markdown(bullets)

    _set_api_key(api_key)
    progress(0.2, desc="Searching for data on your angle...")
    try:
        new_bullets = research_angle(topic=topic, angle=angle_text.strip(), existing_bullets=bullets)
        combined = bullets + (new_bullets or [])
    except Exception:
        combined = bullets

    if len(combined) < 20:
        progress(0.6, desc=f"Only {len(combined)} points, researching more...")
        try:
            extra_bullets = consolidate_topic_data(
                research_text=state["research_text"],
                research_facts=[{"fact": b["bullet"], "source": b.get("source", "")} for b in combined],
                topic={"title": topic["title"], "description": f"{topic['description']}, angle: {angle_text.strip()}"},
                audience=audience,
            )
            existing_texts = {b["bullet"].lower() for b in combined}
            for eb in extra_bullets:
                if eb["bullet"].lower() not in existing_texts:
                    combined.append(eb)
                    existing_texts.add(eb["bullet"].lower())
        except Exception:
            pass

    state["verified_bullets"] = combined
    state["angle_verified"] = True
    progress(1.0, desc="Done")
    return state, f"{len(combined)} data points after angle research.", _bullets_markdown(combined)


def step3_add_user_data(user_facts_text, state):
    """Add user-provided facts to the bullet pool."""
    bullets = state["verified_bullets"]
    user_bullets = []
    for line in user_facts_text.strip().split("\n"):
        line = line.strip()
        if line:
            user_bullets.append({"bullet": line, "value": line, "source": "user-provided", "confidence": "high"})
    if user_bullets:
        state["verified_bullets"] = bullets + user_bullets
        state["user_facts"] = user_facts_text.strip()
    return state, f"Added {len(user_bullets)} user data points. Total: {len(state['verified_bullets'])}", _bullets_markdown(state["verified_bullets"])


def step4_generate_hooks(api_key, tone, audience, state, progress=gr.Progress()):
    """Generate hook options grounded in verified data."""
    topic = state["selected_topic"]
    bullets = state["verified_bullets"]
    angle = state.get("angle", "")
    demo_mode = not api_key

    if demo_mode:
        progress(0.5, desc="Loading demo hooks...")
        time.sleep(0.3)
        state["hook_options"] = list(_DEMO_HOOKS)
        return state, _hook_cards_html(_DEMO_HOOKS), f"Generated {len(_DEMO_HOOKS)} hooks (demo)."

    _set_api_key(api_key)
    progress(0.3, desc="Generating hooks from verified data...")
    try:
        hooks = generate_hooks(
            topic=topic["title"],
            verified_bullets=bullets,
            tone=tone,
            audience=audience,
            angle=angle,
        )
        state["hook_options"] = hooks
        progress(1.0, desc="Done")
        return state, _hook_cards_html(hooks), f"Generated {len(hooks)} hooks."
    except anthropic.AuthenticationError:
        return state, "", "Invalid API key."
    except anthropic.APIError as exc:
        return state, "", f"API error: {exc}"


def step4_select_hook(hook_index, state):
    """Select a hook by number."""
    hooks = state.get("hook_options", [])
    if not hooks:
        return state, "No hooks available."
    idx = max(0, min(int(hook_index) - 1, len(hooks) - 1))
    state["selected_hook"] = hooks[idx]
    state["step"] = 5
    return state, f"Selected hook: **{hooks[idx]['hook']}**"


def step5_generate_slides(api_key, slide_count, tone, audience, style_notes, review_iters, state, progress=gr.Progress()):
    """Run the full generation + verification pipeline."""
    topic = state["selected_topic"]
    hook = state["selected_hook"]
    bullets = state["verified_bullets"]
    angle = state.get("angle", "")
    research_text = state["research_text"]
    hook_text = hook["hook"]
    demo_mode = not api_key

    if demo_mode:
        steps = [
            "Creating slides...", "Reviewing engagement...", "Fact-checking...",
            "Web verification...", "Conclusion check...", "Coherence check...",
            "Final polish...", "TikTok metadata...",
        ]
        for i, desc in enumerate(steps):
            progress((i + 1) / len(steps), desc=desc)
            time.sleep(0.25)

        state["slides"] = list(_DEMO_SLIDES)
        state["fact_check_report"] = list(_DEMO_FACT_REPORT)
        state["conclusion_report"] = {"logic_valid": True, "verdict_slide": 8, "issues": []}
        state["coherence_report"] = {"coherence_score": 9, "arc_analysis": "Strong hook → evidence buildup → contrarian verdict.", "issues": []}
        state["tiktok_metadata"] = dict(_DEMO_METADATA)
        state["step"] = 6
        return state, _render_all_slides_html(state), "Demo slides generated (8 slides, coherence 9/10)."

    _set_api_key(api_key)

    try:
        # 1. Generate
        progress(1 / 8, desc="Creating fact-grounded slides...")
        slides = generate_slide_content(
            topic=topic["title"],
            angle=angle or topic["description"],
            hook=hook_text,
            slide_count=int(slide_count),
            tone=tone,
            audience=audience,
            style_notes=style_notes,
            research_facts=bullets,
        )
        slides = enforce_hook_and_count(slides, hook_text, int(slide_count))

        # 2. Review
        review_iterations = int(review_iters)
        if review_iterations > 0:
            progress(2 / 8, desc=f"Engagement review ({review_iterations} iterations)...")
            slides = review_and_improve(slides=slides, tone=tone, audience=audience, iterations=review_iterations, hook=hook_text)
            slides = enforce_hook_and_count(slides, hook_text, int(slide_count))

        # 3. Layered fact-check
        fact_report = []
        if bullets:
            progress(3 / 8, desc="Layered fact-check...")
            fc_result = layered_fact_check(slides, research_text, bullets, topic["title"], angle or topic["description"])
            if not isinstance(fc_result, dict):
                fc_result = {"corrected_slides": slides}
            for item in fc_result.get("layer_a_report", []):
                fact_report.append({"slide": item.get("slide", "?"), "status": item.get("status", "unknown"), "notes": f"[news-sourced] {item.get('notes', '')}"})
            for item in fc_result.get("layer_b_report", []):
                fact_report.append({"slide": item.get("slide", "?"), "status": item.get("status", "unknown"), "notes": f"[supporting data] {item.get('notes', '')}"})
            slides = _safe_get_slides(fc_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, int(slide_count))

        # 4. Web verification
        progress(4 / 8, desc="Web verification...")
        try:
            ws_result = web_search_fact_check(slides, topic["title"], angle or topic["description"])
            if not isinstance(ws_result, dict):
                ws_result = {"corrected_slides": slides}
            for item in ws_result.get("search_report", []):
                fact_report.append({"slide": item.get("slide", "?"), "status": item.get("status", "unknown"), "notes": f"[web search] {item.get('notes', '')}"})
            slides = _safe_get_slides(ws_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, int(slide_count))
        except Exception:
            pass

        state["fact_check_report"] = fact_report

        # 5. Conclusion validation
        progress(5 / 8, desc="Conclusion validation...")
        conclusion_result = validate_conclusion(slides, bullets, topic["title"], angle or topic["description"])
        if not isinstance(conclusion_result, dict):
            conclusion_result = {"corrected_slides": slides}
        state["conclusion_report"] = conclusion_result
        slides = _safe_get_slides(conclusion_result, slides)
        slides = enforce_hook_and_count(slides, hook_text, int(slide_count))

        # 6. Coherence
        progress(6 / 8, desc="Narrative coherence...")
        coherence_result = check_narrative_coherence(slides, topic["title"], angle or topic["description"], hook_text)
        if not isinstance(coherence_result, dict):
            coherence_result = {"corrected_slides": slides}
        coherence_score = coherence_result.get("coherence_score", 0)
        slides = _safe_get_slides(coherence_result, slides)
        slides = enforce_hook_and_count(slides, hook_text, int(slide_count))

        slides = strip_claim_tags(slides)

        # 7. Final polish
        progress(7 / 8, desc="Final polish...")
        value_result = add_value_pass(slides=slides, topic=topic["title"], angle=angle or topic["description"], audience=audience, hook=hook_text)
        if not isinstance(value_result, dict):
            value_result = {"corrected_slides": _safe_get_slides(value_result, slides)}
        slides = _safe_get_slides(value_result, slides)
        slides = enforce_hook_and_count(slides, hook_text, int(slide_count))
        final_score = value_result.get("coherence_score", coherence_score)
        state["coherence_report"] = {"coherence_score": final_score, "arc_analysis": value_result.get("arc_analysis", ""), "issues": value_result.get("issues", [])}

        # 8. TikTok metadata
        progress(8 / 8, desc="TikTok metadata...")
        metadata = generate_tiktok_metadata(slides=slides, topic=topic["title"], angle=angle or topic["description"], hook=hook["hook"])
        state["tiktok_metadata"] = metadata

        state["slides"] = slides
        state["step"] = 6
        return state, _render_all_slides_html(state), f"Generated {len(slides)} slides (coherence: {final_score}/10)."

    except anthropic.AuthenticationError:
        return state, "", "Invalid API key."
    except anthropic.APIError as exc:
        return state, "", f"API error: {exc}"
    except Exception as exc:
        return state, "", f"Generation failed: {type(exc).__name__}: {exc}"


def _render_all_slides_html(state: dict) -> str:
    """Render all slides as preview HTML."""
    slides = state.get("slides", [])
    colors = _colors_from_state(state)
    handle = state.get("handle", "")
    total = len(slides)
    html_parts = []
    for i, s in enumerate(slides):
        html_parts.append(_slide_preview_html(s, i, total, colors, handle))
    return "".join(html_parts)


# ── Studio callbacks ──────────────────────────────────────────────────────────


def studio_update_slides_from_json(slides_json, state):
    """Update slides from the JSON editor."""
    try:
        parsed = json.loads(slides_json)
        if isinstance(parsed, list) and all(isinstance(s, dict) for s in parsed):
            state["slides"] = parsed
            return state, _render_all_slides_html(state), "Slides updated."
        return state, _render_all_slides_html(state), "Invalid format. Expected a list of {title, body, footer} dicts."
    except json.JSONDecodeError as e:
        return state, _render_all_slides_html(state), f"JSON parse error: {e}"


def studio_build_pptx(state, progress=gr.Progress()):
    """Build PPTX and return file for download."""
    slides = state.get("slides", [])
    if not slides:
        return state, None, "No slides to export."
    colors = _colors_from_state(state)
    progress(0.3, desc="Building PowerPoint...")
    filepath = build_pptx(
        slides=slides, colors=colors,
        aspect_ratio=state.get("aspect_ratio", "9:16"),
        output_dir="./output", handle=state.get("handle", ""),
    )
    state["pptx_path"] = filepath
    progress(1.0, desc="Done")
    return state, filepath, "PPTX built successfully."


def studio_build_pngs(state, progress=gr.Progress()):
    """Build PNGs and return paths + gallery images."""
    slides = state.get("slides", [])
    if not slides:
        return state, [], None, "No slides to export."
    colors = _colors_from_state(state)
    progress(0.3, desc="Rendering PNGs...")
    png_paths = build_pngs(
        slides=slides, colors=colors,
        aspect_ratio=state.get("aspect_ratio", "9:16"),
        output_dir="./output", handle=state.get("handle", ""),
    )
    state["png_paths"] = png_paths
    progress(1.0, desc="Done")
    # Build zip for download
    zip_bytes = _build_zip(png_paths)
    zip_path = os.path.join("./output", "slides.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_bytes)
    return state, png_paths, zip_path, f"Rendered {len(png_paths)} PNGs."


def studio_build_alternatives(num_alts, state, progress=gr.Progress()):
    """Generate alternative slide styles."""
    slides = state.get("slides", [])
    if not slides:
        return state, [], None, "No slides."
    progress(0.3, desc="Generating alternatives...")
    try:
        alts = build_style_alternatives(
            slides=slides,
            aspect_ratio=state.get("aspect_ratio", "9:16"),
            output_dir="./output",
            handle=state.get("handle", ""),
            num_alternatives=int(num_alts),
        )
        state["mcp_alternatives"] = alts
        all_paths = []
        for alt in alts:
            all_paths.extend(alt.get("png_paths", []))
        zip_path = None
        if all_paths:
            zip_bytes = _build_zip(all_paths)
            zip_path = os.path.join("./output", "alternatives.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_bytes)
        progress(1.0, desc="Done")
        return state, all_paths, zip_path, f"Generated {len(alts)} alternative styles."
    except Exception as exc:
        return state, [], None, f"Failed: {exc}"


def studio_generate_ai_images(api_key, google_ai_key, openai_img_key, state, progress=gr.Progress()):
    """Generate AI slide images."""
    slides = state.get("slides", [])
    if not slides:
        return state, [], None, "No slides."

    if google_ai_key:
        os.environ["GOOGLE_AI_API_KEY"] = google_ai_key
        os.environ.pop("OPENAI_API_KEY", None)
    elif openai_img_key:
        os.environ["OPENAI_API_KEY"] = openai_img_key
        os.environ.pop("GOOGLE_AI_API_KEY", None)
    else:
        return state, [], None, "No image API key provided."

    _set_api_key(api_key)
    colors = _colors_from_state(state)
    topic_title = state["selected_topic"]["title"] if state.get("selected_topic") else ""
    angle = state.get("angle", "")

    progress(0.2, desc="Generating image prompts...")
    try:
        image_prompts = generate_image_prompts(slides=slides, topic=topic_title, angle=angle)
        progress(0.5, desc="Generating AI images...")
        ai_paths = generate_slide_images(
            slides=slides, image_prompts=image_prompts, colors=colors,
            aspect_ratio=state.get("aspect_ratio", "9:16"),
            output_dir="./output", handle=state.get("handle", ""),
        )
        state["ai_image_paths"] = ai_paths
        state["ai_image_prompts"] = image_prompts
        zip_bytes = _build_zip(ai_paths)
        zip_path = os.path.join("./output", "ai_slides.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)
        progress(1.0, desc="Done")
        return state, ai_paths, zip_path, f"Generated {len(ai_paths)} AI images."
    except Exception as exc:
        return state, [], None, f"AI image generation failed: {exc}\n{traceback.format_exc()}"


def studio_generate_script(api_key, state, progress=gr.Progress()):
    """Generate voiceover scripts."""
    slides = state.get("slides", [])
    if not slides:
        return state, "", "No slides."
    _set_api_key(api_key)
    topic = state["selected_topic"]["title"] if state.get("selected_topic") else ""
    angle = state.get("angle", "")
    progress(0.3, desc="Generating voiceover scripts...")
    try:
        scripts = generate_video_script(slides=slides, topic=topic, angle=angle)
        state["video_scripts"] = scripts
        state["video_path"] = None
        progress(1.0, desc="Done")
        scripts_text = "\n---\n".join([f"Slide {i+1}:\n{s}" for i, s in enumerate(scripts)])
        return state, scripts_text, f"Generated scripts for {len(scripts)} slides."
    except Exception as exc:
        return state, "", f"Script generation failed: {exc}"


def studio_build_video(api_key, elevenlabs_key, voice_id, bg_option, google_ai_key, openai_img_key, pexels_key, pixabay_key, edited_scripts_text, state, progress=gr.Progress()):
    """Build narrated video."""
    slides = state.get("slides", [])
    if not slides:
        return state, None, "No slides."
    if not elevenlabs_key:
        return state, None, "ElevenLabs API key required."

    os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
    _set_api_key(api_key)
    colors = _colors_from_state(state)

    # Parse edited scripts from text
    raw_blocks = edited_scripts_text.split("---")
    edited_scripts = []
    for block in raw_blocks:
        lines = block.strip().split("\n")
        # Remove "Slide N:" prefix if present
        script_lines = []
        for line in lines:
            if line.strip().startswith("Slide ") and line.strip().endswith(":"):
                continue
            script_lines.append(line)
        edited_scripts.append("\n".join(script_lines).strip())
    edited_scripts = [s for s in edited_scripts if s]

    # Fallback to stored scripts if parsing failed
    if len(edited_scripts) != len(slides):
        edited_scripts = state.get("video_scripts", [])
    if not edited_scripts:
        return state, None, "No scripts. Generate scripts first."

    state["video_scripts"] = edited_scripts
    topic_title = state["selected_topic"]["title"] if state.get("selected_topic") else ""
    angle = state.get("angle", "")

    progress(0.2, desc="Building video...")
    try:
        if bg_option == "Web image search":
            if pexels_key:
                os.environ["PEXELS_API_KEY"] = pexels_key
            if pixabay_key:
                os.environ["PIXABAY_API_KEY"] = pixabay_key
            search_queries = generate_image_search_queries(slides=slides, scripts=edited_scripts, topic=topic_title, angle=angle)
            web_result = build_video_with_searched_images(
                slides=slides, scripts=edited_scripts, search_queries=search_queries,
                colors=colors, aspect_ratio=state.get("aspect_ratio", "9:16"),
                output_dir="./output", handle=state.get("handle", ""),
                voice_id=voice_id,
            )
            state["video_path"] = web_result["video_path"]
            state["video_search_results"] = web_result["search_results"]
        elif bg_option == "AI-generated images":
            if google_ai_key:
                os.environ["GOOGLE_AI_API_KEY"] = google_ai_key
            elif openai_img_key:
                os.environ["OPENAI_API_KEY"] = openai_img_key
            image_prompts = generate_image_prompts(slides=slides, topic=topic_title, angle=angle)
            video_path = build_video_with_ai_images(
                slides=slides, scripts=edited_scripts, image_prompts=image_prompts,
                colors=colors, aspect_ratio=state.get("aspect_ratio", "9:16"),
                output_dir="./output", handle=state.get("handle", ""),
                voice_id=voice_id,
            )
            state["video_path"] = video_path
        else:
            video_path = build_video(
                slides=slides, scripts=edited_scripts,
                colors=colors, aspect_ratio=state.get("aspect_ratio", "9:16"),
                output_dir="./output", handle=state.get("handle", ""),
                voice_id=voice_id,
            )
            state["video_path"] = video_path

        progress(1.0, desc="Done")
        return state, state["video_path"], "Video built successfully."
    except Exception as exc:
        return state, None, f"Video build failed: {exc}"


def start_over(state):
    """Reset all state."""
    return _default_state()


# ══════════════════════════════════════════════════════════════════════════════
# BUILD THE GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
.step-header { font-size: 18px; font-weight: 700; margin-bottom: 8px; }
.card-container { max-height: 500px; overflow-y: auto; }
footer { display: none !important; }
"""

slides_cfg = CONFIG.get("slides", {})
research_cfg = CONFIG.get("research", {})
content_cfg = CONFIG.get("content", {})
video_cfg = CONFIG.get("video", {})
colors_cfg = slides_cfg.get("colors", {})

with gr.Blocks(
    title="Posting: Finance Slides",
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    css=CSS,
) as app:

    # -- State --
    state = gr.State(_default_state())

    # -- Header --
    gr.Markdown("# Posting\n*Generate trending finance slide decks for TikTok & Instagram — Gradio Edition*")

    # ── Settings (Accordion) ─────────────────────────────────────────────────

    with gr.Accordion("Settings", open=False):
        with gr.Tab("Integrations"):
            api_key = gr.Textbox(
                label="Anthropic API Key", type="password",
                value=_get_env_key("ANTHROPIC_API_KEY"),
                info="Required. Set ANTHROPIC_API_KEY env var or paste here.",
            )
            elevenlabs_key = gr.Textbox(
                label="ElevenLabs API Key", type="password",
                value=_get_env_key("ELEVENLABS_API_KEY"),
                info="Required for video export.",
            )
            voice_id = gr.Textbox(
                label="Voice ID",
                value=video_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
                info="ElevenLabs voice ID. Default is 'Adam'.",
            )
            image_provider = gr.Dropdown(
                label="AI Image Provider",
                choices=["Gemini Flash (Free)", "OpenAI DALL-E 3"],
                value="Gemini Flash (Free)",
            )
            google_ai_key = gr.Textbox(
                label="Google AI API Key", type="password",
                value=_get_env_key("GOOGLE_AI_API_KEY"),
                info="Free. Get at aistudio.google.com/apikey",
            )
            openai_img_key = gr.Textbox(
                label="OpenAI API Key", type="password",
                value=_get_env_key("OPENAI_API_KEY"),
                info="Paid. For DALL-E 3.",
            )
            pexels_key = gr.Textbox(
                label="Pexels API Key", type="password",
                value=_get_env_key("PEXELS_API_KEY"),
            )
            pixabay_key = gr.Textbox(
                label="Pixabay API Key", type="password",
                value=_get_env_key("PIXABAY_API_KEY"),
            )

        with gr.Tab("Slides & Branding"):
            slide_count = gr.Slider(
                label="Number of slides", minimum=3, maximum=15, step=1,
                value=min(max(slides_cfg.get("count", 7), 3), 15),
            )
            tone = gr.Dropdown(
                label="Tone",
                choices=["bold", "casual", "professional", "educational"],
                value=slides_cfg.get("tone", "bold"),
            )
            audience = gr.Textbox(
                label="Audience",
                value=slides_cfg.get("audience", "retail investors"),
            )
            aspect_ratio = gr.Dropdown(
                label="Aspect ratio",
                choices=["9:16", "16:9"],
                value=slides_cfg.get("aspect_ratio", "9:16"),
            )
            handle = gr.Textbox(
                label="Account handle",
                value=slides_cfg.get("handle", "@cristian.bojaca"),
            )

        with gr.Tab("Colors"):
            bg_color = gr.ColorPicker(label="Background", value=colors_cfg.get("background", "#0D0D15"))
            title_color = gr.ColorPicker(label="Title", value=colors_cfg.get("title", "#FFFFFF"))
            body_color = gr.ColorPicker(label="Body", value=colors_cfg.get("body", "#C0C0D0"))
            accent_color = gr.ColorPicker(label="Accent", value=colors_cfg.get("accent", "#F7B731"))
            highlight_color = gr.ColorPicker(label="Highlight", value=colors_cfg.get("highlight", "#FF5757"))

        with gr.Tab("Research"):
            sources = gr.CheckboxGroup(
                label="Sources",
                choices=["news", "reddit"],
                value=research_cfg.get("sources", ["news"]),
            )
            research_topics = gr.CheckboxGroup(
                label="Topics",
                choices=["stocks", "crypto", "earnings", "market trends", "economic indicators"],
                value=research_cfg.get("topics", ["stocks"]),
            )
            subreddits_input = gr.Textbox(
                label="Subreddits (comma-separated)",
                value=", ".join(research_cfg.get("subreddits", ["stocks"])),
            )

        with gr.Tab("Review"):
            review_iterations = gr.Slider(
                label="Review iterations", minimum=0, maximum=5, step=1,
                value=content_cfg.get("review_iterations", 2),
            )
            style_notes = gr.Textbox(
                label="Style notes",
                value=content_cfg.get("style_notes", ""),
                lines=4,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Research & Pick a Topic
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("1. Topic"):
        gr.Markdown("## Step 1: Pick a Topic")
        research_mode = gr.Radio(
            label="How do you want to find a topic?",
            choices=["Latest News", "Custom Topic"],
            value="Latest News",
        )
        custom_topic_input = gr.Textbox(
            label="Topic to research",
            placeholder="e.g. Tesla earnings Q4, Bitcoin ETF inflows",
            visible=True,
        )
        research_btn = gr.Button("Research Topics", variant="primary")
        step1_status = gr.Textbox(label="Status", interactive=False)

        topic_cards_display = gr.HTML(visible=False)
        with gr.Row(visible=False) as topic_select_row:
            topic_dropdown = gr.Number(
                label="Select topic number",
                value=1, minimum=1, maximum=10, precision=0,
            )
            select_topic_btn = gr.Button("Select This Topic", variant="primary")
        step1_select_status = gr.Textbox(label="Selection", interactive=False)

        research_btn.click(
            fn=step1_research,
            inputs=[api_key, research_mode, custom_topic_input, sources, research_topics, subreddits_input, audience, state],
            outputs=[state, topic_cards_display, topic_cards_display, topic_select_row, step1_status],
        )
        select_topic_btn.click(
            fn=step1_select_topic,
            inputs=[topic_dropdown, state],
            outputs=[state, step1_select_status],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Consolidate Data
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("2. Data"):
        gr.Markdown("## Step 2: Verified Data")
        gr.Markdown("Automatically consolidates 20+ verified bullet points from all sources.")
        consolidate_btn = gr.Button("Consolidate Data", variant="primary")
        step2_status = gr.Textbox(label="Status", interactive=False)
        bullets_display = gr.Markdown()

        consolidate_btn.click(
            fn=step2_consolidate,
            inputs=[api_key, audience, state],
            outputs=[state, step2_status, bullets_display],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Angle & Additional Data
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("3. Angle"):
        gr.Markdown("## Step 3: Your Angle & Data")
        gr.Markdown("Describe your angle and we'll research it, or add your own facts.")

        with gr.Tab("Provide an Angle"):
            angle_input = gr.Textbox(
                label="Your angle",
                placeholder="e.g. 'Focus on how beginners can take advantage'",
                lines=3,
            )
            research_angle_btn = gr.Button("Research & Verify This Angle", variant="primary")
            angle_status = gr.Textbox(label="Status", interactive=False)
            angle_bullets_display = gr.Markdown()

            research_angle_btn.click(
                fn=step3_research_angle,
                inputs=[api_key, angle_input, audience, state],
                outputs=[state, angle_status, angle_bullets_display],
            )

        with gr.Tab("Add Your Own Data"):
            user_facts_input = gr.Textbox(
                label="Your factual data (one per line)",
                placeholder="Oil is at $66 per barrel\nThe Fed held rates at 4.5%",
                lines=5,
            )
            add_data_btn = gr.Button("Add My Data", variant="primary")
            add_data_status = gr.Textbox(label="Status", interactive=False)
            user_bullets_display = gr.Markdown()

            add_data_btn.click(
                fn=step3_add_user_data,
                inputs=[user_facts_input, state],
                outputs=[state, add_data_status, user_bullets_display],
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Choose a Hook
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("4. Hook"):
        gr.Markdown("## Step 4: Choose a Hook")
        generate_hooks_btn = gr.Button("Generate Hooks", variant="primary")
        step4_status = gr.Textbox(label="Status", interactive=False)
        hooks_display = gr.HTML()
        with gr.Row():
            hook_dropdown = gr.Number(
                label="Select hook number",
                value=1, minimum=1, maximum=10, precision=0,
            )
            select_hook_btn = gr.Button("Use This Hook", variant="primary")
        step4_select_status = gr.Textbox(label="Selection", interactive=False)

        generate_hooks_btn.click(
            fn=step4_generate_hooks,
            inputs=[api_key, tone, audience, state],
            outputs=[state, hooks_display, step4_status],
        )
        select_hook_btn.click(
            fn=step4_select_hook,
            inputs=[hook_dropdown, state],
            outputs=[state, step4_select_status],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Generate & Verify Slides
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("5. Generate"):
        gr.Markdown("## Step 5: Generate & Verify Slides")
        gr.Markdown("Runs an 8-step pipeline: generate → review → fact-check → verify → conclusion → coherence → polish → metadata")
        generate_slides_btn = gr.Button("Generate Slides", variant="primary", size="lg")
        step5_status = gr.Textbox(label="Status", interactive=False)
        slides_preview = gr.HTML(label="Slide Preview")

        generate_slides_btn.click(
            fn=step5_generate_slides,
            inputs=[api_key, slide_count, tone, audience, style_notes, review_iterations, state],
            outputs=[state, slides_preview, step5_status],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Studio
    # ══════════════════════════════════════════════════════════════════════════

    with gr.Tab("6. Studio"):
        gr.Markdown("## Studio")

        # ── Tab: Edit ────────────────────────────────────────────────────

        with gr.Tab("Edit"):
            gr.Markdown("Edit slides as JSON. Each slide has `title`, `body`, and `footer` fields.")

            def _load_slides_json(state):
                slides = state.get("slides", [])
                reports = []
                reports.append(_fact_report_markdown(state.get("fact_check_report", [])))
                cr = state.get("coherence_report")
                if cr:
                    reports.append(_coherence_report_markdown(cr))
                meta = state.get("tiktok_metadata")
                meta_title = meta.get("title", "") if meta else ""
                meta_desc = meta.get("description", "") if meta else ""
                return (
                    json.dumps(slides, indent=2),
                    _render_all_slides_html(state),
                    "\n\n".join(reports),
                    meta_title,
                    meta_desc,
                )

            load_editor_btn = gr.Button("Load Slides into Editor", variant="secondary")
            with gr.Row():
                with gr.Column():
                    slides_json_editor = gr.Code(
                        label="Slides JSON",
                        language="json",
                        lines=20,
                    )
                    update_slides_btn = gr.Button("Update Slides from JSON", variant="primary")
                    edit_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column():
                    edit_preview = gr.HTML(label="Preview")

            reports_md = gr.Markdown(label="Reports")

            gr.Markdown("### TikTok Post Copy")
            tiktok_title = gr.Textbox(label="Video Title")
            tiktok_desc = gr.Textbox(label="Description", lines=5)

            load_editor_btn.click(
                fn=_load_slides_json,
                inputs=[state],
                outputs=[slides_json_editor, edit_preview, reports_md, tiktok_title, tiktok_desc],
            )
            update_slides_btn.click(
                fn=studio_update_slides_from_json,
                inputs=[slides_json_editor, state],
                outputs=[state, edit_preview, edit_status],
            )

        # ── Tab: Slides (Export) ─────────────────────────────────────────

        with gr.Tab("Slides"):
            gr.Markdown("### Export Carousel Slides")
            with gr.Row():
                build_pptx_btn = gr.Button("Build PPTX", variant="primary")
                build_pngs_btn = gr.Button("Build PNGs", variant="primary")
            slides_export_status = gr.Textbox(label="Status", interactive=False)
            pptx_download = gr.File(label="Download PPTX")
            png_gallery = gr.Gallery(label="PNG Slides", columns=3)
            png_zip_download = gr.File(label="Download All PNGs (ZIP)")

            gr.Markdown("### Alternative Styles")
            gr.Markdown("Finance-themed backgrounds (candlestick charts, trend lines, volume bars). Instant, offline.")
            with gr.Row():
                num_alts = gr.Dropdown(label="Number of alternatives", choices=["2", "3", "4"], value="2")
                gen_alts_btn = gr.Button("Generate Alternatives", variant="primary")
            alts_status = gr.Textbox(label="Status", interactive=False)
            alts_gallery = gr.Gallery(label="Alternative Styles", columns=3)
            alts_zip_download = gr.File(label="Download Alternatives (ZIP)")

            build_pptx_btn.click(
                fn=studio_build_pptx,
                inputs=[state],
                outputs=[state, pptx_download, slides_export_status],
            )
            build_pngs_btn.click(
                fn=studio_build_pngs,
                inputs=[state],
                outputs=[state, png_gallery, png_zip_download, slides_export_status],
            )
            gen_alts_btn.click(
                fn=studio_build_alternatives,
                inputs=[num_alts, state],
                outputs=[state, alts_gallery, alts_zip_download, alts_status],
            )

        # ── Tab: AI Images ───────────────────────────────────────────────

        with gr.Tab("AI Images"):
            gr.Markdown("### AI-Generated Slide Images")
            gr.Markdown(
                "Generate cinematic AI backgrounds using Gemini Flash (free) or DALL-E 3. "
                "Claude creates visual prompts, the AI generates photorealistic images."
            )
            gen_ai_images_btn = gr.Button("Generate AI Slide Images", variant="primary")
            ai_images_status = gr.Textbox(label="Status", interactive=False)
            ai_gallery = gr.Gallery(label="AI Slides", columns=3)
            ai_zip_download = gr.File(label="Download AI Slides (ZIP)")

            gen_ai_images_btn.click(
                fn=studio_generate_ai_images,
                inputs=[api_key, google_ai_key, openai_img_key, state],
                outputs=[state, ai_gallery, ai_zip_download, ai_images_status],
            )

        # ── Tab: Video ───────────────────────────────────────────────────

        with gr.Tab("Video"):
            gr.Markdown("### Narrated Video")
            gr.Markdown("Generate a narrated MP4 video with ElevenLabs AI voiceover.")

            video_bg_option = gr.Radio(
                label="Slide backgrounds",
                choices=["Plain slides", "Web image search", "AI-generated images"],
                value="Plain slides",
            )

            gr.Markdown("**Step 1 — Voiceover Script**")
            gen_script_btn = gr.Button("Generate Script", variant="primary")
            script_status = gr.Textbox(label="Status", interactive=False)
            scripts_editor = gr.Textbox(
                label="Voiceover Scripts (edit below, separated by ---)",
                lines=12,
            )

            gr.Markdown("**Step 2 — Build Video**")
            build_video_btn = gr.Button("Build Narrated Video", variant="primary")
            video_status = gr.Textbox(label="Status", interactive=False)
            video_output = gr.Video(label="Preview")

            gen_script_btn.click(
                fn=studio_generate_script,
                inputs=[api_key, state],
                outputs=[state, scripts_editor, script_status],
            )
            build_video_btn.click(
                fn=studio_build_video,
                inputs=[api_key, elevenlabs_key, voice_id, video_bg_option, google_ai_key, openai_img_key, pexels_key, pixabay_key, scripts_editor, state],
                outputs=[state, video_output, video_status],
            )

    # ── Start Over ────────────────────────────────────────────────────────

    with gr.Row():
        restart_btn = gr.Button("Start Over", variant="secondary")
        restart_btn.click(fn=start_over, inputs=[state], outputs=[state])


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("./output", exist_ok=True)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
