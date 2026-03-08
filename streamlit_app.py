"""Streamlit UI for the finance slide generator. Guided 6-step workflow.

Flow:
  1. Research & Pick a Topic (card-based selection)
  2. Consolidate Data (20+ verified bullet points)
  3. Provide Angle & Additional Data
  4. Choose a Hook (card-based selection)
  5. Generate & Verify Slides (verification dashboard)
  6. Studio: Edit + Export (tabbed — Edit | Slides | AI Images | Video)
"""

import gc
import html as html_mod
import io
import os

import anthropic
import streamlit as st
import yaml

from src.topic_store import save_topic, load_topic, list_topics, delete_topic
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
    regenerate_video_script,

    generate_tiktok_script,
    regenerate_tiktok_script,
    generate_image_search_queries,
    generate_overlay_prompts,
    analyze_charts,
    map_charts_to_slides,
    OVERLAY_STYLE_DESCRIPTIONS,
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx
from src.slides.png_builder import build_pngs
from src.slides.png_builder import build_style_alternatives
from src.slides.video_builder import (
    build_video_with_searched_images,
    build_video_with_chart_overlays,
    build_tiktok_video,
)
from src.slides.image_generator import (
    generate_slide_images,
    generate_ai_overlay,
    get_slide_role,
    CINEMATIC_OVERLAY_PRESETS,
)


@st.cache_data
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


st.set_page_config(page_title="Posting: Finance Slides", page_icon="📊", layout="wide")

st.title("Posting")
st.caption("Generate trending finance slide decks for TikTok & Instagram · v2.2-gemini")

# ── Sidebar: Settings (collapsible groups) ────────────────────────────────────

config = load_config()
slides_cfg = config.get("slides", {})
research_cfg = config.get("research", {})
content_cfg = config.get("content", {})

st.sidebar.header("Settings")


# ── Integrations ──────────────────────────────────────────────────────────────

def _get_default_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


def _get_default_elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ELEVENLABS_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


def _get_default_google_ai_key() -> str:
    key = os.environ.get("GOOGLE_AI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_AI_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


def _get_default_openai_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


def _get_default_pexels_key() -> str:
    key = os.environ.get("PEXELS_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("PEXELS_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


def _get_default_pixabay_key() -> str:
    key = os.environ.get("PIXABAY_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("PIXABAY_API_KEY", "")
        except FileNotFoundError:
            pass
    return key


with st.sidebar.expander("Integrations", expanded=True):
    api_key = st.text_input(
        "Anthropic API Key",
        value=_get_default_api_key(),
        type="password",
        help="Required. Set ANTHROPIC_API_KEY env var, add to Streamlit secrets, or paste here.",
    )

    elevenlabs_key = st.text_input(
        "ElevenLabs API Key",
        value=_get_default_elevenlabs_key(),
        type="password",
        help="Required for video export. Get your key at https://elevenlabs.io",
    )

    video_cfg = config.get("video", {})
    elevenlabs_voice = st.text_input(
        "Voice ID",
        value=video_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
        help="ElevenLabs voice ID. Default is 'Adam'.",
    )
    video_enabled = bool(elevenlabs_key)

    st.markdown("**Image Generation**")
    image_provider = st.selectbox(
        "AI Image Provider",
        ["Gemini Flash (Free)", "OpenAI DALL-E 3"],
        index=0,
        help="Gemini Flash is free with a Google AI Studio key. DALL-E 3 requires a paid OpenAI key.",
    )

    google_ai_key = ""
    openai_img_key = ""
    if image_provider.startswith("Gemini"):
        google_ai_key = st.text_input(
            "Google AI API Key",
            value=_get_default_google_ai_key(),
            type="password",
            help="Free. Get your key at https://aistudio.google.com/apikey",
        )
    else:
        openai_img_key = st.text_input(
            "OpenAI API Key",
            value=_get_default_openai_key(),
            type="password",
            help="Paid. Get your key at https://platform.openai.com/api-keys",
        )
    ai_images_enabled = bool(google_ai_key or openai_img_key)

    st.markdown("**Web Image Search**")
    pexels_key = st.text_input(
        "Pexels API Key",
        value=_get_default_pexels_key(),
        type="password",
        help="Free. Get your key at https://www.pexels.com/api/",
    )
    pixabay_key = st.text_input(
        "Pixabay API Key",
        value=_get_default_pixabay_key(),
        type="password",
        help="Free. Get your key at https://pixabay.com/api/docs/",
    )
# ── Demo Mode ─────────────────────────────────────────────────────────────────

demo_mode = not api_key

if demo_mode:
    st.sidebar.warning("No API key — running in **Demo Mode** with sample finance data.")

# Demo data: a coherent Tesla Q4 story used across all steps
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
    "description": "Tesla just reported record Q4 deliveries of 495,570 vehicles, crushing Wall Street estimates by 12,570 units. The stock surged 15% in two days.\n\nBut the real story isn't just cars — it's the $1.4B energy business growing 75% YoY, margins healing to 18.2%, and $2.1B in free cash flow.\n\nPlus: the sub-$25K vehicle is coming in late 2025, FSD is improving fast, and short sellers are at a 4-year low.\n\nFollow for more finance breakdowns.\n\n#Tesla #TSLA #Stocks #Investing #Finance #StockMarket #RetailInvestor #WallStreet #EV",
}

# ── Slides & Branding ─────────────────────────────────────────────────────────

with st.sidebar.expander("Slides & Branding"):
    slide_count = st.slider(
        "Number of slides", 3, 15,
        min(max(slides_cfg.get("count", 7), 3), 15),
    )
    tone = st.selectbox(
        "Tone",
        ["bold", "casual", "professional", "educational"],
        index=["bold", "casual", "professional", "educational"].index(
            slides_cfg.get("tone", "bold")
        ),
    )
    audience = st.text_input("Audience", slides_cfg.get("audience", "retail investors"))
    aspect_ratio = st.selectbox(
        "Aspect ratio",
        ["9:16 (vertical / stories)", "16:9 (landscape)"],
        index=0 if slides_cfg.get("aspect_ratio", "9:16") == "9:16" else 1,
    )
    aspect_ratio_val = "9:16" if aspect_ratio.startswith("9:16") else "16:9"
    handle = st.text_input("Account handle", slides_cfg.get("handle", "@cristian.bojaca"))

# ── Colors ─────────────────────────────────────────────────────────────────────

with st.sidebar.expander("Colors"):
    colors_cfg = slides_cfg.get("colors", {})
    col1, col2 = st.columns(2)
    bg_color = col1.color_picker("Background", colors_cfg.get("background", "#0D0D15"))
    title_color = col2.color_picker("Title", colors_cfg.get("title", "#FFFFFF"))
    body_color = col1.color_picker("Body", colors_cfg.get("body", "#C0C0D0"))
    accent_color = col2.color_picker("Accent", colors_cfg.get("accent", "#F7B731"))
    highlight_color = col1.color_picker("Highlight", colors_cfg.get("highlight", "#FF5757"))

# ── Research ───────────────────────────────────────────────────────────────────

with st.sidebar.expander("Research"):
    available_sources = ["news", "reddit"]
    default_sources = research_cfg.get("sources", ["news"])
    sources = st.multiselect("Sources", available_sources, default=default_sources)

    default_topics = research_cfg.get("topics", ["stocks"])
    topics = st.multiselect(
        "Topics",
        ["stocks", "crypto", "earnings", "market trends", "economic indicators"],
        default=default_topics,
    )

    default_subs = research_cfg.get("subreddits", ["stocks"])
    subreddits_input = st.text_input(
        "Subreddits (comma-separated)",
        ", ".join(default_subs),
    )
    subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]

# ── Review ─────────────────────────────────────────────────────────────────────

with st.sidebar.expander("Review"):
    review_iterations = st.slider(
        "Review iterations", 0, 5, content_cfg.get("review_iterations", 2)
    )
    style_notes = st.text_area(
        "Style notes",
        content_cfg.get("style_notes", ""),
        height=100,
    )

# ── Helpers ────────────────────────────────────────────────────────────────────

def _require_api_key():
    if not api_key:
        st.error("Please provide an Anthropic API key in the sidebar.")
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key


def _safe_get_slides(result, fallback: list[dict]) -> list[dict]:
    """Safely extract corrected_slides from a pipeline result."""
    if isinstance(result, list):
        return result if all(isinstance(s, dict) for s in result) else fallback
    if isinstance(result, dict):
        slides = result.get("corrected_slides", fallback)
        if isinstance(slides, list) and all(isinstance(s, dict) for s in slides):
            return slides
        return fallback
    return fallback


def _slide_preview_html(slide: dict, idx: int, total: int, colors: dict, handle_text: str) -> str:
    """Generate HTML for a live slide preview card."""
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
        min-height: 320px;
        position: relative;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        overflow: hidden;
    ">
        <div style="position:absolute;top:14px;right:18px;color:{ac};font-size:13px;opacity:0.7;">
            {idx + 1}/{total}
        </div>
        <div style="color:{tc};font-size:20px;font-weight:700;margin-top:14px;line-height:1.35;">
            {title}
        </div>
        <div style="border-top:2px solid {ac};width:25%;margin:14px 0;opacity:0.4;"></div>
        <div style="color:{bc};font-size:14px;line-height:1.6;">
            {body}
        </div>
        <div style="position:absolute;bottom:14px;left:0;right:0;text-align:center;
                    color:#4A5568;font-size:11px;">
            {footer_line}
        </div>
    </div>
    """


def _get_live_slides() -> list[dict]:
    """Read current slide values from editor widget session state."""
    base = st.session_state.slides
    live = []
    for i, s in enumerate(base):
        live.append({
            "title": st.session_state.get(f"edit_title_{i}", s.get("title", "")),
            "body": st.session_state.get(f"edit_body_{i}", s.get("body", "")),
            "footer": st.session_state.get(f"edit_footer_{i}", s.get("footer", "")),
        })
    return live


@st.cache_data
def _read_file_bytes(path: str, mtime: float) -> bytes:
    """Read file bytes, cached by path and modification time."""
    with open(path, "rb") as f:
        return f.read()


def _cached_read(path: str) -> bytes:
    """Read file bytes using cache keyed on modification time."""
    return _read_file_bytes(path, os.path.getmtime(path))


@st.cache_data
def _build_zip_bytes(paths: tuple[str, ...], mtimes: tuple[float, ...]) -> bytes:
    """Build a ZIP archive in memory, cached by file paths and mtimes."""
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            zf.write(p, os.path.basename(p))
    return buf.getvalue()


def _cached_zip(paths: list[str]) -> bytes:
    """Build a ZIP archive using cache keyed on file modification times."""
    path_tuple = tuple(paths)
    mtime_tuple = tuple(os.path.getmtime(p) for p in paths)
    return _build_zip_bytes(path_tuple, mtime_tuple)


# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "step": 1,
    "research_text": "",
    "research_facts": [],
    "topic_options": [],
    "selected_topic": None,
    "topic_id": None,
    "verified_bullets": [],
    "angle": "",
    "angle_verified": False,
    "user_facts": "",
    "hook_options": [],
    "selected_hook": None,
    "slides": [],
    "fact_check_report": [],
    "conclusion_report": None,
    "coherence_report": None,
    "tiktok_metadata": None,
    "pptx_path": None,
    "png_paths": [],
    "mcp_alternatives": [],
    "video_path": None,
    "video_scripts": [],
    "video_search_queries": [],
    "video_search_results": {},
    "video_build_error": None,
    "ai_image_paths": [],
    "ai_image_prompts": [],
    "ai_overlay_prompts": [],
    "overlay_style": "auto",
    "overlays_enabled": False,
    "chart_image_paths": [],
    "chart_analyses": [],
    "chart_slide_mapping": [],
    "tiktok_script": None,
    "tiktok_video_path": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Handle transition from old 7-step layout
if st.session_state.step > 6:
    st.session_state.step = 6

# ── Step indicators (clickable for completed steps) ──────────────────────────

# Steps 4+5 merged: step 5 is skipped internally
_NAV_STEPS = [
    (1, "1. Topic"),
    (2, "2. Data"),
    (3, "3. Angle"),
    (4, "4. Hook & Slides"),
    (6, "5. Studio"),
]

current = st.session_state.step
# Map step 5 → treat as step 4 for nav display (merged)
nav_current = 4 if current == 5 else current
cols = st.columns(len(_NAV_STEPS))
for col_idx, (step_num, label) in enumerate(_NAV_STEPS):
    with cols[col_idx]:
        if step_num < nav_current:
            if st.button(
                f"✓ {label}",
                key=f"nav_{step_num}",
                use_container_width=True,
            ):
                st.session_state.step = step_num
                st.rerun()
        elif step_num == nav_current:
            st.markdown(
                f"<div style='background:#1f6feb;color:white;text-align:center;"
                f"padding:8px 4px;border-radius:8px;font-weight:600;font-size:14px;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='color:#555;text-align:center;padding:8px 4px;"
                f"border-radius:8px;font-size:14px;border:1px solid #333;'>"
                f"{label}</div>",
                unsafe_allow_html=True,
            )

st.divider()

# ── Restart button ────────────────────────────────────────────────────────────

if st.session_state.step > 1:
    if st.button("Start Over", type="secondary"):
        for key in list(st.session_state.keys()):
            if key != "step":
                del st.session_state[key]
        st.session_state.step = 1
        # Clear file caches to free memory
        _read_file_bytes.clear()
        _build_zip_bytes.clear()
        gc.collect()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Research & Pick a Topic (card-based selection)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    st.header("Step 1: Pick a Topic")

    # ── Saved Topics ──────────────────────────────────────────────────────
    saved = list_topics()
    if saved:
        with st.expander(f"Saved Topics ({len(saved)})", expanded=False):
            st.caption("Resume a previous topic with its verified data and research.")
            for rec in saved:
                t = rec["topic"]
                bullet_count = len(rec.get("verified_bullets", []))
                angle_text = rec.get("angle", "")
                with st.container(border=True):
                    info_cols = st.columns([8, 2, 2])
                    info_cols[0].markdown(f"**{t['title']}**")
                    badge_parts = []
                    if bullet_count:
                        badge_parts.append(f"{bullet_count} data points")
                    if angle_text:
                        badge_parts.append(f"angle: {angle_text[:40]}")
                    if badge_parts:
                        info_cols[0].caption(" · ".join(badge_parts))
                    else:
                        info_cols[0].caption(t.get("description", "")[:80])
                    if info_cols[1].button(
                        "Resume",
                        key=f"load_{rec['topic_id']}",
                        type="primary",
                        use_container_width=True,
                    ):
                        st.session_state.selected_topic = rec["topic"]
                        st.session_state.topic_id = rec["topic_id"]
                        st.session_state.research_text = rec.get("research_text", "")
                        st.session_state.research_facts = rec.get("research_facts", [])
                        st.session_state.verified_bullets = rec.get("verified_bullets", [])
                        st.session_state.angle = rec.get("angle", "")
                        st.session_state.user_facts = rec.get("user_facts", "")
                        st.session_state.chart_image_paths = rec.get("chart_image_paths", [])
                        st.session_state.chart_analyses = rec.get("chart_analyses", [])
                        # Jump to the furthest meaningful step
                        if rec.get("verified_bullets"):
                            st.session_state.step = 3
                        else:
                            st.session_state.step = 2
                        st.rerun()
                    if info_cols[2].button(
                        "Delete",
                        key=f"del_{rec['topic_id']}",
                        use_container_width=True,
                    ):
                        delete_topic(rec["topic_id"])
                        st.rerun()
        st.divider()

    research_mode = st.radio(
        "How do you want to find a topic?",
        ["Latest News", "Custom Topic", "Chart Analysis"],
        horizontal=True,
        help="Choose 'Latest News' to research trending stories, 'Custom Topic' to provide your own subject, or 'Chart Analysis' to upload charts for AI interpretation.",
    )

    if research_mode == "Chart Analysis":
        st.write("Upload one or more chart images. Claude will analyze them, extract data, and suggest topics.")
        uploaded_charts = st.file_uploader(
            "Upload chart images",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="Upload screenshots or exported charts (PNG, JPG, WEBP). Multiple files supported.",
        )
        chart_context = st.text_input(
            "Context (optional)",
            placeholder="e.g. These are Tesla stock charts from Q4 2025",
            help="Help Claude understand what the charts are about.",
        )
        research_btn = st.button(
            "Analyze Charts",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_charts,
        )
        custom_topic = ""
    elif research_mode == "Custom Topic":
        st.write("Enter a topic and we'll research it across news sources, then suggest 10 angles.")
        custom_topic = st.text_input(
            "Topic to research",
            placeholder="e.g. Tesla earnings Q4, Bitcoin ETF inflows, Fed interest rate decision",
        )
        research_btn = st.button(
            "Research This Topic",
            type="primary",
            use_container_width=True,
            disabled=not custom_topic,
        )
        uploaded_charts = None
    else:
        st.write("We'll research the latest trends and suggest 10 topics for your slide deck.")
        custom_topic = ""
        uploaded_charts = None
        research_btn = st.button("Research Topics", type="primary", use_container_width=True)

    if research_btn:
        if demo_mode:
            # ── Demo: populate with sample data ──
            import time as _time
            with st.spinner("Loading demo data..."):
                _time.sleep(0.5)
            st.session_state.research_text = "Demo research data for Tesla Q4 deliveries."
            st.session_state.research_facts = [
                {"fact": b["bullet"], "source": b["source"]}
                for b in _DEMO_BULLETS[:10]
            ]
            st.session_state.topic_options = list(_DEMO_TOPICS)
            st.toast("Demo mode: loaded 5 sample topics")
            st.rerun()

        _require_api_key()

        # ── Chart Analysis Mode ──────────────────────────────────────
        if research_mode == "Chart Analysis" and uploaded_charts:
            chart_bytes_list = [f.read() for f in uploaded_charts]

            with st.spinner(f"Analyzing {len(chart_bytes_list)} chart(s) with Claude..."):
                chart_result = analyze_charts(chart_bytes_list, context=chart_context or "")
                research_text = chart_result["research_text"]
                research_facts = chart_result["research_facts"]
                chart_analyses = chart_result["chart_analyses"]

                st.session_state.research_text = research_text
                st.session_state.research_facts = research_facts
                st.session_state.chart_analyses = chart_analyses

                st.toast(f"Extracted {len(research_facts)} facts from {len(chart_analyses)} chart(s)")

            # Save chart images to disk for later use in video
            chart_dir = os.path.join("saved_topics", "charts")
            os.makedirs(chart_dir, exist_ok=True)
            import hashlib
            chart_paths = []
            for i, img_bytes in enumerate(chart_bytes_list):
                ext = uploaded_charts[i].name.rsplit(".", 1)[-1] if "." in uploaded_charts[i].name else "png"
                h = hashlib.sha256(img_bytes).hexdigest()[:12]
                chart_path = os.path.join(chart_dir, f"chart_{h}.{ext}")
                with open(chart_path, "wb") as fp:
                    fp.write(img_bytes)
                chart_paths.append(chart_path)
            st.session_state.chart_image_paths = chart_paths

            # Additional web verification of chart-derived facts
            with st.spinner("Verifying chart data with web search..."):
                # Use the combined narrative as a search query
                verify_query = " ".join(
                    c.get("subject", "") for c in chart_analyses
                )[:100]
                if verify_query.strip():
                    web_results = search_claim(verify_query + " finance", max_results=5)
                    if web_results:
                        lines = ["\n=== Web Verification ===\n"]
                        for idx, r in enumerate(web_results, 1):
                            lines.append(f"{idx}. [{r['source']}] {r['title']}")
                            if r["summary"]:
                                lines.append(f"   {r['summary'][:200]}")
                        research_text += "\n".join(lines)
                        st.session_state.research_text = research_text
                        st.toast(f"Found {len(web_results)} verification sources")

            # Suggest topics based on chart analysis
            try:
                with st.spinner("Generating topic suggestions from charts..."):
                    topic_options = suggest_topics(research_text, audience)
                    st.session_state.topic_options = topic_options
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
                st.stop()
            except anthropic.APIError as exc:
                st.error(f"API error: {exc}")
                st.stop()

            st.rerun()

        # ── News / Custom Topic Mode ─────────────────────────────────
        elif research_mode in ("Latest News", "Custom Topic"):
            with st.spinner(
                f"Researching '{custom_topic}'..."
                if custom_topic
                else "Fetching latest trends (last 48 hours only)..."
            ):
                research_parts = []

                if custom_topic:
                    custom_news = fetch_news_topics([custom_topic], max_per_topic=10)
                    if custom_news:
                        st.toast(f"Found {len(custom_news)} articles about '{custom_topic}'")
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
                        st.toast(f"Found {len(web_results)} additional web results")

                    if not research_parts:
                        st.error(f"No results found for '{custom_topic}'. Try a different query.")
                        st.stop()
                else:
                    if "news" in sources:
                        news_items = fetch_news_topics(topics)
                        if news_items:
                            st.toast(f"Found {len(news_items)} recent articles, fact checking...")

                            raw_news = format_news_for_prompt(news_items)
                            try:
                                verdicts = fact_check_news(raw_news)
                                corrections = {
                                    v["index"]: v for v in verdicts
                                    if v.get("status") == "corrected"
                                }
                                for v_idx, verdict in corrections.items():
                                    if 1 <= v_idx <= len(news_items):
                                        item = news_items[v_idx - 1]
                                        item.title = verdict.get("corrected_title", item.title)
                                        item.summary = verdict.get("corrected_summary", item.summary)

                                if corrections:
                                    st.toast(
                                        f"Corrected {len(corrections)} article(s), "
                                        f"all {len(news_items)} now factual"
                                    )
                                else:
                                    st.toast(f"All {len(news_items)} articles verified")
                            except Exception:
                                st.toast("Fact check unavailable, using articles as is")

                            research_parts.append(format_news_for_prompt(news_items))
                        else:
                            st.toast("No news articles found in the last 48 hours")

                    if "reddit" in sources:
                        reddit_posts = fetch_reddit_topics(subreddits)
                        research_parts.append(format_reddit_for_prompt(reddit_posts))
                        st.toast(f"Found {len(reddit_posts)} Reddit posts")

                research_text = "\n\n".join(research_parts)
                empty_markers = {"No news articles found.", "No Reddit posts found."}

                if not research_parts or all(p.strip() in empty_markers for p in research_parts):
                    st.error("No research data found. Check your network connection and config.")
                    st.stop()

                st.session_state.research_text = research_text

            try:
                with st.spinner("Extracting structured facts from research..."):
                    research_facts = extract_news_facts(research_text)
                    st.session_state.research_facts = research_facts
                    st.toast(f"Extracted {len(research_facts)} verifiable facts")
            except Exception:
                st.toast("Facts extraction unavailable, continuing without grounding")
                st.session_state.research_facts = []

            try:
                with st.spinner("Generating topic suggestions..."):
                    topic_options = suggest_topics(research_text, audience)
                    st.session_state.topic_options = topic_options
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
                st.stop()
            except anthropic.APIError as exc:
                st.error(f"API error: {exc}")
                st.stop()

            st.rerun()

    # ── Card-based topic selection ──────────────────────────────────────────
    if st.session_state.topic_options:
        st.subheader("Select a topic")
        topic_options = st.session_state.topic_options

        for i, t in enumerate(topic_options):
            with st.container(border=True):
                card_cols = st.columns([10, 2])
                card_cols[0].markdown(f"**{t['title']}**")
                card_cols[0].caption(t["description"])
                if card_cols[1].button(
                    "Select",
                    key=f"topic_{i}",
                    type="primary" if i == 0 else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_topic = t
                    tid = save_topic(
                        t,
                        research_text=st.session_state.research_text,
                        research_facts=st.session_state.research_facts,
                        chart_image_paths=st.session_state.chart_image_paths or None,
                        chart_analyses=st.session_state.chart_analyses or None,
                    )
                    st.session_state.topic_id = tid
                    st.session_state.step = 2
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Consolidate Data (20+ verified bullet points)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("Step 2: Verified Data")
    topic = st.session_state.selected_topic
    st.info(f"**Topic:** {topic['title']}  \n{topic['description']}")

    if not st.session_state.verified_bullets:
        if demo_mode:
            import time as _time
            with st.spinner("Loading demo verified data..."):
                _time.sleep(0.3)
            st.session_state.verified_bullets = list(_DEMO_BULLETS)
            st.rerun()

        _require_api_key()
        try:
            with st.spinner("Consolidating all relevant data and verifying (web search + research)..."):
                bullets = consolidate_topic_data(
                    research_text=st.session_state.research_text,
                    research_facts=st.session_state.research_facts,
                    topic=topic,
                    audience=audience,
                )
                st.session_state.verified_bullets = bullets
                tid = save_topic(
                    topic,
                    verified_bullets=bullets,
                    research_text=st.session_state.research_text,
                    research_facts=st.session_state.research_facts,
                )
                st.session_state.topic_id = tid
                st.rerun()
        except anthropic.APIError as exc:
            st.error(f"API error: {exc}")
            st.stop()

    bullets = st.session_state.verified_bullets
    st.success(f"Found {len(bullets)} verified data points for this topic.")

    high_conf = [b for b in bullets if b.get("confidence") == "high"]
    med_conf = [b for b in bullets if b.get("confidence") != "high"]

    if high_conf:
        st.subheader(f"High confidence ({len(high_conf)})")
        for b in high_conf:
            st.markdown(f"- **{b['bullet']}** , _source: {b.get('source', 'unknown')}_")

    if med_conf:
        st.subheader(f"Medium confidence ({len(med_conf)})")
        for b in med_conf:
            st.markdown(f"- {b['bullet']} , _source: {b.get('source', 'unknown')}_")

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.verified_bullets = []
        st.session_state.step = 1
        st.rerun()

    if col_next.button("Continue to Angle", type="primary"):
        st.session_state.step = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Provide Angle & Additional Data
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("Step 3: Your Angle & Data")
    topic = st.session_state.selected_topic
    bullets = st.session_state.verified_bullets

    st.info(
        f"**Topic:** {topic['title']}  \n"
        f"**Data points:** {len(bullets)} verified"
    )

    st.write("Choose how to add your perspective (this will help generate better hooks):")

    tab_angle, tab_data = st.tabs(["Provide an Angle", "Add Your Own Data"])

    with tab_angle:
        st.write(
            "Describe your angle and we'll research it, verify all data is factual, "
            "and ensure at least 20 verified bullet points before building slides."
        )
        angle = st.text_area(
            "Your angle",
            value=st.session_state.angle,
            placeholder="e.g. 'Focus on how beginners can take advantage of this trend, include 3 actionable steps'",
            height=120,
            key="angle_input",
        )

        if st.button("Research & Verify This Angle", disabled=not angle.strip()):
            st.session_state.angle = angle.strip()

            if demo_mode:
                st.session_state["angle_verified"] = True
                st.toast("Demo mode: angle accepted without API verification")
                st.rerun()

            _require_api_key()

            with st.spinner("Searching for data on your angle..."):
                try:
                    new_bullets = research_angle(
                        topic=topic,
                        angle=angle.strip(),
                        existing_bullets=bullets,
                    )
                    if new_bullets:
                        combined = bullets + new_bullets
                        st.toast(f"Found {len(new_bullets)} additional data points for your angle")
                    else:
                        combined = bullets
                        st.toast("No additional data found from angle search")
                except Exception as e:
                    combined = bullets
                    st.toast(f"Angle research failed: {e}, using existing data")

            if len(combined) < 20:
                with st.spinner(
                    f"Only {len(combined)} data points, researching more to reach 20+..."
                ):
                    try:
                        extra_bullets = consolidate_topic_data(
                            research_text=st.session_state.research_text,
                            research_facts=[
                                {"fact": b["bullet"], "source": b.get("source", "")}
                                for b in combined
                            ],
                            topic={
                                "title": topic["title"],
                                "description": f"{topic['description']}, angle: {angle.strip()}",
                            },
                            audience=audience,
                        )
                        existing_texts = {b["bullet"].lower() for b in combined}
                        for eb in extra_bullets:
                            if eb["bullet"].lower() not in existing_texts:
                                combined.append(eb)
                                existing_texts.add(eb["bullet"].lower())
                        st.toast(f"Consolidated to {len(combined)} total data points")
                    except Exception as e:
                        st.toast(f"Extra consolidation failed: {e}")

            st.session_state.verified_bullets = combined
            st.session_state["angle_verified"] = True
            save_topic(
                topic,
                verified_bullets=combined,
                research_text=st.session_state.research_text,
                research_facts=st.session_state.research_facts,
                angle=angle.strip(),
            )
            st.rerun()

    with tab_data:
        st.write("Paste factual data you know to be true. One fact per line.")
        user_facts = st.text_area(
            "Your factual data",
            value=st.session_state.user_facts,
            placeholder="Oil is at $66 per barrel as of today\nThe Fed held rates at 4.5% in January\nTesla delivered 495,000 vehicles in Q4",
            height=160,
            key="user_facts_input",
        )

        if st.button("Add My Data", disabled=not user_facts.strip()):
            st.session_state.user_facts = user_facts.strip()
            user_bullets = []
            for line in user_facts.strip().split("\n"):
                line = line.strip()
                if line:
                    user_bullets.append({
                        "bullet": line,
                        "value": line,
                        "source": "user-provided",
                        "confidence": "high",
                    })
            if user_bullets:
                st.session_state.verified_bullets = bullets + user_bullets
                st.toast(f"Added {len(user_bullets)} user-provided data points")
                save_topic(
                    topic,
                    verified_bullets=bullets + user_bullets,
                    research_text=st.session_state.research_text,
                    research_facts=st.session_state.research_facts,
                    angle=st.session_state.angle,
                    user_facts=user_facts.strip(),
                )
            st.rerun()

    current_bullets = st.session_state.verified_bullets
    bullet_count = len(current_bullets)

    if bullet_count < 20:
        st.warning(
            f"Only {bullet_count} data points, at least 20 recommended. "
            "Research an angle or add your own data above."
        )
    else:
        st.success(f"{bullet_count} verified data points ready.")

    with st.expander(f"Current data pool ({bullet_count} points)", expanded=False):
        for b in current_bullets:
            src = b.get("source", "unknown")
            conf = b.get("confidence", "medium")
            icon = "**" if conf == "high" else ""
            st.markdown(f"- {icon}{b['bullet']}{icon}, _{src}_")

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.step = 2
        st.rerun()

    angle_verified = st.session_state.get("angle_verified", False)
    has_user_facts = bool(st.session_state.user_facts)
    can_proceed = angle_verified or has_user_facts or bullet_count >= 20

    if col_next.button(
        "Continue to Hooks",
        type="primary",
        disabled=not can_proceed,
        help="Research an angle or add your data first" if not can_proceed else "",
    ):
        if not st.session_state.angle:
            st.session_state.angle = ""
        st.session_state.hook_options = []
        st.session_state.selected_hook = None
        st.session_state.step = 4
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Choose a Hook → Generate & Verify Slides (merged)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 4: Hook & Slides")
    topic = st.session_state.selected_topic
    bullets = st.session_state.verified_bullets
    current_angle = st.session_state.angle
    research_text = st.session_state.research_text

    angle_display = f"**Angle:** {current_angle}  \n" if current_angle else ""
    st.info(
        f"**Topic:** {topic['title']}  \n"
        f"{angle_display}"
        f"**Data points:** {len(bullets)} verified"
    )

    # ── Generate hooks if not yet available ────────────────────────────────
    if not st.session_state.hook_options:
        if demo_mode:
            import time as _time
            with st.spinner("Loading demo hooks..."):
                _time.sleep(0.3)
            st.session_state.hook_options = list(_DEMO_HOOKS)
            st.rerun()

        _require_api_key()
        try:
            with st.spinner("Generating hooks grounded in your verified data..."):
                hooks = generate_hooks(
                    topic=topic["title"],
                    verified_bullets=bullets,
                    tone=tone,
                    audience=audience,
                    angle=current_angle,
                )
                st.session_state.hook_options = hooks
                st.rerun()
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
            st.stop()
        except anthropic.APIError as exc:
            st.error(f"API error: {exc}")
            st.stop()

    # ── Hook selection (only show if slides not yet generated) ─────────────
    if not st.session_state.selected_hook:
        hooks = st.session_state.hook_options
        st.subheader("Pick the hook for your opening slide")
        st.caption("Sorted by best fit to your data. Selecting a hook immediately generates slides.")

        if st.button("Back", key="hook_back"):
            st.session_state.hook_options = []
            st.session_state.selected_hook = None
            st.session_state.step = 3
            st.rerun()

        for i, h in enumerate(hooks):
            fit = h.get("fit_score", 0)
            with st.container(border=True):
                hook_cols = st.columns([1, 9, 2])

                # Fit score badge
                fit_color = "#22c55e" if fit >= 8 else "#f59e0b" if fit >= 5 else "#ef4444"
                hook_cols[0].markdown(
                    f"<div style='text-align:center;padding:8px 0;'>"
                    f"<span style='font-size:24px;font-weight:700;color:{fit_color};'>{fit}</span>"
                    f"<br><span style='font-size:11px;color:#888;'>/10</span></div>",
                    unsafe_allow_html=True,
                )

                # Hook text + metadata
                hook_cols[1].markdown(f"**{h['hook']}**")
                meta_parts = [f"Style: {h['style']}"]
                if h.get("data_used"):
                    meta_parts.append(f"Data: {h['data_used']}")
                hook_cols[1].caption(" | ".join(meta_parts))

                # Select button — picking a hook triggers slide generation
                if hook_cols[2].button(
                    "Use This",
                    key=f"hook_{i}",
                    type="primary" if i == 0 else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.selected_hook = h
                    st.rerun()

    # ── Hook selected → generate slides (or show existing) ────────────────
    if st.session_state.selected_hook:
        hook = st.session_state.selected_hook

        st.info(
            f"**Hook:** {hook['hook']}  \n"
            f"**Data points:** {len(bullets)} verified"
        )

    # If slides already exist (navigated back), show summary
    if st.session_state.slides:
        st.success(f"Slides already generated ({len(st.session_state.slides)} slides).")

        coherence_report = st.session_state.coherence_report
        if coherence_report:
            score = coherence_report.get("coherence_score", "?")
            st.metric("Coherence Score", f"{score}/10")

        gen_cols = st.columns(2)
        if gen_cols[0].button("Regenerate Slides", type="secondary", use_container_width=True):
            st.session_state.slides = []
            st.session_state.fact_check_report = []
            st.session_state.conclusion_report = None
            st.session_state.coherence_report = None
            st.session_state.tiktok_metadata = None
            # Free Studio assets from previous generation
            st.session_state.ai_image_paths = []
            st.session_state.png_paths = []
            st.session_state.mcp_alternatives = []
            st.session_state.video_path = None
            st.session_state.video_scripts = []
            _read_file_bytes.clear()
            _build_zip_bytes.clear()
            gc.collect()
            st.rerun()
        if gen_cols[1].button("Continue to Studio", type="primary", use_container_width=True):
            st.session_state.step = 6
            st.rerun()

    else:
        if demo_mode:
            import time as _time

            hook_text = hook["hook"]
            check_names = [
                ("Slide generation", "Creating fact-grounded slides"),
                ("Engagement review", f"Tightening copy ({review_iterations} iterations)"),
                ("Layered fact-check", "Verifying claims against research"),
                ("Web verification", "Cross-checking with live data"),
                ("Conclusion check", "Validating verdict logic"),
                ("Coherence check", "Analyzing narrative flow"),
                ("Final polish", "Value + cohesion pass"),
                ("TikTok metadata", "Generating title & description"),
            ]
            dashboard = st.container()
            placeholders = []
            with dashboard:
                for name, desc in check_names:
                    ph = st.empty()
                    ph.markdown(f"⬜ **{name}** — _{desc}_")
                    placeholders.append(ph)

            demo_results = [
                f"✅ **Slide generation** — {len(_DEMO_SLIDES)} slides created",
                f"✅ **Engagement review** — {review_iterations} passes complete",
                f"✅ **Layered fact-check** — 7/7 claims verified",
                "✅ **Web verification** — All claims current",
                "✅ **Conclusion check** — Logic sound",
                "✅ **Coherence check** — Score: 9/10",
                "✅ **Final polish** — Coherence: 9/10",
                "✅ **TikTok metadata** — Ready",
            ]

            for i, result_text in enumerate(demo_results):
                placeholders[i].markdown(f"🔄 **{check_names[i][0]}** — _Working..._")
                _time.sleep(0.25)
                placeholders[i].markdown(result_text)

            st.session_state.slides = list(_DEMO_SLIDES)
            st.session_state.fact_check_report = list(_DEMO_FACT_REPORT)
            st.session_state.conclusion_report = {
                "logic_valid": True,
                "verdict_slide": 8,
                "issues": [],
            }
            st.session_state.coherence_report = {
                "coherence_score": 9,
                "arc_analysis": "Strong hook → evidence buildup → contrarian verdict. Clean narrative arc.",
                "issues": [],
            }
            st.session_state.tiktok_metadata = dict(_DEMO_METADATA)
            st.session_state.step = 6
            st.rerun()

        _require_api_key()

        try:
            hook_text = hook["hook"]

            # ── Verification Dashboard ────────────────────────────────────
            check_names = [
                ("Slide generation", "Creating fact-grounded slides"),
                ("Engagement review", f"Tightening copy ({review_iterations} iterations)"),
                ("Layered fact-check", "Verifying claims against research"),
                ("Web verification", "Cross-checking with live data"),
                ("Conclusion check", "Validating verdict logic"),
                ("Coherence check", "Analyzing narrative flow"),
                ("Final polish", "Value + cohesion pass"),
                ("TikTok metadata", "Generating title & description"),
            ]

            dashboard = st.container()
            placeholders = []
            with dashboard:
                for name, desc in check_names:
                    ph = st.empty()
                    ph.markdown(f"⬜ **{name}** — _{desc}_")
                    placeholders.append(ph)

            # 1. Generate slides
            placeholders[0].markdown("🔄 **Slide generation** — _Creating fact-grounded slides..._")
            slides = generate_slide_content(
                topic=topic["title"],
                angle=angle or topic["description"],
                hook=hook_text,
                slide_count=slide_count,
                tone=tone,
                audience=audience,
                style_notes=style_notes,
                research_facts=bullets,
            )
            slides = enforce_hook_and_count(slides, hook_text, slide_count)
            placeholders[0].markdown(f"✅ **Slide generation** — {len(slides)} slides created")

            # 2. Engagement review
            if review_iterations > 0:
                placeholders[1].markdown("🔄 **Engagement review** — _Improving copy..._")
                slides = review_and_improve(
                    slides=slides,
                    tone=tone,
                    audience=audience,
                    iterations=review_iterations,
                    hook=hook_text,
                )
                slides = enforce_hook_and_count(slides, hook_text, slide_count)
                placeholders[1].markdown(f"✅ **Engagement review** — {review_iterations} passes complete")
            else:
                placeholders[1].markdown("⏭️ **Engagement review** — _Skipped (0 iterations)_")

            # 3. Layered fact-check
            fact_report = []
            if bullets:
                placeholders[2].markdown("🔄 **Layered fact-check** — _Verifying claims..._")
                fc_result = layered_fact_check(
                    slides, research_text, bullets,
                    topic["title"], angle or topic["description"],
                )
                if not isinstance(fc_result, dict):
                    fc_result = {"corrected_slides": slides}
                layer_a = fc_result.get("layer_a_report", [])
                layer_b = fc_result.get("layer_b_report", [])
                slides = _safe_get_slides(fc_result, slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

                for item in layer_a:
                    fact_report.append({
                        "slide": item.get("slide", "?"),
                        "status": item.get("status", "unknown"),
                        "notes": f"[news-sourced] {item.get('notes', '')}",
                    })
                for item in layer_b:
                    fact_report.append({
                        "slide": item.get("slide", "?"),
                        "status": item.get("status", "unknown"),
                        "notes": f"[supporting data] {item.get('notes', '')}",
                    })

                verified = sum(1 for r in fact_report if r.get("status") == "verified")
                placeholders[2].markdown(f"✅ **Layered fact-check** — {verified}/{len(fact_report)} claims verified")
            else:
                placeholders[2].markdown("⏭️ **Layered fact-check** — _Skipped (no bullets)_")

            # 4. Web search verification
            placeholders[3].markdown("🔄 **Web verification** — _Cross-checking live data..._")
            try:
                ws_result = web_search_fact_check(
                    slides, topic["title"], angle or topic["description"],
                )
                if not isinstance(ws_result, dict):
                    ws_result = {"corrected_slides": slides}
                ws_report = ws_result.get("search_report", [])
                slides = _safe_get_slides(ws_result, slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

                corrected_count = sum(
                    1 for r in ws_report if r.get("status") == "corrected"
                )

                for item in ws_report:
                    fact_report.append({
                        "slide": item.get("slide", "?"),
                        "status": item.get("status", "unknown"),
                        "notes": f"[web search] {item.get('notes', '')}",
                    })

                if corrected_count:
                    placeholders[3].markdown(f"✅ **Web verification** — Corrected {corrected_count} claim(s)")
                else:
                    placeholders[3].markdown("✅ **Web verification** — All claims current")
            except Exception:
                placeholders[3].markdown("⚠️ **Web verification** — _Unavailable, continuing_")

            st.session_state.fact_check_report = fact_report

            # 5. Conclusion validation
            placeholders[4].markdown("🔄 **Conclusion check** — _Validating logic..._")
            conclusion_result = validate_conclusion(
                slides, bullets, topic["title"],
                angle or topic["description"],
            )
            if not isinstance(conclusion_result, dict):
                conclusion_result = {"corrected_slides": slides}
            logic_valid = conclusion_result.get("logic_valid", True)
            if logic_valid:
                placeholders[4].markdown("✅ **Conclusion check** — Logic sound")
            else:
                issues = conclusion_result.get("issues", [])
                placeholders[4].markdown(f"✅ **Conclusion check** — Fixed {len(issues)} logic gap(s)")
            slides = _safe_get_slides(conclusion_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, slide_count)
            # Update verdict_slide to reflect actual position after reorder
            conclusion_result["verdict_slide"] = max(len(slides) - 1, 1)
            st.session_state.conclusion_report = conclusion_result

            # 6. Narrative coherence
            placeholders[5].markdown("🔄 **Coherence check** — _Analyzing narrative flow..._")
            coherence_result = check_narrative_coherence(
                slides, topic["title"],
                angle or topic["description"], hook_text,
            )
            if not isinstance(coherence_result, dict):
                coherence_result = {"corrected_slides": slides}
            st.session_state.coherence_report = coherence_result
            coherence_score = coherence_result.get("coherence_score", 0)
            placeholders[5].markdown(f"✅ **Coherence check** — Score: {coherence_score}/10")
            slides = _safe_get_slides(coherence_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Strip claim tags
            slides = strip_claim_tags(slides)

            # 7. Final polish
            placeholders[6].markdown("🔄 **Final polish** — _Value + cohesion pass..._")
            value_result = add_value_pass(
                slides=slides,
                topic=topic["title"],
                angle=angle or topic["description"],
                audience=audience,
                hook=hook_text,
            )
            if not isinstance(value_result, dict):
                value_result = {"corrected_slides": _safe_get_slides(value_result, slides)}
            slides = _safe_get_slides(value_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, slide_count)
            final_score = value_result.get("coherence_score", coherence_score)
            st.session_state.coherence_report = {
                "coherence_score": final_score,
                "arc_analysis": value_result.get("arc_analysis", ""),
                "issues": value_result.get("issues", []),
            }
            placeholders[6].markdown(f"✅ **Final polish** — Coherence: {final_score}/10")

            # 8. TikTok metadata
            placeholders[7].markdown("🔄 **TikTok metadata** — _Generating title & description..._")
            metadata = generate_tiktok_metadata(
                slides=slides,
                topic=topic["title"],
                angle=angle or topic["description"],
                hook=hook["hook"],
            )
            st.session_state.tiktok_metadata = metadata
            placeholders[7].markdown("✅ **TikTok metadata** — Ready")

            st.session_state.slides = slides
            st.session_state.step = 6
            st.rerun()

        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
            st.stop()
        except anthropic.APIError as exc:
            st.error(f"API error: {exc}")
            st.stop()
        except Exception as exc:
            st.error(f"Slide generation failed: {type(exc).__name__}: {exc}")
            st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Studio (Edit + Export, tabbed interface)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 6:
    st.header("Studio")
    slides = st.session_state.slides
    fact_report = st.session_state.fact_check_report
    metadata = st.session_state.tiktok_metadata

    colors = {
        "background": bg_color,
        "title": title_color,
        "body": body_color,
        "accent": accent_color,
        "highlight": highlight_color,
    }

    # Build tab list (Video-first: most common output right after editing)
    tab_labels = ["Edit", "Video", "AI Images", "Slides"]
    studio_tabs = st.tabs(tab_labels)

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Edit (side-by-side editor + live preview)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[0]:
        st.success(f"{len(slides)} slides ready. Edit below, preview updates live.")

        # ── Reports ───────────────────────────────────────────────────────
        report_cols = st.columns(3)
        with report_cols[0]:
            if fact_report:
                with st.expander("Fact-Check Report"):
                    for item in fact_report:
                        slide_num = item.get("slide", "?")
                        status = item.get("status", "unknown")
                        notes = item.get("notes", "")
                        if status == "verified":
                            st.markdown(f"**Slide {slide_num}**: :green[verified]  \n{notes}")
                        elif status == "corrected":
                            st.markdown(f"**Slide {slide_num}**: :orange[corrected]  \n{notes}")
                        else:
                            st.markdown(f"**Slide {slide_num}**: :red[flagged]  \n{notes}")

        with report_cols[1]:
            conclusion_report = st.session_state.conclusion_report
            if conclusion_report:
                with st.expander("Conclusion Validation"):
                    logic_valid = conclusion_report.get("logic_valid", True)
                    verdict_slide = conclusion_report.get("verdict_slide", "?")
                    if logic_valid:
                        st.markdown(f":green[Verdict (Slide {verdict_slide}) logically follows from evidence]")
                    else:
                        st.markdown(f":orange[Verdict (Slide {verdict_slide}) had logic gaps, corrected]")
                    issues = conclusion_report.get("issues", [])
                    if issues:
                        for issue in issues:
                            st.markdown(f"- :orange[{issue}]")

        with report_cols[2]:
            coherence_report = st.session_state.coherence_report
            if coherence_report:
                with st.expander("Narrative Coherence"):
                    score = coherence_report.get("coherence_score", "?")
                    st.markdown(f"**Coherence Score:** {score}/10")
                    arc = coherence_report.get("arc_analysis", "")
                    if arc:
                        st.markdown(f"**Arc:** {arc}")

        # ── Side-by-side Slide Editor ─────────────────────────────────────
        st.subheader("Slide Editor")
        BODY_MIN, BODY_IDEAL, BODY_MAX = 20, 50, 90
        st.caption(
            f"Body length guide: :red[< {BODY_MIN} too short] · "
            f":green[{BODY_MIN}–{BODY_MAX} ideal] · "
            f":red[> {BODY_MAX} too long]"
        )

        total_slides = len(slides)
        for i, slide in enumerate(slides):
            col_edit, col_preview = st.columns([1, 1], gap="medium")

            with col_edit:
                st.markdown(f"**Slide {i + 1}**")
                title_val = st.text_input(
                    "Title",
                    value=slide.get("title", ""),
                    key=f"edit_title_{i}",
                    label_visibility="collapsed",
                    placeholder="Slide title",
                )
                body_val = st.text_area(
                    "Body",
                    value=slide.get("body", ""),
                    key=f"edit_body_{i}",
                    height=68,
                    label_visibility="collapsed",
                    placeholder="Slide body text",
                )
                footer_val = st.text_input(
                    "Footer",
                    value=slide.get("footer", ""),
                    key=f"edit_footer_{i}",
                    label_visibility="collapsed",
                    placeholder="source: ...",
                )

                char_count = len(body_val)
                bar_value = min(char_count / BODY_MAX, 1.0)
                if char_count < BODY_MIN:
                    label = f":red[{char_count} chars, too short]"
                elif char_count <= BODY_MAX:
                    label = f":green[{char_count} chars]"
                else:
                    label = f":red[{char_count} chars, too long]"
                st.progress(bar_value)
                st.caption(label)

            with col_preview:
                st.markdown(f"**Preview**")
                preview_slide = {"title": title_val, "body": body_val, "footer": footer_val}
                st.markdown(
                    _slide_preview_html(preview_slide, i, total_slides, colors, handle),
                    unsafe_allow_html=True,
                )

            if i < total_slides - 1:
                st.divider()

        # ── TikTok Metadata Editing ───────────────────────────────────────
        if metadata:
            st.subheader("TikTok Post Copy")
            tiktok_title = st.text_input(
                "Video Title", value=metadata.get("title", ""), key="tiktok_title"
            )
            tiktok_desc = st.text_area(
                "Description", value=metadata.get("description", ""),
                height=160, key="tiktok_desc"
            )
            char_count = len(tiktok_desc)
            if char_count >= 200:
                st.caption(f":green[{char_count} characters] (meets 200+ requirement)")
            else:
                st.caption(f":red[{char_count} characters] (below 200 minimum)")

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Slides (PPTX + PNG + Alternatives)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[3]:
        st.subheader("Export Carousel Slides")
        st.caption("Build static slide decks as PowerPoint or PNG images.")

        live_slides = _get_live_slides()

        build_cols = st.columns(2)

        # Build PPTX
        if build_cols[0].button("Build PPTX", type="primary", use_container_width=True, key="build_pptx"):
            with st.spinner("Building PowerPoint..."):
                filepath = build_pptx(
                    slides=live_slides,
                    colors=colors,
                    aspect_ratio=aspect_ratio_val,
                    output_dir="./output",
                    handle=handle,
                )
                st.session_state.pptx_path = filepath
            st.rerun()

        # Build PNGs
        if build_cols[1].button("Build PNGs", type="primary", use_container_width=True, key="build_pngs"):
            with st.spinner("Rendering PNG slides..."):
                png_paths = build_pngs(
                    slides=live_slides,
                    colors=colors,
                    aspect_ratio=aspect_ratio_val,
                    output_dir="./output",
                    handle=handle,
                )
                st.session_state.png_paths = png_paths
            st.rerun()

        # Downloads
        filepath = st.session_state.pptx_path
        png_paths = st.session_state.png_paths

        if filepath:
            st.divider()
            pptx_bytes = _cached_read(filepath)
            st.download_button(
                label="Download PPTX",
                data=pptx_bytes,
                file_name=os.path.basename(filepath),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True,
                key="dl_pptx",
            )

        if png_paths:
            st.divider()
            # Show images in grid
            img_cols = st.columns(min(len(png_paths), 3))
            for i, p in enumerate(png_paths):
                col = img_cols[i % 3]
                col.image(p, caption=f"Slide {i + 1}", use_container_width=True)

            # ZIP download
            zip_bytes = _cached_zip(png_paths)
            st.download_button(
                label="Download All PNGs (ZIP)",
                data=zip_bytes,
                file_name="slides.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
                key="dl_png_zip",
            )

            # Individual downloads
            with st.expander("Individual slide downloads"):
                dl_cols = st.columns(min(len(png_paths), 3))
                for i, p in enumerate(png_paths):
                    col = dl_cols[i % 3]
                    col.download_button(
                        label=f"Slide {i + 1}",
                        data=_cached_read(p),
                        file_name=os.path.basename(p),
                        mime="image/png",
                        key=f"png_dl_{i}",
                        use_container_width=True,
                    )

        # ── Alternative Versions ──────────────────────────────────────────
        st.divider()
        st.subheader("Alternative Styles")
        st.caption(
            "Generate multiple design variations with finance-themed backgrounds "
            "(candlestick charts, trend lines, volume bars). "
            "Instant, offline — no API keys needed."
        )
        alt_cols = st.columns([2, 1])
        num_alts = alt_cols[1].selectbox(
            "Number of alternatives",
            [2, 3, 4],
            index=0,
            key="num_alts",
        )
        if alt_cols[0].button(
            "Generate Alternatives",
            type="primary",
            use_container_width=True,
            key="gen_alts",
        ):
            with st.spinner("Generating style alternatives..."):
                try:
                    alts = build_style_alternatives(
                        slides=live_slides,
                        aspect_ratio=aspect_ratio_val,
                        output_dir="./output",
                        handle=handle,
                        num_alternatives=num_alts,
                    )
                    st.session_state.mcp_alternatives = alts
                except Exception as exc:
                    st.error(f"Alternative generation failed: {exc}")
            st.rerun()

        mcp_alts = st.session_state.mcp_alternatives
        if mcp_alts:
            for alt in mcp_alts:
                st.markdown(f"**{alt['version']}** — _{alt['style']}_")
                alt_png_paths = alt.get("png_paths", [])
                if alt_png_paths:
                    alt_img_cols = st.columns(min(len(alt_png_paths), 3))
                    for j, p in enumerate(alt_png_paths):
                        col = alt_img_cols[j % len(alt_img_cols)]
                        col.image(p, caption=f"Slide {j + 1}", use_container_width=True)
                    alt_zip_bytes = _cached_zip(alt_png_paths)
                    st.download_button(
                        label=f"Download {alt['version']} (ZIP)",
                        data=alt_zip_bytes,
                        file_name=f"{alt['version'].replace(' ', '_').lower()}.zip",
                        mime="application/zip",
                        key=f"alt_zip_{alt['version']}",
                        use_container_width=True,
                    )
                st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # TAB: AI Images (Gemini Flash / OpenAI DALL-E 3)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[2]:
        st.subheader("AI-Generated Slide Images")

        if not ai_images_enabled:
            st.info(
                "Add an image generation API key in the sidebar under Integrations "
                "to generate cinematic AI background images for each slide.\n\n"
                "**Gemini Flash** is free — get a key at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
            )
        else:
            provider_label = "Gemini Flash" if google_ai_key else "OpenAI DALL-E 3"
            st.caption(
                f"Generate cinematic AI backgrounds using {provider_label}. "
                "Claude creates visual prompts, the AI generates photorealistic images, "
                "then slide text is composited on top."
            )

            live_slides = _get_live_slides()

            # ── Cinematic Overlay Options ──────────────────────────────
            with st.expander("Cinematic Overlay Options", expanded=False):
                st.markdown(
                    "Add AI-generated cinematic overlays on top of backgrounds for "
                    "professional depth and atmosphere. Inspired by cinema camera "
                    "techniques: bokeh, light leaks, film grain, volumetric lighting."
                )
                overlay_enabled = st.checkbox(
                    "Enable AI cinematic overlays",
                    value=st.session_state.overlays_enabled,
                    key="overlay_toggle",
                    help="Generates a second AI image per slide for atmospheric effects.",
                )
                st.session_state.overlays_enabled = overlay_enabled

                if overlay_enabled:
                    overlay_style_options = list(OVERLAY_STYLE_DESCRIPTIONS.keys())
                    overlay_style_labels = {
                        "auto": "Auto (recommended) — best style per slide",
                        "cinematic_bokeh": "Cinematic Bokeh — shallow DOF, 85mm f/1.2",
                        "light_leak": "Light Leak — warm amber film exposure",
                        "film_noir": "Film Noir — high contrast, deep shadows",
                        "volumetric_light": "Volumetric Light — god rays, atmospheric haze",
                        "neon_glow": "Neon Glow — cyberpunk neon reflections",
                        "golden_hour": "Golden Hour — warm sunset, lens flare",
                        "film_grain": "Film Grain — 35mm Kodak analog texture",
                    }
                    selected_style = st.selectbox(
                        "Overlay style",
                        options=overlay_style_options,
                        format_func=lambda x: overlay_style_labels.get(x, x),
                        index=0,
                        key="overlay_style_select",
                    )
                    st.session_state.overlay_style = selected_style

                    st.caption(
                        "**How it works:** Claude generates cinematic overlay prompts "
                        "based on each slide's content and mood. The AI creates atmospheric "
                        "images (bokeh orbs, light rays, film grain, etc.) which are "
                        "post-processed with blur, color temperature shifts, bloom, and "
                        "edge fading, then composited at ~30-40% opacity to add depth "
                        "without obscuring text. Overlays are strongest at edges and "
                        "subtler in the center where text sits."
                    )
            # ──────────────────────────────────────────────────────────

            if st.button("Generate AI Slide Images", type="primary", use_container_width=True, key="gen_ai_imgs"):
                if google_ai_key:
                    os.environ["GOOGLE_AI_API_KEY"] = google_ai_key
                    os.environ.pop("OPENAI_API_KEY", None)
                elif openai_img_key:
                    os.environ["OPENAI_API_KEY"] = openai_img_key
                    os.environ.pop("GOOGLE_AI_API_KEY", None)
                _require_api_key()

                overlay_prompts_list = None
                effective_overlay_style = st.session_state.get("overlay_style", "auto")

                if st.session_state.overlays_enabled:
                    with st.spinner("Generating cinematic overlay prompts..."):
                        try:
                            overlay_prompts_list = generate_overlay_prompts(
                                slides=live_slides,
                                topic=st.session_state.selected_topic["title"] if st.session_state.selected_topic else "",
                                angle=st.session_state.get("angle", ""),
                                overlay_style=effective_overlay_style,
                            )
                            st.session_state.ai_overlay_prompts = overlay_prompts_list
                        except Exception as exc:
                            st.warning(f"Overlay prompt generation failed (will proceed without overlays): {exc}")
                            overlay_prompts_list = None

                spinner_msg = "Generating image prompts and AI images"
                if overlay_prompts_list:
                    spinner_msg += " + cinematic overlays"
                spinner_msg += "... This may take 30-60s."

                with st.spinner(spinner_msg):
                    try:
                        image_prompts = generate_image_prompts(
                            slides=live_slides,
                            topic=st.session_state.selected_topic["title"] if st.session_state.selected_topic else "",
                            angle=st.session_state.get("angle", ""),
                        )
                        ai_paths = generate_slide_images(
                            slides=live_slides,
                            image_prompts=image_prompts,
                            colors=colors,
                            aspect_ratio=aspect_ratio_val,
                            output_dir="./output",
                            handle=handle,
                            overlay_prompts=overlay_prompts_list,
                            overlay_style=effective_overlay_style,
                        )
                        st.session_state.ai_image_paths = ai_paths
                        st.session_state.ai_image_prompts = image_prompts
                        gc.collect()
                        st.rerun()
                    except Exception as exc:
                        st.error(f"AI image generation failed: {exc}")
                        import traceback
                        with st.expander("Full error details"):
                            st.code(traceback.format_exc())

            if st.session_state.ai_image_paths:
                ai_img_cols = st.columns(min(len(st.session_state.ai_image_paths), 3))
                for i, p in enumerate(st.session_state.ai_image_paths):
                    col = ai_img_cols[i % 3]
                    col.image(p, caption=f"Slide {i + 1}", use_container_width=True)

                with st.expander("Image Prompts"):
                    for i, prompt in enumerate(st.session_state.ai_image_prompts):
                        st.markdown(f"**Slide {i + 1}:** {prompt}")

                if st.session_state.ai_overlay_prompts:
                    with st.expander("Cinematic Overlay Prompts"):
                        for i, slide_prompts in enumerate(st.session_state.ai_overlay_prompts):
                            if isinstance(slide_prompts, list):
                                st.markdown(f"**Slide {i + 1}** ({len(slide_prompts)} overlays):")
                                for j, p in enumerate(slide_prompts):
                                    st.markdown(f"  - Sentence {j + 1}: {p}")
                            else:
                                st.markdown(f"**Slide {i + 1}:** {slide_prompts}")

                ai_zip_bytes = _cached_zip(st.session_state.ai_image_paths)
                st.download_button(
                    label="Download AI Slides (ZIP)",
                    data=ai_zip_bytes,
                    file_name="ai_slides.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                    key="ai_slides_zip_dl",
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Video (Carousel Narration + TikTok Script, unified)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[1]:
        # ── Mode selector ─────────────────────────────────────────────
        video_mode = st.radio(
            "Video type",
            ["Carousel Narration", "TikTok Script"],
            horizontal=True,
            key="video_mode_selector",
            help="Carousel: narrated slide-by-slide deck. TikTok: 130-160 word short-form script.",
        )

        # ══════════════════════════════════════════════════════════════
        # MODE: Carousel Narration (multi-slide narrated video)
        # ══════════════════════════════════════════════════════════════
        if video_mode == "Carousel Narration":
            st.subheader("Carousel Narration")

            if not video_enabled:
                st.info(
                    "Add your ElevenLabs API Key in the sidebar under Integrations "
                    "to generate narrated MP4 videos with AI voiceover."
                )
            else:
                st.caption(
                    "Generate a narrated MP4 video with AI voiceover (ElevenLabs). "
                    "Each slide is read aloud with a natural script generated by Claude."
                )

            if st.session_state.get("chart_image_paths"):
                n_charts = len(st.session_state.chart_image_paths)
                st.caption(
                    f"Chart Analysis mode: {n_charts} chart(s) will be used as overlays "
                    "on relevant slides with predetermined Ken Burns backgrounds (no image API needed)."
                )
            elif ai_images_enabled:
                ai_provider = "Gemini Flash" if google_ai_key else "DALL-E 3"
                st.caption(
                    f"Slides use AI-generated cinematic realistic backgrounds ({ai_provider})."
                )
            else:
                provider_hint = "Google Images"
                if pexels_key:
                    provider_hint = "Pexels"
                elif pixabay_key:
                    provider_hint = "Pixabay"
                st.caption(
                    f"Slides use web image backgrounds ({provider_hint}). "
                    "Add a Google AI or OpenAI key for AI-generated images."
                )

            live_slides = _get_live_slides()

            # Detect stale data: if slides changed since scripts/mapping were generated
            import hashlib as _hashlib
            _slides_hash = _hashlib.md5(str(live_slides).encode()).hexdigest()
            _slides_changed = (
                st.session_state.get("_video_scripts_slides_hash")
                and st.session_state["_video_scripts_slides_hash"] != _slides_hash
            )
            if _slides_changed:
                if st.session_state.video_scripts:
                    st.warning("Slides have changed since scripts were generated. Regenerate scripts to match.")
                # Invalidate chart mapping when slide content changes
                if st.session_state.get("chart_slide_mapping"):
                    st.session_state.chart_slide_mapping = []

            # ── Step 1: Generate voiceover script ──────────────────────────
            st.markdown("---")
            st.markdown("**Step 1 — Voiceover Script**")

            gen_col, reset_col = st.columns([3, 1])
            with gen_col:
                if st.button("Generate Script", type="primary", use_container_width=True, key="gen_script"):
                    _require_api_key()
                    with st.spinner("Generating voiceover scripts..."):
                        try:
                            topic = st.session_state.selected_topic["title"] if st.session_state.selected_topic else ""
                            angle = st.session_state.get("angle", "")
                            scripts = generate_video_script(
                                slides=live_slides, topic=topic, angle=angle,
                            )
                            st.session_state.video_scripts = scripts
                            st.session_state["_video_scripts_slides_hash"] = _slides_hash
                            # Clear previous video when regenerating scripts
                            st.session_state.video_path = None
                            st.session_state.video_search_queries = []
                            st.session_state.video_search_results = {}
                        except Exception as exc:
                            st.error(f"Script generation failed: {exc}")
                    st.rerun()
            with reset_col:
                if st.session_state.video_scripts and st.button(
                    "Clear", use_container_width=True, key="clear_script"
                ):
                    st.session_state.video_scripts = []
                    st.session_state.video_path = None
                    st.rerun()

            # Show editable script text areas
            if st.session_state.video_scripts:
                st.caption("Edit the scripts below before building the video.")
                edited_scripts = []
                for i, script in enumerate(st.session_state.video_scripts):
                    edited = st.text_area(
                        f"Slide {i + 1}",
                        value=script,
                        height=80,
                        key=f"script_edit_{i}",
                    )
                    edited_scripts.append(edited)

                # ── Feedback & Regenerate ──────────────────────────────
                st.markdown("---")
                st.markdown("**Feedback — Regenerate Script**")
                st.caption(
                    "Provide feedback on the scripts above and click Regenerate "
                    "to get improved scripts before building the video."
                )
                script_feedback = st.text_area(
                    "Your feedback",
                    placeholder="e.g. Make the hook more dramatic, use a more casual tone, shorten slide 3...",
                    height=80,
                    key="script_feedback",
                )
                if st.button(
                    "Regenerate Script with Feedback",
                    use_container_width=True,
                    key="regen_script",
                    disabled=not script_feedback.strip(),
                ):
                    _require_api_key()
                    with st.spinner("Regenerating scripts with your feedback..."):
                        try:
                            topic = st.session_state.selected_topic["title"] if st.session_state.selected_topic else ""
                            angle = st.session_state.get("angle", "")
                            new_scripts = regenerate_video_script(
                                slides=live_slides,
                                current_scripts=edited_scripts,
                                feedback=script_feedback,
                                topic=topic,
                                angle=angle,
                            )
                            st.session_state.video_scripts = new_scripts
                            st.session_state["_video_scripts_slides_hash"] = _slides_hash
                            st.session_state.video_path = None
                            st.session_state.video_search_queries = []
                            st.session_state.video_search_results = {}
                        except Exception as exc:
                            st.error(f"Script regeneration failed: {exc}")
                    st.rerun()

                # ── Step 2: Build video with edited scripts ────────────
                st.markdown("---")
                st.markdown("**Step 2 — Build Video**")

                if st.button("Build Narrated Video", type="primary", use_container_width=True, key="build_video"):
                    os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
                    if pexels_key:
                        os.environ["PEXELS_API_KEY"] = pexels_key
                    if pixabay_key:
                        os.environ["PIXABAY_API_KEY"] = pixabay_key
                    if openai_img_key:
                        os.environ["OPENAI_API_KEY"] = openai_img_key
                    if google_ai_key:
                        os.environ["GOOGLE_AI_API_KEY"] = google_ai_key

                    # Save edits back to session state
                    st.session_state.video_scripts = edited_scripts

                    # Determine video build mode
                    use_ai = ai_images_enabled
                    has_charts = bool(st.session_state.get("chart_image_paths"))
                    if has_charts:
                        spinner_msg = "Building video with chart overlays..."
                    elif use_ai:
                        spinner_msg = "Generating AI images and building video..."
                    else:
                        spinner_msg = "Searching for images and building video..."

                    video_success = False
                    with st.spinner(spinner_msg):
                        try:
                            _topic = st.session_state.selected_topic["title"] if st.session_state.selected_topic else ""
                            _angle = st.session_state.get("angle", "")

                            _require_api_key()

                            # Generate cinematic overlays if enabled
                            cinematic_overlay_images = None
                            if st.session_state.overlays_enabled:
                                img_w = 1080 if aspect_ratio_val == "9:16" else 1920
                                img_h = 1920 if aspect_ratio_val == "9:16" else 1080
                                effective_overlay_style = st.session_state.get("overlay_style", "auto")
                                auto_styles = {
                                    "hook": "volumetric_light",
                                    "context": "cinematic_bokeh",
                                    "payoff": "light_leak",
                                    "cta": "golden_hour",
                                }
                                try:
                                    overlay_prompts_list = generate_overlay_prompts(
                                        slides=live_slides,
                                        topic=_topic,
                                        angle=_angle,
                                        overlay_style=effective_overlay_style,
                                    )
                                    cinematic_overlay_images = []
                                    for si, slide_prompts in enumerate(overlay_prompts_list):
                                        role = get_slide_role(si, len(live_slides))
                                        style = auto_styles.get(role, "cinematic_bokeh") if effective_overlay_style == "auto" else effective_overlay_style
                                        slide_overlays = []
                                        prompts = slide_prompts if isinstance(slide_prompts, list) else [slide_prompts]
                                        for prompt in prompts:
                                            try:
                                                import time as _t
                                                _t.sleep(1)
                                                overlay_img = generate_ai_overlay(
                                                    prompt=prompt,
                                                    width=img_w,
                                                    height=img_h,
                                                    style=style,
                                                    role=role,
                                                )
                                                slide_overlays.append(overlay_img)
                                            except Exception:
                                                slide_overlays.append(None)
                                        cinematic_overlay_images.append(slide_overlays)
                                except Exception as exc:
                                    st.warning(f"Overlay generation failed (proceeding without): {exc}")
                                    cinematic_overlay_images = None

                            # Check if chart images are available
                            _chart_paths = st.session_state.get("chart_image_paths", [])
                            _chart_analyses = st.session_state.get("chart_analyses", [])

                            if _chart_paths:
                                # Chart Analysis mode: predetermined backgrounds + chart overlays
                                # Map charts to slides
                                if not st.session_state.get("chart_slide_mapping"):
                                    chart_mapping = map_charts_to_slides(
                                        slides=live_slides,
                                        chart_analyses=_chart_analyses,
                                        num_charts=len(_chart_paths),
                                    )
                                    st.session_state.chart_slide_mapping = chart_mapping
                                else:
                                    chart_mapping = st.session_state.chart_slide_mapping

                                web_result = build_video_with_chart_overlays(
                                    slides=live_slides,
                                    scripts=edited_scripts,
                                    chart_image_paths=_chart_paths,
                                    chart_slide_mapping=chart_mapping,
                                    colors=colors,
                                    aspect_ratio=aspect_ratio_val,
                                    output_dir="./output",
                                    handle=handle,
                                    voice_id=elevenlabs_voice,
                                    topic=_topic,
                                    angle=_angle,
                                )
                                st.session_state.video_search_queries = []
                            elif use_ai:
                                # Generate proper AI image prompts (Cinematic realistic)
                                image_prompts = generate_image_prompts(
                                    slides=live_slides, topic=_topic, angle=_angle,
                                )
                                web_result = build_video_with_searched_images(
                                    slides=live_slides,
                                    scripts=edited_scripts,
                                    search_queries=image_prompts,
                                    colors=colors,
                                    aspect_ratio=aspect_ratio_val,
                                    output_dir="./output",
                                    handle=handle,
                                    voice_id=elevenlabs_voice,
                                    image_source="ai",
                                    cinematic_overlays=cinematic_overlay_images,
                                )
                                st.session_state.video_search_queries = image_prompts
                            else:
                                search_queries = generate_image_search_queries(
                                    slides=live_slides, scripts=edited_scripts,
                                    topic=_topic, angle=_angle,
                                )
                                web_result = build_video_with_searched_images(
                                    slides=live_slides,
                                    scripts=edited_scripts,
                                    search_queries=search_queries,
                                    colors=colors,
                                    aspect_ratio=aspect_ratio_val,
                                    output_dir="./output",
                                    handle=handle,
                                    voice_id=elevenlabs_voice,
                                    cinematic_overlays=cinematic_overlay_images,
                                )
                                st.session_state.video_search_queries = search_queries

                            st.session_state.video_path = web_result["video_path"]
                            st.session_state.video_search_results = web_result.get("search_results", {})
                            st.session_state.video_build_error = None
                            video_success = True

                        except Exception as exc:
                            import traceback
                            tb = traceback.format_exc()
                            st.session_state.video_build_error = f"{exc}\n\n{tb}"
                            st.error(f"Video build failed: {exc}")
                    gc.collect()
                    if video_success:
                        st.rerun()

            # ── Show persistent error ─────────────────────────────────
            if st.session_state.get("video_build_error"):
                st.error(f"Video build failed: {st.session_state.video_build_error}")
                if st.button("Dismiss Error", key="dismiss_video_err"):
                    st.session_state.video_build_error = None
                    st.rerun()

            # ── Show result ────────────────────────────────────────────
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.markdown("---")
                st.video(st.session_state.video_path)
                st.download_button(
                    label="Download MP4",
                    data=_cached_read(st.session_state.video_path),
                    file_name="narrated_slides.mp4",
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                    key="video_dl",
                )
                if st.session_state.video_search_results:
                    with st.expander("Image Search Results"):
                        for query, status in st.session_state.video_search_results.items():
                            if "found" in status and "not" not in status:
                                icon = status
                            elif "dalle" in status.lower():
                                icon = "DALL-E fallback"
                            else:
                                icon = "not found (used plain slide)"
                            st.markdown(f"- **\"{query}\"** — {icon}")

        # ══════════════════════════════════════════════════════════════
        # MODE: TikTok Script (standalone 130-160 word video script)
        # ══════════════════════════════════════════════════════════════
        else:
            st.subheader("TikTok Script")
            st.caption(
                "Generate a standalone 130-160 word script optimized for "
                "faceless TikTok finance videos, using your hook from Step 4."
            )

            _tt_topic = (
                st.session_state.selected_topic["title"]
                if st.session_state.selected_topic
                else ""
            )
            _tt_angle = st.session_state.get("angle", "")
            _tt_bullets = st.session_state.get("verified_bullets", [])

            # ── Reuse Step 4 hook (editable) ──────────────────────────
            step4_hook = st.session_state.selected_hook or {}
            default_hook_text = step4_hook.get("hook", "")
            default_hook_style = step4_hook.get("style", "")

            st.markdown("**Your hook** (from Step 4 — edit to customize for TikTok)")
            tt_hook_text = st.text_input(
                "Hook text",
                value=default_hook_text,
                key="tiktok_hook_input",
                placeholder="e.g. Tesla just lost $80 billion in one week",
            )

            if not tt_hook_text.strip():
                st.warning("Enter a hook above or go back to Step 4 to pick one.")
            else:
                # ── Generate full script ──────────────────────────────
                gen_col, reset_col = st.columns([3, 1])
                with gen_col:
                    if st.button(
                        "Generate Script",
                        type="primary",
                        use_container_width=True,
                        key="gen_tiktok_full",
                        disabled=not tt_hook_text.strip(),
                    ):
                        _require_api_key()
                        with st.spinner("Writing 130-160 word script..."):
                            try:
                                script = generate_tiktok_script(
                                    topic=_tt_topic,
                                    hook=tt_hook_text.strip(),
                                    hook_framework=default_hook_style,
                                    verified_bullets=_tt_bullets,
                                    angle=_tt_angle,
                                )
                                st.session_state.tiktok_script = script
                                st.session_state.tiktok_video_path = None
                            except Exception as exc:
                                st.error(f"Script generation failed: {exc}")
                        st.rerun()
                with reset_col:
                    if st.session_state.tiktok_script and st.button(
                        "Reset", use_container_width=True, key="clear_tiktok"
                    ):
                        st.session_state.tiktok_script = None
                        st.session_state.tiktok_video_path = None
                        st.rerun()

            # ── Display, validate & edit the script ───────────────────
            if st.session_state.tiktok_script:
                script = st.session_state.tiktok_script
                st.divider()

                # Validation dashboard
                word_count = script.get("word_count", 0)
                abstract_nouns = script.get("abstract_nouns", [])
                concrete_nouns = script.get("concrete_nouns", [])
                wc_ok = 130 <= word_count <= 160
                abs_ok = len(abstract_nouns) <= 2

                v_cols = st.columns(5)
                v_cols[0].metric("Words", word_count, delta="pass" if wc_ok else "miss", delta_color="normal" if wc_ok else "inverse")
                v_cols[1].metric("Target", "130-160", delta="in range" if wc_ok else "out of range", delta_color="normal" if wc_ok else "inverse")
                v_cols[2].metric("Abstract", f"{len(abstract_nouns)}/2", delta="pass" if abs_ok else "over limit", delta_color="normal" if abs_ok else "inverse")
                v_cols[3].metric("Concrete", len(concrete_nouns))
                v_cols[4].metric("Hook Style", default_hook_style[:18] if default_hook_style else "Custom")

                # Two-column layout: editor + preview
                edit_col, preview_col = st.columns([1, 1])

                with edit_col:
                    st.markdown("**Edit Script**")
                    hook_val = st.text_input(
                        "Hook (8 words max)",
                        value=script.get("hook", ""),
                        key="tiktok_hook_edit",
                    )
                    body_val = st.text_area(
                        "Body",
                        value=script.get("body", ""),
                        height=180,
                        key="tiktok_body_edit",
                    )
                    cta_val = st.text_input(
                        "CTA (one action)",
                        value=script.get("cta", ""),
                        key="tiktok_cta_edit",
                    )
                    edited_tt_script = f"{hook_val}\n\n{body_val}\n\n{cta_val}"

                    # Live word count
                    live_wc = len(edited_tt_script.split())
                    live_wc_ok = 130 <= live_wc <= 160
                    wc_color = "green" if live_wc_ok else "red"
                    st.markdown(f"Live word count: :{wc_color}[**{live_wc}**/160]")

                with preview_col:
                    st.markdown("**Preview**")
                    bg = colors.get("background", "#0D0D15")
                    bc = colors.get("body", "#C0C0D0")
                    ac = colors.get("accent", "#F7B731")
                    hl = colors.get("highlight", "#FF5757")

                    hook_esc = html_mod.escape(hook_val)
                    body_esc = html_mod.escape(body_val).replace("\n", "<br>")
                    cta_esc = html_mod.escape(cta_val)
                    handle_esc = html_mod.escape(handle)

                    phone_html = f"""
                    <div style="
                        background: {bg};
                        border: 2px solid #333;
                        border-radius: 24px;
                        padding: 32px 20px 24px 20px;
                        max-width: 360px;
                        margin: 0 auto;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    ">
                        <div style="color:{hl};font-size:18px;font-weight:800;line-height:1.3;
                            margin-bottom:16px;text-transform:uppercase;letter-spacing:0.5px;">
                            {hook_esc}</div>
                        <div style="border-top:2px solid {ac};width:30%;margin-bottom:14px;opacity:0.5;"></div>
                        <div style="color:{bc};font-size:13px;line-height:1.7;margin-bottom:20px;">
                            {body_esc}</div>
                        <div style="color:{ac};font-size:13px;font-weight:600;">
                            {cta_esc}</div>
                        <div style="text-align:center;color:#4A5568;font-size:11px;
                            margin-top:20px;padding-top:12px;border-top:1px solid #222;">
                            {handle_esc}</div>
                    </div>
                    """
                    st.markdown(phone_html, unsafe_allow_html=True)
                    st.caption(f"Argument: {script.get('argument_summary', '—')}")

                # Quality details
                with st.expander("Quality Details"):
                    q_cols = st.columns(2)
                    with q_cols[0]:
                        st.markdown("**Concrete nouns:**")
                        if concrete_nouns:
                            st.markdown(", ".join(f"`{n}`" for n in concrete_nouns))
                        else:
                            st.warning("No concrete nouns detected")
                    with q_cols[1]:
                        st.markdown("**Abstract nouns:**")
                        if abstract_nouns:
                            for an in abstract_nouns:
                                st.markdown(f"- `{an}`")
                        else:
                            st.success("None used")

                # Copyable script
                with st.expander("Copy Full Script"):
                    st.code(edited_tt_script, language=None)

                # Feedback & Regenerate
                st.divider()
                regen_cols = st.columns([3, 1])
                with regen_cols[0]:
                    tiktok_feedback = st.text_input(
                        "Feedback",
                        placeholder="e.g. more aggressive hook, add the Tesla data, simplify jargon...",
                        key="tiktok_script_feedback",
                    )
                with regen_cols[1]:
                    feedback_text = tiktok_feedback.strip() if isinstance(tiktok_feedback, str) else ""
                    if st.button(
                        "Regenerate",
                        use_container_width=True,
                        key="regen_tiktok_script",
                        disabled=not feedback_text,
                    ):
                        _require_api_key()
                        with st.spinner("Regenerating script..."):
                            try:
                                new_script = regenerate_tiktok_script(
                                    topic=_tt_topic,
                                    hook=tt_hook_text.strip(),
                                    hook_framework=default_hook_style,
                                    current_script=script,
                                    feedback=tiktok_feedback,
                                    verified_bullets=_tt_bullets,
                                    angle=_tt_angle,
                                )
                                st.session_state.tiktok_script = new_script
                                st.session_state.tiktok_video_path = None
                            except Exception as exc:
                                st.error(f"Script regeneration failed: {exc}")
                        st.rerun()

                # Build TikTok Video
                st.divider()
                vid_cols = st.columns([3, 1])
                with vid_cols[0]:
                    st.markdown("**Build Narrated Video**")
                    if not elevenlabs_key:
                        st.caption("Add your ElevenLabs API key in the sidebar to enable video.")
                with vid_cols[1]:
                    if st.button(
                        "Build MP4",
                        type="primary",
                        use_container_width=True,
                        key="build_tiktok_video",
                        disabled=not elevenlabs_key,
                    ):
                        os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
                        with st.spinner("Synthesizing audio and building video..."):
                            try:
                                tiktok_vid_path = build_tiktok_video(
                                    script_text=edited_tt_script,
                                    topic=_tt_topic,
                                    colors=colors,
                                    aspect_ratio=aspect_ratio_val,
                                    output_dir="./output",
                                    handle=handle,
                                    voice_id=elevenlabs_voice,
                                )
                                st.session_state.tiktok_video_path = tiktok_vid_path
                            except Exception as exc:
                                st.error(f"TikTok video build failed: {exc}")
                        st.rerun()

                # Show TikTok video result
                if st.session_state.tiktok_video_path and os.path.exists(
                    st.session_state.tiktok_video_path
                ):
                    st.divider()
                    vid_preview_col, vid_dl_col = st.columns([3, 1])
                    with vid_preview_col:
                        st.video(st.session_state.tiktok_video_path)
                    with vid_dl_col:
                        st.download_button(
                            label="Download MP4",
                            data=_cached_read(st.session_state.tiktok_video_path),
                            file_name="tiktok_script.mp4",
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                            key="tiktok_video_dl",
                        )

    # ── Back to edit hook/regenerate ───────────────────────────────────────
    st.divider()
    if st.button("Back to Hook Selection"):
        # Clear hook + slides so user returns to hook picker
        st.session_state.selected_hook = None
        st.session_state.slides = []
        st.session_state.fact_check_report = []
        st.session_state.conclusion_report = None
        st.session_state.coherence_report = None
        st.session_state.tiktok_metadata = None
        # Free large Studio data (images, video, alternatives) to reclaim memory
        for key in ["ai_image_paths", "ai_image_prompts", "ai_overlay_prompts",
                     "png_paths", "pptx_path", "mcp_alternatives",
                     "video_path", "video_scripts", "video_search_queries",
                     "video_search_results", "video_build_error",
                     "tiktok_script", "tiktok_video_path"]:
            if key in st.session_state:
                st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], (list, dict)) else None
        _read_file_bytes.clear()
        _build_zip_bytes.clear()
        gc.collect()
        st.session_state.step = 4
        st.rerun()
