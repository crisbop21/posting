"""Streamlit UI for the finance slide generator. Guided 6-step workflow.

Flow:
  1. Research & Pick a Topic (card-based selection)
  2. Consolidate Data (20+ verified bullet points)
  3. Provide Angle & Additional Data
  4. Choose a Hook (card-based selection)
  5. Generate & Verify Slides (verification dashboard)
  6. Studio: Edit + Export (tabbed — Edit | Slides | AI Images | Video | Canva)
"""

import html as html_mod
import io
import os

import anthropic
import streamlit as st
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
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx
from src.slides.png_builder import build_pngs
from src.slides.canva_builder import (
    get_oauth_url,
    exchange_code_for_token,
    refresh_access_token,
    build_canva_slides,
)
from src.slides.png_builder import build_style_alternatives
from src.slides.video_builder import build_video_from_slides
from src.slides.image_generator import generate_slide_images


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


st.set_page_config(page_title="Posting: Finance Slides", page_icon="📊", layout="wide")

st.title("Posting")
st.caption("Generate trending finance slide decks for TikTok & Instagram")

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
        ["Google Imagen 3 (Free)", "OpenAI DALL-E 3"],
        index=0,
        help="Google Imagen 3 is free (~50 images/day). DALL-E 3 requires a paid OpenAI key.",
    )

    google_ai_key = ""
    openai_img_key = ""
    if image_provider.startswith("Google"):
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

    canva_cfg = config.get("canva", {})
    canva_client_id = st.text_input(
        "Canva Client ID",
        value=canva_cfg.get("client_id", ""),
        type="password",
    )
    canva_client_secret = st.text_input(
        "Canva Client Secret",
        value=canva_cfg.get("client_secret", ""),
        type="password",
    )
    canva_enabled = bool(canva_client_id and canva_client_secret)

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
    bg_color = col1.color_picker("Background", colors_cfg.get("background", "#0D1117"))
    title_color = col2.color_picker("Title", colors_cfg.get("title", "#FFFFFF"))
    body_color = col1.color_picker("Body", colors_cfg.get("body", "#C9D1D9"))
    accent_color = col2.color_picker("Accent", colors_cfg.get("accent", "#58A6FF"))
    highlight_color = col1.color_picker("Highlight", colors_cfg.get("highlight", "#F0883E"))

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

# ── Canva OAuth handling ──────────────────────────────────────────────────────

_query_params = st.query_params
if "code" in _query_params and "state" in _query_params:
    canva_code = _query_params.get("code", "")
    canva_state = _query_params.get("state", "")
    if canva_state == "canva" and canva_code and canva_enabled:
        try:
            _app_url = st.context.headers.get("Origin", "http://localhost:8501")
            token_data = exchange_code_for_token(
                canva_code, canva_client_id, canva_client_secret,
                redirect_uri=_app_url,
            )
            st.session_state["canva_access_token"] = token_data["access_token"]
            st.session_state["canva_refresh_token"] = token_data.get("refresh_token", "")
            st.query_params.clear()
            st.toast("Canva connected!")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"Canva OAuth failed: {exc}")

if canva_enabled:
    if st.session_state.get("canva_access_token"):
        st.sidebar.success("Canva connected")
    else:
        _app_url = st.context.headers.get("Origin", "http://localhost:8501")
        _oauth_url = get_oauth_url(canva_client_id, redirect_uri=_app_url)
        st.sidebar.link_button("Connect Canva", _oauth_url)


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
    bg = colors.get("background", "#0D1117")
    tc = colors.get("title", "#FFFFFF")
    bc = colors.get("body", "#C9D1D9")
    ac = colors.get("accent", "#58A6FF")

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


# ── Session state defaults ────────────────────────────────────────────────────

for key, default in {
    "step": 1,
    "research_text": "",
    "research_facts": [],
    "topic_options": [],
    "selected_topic": None,
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
    "canva_result": None,
    "mcp_alternatives": [],
    "video_path": None,
    "video_scripts": [],
    "ai_image_paths": [],
    "ai_image_prompts": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Handle transition from old 7-step layout
if st.session_state.step > 6:
    st.session_state.step = 6

# ── Step indicators (clickable for completed steps) ──────────────────────────

step_labels = [
    "1. Topic",
    "2. Data",
    "3. Angle",
    "4. Hook",
    "5. Generate",
    "6. Studio",
]

current = st.session_state.step
cols = st.columns(len(step_labels))
for i, label in enumerate(step_labels):
    step_num = i + 1
    with cols[i]:
        if step_num < current:
            if st.button(
                f"✓ {label}",
                key=f"nav_{step_num}",
                use_container_width=True,
            ):
                st.session_state.step = step_num
                st.rerun()
        elif step_num == current:
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
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Research & Pick a Topic (card-based selection)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    st.header("Step 1: Pick a Topic")

    research_mode = st.radio(
        "How do you want to find a topic?",
        ["Latest News", "Custom Topic"],
        horizontal=True,
        help="Choose 'Latest News' to research trending stories, or 'Custom Topic' to provide your own subject.",
    )

    if research_mode == "Custom Topic":
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
    else:
        st.write("We'll research the latest trends and suggest 10 topics for your slide deck.")
        custom_topic = ""
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
# STEP 4: Choose a Hook (card-based selection)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 4: Choose a Hook")
    topic = st.session_state.selected_topic
    bullets = st.session_state.verified_bullets
    current_angle = st.session_state.angle

    angle_display = f"**Angle:** {current_angle}  \n" if current_angle else ""
    st.info(
        f"**Topic:** {topic['title']}  \n"
        f"{angle_display}"
        f"**Data points:** {len(bullets)} verified"
    )

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

    hooks = st.session_state.hook_options
    st.subheader("Pick the hook for your opening slide")
    st.caption("Sorted by best fit to your data")

    if st.button("Back", key="hook_back"):
        st.session_state.hook_options = []
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

            # Select button
            if hook_cols[2].button(
                "Use This",
                key=f"hook_{i}",
                type="primary" if i == 0 else "secondary",
                use_container_width=True,
            ):
                st.session_state.selected_hook = h
                st.session_state.step = 5
                st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Generate & Verify Slides (verification dashboard)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 5:
    st.header("Step 5: Generating Slides")
    topic = st.session_state.selected_topic
    angle = st.session_state.angle
    hook = st.session_state.selected_hook
    bullets = st.session_state.verified_bullets
    research_text = st.session_state.research_text

    st.info(
        f"**Topic:** {topic['title']}  \n"
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
            st.session_state.conclusion_report = conclusion_result
            logic_valid = conclusion_result.get("logic_valid", True)
            if logic_valid:
                placeholders[4].markdown("✅ **Conclusion check** — Logic sound")
            else:
                issues = conclusion_result.get("issues", [])
                placeholders[4].markdown(f"✅ **Conclusion check** — Fixed {len(issues)} logic gap(s)")
            slides = _safe_get_slides(conclusion_result, slides)
            slides = enforce_hook_and_count(slides, hook_text, slide_count)

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

    has_canva = canva_enabled and st.session_state.get("canva_access_token")

    # Build tab list
    tab_labels = ["Edit", "Slides", "AI Images", "Video"]
    if canva_enabled:
        tab_labels.append("Canva")

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

    with studio_tabs[1]:
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
            with open(filepath, "rb") as f:
                pptx_bytes = f.read()
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
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in png_paths:
                    zf.write(p, os.path.basename(p))
            zip_buf.seek(0)
            st.download_button(
                label="Download All PNGs (ZIP)",
                data=zip_buf,
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
                    with open(p, "rb") as f:
                        col.download_button(
                            label=f"Slide {i + 1}",
                            data=f.read(),
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
                    import zipfile as _zf
                    alt_zip = io.BytesIO()
                    with _zf.ZipFile(alt_zip, "w", _zf.ZIP_DEFLATED) as zf:
                        for p in alt_png_paths:
                            zf.write(p, os.path.basename(p))
                    alt_zip.seek(0)
                    st.download_button(
                        label=f"Download {alt['version']} (ZIP)",
                        data=alt_zip,
                        file_name=f"{alt['version'].replace(' ', '_').lower()}.zip",
                        mime="application/zip",
                        key=f"alt_zip_{alt['version']}",
                        use_container_width=True,
                    )
                st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # TAB: AI Images (Google Imagen 3 / OpenAI DALL-E 3)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[2]:
        st.subheader("AI-Generated Slide Images")

        if not ai_images_enabled:
            st.info(
                "Add an image generation API key in the sidebar under Integrations "
                "to generate cinematic AI background images for each slide.\n\n"
                "**Google Imagen 3** is free (~50 images/day) — get a key at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey)"
            )
        else:
            provider_label = "Google Imagen 3" if google_ai_key else "OpenAI DALL-E 3"
            st.caption(
                f"Generate cinematic AI backgrounds using {provider_label}. "
                "Claude creates visual prompts, the AI generates photorealistic images, "
                "then slide text is composited on top."
            )

            live_slides = _get_live_slides()

            if st.button("Generate AI Slide Images", type="primary", use_container_width=True, key="gen_ai_imgs"):
                if google_ai_key:
                    os.environ["GOOGLE_AI_API_KEY"] = google_ai_key
                elif openai_img_key:
                    os.environ["OPENAI_API_KEY"] = openai_img_key
                _require_api_key()
                with st.spinner("Generating image prompts and AI images... This may take 30-60s."):
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
                        )
                        st.session_state.ai_image_paths = ai_paths
                        st.session_state.ai_image_prompts = image_prompts
                    except Exception as exc:
                        st.error(f"AI image generation failed: {exc}")
                st.rerun()

            if st.session_state.ai_image_paths:
                ai_img_cols = st.columns(min(len(st.session_state.ai_image_paths), 3))
                for i, p in enumerate(st.session_state.ai_image_paths):
                    col = ai_img_cols[i % 3]
                    col.image(p, caption=f"Slide {i + 1}", use_container_width=True)

                with st.expander("Image Prompts"):
                    for i, prompt in enumerate(st.session_state.ai_image_prompts):
                        st.markdown(f"**Slide {i + 1}:** {prompt}")

                import zipfile as _ai_zf
                ai_zip = io.BytesIO()
                with _ai_zf.ZipFile(ai_zip, "w", _ai_zf.ZIP_DEFLATED) as zf:
                    for p in st.session_state.ai_image_paths:
                        zf.write(p, os.path.basename(p))
                ai_zip.seek(0)
                st.download_button(
                    label="Download AI Slides (ZIP)",
                    data=ai_zip,
                    file_name="ai_slides.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True,
                    key="ai_slides_zip_dl",
                )

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Video (Narrated video with ElevenLabs)
    # ══════════════════════════════════════════════════════════════════════

    with studio_tabs[3]:
        st.subheader("Narrated Video")

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

            use_ai_bg = False
            if ai_images_enabled:
                provider_label = "Google Imagen 3" if google_ai_key else "DALL-E 3"
                use_ai_bg = st.checkbox(
                    "Use AI-generated background images",
                    value=True,
                    help=f"Generate cinematic {provider_label} images as slide backgrounds in the video.",
                    key="video_use_ai_bg",
                )

            live_slides = _get_live_slides()

            if st.button("Build Narrated Video", type="primary", use_container_width=True, key="build_video"):
                os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
                if use_ai_bg:
                    if google_ai_key:
                        os.environ["GOOGLE_AI_API_KEY"] = google_ai_key
                    elif openai_img_key:
                        os.environ["OPENAI_API_KEY"] = openai_img_key
                _require_api_key()
                with st.spinner("Generating voiceover scripts and building video... This may take a minute."):
                    try:
                        result = build_video_from_slides(
                            slides=live_slides,
                            colors=colors,
                            topic=st.session_state.selected_topic["title"] if st.session_state.selected_topic else "",
                            angle=st.session_state.get("angle", ""),
                            aspect_ratio=aspect_ratio_val,
                            output_dir="./output",
                            handle=handle,
                            voice_id=elevenlabs_voice,
                            use_ai_images=use_ai_bg,
                        )
                        st.session_state.video_path = result["video_path"]
                        st.session_state.video_scripts = result["scripts"]
                    except Exception as exc:
                        st.error(f"Video build failed: {exc}")
                st.rerun()

            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.video(st.session_state.video_path)
                with open(st.session_state.video_path, "rb") as f:
                    st.download_button(
                        label="Download MP4",
                        data=f.read(),
                        file_name="narrated_slides.mp4",
                        mime="video/mp4",
                        type="primary",
                        use_container_width=True,
                        key="video_dl",
                    )
                if st.session_state.video_scripts:
                    with st.expander("Voiceover Scripts"):
                        for i, script in enumerate(st.session_state.video_scripts):
                            st.markdown(f"**Slide {i + 1}:** {script}")

    # ══════════════════════════════════════════════════════════════════════
    # TAB: Canva (optional)
    # ══════════════════════════════════════════════════════════════════════

    if canva_enabled:
        with studio_tabs[4]:
            st.subheader("Canva Export")

            if not has_canva:
                st.info(
                    "Connect your Canva account in the sidebar to export designs. "
                    "Your Client ID and Secret are configured — click 'Connect Canva' in the sidebar."
                )
            else:
                st.caption("Generate a Canva design from your slides. Edit further in Canva's visual editor.")

                live_slides = _get_live_slides()

                if st.button("Build with Canva", type="primary", use_container_width=True, key="build_canva"):
                    with st.spinner("Generating Canva design (this may take a moment)..."):
                        try:
                            canva_result = build_canva_slides(
                                slides=live_slides,
                                access_token=st.session_state["canva_access_token"],
                                topic=st.session_state.selected_topic["title"],
                                aspect_ratio=aspect_ratio_val,
                                colors=colors,
                            )
                            st.session_state.canva_result = canva_result
                        except Exception as exc:
                            st.error(f"Canva build failed: {exc}")
                    st.rerun()

                canva_result = st.session_state.canva_result
                if canva_result:
                    canva_col1, canva_col2 = st.columns(2)
                    edit_url = canva_result.get("edit_url", "")
                    export_url = canva_result.get("export_url", "")
                    if edit_url:
                        canva_col1.link_button(
                            "Edit in Canva",
                            edit_url,
                            type="primary",
                            use_container_width=True,
                        )
                    if export_url:
                        canva_col2.link_button(
                            "Download Canva PNG",
                            export_url,
                            use_container_width=True,
                        )
                    if not edit_url and not export_url:
                        st.warning("Canva design was created but no URLs were returned.")
                    st.caption(
                        "Open in Canva to customize fonts, colors, and layouts."
                    )

    # ── Back to edit hook/regenerate ───────────────────────────────────────
    st.divider()
    if st.button("Back to Hook Selection"):
        st.session_state.step = 4
        st.rerun()
