"""Streamlit UI for the finance slide generator. Guided 7-step workflow.

Flow:
  1. Research & Pick a Topic
  2. Consolidate Data (20+ verified bullet points)
  3. Provide Angle & Additional Data
  4. Choose a Hook (grounded in verified data, informed by angle)
  5. Generate & Verify Slides (consolidated checks)
  6. Edit Slides (inline editing)
  7. Export & Visualize (on-demand)
"""

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

# ── Sidebar: Settings ─────────────────────────────────────────────────────────

st.sidebar.header("Settings")

def _get_default_api_key() -> str:
    """Read API key from env var or Streamlit Cloud secrets."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except FileNotFoundError:
            pass
    return key

api_key = st.sidebar.text_input(
    "Anthropic API Key",
    value=_get_default_api_key(),
    type="password",
    help="Required. Set ANTHROPIC_API_KEY env var, add to Streamlit secrets, or paste here.",
)

config = load_config()
slides_cfg = config.get("slides", {})
research_cfg = config.get("research", {})
content_cfg = config.get("content", {})

st.sidebar.subheader("Slides")
slide_count = st.sidebar.slider("Number of slides", 3, 15, min(max(slides_cfg.get("count", 7), 3), 15))
tone = st.sidebar.selectbox(
    "Tone",
    ["bold", "casual", "professional", "educational"],
    index=["bold", "casual", "professional", "educational"].index(
        slides_cfg.get("tone", "bold")
    ),
)
audience = st.sidebar.text_input("Audience", slides_cfg.get("audience", "retail investors"))
aspect_ratio = st.sidebar.selectbox(
    "Aspect ratio",
    ["9:16 (vertical / stories)", "16:9 (landscape)"],
    index=0 if slides_cfg.get("aspect_ratio", "9:16") == "9:16" else 1,
)
aspect_ratio_val = "9:16" if aspect_ratio.startswith("9:16") else "16:9"

st.sidebar.subheader("Colors")
col1, col2 = st.sidebar.columns(2)
colors_cfg = slides_cfg.get("colors", {})
bg_color = col1.color_picker("Background", colors_cfg.get("background", "#0D1117"))
title_color = col2.color_picker("Title", colors_cfg.get("title", "#FFFFFF"))
body_color = col1.color_picker("Body", colors_cfg.get("body", "#C9D1D9"))
accent_color = col2.color_picker("Accent", colors_cfg.get("accent", "#58A6FF"))
highlight_color = col1.color_picker("Highlight", colors_cfg.get("highlight", "#F0883E"))

st.sidebar.subheader("Branding")
handle = st.sidebar.text_input("Account handle", slides_cfg.get("handle", "@cristian.bojaca"))

st.sidebar.subheader("Research")
available_sources = ["news", "reddit"]
default_sources = research_cfg.get("sources", ["news"])
sources = st.sidebar.multiselect("Sources", available_sources, default=default_sources)

default_topics = research_cfg.get("topics", ["stocks"])
topics = st.sidebar.multiselect(
    "Topics",
    ["stocks", "crypto", "earnings", "market trends", "economic indicators"],
    default=default_topics,
)

default_subs = research_cfg.get("subreddits", ["stocks"])
subreddits_input = st.sidebar.text_input(
    "Subreddits (comma-separated)",
    ", ".join(default_subs),
)
subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]

st.sidebar.subheader("Review")
review_iterations = st.sidebar.slider(
    "Review iterations", 0, 5, content_cfg.get("review_iterations", 2)
)
style_notes = st.sidebar.text_area(
    "Style notes",
    content_cfg.get("style_notes", ""),
    height=100,
)

st.sidebar.subheader("Video / ElevenLabs (optional)")
video_cfg = config.get("video", {})

def _get_default_elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ELEVENLABS_API_KEY", "")
        except FileNotFoundError:
            pass
    return key

elevenlabs_key = st.sidebar.text_input(
    "ElevenLabs API Key",
    value=_get_default_elevenlabs_key(),
    type="password",
    help="Required for video export. Get your key at https://elevenlabs.io",
)
elevenlabs_voice = st.sidebar.text_input(
    "Voice ID",
    value=video_cfg.get("voice_id", "pNInz6obpgDQGcFmaJgB"),
    help="ElevenLabs voice ID. Default is 'Adam'.",
)
video_enabled = bool(elevenlabs_key)

st.sidebar.subheader("AI Images / Replicate (optional)")

def _get_default_replicate_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        try:
            token = st.secrets.get("REPLICATE_API_TOKEN", "")
        except FileNotFoundError:
            pass
    return token

replicate_token = st.sidebar.text_input(
    "Replicate API Token",
    value=_get_default_replicate_token(),
    type="password",
    help="Required for AI-generated slide images (Flux). Get your token at https://replicate.com",
)
ai_images_enabled = bool(replicate_token)

st.sidebar.subheader("Canva (optional)")
canva_cfg = config.get("canva", {})
canva_client_id = st.sidebar.text_input(
    "Canva Client ID",
    value=canva_cfg.get("client_id", ""),
    type="password",
)
canva_client_secret = st.sidebar.text_input(
    "Canva Client Secret",
    value=canva_cfg.get("client_secret", ""),
    type="password",
)
canva_enabled = bool(canva_client_id and canva_client_secret)

# Handle Canva OAuth callback
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

# ── Helper: ensure API key is set ─────────────────────────────────────────────

def _require_api_key():
    if not api_key:
        st.error("Please provide an Anthropic API key in the sidebar.")
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key


def _safe_get_slides(result, fallback: list[dict]) -> list[dict]:
    """Safely extract corrected_slides from a pipeline result.

    Handles cases where _parse_json returns a list instead of a dict,
    or where corrected_slides is missing or has an unexpected type.
    """
    if isinstance(result, list):
        # LLM returned a bare list of slides
        return result if all(isinstance(s, dict) for s in result) else fallback
    if isinstance(result, dict):
        slides = result.get("corrected_slides", fallback)
        if isinstance(slides, list) and all(isinstance(s, dict) for s in slides):
            return slides
        return fallback
    return fallback


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

# ── Step indicators ───────────────────────────────────────────────────────────

step_labels = [
    "1. Topic",
    "2. Data",
    "3. Angle",
    "4. Hook",
    "5. Generate",
    "6. Edit",
    "7. Export",
]

current = st.session_state.step
cols = st.columns(len(step_labels))
for i, label in enumerate(step_labels):
    step_num = i + 1
    if step_num < current:
        cols[i].success(label)
    elif step_num == current:
        cols[i].info(label)
    else:
        cols[i].markdown(f"<span style='color:grey'>{label}</span>", unsafe_allow_html=True)

st.divider()

# ── Restart button ────────────────────────────────────────────────────────────

if st.session_state.step > 1:
    if st.button("Start Over"):
        for key in list(st.session_state.keys()):
            if key != "step":
                del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: Research & Pick a Topic
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
        _require_api_key()

        with st.spinner(
            f"Researching '{custom_topic}'..."
            if custom_topic
            else "Fetching latest trends (last 48 hours only)..."
        ):
            research_parts = []

            if custom_topic:
                # Custom topic: search news specifically for this subject
                custom_query = custom_topic.replace(" ", "+")
                custom_news = fetch_news_topics([custom_topic], max_per_topic=10)
                if custom_news:
                    st.toast(f"Found {len(custom_news)} articles about '{custom_topic}'")
                    research_parts.append(format_news_for_prompt(custom_news))

                # Also search via web_search for broader coverage
                web_results = search_claim(custom_topic + " finance", max_results=10)
                if web_results:
                    lines = [f"=== Web Search: {custom_topic} ===\n"]
                    for i, r in enumerate(web_results, 1):
                        lines.append(f"{i}. [{r['source']}] {r['title']}")
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
                # Original flow: latest news + Reddit
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
                            for idx, verdict in corrections.items():
                                if 1 <= idx <= len(news_items):
                                    item = news_items[idx - 1]
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

        # Extract structured facts from research
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

    if st.session_state.topic_options:
        topic_options = st.session_state.topic_options
        labels = [f"{t['title']}: {t['description']}" for t in topic_options]

        selected_idx = st.selectbox(
            "Select a topic for your deck",
            range(len(labels)),
            format_func=lambda i: labels[i],
        )

        if st.button("Use This Topic", type="primary"):
            chosen = topic_options[selected_idx]
            st.session_state.selected_topic = chosen
            st.session_state.step = 2
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: Consolidate Data (20+ verified bullet points)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("Step 2: Verified Data")
    topic = st.session_state.selected_topic
    st.info(f"**Topic:** {topic['title']}  \n{topic['description']}")

    # Generate bullet points if we don't have them yet
    if not st.session_state.verified_bullets:
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

    # Display bullets grouped by confidence
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
            _require_api_key()
            st.session_state.angle = angle.strip()

            # Step A: Research the angle
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

            # Step B: If under 20 bullets, do a second round of consolidation
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
                        # Merge without duplicating (by bullet text)
                        existing_texts = {b["bullet"].lower() for b in combined}
                        for eb in extra_bullets:
                            if eb["bullet"].lower() not in existing_texts:
                                combined.append(eb)
                                existing_texts.add(eb["bullet"].lower())
                        st.toast(f"Consolidated to {len(combined)} total data points")
                    except Exception as e:
                        st.toast(f"Extra consolidation failed: {e}")

            st.session_state.verified_bullets = combined
            # Mark that angle research is done
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
            # Convert user facts to bullet format
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

    # Show current data pool
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

    # Only allow proceeding if angle was verified or user added data
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
        # Clear hooks so they regenerate with the angle
        st.session_state.hook_options = []
        st.session_state.selected_hook = None
        st.session_state.step = 4
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Choose a Hook (grounded in verified data, informed by angle)
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
    hook_labels = []
    for h in hooks:
        fit = h.get("fit_score", "?")
        hook_labels.append(f"[{h['style']} | fit: {fit}/10] {h['hook']}")

    selected_hook_idx = st.radio(
        "Pick the hook for your opening slide (sorted by best fit)",
        range(len(hook_labels)),
        format_func=lambda i: hook_labels[i],
    )

    # Show which data the selected hook uses
    selected = hooks[selected_hook_idx]
    if selected.get("data_used"):
        st.caption(f"Data used: {selected['data_used']}")

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.hook_options = []
        st.session_state.step = 3
        st.rerun()

    if col_next.button("Generate Slides", type="primary"):
        st.session_state.selected_hook = hooks[selected_hook_idx]
        st.session_state.step = 5
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Generate & Verify Slides (consolidated checks)
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

    if not st.session_state.slides:
        _require_api_key()

        try:
            progress = st.progress(0, text="Generating slides...")

            hook_text = hook["hook"]

            # Generate slides grounded in verified bullets
            with st.spinner("Creating slides from verified data..."):
                progress.progress(10, text="Generating fact-grounded slides...")
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

            # Engagement review
            if review_iterations > 0:
                with st.spinner(f"Improving engagement ({review_iterations} iterations)..."):
                    progress.progress(25, text="Reviewing engagement...")
                    slides = review_and_improve(
                        slides=slides,
                        tone=tone,
                        audience=audience,
                        iterations=review_iterations,
                        hook=hook_text,
                    )
                    slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Fact-check against research
            if bullets:
                with st.spinner("Fact-checking against verified data..."):
                    progress.progress(40, text="Layered fact-check...")
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

                fact_report = []
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
            else:
                fact_report = []

            # Web search verification
            with st.spinner("Verifying claims against live data..."):
                progress.progress(55, text="Web search verification...")
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
                    if corrected_count:
                        st.toast(f"Web search corrected {corrected_count} outdated claim(s)")

                    for item in ws_report:
                        fact_report.append({
                            "slide": item.get("slide", "?"),
                            "status": item.get("status", "unknown"),
                            "notes": f"[web search] {item.get('notes', '')}",
                        })
                except Exception:
                    st.toast("Web search verification unavailable, continuing")

            st.session_state.fact_check_report = fact_report

            # Conclusion validation
            with st.spinner("Validating conclusion logic..."):
                progress.progress(65, text="Validating conclusion...")
                conclusion_result = validate_conclusion(
                    slides, bullets, topic["title"],
                    angle or topic["description"],
                )
                if not isinstance(conclusion_result, dict):
                    conclusion_result = {"corrected_slides": slides}
                st.session_state.conclusion_report = conclusion_result
                if not conclusion_result.get("logic_valid", True):
                    issues = conclusion_result.get("issues", [])
                    st.toast(f"Fixed {len(issues)} logic gap(s) in conclusion")
                slides = _safe_get_slides(conclusion_result, slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Narrative coherence check
            with st.spinner("Checking narrative coherence..."):
                progress.progress(70, text="Checking narrative coherence...")
                coherence_result = check_narrative_coherence(
                    slides, topic["title"],
                    angle or topic["description"], hook_text,
                )
                if not isinstance(coherence_result, dict):
                    coherence_result = {"corrected_slides": slides}
                st.session_state.coherence_report = coherence_result
                coherence_score = coherence_result.get("coherence_score", 0)
                st.toast(f"Narrative coherence: {coherence_score}/10")
                slides = _safe_get_slides(coherence_result, slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Strip claim tags
            slides = strip_claim_tags(slides)

            # Final combined value + cohesion pass
            with st.spinner("Final polish (value + cohesion)..."):
                progress.progress(85, text="Final polish...")
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
                # Update coherence report with the final score
                final_score = value_result.get("coherence_score", coherence_score)
                st.session_state.coherence_report = {
                    "coherence_score": final_score,
                    "arc_analysis": value_result.get("arc_analysis", ""),
                    "issues": value_result.get("issues", []),
                }
                st.toast(f"Final coherence: {final_score}/10")

            # TikTok metadata
            with st.spinner("Generating TikTok metadata..."):
                progress.progress(95, text="Generating metadata...")
                metadata = generate_tiktok_metadata(
                    slides=slides,
                    topic=topic["title"],
                    angle=angle or topic["description"],
                    hook=hook["hook"],
                )
                st.session_state.tiktok_metadata = metadata

            progress.progress(100, text="Done!")

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
# STEP 6: Edit Slides (inline editing)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 6:
    st.header("Step 6: Edit Your Slides")
    slides = st.session_state.slides
    fact_report = st.session_state.fact_check_report

    st.success(f"Generated {len(slides)} slides. Edit below, then export.")

    # ── Fact-Check Report ──────────────────────────────────────────────────
    if fact_report:
        with st.expander("Fact-Check Report", expanded=False):
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

    # ── Conclusion Report ──────────────────────────────────────────────────
    conclusion_report = st.session_state.conclusion_report
    if conclusion_report:
        with st.expander("Conclusion Validation", expanded=False):
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

    # ── Coherence Report ───────────────────────────────────────────────────
    coherence_report = st.session_state.coherence_report
    if coherence_report:
        with st.expander("Narrative Coherence", expanded=False):
            score = coherence_report.get("coherence_score", "?")
            st.markdown(f"**Coherence Score:** {score}/10")
            arc = coherence_report.get("arc_analysis", "")
            if arc:
                st.markdown(f"**Arc:** {arc}")

    # ── Inline Slide Editing ───────────────────────────────────────────────
    st.subheader("Slide Editor")

    BODY_MIN, BODY_IDEAL, BODY_MAX = 20, 50, 90
    st.caption(
        f"Body length guide: :red[< {BODY_MIN} too short] · "
        f":green[{BODY_MIN}–{BODY_MAX} ideal] · "
        f":red[> {BODY_MAX} too long]"
    )

    edited_slides = []
    for i, slide in enumerate(slides):
        with st.container(border=True):
            st.markdown(f"**Slide {i + 1}**")

            title = st.text_input(
                "Title",
                value=slide.get("title", ""),
                key=f"edit_title_{i}",
                label_visibility="collapsed",
                placeholder="Slide title",
            )
            body = st.text_area(
                "Body",
                value=slide.get("body", ""),
                key=f"edit_body_{i}",
                height=68,
                label_visibility="collapsed",
                placeholder="Slide body text",
            )
            footer = st.text_input(
                "Footer",
                value=slide.get("footer", ""),
                key=f"edit_footer_{i}",
                label_visibility="collapsed",
                placeholder="source: ...",
            )

            # Character count
            char_count = len(body)
            bar_value = min(char_count / BODY_MAX, 1.0)
            if char_count < BODY_MIN:
                label = f":red[{char_count} chars, too short]"
            elif char_count <= BODY_MAX:
                label = f":green[{char_count} chars]"
            else:
                label = f":red[{char_count} chars, too long]"
            st.progress(bar_value)
            st.caption(label)

            edited_slides.append({"title": title, "body": body, "footer": footer})

    # ── TikTok Metadata Editing ────────────────────────────────────────────
    metadata = st.session_state.tiktok_metadata
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

    # ── Save edits & advance ───────────────────────────────────────────────
    col_back, col_save = st.columns(2)

    if col_back.button("Back (regenerate)"):
        st.session_state.slides = []
        st.session_state.step = 5
        st.rerun()

    if col_save.button("Save & Export", type="primary"):
        # Save edited slides back to session state
        st.session_state.slides = edited_slides
        if metadata:
            st.session_state.tiktok_metadata = {
                "title": tiktok_title,
                "description": tiktok_desc,
            }
        st.session_state.step = 7
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7: Export & Visualize (on demand)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 7:
    st.header("Step 7: Export & Visualize")
    slides = st.session_state.slides
    metadata = st.session_state.tiktok_metadata

    colors = {
        "background": bg_color,
        "title": title_color,
        "body": body_color,
        "accent": accent_color,
        "highlight": highlight_color,
    }

    # ── Slide Preview ──────────────────────────────────────────────────────
    st.subheader("Final Slides")
    for i, slide in enumerate(slides):
        with st.container(border=True):
            slide_cols = st.columns([1, 12])
            slide_cols[0].markdown(f"**{i + 1}**")
            slide_cols[1].markdown(f"### {slide.get('title', '')}")
            slide_cols[1].write(slide.get("body", ""))
            if slide.get("footer"):
                slide_cols[1].caption(slide.get("footer", ""))

    # ── TikTok Metadata ───────────────────────────────────────────────────
    if metadata:
        st.subheader("TikTok Post Copy")
        st.markdown(f"**Title:** {metadata.get('title', '')}")
        st.text_area(
            "Description (copy this)",
            value=metadata.get("description", ""),
            height=120,
            disabled=True,
            key="final_tiktok_desc",
        )

    # ── Export Buttons ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Export")

    has_canva = canva_enabled and st.session_state.get("canva_access_token")
    if has_canva:
        export_col1, export_col2, export_col3 = st.columns(3)
    else:
        export_col1, export_col2 = st.columns(2)

    # Build PPTX on demand
    if export_col1.button("Build PPTX", type="primary", use_container_width=True):
        with st.spinner("Building PowerPoint..."):
            filepath = build_pptx(
                slides=slides,
                colors=colors,
                aspect_ratio=aspect_ratio_val,
                output_dir="./output",
                handle=handle,
            )
            st.session_state.pptx_path = filepath
        st.rerun()

    # Build PNGs on demand
    if export_col2.button("Build PNGs", type="primary", use_container_width=True):
        with st.spinner("Rendering PNG slides..."):
            png_paths = build_pngs(
                slides=slides,
                colors=colors,
                aspect_ratio=aspect_ratio_val,
                output_dir="./output",
                handle=handle,
            )
            st.session_state.png_paths = png_paths
        st.rerun()

    # Build with Canva on demand
    if has_canva:
        if export_col3.button("Build with Canva", type="secondary", use_container_width=True):
            with st.spinner("Generating Canva design (this may take a moment)..."):
                try:
                    canva_result = build_canva_slides(
                        slides=slides,
                        access_token=st.session_state["canva_access_token"],
                        topic=st.session_state.selected_topic["title"],
                        aspect_ratio=aspect_ratio_val,
                        colors=colors,
                    )
                    st.session_state.canva_result = canva_result
                except Exception as exc:
                    st.error(f"Canva build failed: {exc}")
            st.rerun()
    elif canva_enabled:
        st.caption("Connect your Canva account in the sidebar to enable Canva export.")

    # AI-generated slide images (standalone, without video)
    if ai_images_enabled:
        st.divider()
        st.subheader("AI-Generated Slide Images")
        st.caption(
            "Generate cinematic AI background images for each slide using Flux (via Replicate). "
            "Claude creates visual prompts, Flux generates photorealistic images, "
            "then slide text is composited on top."
        )
        if st.button("Generate AI Slide Images", type="primary", use_container_width=True):
            os.environ["REPLICATE_API_TOKEN"] = replicate_token
            _require_api_key()
            with st.spinner("Generating image prompts and AI images... This may take 30-60s."):
                try:
                    image_prompts = generate_image_prompts(
                        slides=slides,
                        topic=st.session_state.selected_topic["title"] if st.session_state.selected_topic else "",
                        angle=st.session_state.get("angle", ""),
                    )
                    ai_paths = generate_slide_images(
                        slides=slides,
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
            # ZIP download
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

    # Build narrated video on demand
    if video_enabled:
        st.divider()
        st.subheader("Narrated Video")
        st.caption(
            "Generate a narrated MP4 video with AI voiceover (ElevenLabs). "
            "Each slide is read aloud with a natural script generated by Claude."
        )
        use_ai_bg = False
        if ai_images_enabled:
            use_ai_bg = st.checkbox(
                "Use AI-generated background images",
                value=True,
                help="Generate cinematic Flux images as slide backgrounds in the video.",
            )
        if st.button("Build Narrated Video", type="primary", use_container_width=True):
            os.environ["ELEVENLABS_API_KEY"] = elevenlabs_key
            if use_ai_bg:
                os.environ["REPLICATE_API_TOKEN"] = replicate_token
            _require_api_key()
            with st.spinner("Generating voiceover scripts and building video... This may take a minute."):
                try:
                    result = build_video_from_slides(
                        slides=slides,
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

    # Generate alternative style versions (fully local, no login needed)
    st.divider()
    st.subheader("Alternative Style Versions")
    st.caption(
        "Generate multiple design variations with finance-themed backgrounds "
        "(candlestick charts, trend lines, volume bars). "
        "Works instantly offline — no login, no API keys, no internet needed."
    )
    mcp_cols = st.columns([2, 1])
    num_alts = mcp_cols[1].selectbox(
        "Number of alternatives",
        [2, 3, 4],
        index=0,
        key="mcp_num_alts",
    )
    if mcp_cols[0].button(
        "Generate Alternative Versions",
        type="primary",
        use_container_width=True,
        key="mcp_generate_btn",
    ):
        with st.spinner("Generating style alternatives..."):
            try:
                alts = build_style_alternatives(
                    slides=slides,
                    aspect_ratio=aspect_ratio_val,
                    output_dir="./output",
                    handle=handle,
                    num_alternatives=num_alts,
                )
                st.session_state.mcp_alternatives = alts
            except Exception as exc:
                st.error(f"Alternative generation failed: {exc}")
        st.rerun()

    # ── Download Buttons ───────────────────────────────────────────────────
    filepath = st.session_state.pptx_path
    png_paths = st.session_state.png_paths

    if filepath or png_paths:
        st.divider()
        st.subheader("Download")
        dl_col1, dl_col2 = st.columns(2)

        if filepath:
            with open(filepath, "rb") as f:
                pptx_bytes = f.read()
            dl_col1.download_button(
                label="Download PPTX",
                data=pptx_bytes,
                file_name=os.path.basename(filepath),
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                type="primary",
                use_container_width=True,
            )

        if png_paths:
            import zipfile
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in png_paths:
                    zf.write(p, os.path.basename(p))
            zip_buf.seek(0)
            dl_col2.download_button(
                label="Download All PNGs (ZIP)",
                data=zip_buf,
                file_name="slides.zip",
                mime="application/zip",
                type="primary",
                use_container_width=True,
            )

            # Individual slide images
            with st.expander("Individual slide images", expanded=False):
                img_cols = st.columns(min(len(png_paths), 3))
                for i, p in enumerate(png_paths):
                    col = img_cols[i % 3]
                    col.image(p, caption=f"Slide {i + 1}", use_container_width=True)
                    with open(p, "rb") as f:
                        col.download_button(
                            label=f"Slide {i + 1}",
                            data=f.read(),
                            file_name=os.path.basename(p),
                            mime="image/png",
                            key=f"png_dl_{i}",
                            use_container_width=True,
                        )

    # ── Canva Result ──────────────────────────────────────────────────────
    canva_result = st.session_state.canva_result
    if canva_result:
        st.divider()
        st.subheader("Canva Design")
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
            "Open in Canva to customize fonts, colors, and layouts. "
            "Compare with the local PNGs above to choose your favorite."
        )

    # ── Local Alternative Versions ────────────────────────────────────────
    mcp_alts = st.session_state.mcp_alternatives
    if mcp_alts:
        st.divider()
        st.subheader("Alternative Design Versions")
        for alt in mcp_alts:
            st.markdown(f"**{alt['version']}** — _{alt['style']}_")
            alt_png_paths = alt.get("png_paths", [])
            if alt_png_paths:
                alt_img_cols = st.columns(min(len(alt_png_paths), 3))
                for j, p in enumerate(alt_png_paths):
                    col = alt_img_cols[j % len(alt_img_cols)]
                    col.image(p, caption=f"Slide {j + 1}", use_container_width=True)
                # ZIP download for this alternative
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
        st.caption(
            "Each alternative uses a different color scheme and layout. "
            "Compare side-by-side to pick the best one."
        )

    # ── Back to edit ───────────────────────────────────────────────────────
    if st.button("Back to Edit"):
        st.session_state.pptx_path = None
        st.session_state.png_paths = []
        st.session_state.canva_result = None
        st.session_state.mcp_alternatives = []
        st.session_state.video_path = None
        st.session_state.video_scripts = []
        st.session_state.ai_image_paths = []
        st.session_state.ai_image_prompts = []
        st.session_state.step = 6
        st.rerun()
