"""Streamlit UI for the finance slide generator — guided 7-step workflow.

Flow:
  1. Research & Pick a Topic
  2. Consolidate Data (20+ verified bullet points)
  3. Choose a Hook (grounded in verified data)
  4. Provide Angle & Additional Data
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
)
from src.content.reviewer import review_and_improve
from src.slides.pptx_builder import build_pptx
from src.slides.png_builder import build_pngs


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


st.set_page_config(page_title="Posting — Finance Slides", page_icon="📊", layout="wide")

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
handle = st.sidebar.text_input("Account handle", slides_cfg.get("handle", "@posting"))

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

# ── Helper: ensure API key is set ─────────────────────────────────────────────

def _require_api_key():
    if not api_key:
        st.error("Please provide an Anthropic API key in the sidebar.")
        st.stop()
    os.environ["ANTHROPIC_API_KEY"] = api_key


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
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Step indicators ───────────────────────────────────────────────────────────

step_labels = [
    "1. Topic",
    "2. Data",
    "3. Hook",
    "4. Angle",
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
# STEP 1 — Research & Pick a Topic
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.step == 1:
    st.header("Step 1: Pick a Topic")
    st.write("We'll research the latest trends and suggest 10 topics for your slide deck.")

    if st.button("Research Topics", type="primary", use_container_width=True):
        _require_api_key()

        with st.spinner("Fetching latest trends (last 48 hours only)..."):
            research_parts = []

            if "news" in sources:
                news_items = fetch_news_topics(topics)
                if news_items:
                    st.toast(f"Found {len(news_items)} recent articles — fact-checking...")

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
                                f"Corrected {len(corrections)} article(s) — "
                                f"all {len(news_items)} now factual"
                            )
                        else:
                            st.toast(f"All {len(news_items)} articles verified")
                    except Exception:
                        st.toast("Fact-check unavailable — using articles as-is")

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
            st.toast("Facts extraction unavailable — continuing without grounding")
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
        labels = [f"{t['title']} — {t['description']}" for t in topic_options]

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
# STEP 2 — Consolidate Data (20+ verified bullet points)
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
            st.markdown(f"- **{b['bullet']}** — _source: {b.get('source', 'unknown')}_")

    if med_conf:
        st.subheader(f"Medium confidence ({len(med_conf)})")
        for b in med_conf:
            st.markdown(f"- {b['bullet']} — _source: {b.get('source', 'unknown')}_")

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.verified_bullets = []
        st.session_state.step = 1
        st.rerun()

    if col_next.button("Continue to Hooks", type="primary"):
        st.session_state.step = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Choose a Hook (grounded in verified data)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("Step 3: Choose a Hook")
    topic = st.session_state.selected_topic
    bullets = st.session_state.verified_bullets

    st.info(f"**Topic:** {topic['title']}  \n**Data points:** {len(bullets)} verified")

    if not st.session_state.hook_options:
        _require_api_key()
        try:
            with st.spinner("Generating hooks grounded in your verified data..."):
                hooks = generate_hooks(
                    topic=topic["title"],
                    verified_bullets=bullets,
                    tone=tone,
                    audience=audience,
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
        st.session_state.step = 2
        st.rerun()

    if col_next.button("Continue", type="primary"):
        st.session_state.selected_hook = hooks[selected_hook_idx]
        st.session_state.step = 4
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Provide Angle & Additional Data
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 4: Your Angle & Data")
    topic = st.session_state.selected_topic
    hook = st.session_state.selected_hook
    bullets = st.session_state.verified_bullets

    st.info(
        f"**Topic:** {topic['title']}  \n"
        f"**Hook:** {hook['hook']}  \n"
        f"**Data points:** {len(bullets)} verified"
    )

    st.write("Choose how to add your perspective:")

    tab_angle, tab_data = st.tabs(["Provide an Angle", "Add Your Own Data"])

    with tab_angle:
        st.write(
            "Describe your angle — we'll research it, verify all data is factual, "
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
                    st.toast(f"Angle research failed: {e} — using existing data")

            # Step B: If under 20 bullets, do a second round of consolidation
            if len(combined) < 20:
                with st.spinner(
                    f"Only {len(combined)} data points — researching more to reach 20+..."
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
                                "description": f"{topic['description']} — angle: {angle.strip()}",
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
            f"Only {bullet_count} data points — at least 20 recommended. "
            "Research an angle or add your own data above."
        )
    else:
        st.success(f"{bullet_count} verified data points ready.")

    with st.expander(f"Current data pool ({bullet_count} points)", expanded=False):
        for b in current_bullets:
            src = b.get("source", "unknown")
            conf = b.get("confidence", "medium")
            icon = "**" if conf == "high" else ""
            st.markdown(f"- {icon}{b['bullet']}{icon} — _{src}_")

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.step = 3
        st.rerun()

    # Only allow proceeding if angle was verified or user added data
    angle_verified = st.session_state.get("angle_verified", False)
    has_user_facts = bool(st.session_state.user_facts)
    can_proceed = angle_verified or has_user_facts or bullet_count >= 20

    if col_next.button(
        "Generate Slides",
        type="primary",
        disabled=not can_proceed,
        help="Research an angle or add your data first" if not can_proceed else "",
    ):
        if not st.session_state.angle:
            st.session_state.angle = ""
        st.session_state.step = 5
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Generate & Verify Slides (consolidated checks)
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
                    layer_a = fc_result.get("layer_a_report", [])
                    layer_b = fc_result.get("layer_b_report", [])
                    slides = fc_result.get("corrected_slides", slides)
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
                    ws_report = ws_result.get("search_report", [])
                    slides = ws_result.get("corrected_slides", slides)
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
                    st.toast("Web search verification unavailable — continuing")

            st.session_state.fact_check_report = fact_report

            # Conclusion validation
            with st.spinner("Validating conclusion logic..."):
                progress.progress(65, text="Validating conclusion...")
                conclusion_result = validate_conclusion(
                    slides, bullets, topic["title"],
                    angle or topic["description"],
                )
                st.session_state.conclusion_report = conclusion_result
                if not conclusion_result.get("logic_valid", True):
                    issues = conclusion_result.get("issues", [])
                    st.toast(f"Fixed {len(issues)} logic gap(s) in conclusion")
                slides = conclusion_result.get("corrected_slides", slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Narrative coherence
            with st.spinner("Ensuring story cohesion..."):
                progress.progress(75, text="Checking narrative coherence...")
                coherence_result = check_narrative_coherence(
                    slides, topic["title"],
                    angle or topic["description"], hook_text,
                )
                st.session_state.coherence_report = coherence_result
                coherence_score = coherence_result.get("coherence_score", 0)
                st.toast(f"Narrative coherence: {coherence_score}/10")
                slides = coherence_result.get("corrected_slides", slides)
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

            # Strip claim tags
            slides = strip_claim_tags(slides)

            # Final value + cohesion pass
            with st.spinner("Final polish..."):
                progress.progress(85, text="Final polish...")
                slides = add_value_pass(
                    slides=slides,
                    topic=topic["title"],
                    angle=angle or topic["description"],
                    audience=audience,
                )
                slides = enforce_hook_and_count(slides, hook_text, slide_count)

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

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Edit Slides (inline editing)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 6:
    st.header("Step 6: Edit Your Slides")
    slides = st.session_state.slides
    fact_report = st.session_state.fact_check_report

    st.success(f"Generated {len(slides)} slides — edit below, then export.")

    # ── Fact-Check Report ──────────────────────────────────────────────────
    if fact_report:
        with st.expander("Fact-Check Report", expanded=False):
            for item in fact_report:
                slide_num = item.get("slide", "?")
                status = item.get("status", "unknown")
                notes = item.get("notes", "")
                if status == "verified":
                    st.markdown(f"**Slide {slide_num}** — :green[verified]  \n{notes}")
                elif status == "corrected":
                    st.markdown(f"**Slide {slide_num}** — :orange[corrected]  \n{notes}")
                else:
                    st.markdown(f"**Slide {slide_num}** — :red[flagged]  \n{notes}")

    # ── Conclusion Report ──────────────────────────────────────────────────
    conclusion_report = st.session_state.conclusion_report
    if conclusion_report:
        with st.expander("Conclusion Validation", expanded=False):
            logic_valid = conclusion_report.get("logic_valid", True)
            verdict_slide = conclusion_report.get("verdict_slide", "?")
            if logic_valid:
                st.markdown(f":green[Verdict (Slide {verdict_slide}) logically follows from evidence]")
            else:
                st.markdown(f":orange[Verdict (Slide {verdict_slide}) had logic gaps — corrected]")
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
                label = f":red[{char_count} chars — too short]"
            elif char_count <= BODY_MAX:
                label = f":green[{char_count} chars]"
            else:
                label = f":red[{char_count} chars — too long]"
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
# STEP 7 — Export & Visualize (on-demand)
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

    # ── Back to edit ───────────────────────────────────────────────────────
    if st.button("Back to Edit"):
        st.session_state.pptx_path = None
        st.session_state.png_paths = []
        st.session_state.step = 6
        st.rerun()
