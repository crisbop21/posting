"""Streamlit UI for the finance slide generator — guided 4-step workflow."""

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
    extract_news_facts,
    fact_check_news_claims,
    fact_check_supporting_claims,
    validate_conclusions,
    check_narrative_coherence,
    add_value_pass,
    fact_check_slides,
    fact_check_news,
    generate_tiktok_metadata,
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
    "topic_options": [],
    "selected_topic": None,
    "angle": "",
    "hook_options": [],
    "selected_hook": None,
    "key_facts": [],
    "slides": [],
    "fact_check_report": [],
    "tiktok_metadata": None,
    "pptx_path": None,
    "png_paths": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Step indicators ───────────────────────────────────────────────────────────

step_labels = [
    "1. Pick a Topic",
    "2. Define Your Angle",
    "3. Choose a Hook",
    "4. Generate Slides",
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
        for key in ["step", "research_text", "topic_options", "selected_topic",
                     "angle", "hook_options", "selected_hook", "key_facts",
                     "slides", "fact_check_report", "tiktok_metadata",
                     "pptx_path", "png_paths"]:
            del st.session_state[key]
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

                    # Fact-check and correct news before using it
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

        try:
            with st.spinner("Generating topic suggestions with Claude..."):
                topic_options = suggest_topics(research_text, audience)
                st.session_state.topic_options = topic_options
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
            st.stop()

        st.rerun()

    # Show topic dropdown if we have options
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
# STEP 2 — Define Your Angle
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 2:
    st.header("Step 2: Define Your Angle")
    topic = st.session_state.selected_topic
    st.info(f"**Topic:** {topic['title']}  \n{topic['description']}")

    angle = st.text_area(
        "What angle or specific information do you want to provide?",
        placeholder="e.g. 'Focus on how beginners can take advantage of this trend, include 3 actionable steps'",
        height=120,
    )

    if st.button("Continue", type="primary", disabled=not angle.strip()):
        st.session_state.angle = angle.strip()
        st.session_state.step = 3
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Choose a Hook
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 3:
    st.header("Step 3: Choose a Hook")
    topic = st.session_state.selected_topic
    angle = st.session_state.angle

    st.info(f"**Topic:** {topic['title']}  \n**Angle:** {angle}")

    # Generate hooks if we don't have them yet
    if not st.session_state.hook_options:
        _require_api_key()
        try:
            with st.spinner("Generating 10 hook options using proven formulas..."):
                hooks = generate_hooks(
                    topic=topic["title"],
                    angle=angle,
                    tone=tone,
                    audience=audience,
                )
                st.session_state.hook_options = hooks
                st.rerun()
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
            st.stop()

    hooks = st.session_state.hook_options
    hook_labels = [f"[{h['style']}] {h['hook']}" for h in hooks]

    selected_hook_idx = st.radio(
        "Pick the hook for your opening slide",
        range(len(hook_labels)),
        format_func=lambda i: hook_labels[i],
    )

    col_back, col_next = st.columns(2)
    if col_back.button("Back"):
        st.session_state.hook_options = []
        st.session_state.step = 2
        st.rerun()

    if col_next.button("Generate Slides", type="primary"):
        st.session_state.selected_hook = hooks[selected_hook_idx]
        st.session_state.step = 4
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Generate, Fact-Check & Download Slides
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state.step == 4:
    st.header("Step 4: Your Slides")
    topic = st.session_state.selected_topic
    angle = st.session_state.angle
    hook = st.session_state.selected_hook

    st.info(
        f"**Topic:** {topic['title']}  \n"
        f"**Angle:** {angle}  \n"
        f"**Hook:** {hook['hook']}"
    )

    # Generate slides if we haven't yet
    if not st.session_state.slides:
        _require_api_key()

        colors = {
            "background": bg_color,
            "title": title_color,
            "body": body_color,
            "accent": accent_color,
            "highlight": highlight_color,
        }

        try:
            progress = st.progress(0, text="Generating slides...")
            research_text = st.session_state.research_text

            # 0) Extract structured facts from research
            with st.spinner("Extracting facts from research..."):
                progress.progress(5, text="Extracting key facts...")
                key_facts = extract_news_facts(research_text)
                st.session_state.key_facts = key_facts
                st.toast(f"Extracted {len(key_facts)} verified facts from research")

            # 1) Generate slides grounded in facts
            with st.spinner("Claude is writing your slides..."):
                progress.progress(12, text="Generating grounded slides...")
                slides = generate_slide_content(
                    topic=topic["title"],
                    angle=angle,
                    hook=hook["hook"],
                    slide_count=slide_count,
                    tone=tone,
                    audience=audience,
                    style_notes=style_notes,
                    key_facts=key_facts,
                    research_text=research_text,
                )

            # 2) Review iterations
            if review_iterations > 0:
                with st.spinner(f"Reviewing ({review_iterations} iterations)..."):
                    progress.progress(22, text="Reviewing and improving engagement...")
                    slides = review_and_improve(
                        slides=slides,
                        tone=tone,
                        audience=audience,
                        iterations=review_iterations,
                        hook=hook["hook"],
                    )

            # 3A) Fact-check news-sourced claims against research
            with st.spinner("Checking news-sourced claims against research..."):
                progress.progress(32, text="Layer A: verifying news claims...")
                fc_news = fact_check_news_claims(slides, key_facts, research_text)
                news_report = fc_news.get("fact_check_report", [])
                slides = fc_news.get("corrected_slides", slides)
                news_issues = [r for r in news_report if r.get("status") in ("corrected", "flagged")]
                if news_issues:
                    st.toast(f"Fixed {len(news_issues)} news-sourced claim(s)")

            # 3B) Fact-check supporting data claims independently
            with st.spinner("Checking supporting data claims..."):
                progress.progress(40, text="Layer B: verifying supporting data...")
                fc_support = fact_check_supporting_claims(slides, topic["title"])
                support_report = fc_support.get("fact_check_report", [])
                slides = fc_support.get("corrected_slides", slides)
                support_issues = [r for r in support_report if r.get("status") in ("corrected", "flagged")]
                if support_issues:
                    st.toast(f"Fixed {len(support_issues)} supporting claim(s)")

            # Merge fact-check reports
            fact_report = [
                r for r in news_report + support_report
                if r.get("status") != "skipped"
            ]
            st.session_state.fact_check_report = fact_report

            # 4) Validate conclusions
            with st.spinner("Validating conclusions..."):
                progress.progress(50, text="Checking logic & conclusions...")
                vc_result = validate_conclusions(
                    slides, key_facts, topic["title"], hook["hook"],
                )
                if not vc_result.get("valid", True):
                    issues = vc_result.get("issues", [])
                    st.toast(f"Fixed {len(issues)} logical issue(s) in conclusions")
                slides = vc_result.get("corrected_slides", slides)

            # 5) Narrative coherence check
            with st.spinner("Checking narrative coherence..."):
                progress.progress(58, text="Verifying story arc...")
                slides = check_narrative_coherence(
                    slides, hook["hook"], topic["title"],
                )

            # 6) Value-add pass
            with st.spinner("Maximizing reader value..."):
                progress.progress(64, text="Adding sharper insights...")
                slides = add_value_pass(
                    slides=slides,
                    topic=topic["title"],
                    angle=angle,
                    audience=audience,
                )

            # 7) Final engagement polish — 3 iterations with hook alignment
            with st.spinner("Final engagement polish (3 iterations)..."):
                progress.progress(70, text="Engagement polish...")
                slides = review_and_improve(
                    slides=slides,
                    tone=tone,
                    audience=audience,
                    iterations=3,
                    hook=hook["hook"],
                )

            # 6) Generate TikTok metadata
            with st.spinner("Generating TikTok title & description..."):
                progress.progress(82, text="Generating TikTok metadata...")
                metadata = generate_tiktok_metadata(
                    slides=slides,
                    topic=topic["title"],
                    angle=angle,
                    hook=hook["hook"],
                )
                st.session_state.tiktok_metadata = metadata

            # 7) Build PPTX
            progress.progress(88, text="Building PPTX...")
            filepath = build_pptx(
                slides=slides,
                colors=colors,
                aspect_ratio=aspect_ratio_val,
                output_dir="./output",
                handle=handle,
            )

            # 8) Build PNG images
            progress.progress(95, text="Rendering PNG slides...")
            png_paths = build_pngs(
                slides=slides,
                colors=colors,
                aspect_ratio=aspect_ratio_val,
                output_dir="./output",
                handle=handle,
            )

            progress.progress(100, text="Done!")

            st.session_state.slides = slides
            st.session_state.pptx_path = filepath
            st.session_state.png_paths = png_paths
            st.rerun()

        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your Anthropic API key in the sidebar.")
            st.stop()

    # ── Results ────────────────────────────────────────────────────────────
    slides = st.session_state.slides
    filepath = st.session_state.pptx_path
    fact_report = st.session_state.fact_check_report
    metadata = st.session_state.tiktok_metadata

    st.success(f"Generated {len(slides)} slides — fact-checked and ready to post!")

    # ── Slide Preview ──────────────────────────────────────────────────────
    st.subheader("Slide Preview")

    # Character count guidance
    BODY_MIN, BODY_IDEAL, BODY_MAX = 20, 50, 90
    st.caption(
        f"Body length guide: :red[< {BODY_MIN} too short] · "
        f":green[{BODY_MIN}–{BODY_MAX} ideal] · "
        f":red[> {BODY_MAX} too long]"
    )

    for i, slide in enumerate(slides):
        with st.container(border=True):
            slide_cols = st.columns([1, 12])
            slide_cols[0].markdown(f"**{i + 1}**")
            slide_cols[1].markdown(f"### {slide.get('title', '')}")
            slide_cols[1].write(slide.get("body", ""))
            if slide.get("footer"):
                slide_cols[1].caption(slide.get("footer", ""))

            # Character count bar for body text
            body_text = slide.get("body", "")
            char_count = len(body_text)
            bar_value = min(char_count / BODY_MAX, 1.0)

            if char_count < BODY_MIN:
                color = "red"
                label = f":red[{char_count} chars — too short]"
            elif char_count <= BODY_MAX:
                color = "green"
                label = f":green[{char_count} chars]"
            else:
                color = "red"
                label = f":red[{char_count} chars — too long]"

            slide_cols[1].progress(bar_value)
            slide_cols[1].caption(label)

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

    # ── TikTok Metadata ───────────────────────────────────────────────────
    if metadata:
        st.subheader("TikTok Post Copy")
        st.text_input("Video Title", value=metadata.get("title", ""), key="tiktok_title")
        description = metadata.get("description", "")
        st.text_area("Description", value=description, height=160, key="tiktok_desc")
        char_count = len(description)
        if char_count >= 200:
            st.caption(f":green[{char_count} characters] (meets 200+ requirement)")
        else:
            st.caption(f":red[{char_count} characters] (below 200 minimum — add more text)")

    # ── Download ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Download")

    dl_col1, dl_col2 = st.columns(2)

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

    # ZIP all PNGs for easy camera-roll save
    png_paths = st.session_state.png_paths
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
            file_name=f"slides_{os.path.basename(filepath).replace('.pptx', '')}.zip",
            mime="application/zip",
            type="primary",
            use_container_width=True,
        )

        # Individual slide PNGs
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
