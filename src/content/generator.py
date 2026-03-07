"""Generate slide content from research using Claude."""

import json
import re

from src.api import create_message

HOOK_FORMULAS = """
TOP 10 HOOK FORMULAS (use these structural templates):

1. DATA DIG: "[Stock/event] just [happened]. I analyzed [X] days/years of data to find out what happens every time."
   Why it works: leads with a specific event and promises a data backed reveal.

2. COMPARISON SHOCK: "[Thing A] is up [X%]. [Thing B] is up [Y%]. You think [belief]. You're wrong."
   Why it works: contrasts two data points and challenges the reader's assumption.

3. HIDDEN GEM LIST: "I [research effort] to find [X] [things] that [remarkable outcome] since [time span]."
   Why it works: implies effort and exclusivity, promising curated insight.

4. CONTRARIAN CALL: "Stop [common action]. This is what [topic] actually [truth/needs]."
   Why it works: challenges conventional wisdom, triggering curiosity.

5. COUNTDOWN TRIGGER: "Next week could [dramatic consequence]. These [X] events matter more than you think."
   Why it works: adds urgency with a time element and promises insider context.

6. AUTHORITY INSIDER: "[Institution/Billionaire] just [dramatic action]. [Bold claim about implications]."
   Why it works: borrows credibility from a known authority figure.

7. OBSCURE STORY: "[Country/Company] [dramatic event X months ago]. The [name] trade explained."
   Why it works: introduces a little known event and promises an explanation.

8. EXPERIENCE PLAY: "After [X] years in [field]. Here is what they don't tell you about [topic]."
   Why it works: uses personal authority and promises hidden knowledge.

9. SMART MONEY TRACKER: "[Billionaire/Fund] just put $[X]B into [stock] while everyone is selling."
   Why it works: contrasts smart money vs. crowd behavior, triggering FOMO.

10. PANIC REVERSAL: "[Stock/Market] is down [X%] you panic sell. Here is the [time] that save you."
    Why it works: addresses fear directly and promises a calming data point.
"""

HOOK_RULES = """
HOOK RULES (best practices for engagement):
- If a specific number, percentage, or dollar amount is available from the topic/angle, USE IT. Concrete data outperforms vague claims
- If NO verified number is available, use directional language ("surging", "at record levels", "plummeting"). Do NOT invent a number
- Use "you/your" direct address to make it personal
- Keep hooks 11 to 18 words, long enough for substance, short enough to scan
- Use data signaling words ("analyzed", "data", "years") to imply research
- NEVER fabricate specific numbers, prices, or percentages that aren't in the topic or angle
- NEVER use generic hype language ("this stock went up 800%")
- ALWAYS create an "open loop" that promises a specific reveal
- Accuracy over impact: a vague but truthful hook is ALWAYS better than a specific but wrong one
"""

SLIDE_RULES = """
SLIDE RULES (strict, follow exactly):
- Maximum 6 to 8 slides per carousel.
- Each slide: ONE idea, ONE sentence, under 15 words.
- Data slides SHOULD have a specific number, percentage, or dollar amount, but ONLY if the data
  is verified from research. If no verified number is available, use directional language
  (e.g. "rising sharply", "near all time highs") instead of making one up.
- Slide 1 = Hook. Slide 2 = Re-hook (works standalone for mid scroll entry). Final slide = CTA.
- No filler slides. Every slide must make the viewer want to swipe.
- Text style: all lowercase except ticker symbols, numbers, proper nouns (Nasdaq, Tesla, Fed), and acronyms (AI, GDP, IPO).

Slide structure:
- Slide 1: Hook. The headline event + open loop.
- Slide 2: Re-hook. Standalone entry point with a different angle.
- Slides 3 to N: Data. One key fact per slide, comparative or surprising.
- Second to last slide: Verdict. The takeaway in one sentence.
- Last slide: CTA. Lead with a question or prompt that drives comments ("what do you think?",
  "drop your take below", "agree or disagree?"). Then include a save or follow prompt.
- This structure is a GUIDELINE. If the story flows better with a different order or fewer data
  slides, adapt it. Cohesion matters more than rigid structure.

STORYTELLING (critical):
- The slides MUST tell a cohesive story with a clear narrative arc.
- Each slide should logically lead to the next. The viewer should NEED to swipe.
- Structure the data slides as a build up: context, then tension, then insight, then payoff.
- Never repeat the same type of fact back to back; alternate between comparison, trend, and surprise.
- The verdict slide should feel like a satisfying conclusion to the story, not a random opinion.
- If the available facts don't fit the standard structure, CHANGE THE STRUCTURE to fit the facts.
  Never force facts into a structure that makes the story feel disconnected.

FACTUAL ACCURACY (overrides all other rules):
- Accuracy always beats specificity. A slide with directional language is better than one with a wrong number.
- NEVER invent, guess, or round numbers that aren't from verified research.
- If you don't have enough verified data points for the requested slide count, reduce the number of
  data slides rather than filling them with made up stats.

FOOTER RULES:
- Footer is a SHORT source attribution (e.g. "source: Bloomberg", "data: Federal Reserve", "per SEC filing").
- NO hashtags. NO emojis. NO self promotion.
- Keep footers to 3 to 5 words max. Leave blank if no source needed.
"""


# ── Layer 1: News Facts Extraction ────────────────────────────────────────────


def consolidate_topic_data(
    research_text: str,
    research_facts: list[dict],
    topic: dict,
    audience: str,
) -> list[dict]:
    """Consolidate all relevant data for a specific topic into 20+ verified bullet points.

    Filters the broad research to the selected topic, enriches with web search,
    and returns a structured list of verified data points.

    Args:
        research_text: Raw research text from news and Reddit.
        research_facts: Previously extracted facts (may cover multiple topics).
        topic: Selected topic dict with 'title' and 'description'.
        audience: Target audience description.

    Returns:
        List of dicts, each with:
            - "bullet": concise factual statement
            - "value": the specific number/date/event
            - "source": where it came from
            - "confidence": "high" or "medium"
    """
    from src.research.web_search import search_claim

    # Step 1: Web search for additional data on this specific topic
    search_results = search_claim(topic["title"] + " finance", max_results=5)
    search_context = ""
    if search_results:
        lines = ["ADDITIONAL WEB SEARCH RESULTS:"]
        for r in search_results:
            lines.append(f"- [{r['source']}] {r['title']}")
            if r["summary"]:
                lines.append(f"  {r['summary'][:200]}")
        search_context = "\n".join(lines)

    prompt = f"""You are a financial research analyst. Your job is to consolidate ALL relevant,
verifiable data about one specific topic into a structured briefing.

TOPIC: {topic['title']}
DESCRIPTION: {topic['description']}
AUDIENCE: {audience}

RESEARCH DATA (from news and social media):
{research_text}

PREVIOUSLY EXTRACTED FACTS:
{json.dumps(research_facts, indent=2)}

{search_context}

Your task:
1. Filter ONLY the facts relevant to "{topic['title']}" from the research above.
2. Add any additional relevant facts from the web search results.
3. Produce AT LEAST 20 verified bullet points about this topic.
4. Each bullet must be a specific, verifiable claim (with a number, date, name, or event).
5. Discard opinions, predictions without data, and vague statements.
6. If you cannot reach 20 bullets from the research alone, note which bullets come from
   general knowledge and mark them with "medium" confidence.

Return a JSON array of at least 20 objects:
[
  {{
    "bullet": "S&P 500 rose 2.3% this week, its best weekly gain since March",
    "value": "2.3%",
    "source": "Bloomberg",
    "confidence": "high"
  }}
]

Confidence levels:
- "high": directly stated in the research or web search results with a named source
- "medium": inferred, aggregated, or from general knowledge

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def research_angle(
    topic: dict,
    angle: str,
    existing_bullets: list[dict],
) -> list[dict]:
    """Research a user-provided angle and return additional verified bullet points.

    Uses web search to find data supporting the specific angle, then merges
    with existing bullets (deduplicating).

    Returns:
        List of NEW bullet dicts to add to the existing pool.
    """
    from src.research.web_search import search_claim

    # Search for data related to the angle
    query = f"{topic['title']} {angle}"
    search_results = search_claim(query, max_results=5)

    search_context = ""
    if search_results:
        lines = []
        for r in search_results:
            lines.append(f"- [{r['source']}] {r['title']}")
            if r["summary"]:
                lines.append(f"  {r['summary'][:200]}")
        search_context = "\n".join(lines)

    prompt = f"""You are a financial research analyst. You need to find additional data
to support a specific angle on a topic.

TOPIC: {topic['title']}
ANGLE: {angle}

WEB SEARCH RESULTS FOR THIS ANGLE:
{search_context}

EXISTING VERIFIED DATA (do not duplicate):
{json.dumps(existing_bullets, indent=2)}

Find additional verifiable facts that support the angle "{angle}".
Focus on specific numbers, dates, comparisons, and events.
Do NOT duplicate facts already in the existing data above.

Return a JSON array of new bullet points (may be fewer than 20 if data is limited):
[
  {{
    "bullet": "the factual claim",
    "value": "the specific number/date/event",
    "source": "where it came from",
    "confidence": "high" or "medium"
  }}
]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def analyze_charts(chart_images: list[bytes], context: str = "") -> dict:
    """Analyze uploaded chart images using Claude's vision capability.

    Sends each chart image to Claude for interpretation, extracting:
    - Chart type, subject, time period
    - Key data points, trends, and takeaways
    - Notable patterns (breakouts, divergences, support/resistance)

    Args:
        chart_images: List of image bytes (PNG/JPG/WEBP).
        context: Optional user-provided context about the charts.

    Returns:
        Dict with:
            - "research_text": Combined narrative analysis of all charts
            - "research_facts": List of extracted fact dicts
            - "chart_analyses": Per-chart analysis list
    """
    import base64

    content_blocks = []

    if context:
        content_blocks.append({
            "type": "text",
            "text": f"User context about these charts: {context}\n\n",
        })

    content_blocks.append({
        "type": "text",
        "text": (
            "You are a financial analyst. Analyze the following chart image(s) in detail.\n\n"
            "For EACH chart, extract:\n"
            "1. Chart type (line, bar, candlestick, pie, area, etc.)\n"
            "2. Subject/title — what is being measured\n"
            "3. Time period covered\n"
            "4. Key data points (specific numbers, peaks, troughs)\n"
            "5. Trends and patterns (uptrend, downtrend, consolidation, breakout)\n"
            "6. Notable observations (support/resistance levels, divergences, anomalies)\n"
            "7. Key takeaways — what story does this chart tell?\n\n"
            "Be specific with numbers. If you can read values from axes, include them.\n"
            "If you cannot read exact values, describe the approximate range.\n\n"
        ),
    })

    for i, img_bytes in enumerate(chart_images):
        # Detect media type from header bytes
        if img_bytes[:8].startswith(b'\x89PNG'):
            media_type = "image/png"
        elif img_bytes[:2] == b'\xff\xd8':
            media_type = "image/jpeg"
        elif img_bytes[:4] == b'RIFF':
            media_type = "image/webp"
        else:
            media_type = "image/png"  # fallback

        content_blocks.append({
            "type": "text",
            "text": f"\n--- Chart {i + 1} of {len(chart_images)} ---\n",
        })
        content_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": base64.b64encode(img_bytes).decode(),
            },
        })

    content_blocks.append({
        "type": "text",
        "text": (
            "\n\nNow provide your complete analysis. Return a JSON object:\n"
            "{\n"
            '  "charts": [\n'
            "    {\n"
            '      "chart_number": 1,\n'
            '      "chart_type": "line chart",\n'
            '      "subject": "S&P 500 performance",\n'
            '      "time_period": "Jan 2024 - Dec 2024",\n'
            '      "key_data_points": ["Started at 4,770", "Peaked at 5,880 in July", "Ended at 5,400"],\n'
            '      "trends": ["Strong uptrend Jan-Jul", "Correction Aug-Sep", "Recovery Oct-Dec"],\n'
            '      "notable_patterns": ["Double top at 5,880", "Support at 5,100"],\n'
            '      "takeaways": "S&P 500 gained ~13% in 2024 despite mid-year correction"\n'
            "    }\n"
            "  ],\n"
            '  "combined_narrative": "A paragraph summarizing all charts together and the overall story they tell",\n'
            '  "facts": [\n'
            "    {\n"
            '      "fact": "S&P 500 gained approximately 13% in 2024",\n'
            '      "type": "number",\n'
            '      "value": "13%",\n'
            '      "source": "chart analysis",\n'
            '      "confidence": "medium"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Extract at least 5 facts per chart. Mark confidence as 'medium' since "
            "values are read from charts (not primary sources).\n"
            "Return ONLY the JSON object, no other text."
        ),
    })

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": content_blocks}],
    )

    text = _extract_text(response)
    parsed = _parse_json(text)

    # Build research_text from the analysis
    research_parts = []
    if isinstance(parsed, dict):
        charts = parsed.get("charts", [])
        narrative = parsed.get("combined_narrative", "")
        facts = parsed.get("facts", [])

        for chart in charts:
            part = f"=== Chart: {chart.get('subject', 'Unknown')} ({chart.get('chart_type', 'chart')}) ===\n"
            part += f"Time period: {chart.get('time_period', 'Unknown')}\n"
            for dp in chart.get("key_data_points", []):
                part += f"  - {dp}\n"
            for trend in chart.get("trends", []):
                part += f"  Trend: {trend}\n"
            for pattern in chart.get("notable_patterns", []):
                part += f"  Pattern: {pattern}\n"
            part += f"  Takeaway: {chart.get('takeaways', '')}\n"
            research_parts.append(part)

        if narrative:
            research_parts.append(f"\nOverall: {narrative}")

        return {
            "research_text": "\n\n".join(research_parts),
            "research_facts": facts if isinstance(facts, list) else [],
            "chart_analyses": charts if isinstance(charts, list) else [],
        }

    # Fallback if parsing didn't return expected structure
    return {
        "research_text": text,
        "research_facts": [],
        "chart_analyses": [],
    }


def map_charts_to_slides(
    slides: list[dict],
    chart_analyses: list[dict],
    num_charts: int,
) -> list[int | None]:
    """Map chart images to the most relevant slides using Claude.

    Args:
        slides: List of slide dicts (title, body, footer).
        chart_analyses: Per-chart analysis from analyze_charts().
        num_charts: Total number of chart images available.

    Returns:
        List of length len(slides), where each element is a chart index (0-based)
        or None if no chart is relevant for that slide.
    """
    slides_desc = []
    for i, s in enumerate(slides):
        slides_desc.append(f"Slide {i + 1}: {s.get('title', '')} — {s.get('body', '')}")

    charts_desc = []
    for i, c in enumerate(chart_analyses):
        charts_desc.append(
            f"Chart {i + 1}: {c.get('subject', 'Unknown')} "
            f"({c.get('chart_type', 'chart')}) — {c.get('takeaways', '')}"
        )

    prompt = f"""You have {num_charts} chart image(s) and {len(slides)} slides.
Assign each chart to the slide(s) where it is most relevant as a visual overlay.
A chart can be used on multiple slides if relevant. Not every slide needs a chart.
Hook slides (slide 1) and CTA slides (last slide) typically should NOT have charts.

CHARTS:
{chr(10).join(charts_desc)}

SLIDES:
{chr(10).join(slides_desc)}

Return a JSON array of length {len(slides)}, where each element is either:
- A chart number (1-based) if a chart is relevant for that slide
- null if no chart should be shown on that slide

Example for 6 slides with 2 charts: [null, 1, 1, 2, 2, null]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    mapping = _parse_json(text)

    if isinstance(mapping, list) and len(mapping) == len(slides):
        # Convert 1-based to 0-based, keep None as None
        return [
            (m - 1) if isinstance(m, (int, float)) and m is not None and 1 <= m <= num_charts
            else None
            for m in mapping
        ]

    # Fallback: no mapping
    return [None] * len(slides)


def extract_news_facts(research_text: str) -> list[dict]:
    """Pull structured facts (numbers, events, dates, sources) from research text.

    Returns a list of fact dicts, each with:
        - "fact": the factual claim as a concise sentence
        - "type": "number", "event", "date", or "quote"
        - "value": the specific number/date/event name
        - "source": which article or source it came from
        - "confidence": "high" (directly stated) or "medium" (inferred)
    """
    prompt = f"""You are a financial research analyst. Extract every verifiable fact from the research below.

{research_text}

For EACH distinct fact, extract:
1. The factual claim as a concise sentence
2. The type: "number" (percentages, dollar amounts, statistics), "event" (mergers, launches, policy changes), "date" (specific dates or timeframes), or "quote" (attributed statements)
3. The specific value (the number, date, event name, or quoted text)
4. The source it came from (article name, outlet, or "multiple sources")
5. Confidence: "high" if directly stated with a source, "medium" if inferred or aggregated

Focus on VERIFIABLE claims. Skip opinions, predictions, and vague statements.

Return your response as a JSON array:
[
  {{
    "fact": "S&P 500 rose 2.3% this week",
    "type": "number",
    "value": "2.3%",
    "source": "Bloomberg",
    "confidence": "high"
  }}
]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def fact_check_news(news_text: str) -> list[dict]:
    """Fact-check news articles and return corrected versions of any that are inaccurate.

    Returns a list of dicts (one per article), each with:
        - "index": 1-based article number matching the input list
        - "status": "verified" or "corrected"
        - "corrected_title": factually accurate title (same as original if verified)
        - "corrected_summary": factually accurate summary (same as original if verified)
        - "reason": brief explanation of what was checked or changed
    """
    prompt = f"""You are a rigorous financial news fact-checker. Review each numbered article below.

{news_text}

For EACH article:
1. Check if the headline and summary are consistent (no clickbait mismatch).
2. Check if any specific claims (numbers, %, $, company names, events) seem plausible.
3. If an article has inaccurate claims, misleading framing, or outdated info, CORRECT it.
   Rewrite the title and summary so they are factually accurate while keeping the same trending topic.
4. If the article is accurate, keep the original title and summary as-is.

Return your response as a JSON array, one object per article:
[
  {{
    "index": 1,
    "status": "verified" or "corrected",
    "corrected_title": "factually accurate title (unchanged if verified)",
    "corrected_summary": "factually accurate summary (unchanged if verified)",
    "reason": "what was checked or what you fixed"
  }}
]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def suggest_topics(research_text: str, audience: str) -> list[dict]:
    """Analyse research data and return 10 potential topic ideas."""
    prompt = f"""You are a finance content strategist. Your audience is: {audience}.

Based on the trending research below, suggest exactly 10 potential slide-deck topics.
Pick the most timely, engaging, and shareable angles from the data.
Prioritise topics that lend themselves to specific numbers, data points, and open-loop hooks.

{research_text}

Return your response as a JSON array of 10 objects, each with:
- "title": A concise topic title (max 10 words)
- "description": One sentence explaining what this deck would cover

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def generate_hooks(
    topic: str,
    verified_bullets: list[dict],
    tone: str,
    audience: str,
    angle: str = "",
) -> list[dict]:
    """Generate 10 hook options grounded in verified data.

    Args:
        topic: Topic title.
        verified_bullets: List of verified bullet point dicts from consolidate_topic_data.
        tone: Desired tone.
        audience: Target audience.
        angle: Optional user-provided angle for additional context.
    """
    bullets_text = "\n".join(
        f"- {b['bullet']} (source: {b.get('source', 'unknown')}, confidence: {b.get('confidence', 'medium')})"
        for b in verified_bullets
    )
    angle_block = f"\nUser's angle: {angle}\n" if angle else ""

    prompt = f"""You are a viral finance content creator. Your audience is: {audience}.
Tone: {tone}.

Topic: {topic}
{angle_block}
VERIFIED DATA AVAILABLE (use ONLY these numbers in your hooks):
{bullets_text}

{HOOK_FORMULAS}

{HOOK_RULES}

Generate exactly 10 different hook options for the opening slide of a slide deck on this topic.
Each hook MUST follow one of the 10 structural formulas above. Use a different formula for each hook.
Every hook MUST be 11-18 words long (the sweet spot).
Every hook MUST use "you/your" direct address where possible.
Every hook MUST create an open loop (promise a reveal).

FACTUAL ACCURACY (critical):
- You MUST ONLY use numbers, percentages, and dollar amounts from the VERIFIED DATA above.
- Pick the most compelling data points from the bullet list to embed in each hook.
- If a formula needs a number that isn't in the verified data, use directional language instead.
- NEVER fabricate a price, percentage, or statistic not in the verified data above.

Also evaluate which hook formulas are the BEST FIT for the available data and rank them.

Return your response as a JSON array of 10 objects, each with:
- "hook": The hook text (11-18 words, using numbers from verified data only)
- "style": The formula name used (e.g. "Data Dig", "Comparison Shock", etc.)
- "fit_score": 1-10 rating of how well this formula fits the available data (10 = perfect fit)
- "data_used": which verified bullet(s) this hook references

Return ONLY the JSON array, no other text. Sort by fit_score descending (best fit first)."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def generate_slide_content(
    topic: str,
    angle: str,
    hook: str,
    slide_count: int,
    tone: str,
    audience: str,
    style_notes: str,
    research_facts: list[dict] | None = None,
) -> list[dict]:
    """Generate structured slide content grounded in extracted facts.

    When research_facts is provided (Layer 2), each slide's claims are tagged
    as "news_source" (from research) or "supporting_data" (general knowledge).
    """
    facts_block = ""
    claims_instruction = ""
    claims_schema = ""

    if research_facts:
        facts_block = f"""
VERIFIED FACTS FROM RESEARCH (use these as your primary data source):
{json.dumps(research_facts, indent=2)}

GROUNDING RULES:
- ONLY use facts from the research above. Every data slide MUST reference at least one research fact.
- If you use a fact from the research, tag it as "news_source".
- If you MUST add context not in the research, tag it as "supporting_data", but NEVER invent
  specific prices, percentages, or dollar amounts from general knowledge. These go stale quickly.
- For supporting_data claims, use relative/directional language (e.g. "up significantly") rather than
  specific numbers you are not certain about. Only use exact figures that come from the research.
- Do NOT guess current prices, index levels, or commodity prices. If the research doesn't provide
  a specific number, do NOT make one up.
"""
        claims_instruction = """
- "claims": An array of claim objects for each factual assertion in the slide"""
        claims_schema = """
Each claim object must have:
- "text": the specific factual assertion (e.g. "S&P 500 up 2.3%")
- "type": "news_source" (from research above) or "supporting_data" (general knowledge)
- "source_ref": source name for news_source claims, or "general knowledge" for supporting_data"""

    system_prompt = f"""You are an expert social media content creator specializing in finance.
You create slide decks that go viral on TikTok and Instagram.

Target audience: {audience}
Tone: {tone}

Style guidelines:
{style_notes}

{SLIDE_RULES}"""

    user_prompt = f"""Create a {slide_count}-slide deck on the following:

Topic: {topic}
Angle / key information: {angle}
Opening hook (use this EXACTLY as the first slide title): {hook}
{facts_block}
STRICT REQUIREMENTS:
- You MUST produce EXACTLY {slide_count} slides, no more, no fewer.
- Slide 1 title MUST be the hook above (copy it EXACTLY, character for character, do NOT rephrase).
- Slide 2 MUST be a re-hook, a standalone entry point with a different angle on the same topic.
- Data slides SHOULD each contain one key data fact, but ONLY use verified numbers from the research.
- For data slides where you don't have a verified number, use insight, analysis, context, or
  directional language ("rising sharply", "outpacing rivals"), but you MUST still produce the slide.
- Slide {slide_count - 1} MUST be the verdict, the one sentence takeaway.
- Slide {slide_count} MUST be a CTA that drives comments first ("what would you do?", "drop your take below", "agree or disagree?"). Can also include a save or follow prompt.
- ALL text must be lowercase except ticker symbols ($AAPL, $BTC), numbers, proper nouns (Nasdaq, Tesla, Fed), and acronyms (AI, GDP, IPO).
- Each slide body: ONE sentence, under 15 words.
- No filler. Every slide must make the viewer want to swipe.

FACTUAL ACCURACY (overrides everything EXCEPT slide count):
- NEVER invent a specific price, percentage, or dollar amount that is not in the research facts.
- If a data point is missing, use directional/comparative language or insightful analysis instead.
- You MUST still produce exactly {slide_count} slides. Fill non-data slides with context, insight,
  or analysis rather than skipping them.

STORYTELLING AND COHESION:
- The slides must tell ONE cohesive story where each slide builds on the previous.
- Data slides should follow: context, then tension, then insight, then payoff.
- Never repeat the same type of fact back to back; alternate comparison, trend, and surprise.
- A reader who sees all slides should feel like they followed a narrative, not read a list.
- If the available verified facts don't fit the standard slide structure, adapt the content type
  per slide (data, insight, context, analysis) but ALWAYS produce exactly {slide_count} slides.

Return your response as a JSON array of slide objects. Each slide must have:
- "title": The headline (under 15 words, lowercase except tickers, numbers, proper nouns, and acronyms)
- "body": One sentence of content (under 15 words, include a verified number if available)
- "footer": Short source attribution only (e.g. "source: Bloomberg"). NO hashtags, NO emojis. Leave blank if no source.{claims_instruction}
{claims_schema}
Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def add_value_pass(
    slides: list[dict], topic: str, angle: str, audience: str, hook: str = "",
) -> dict:
    """Final combined pass: maximize reader value AND narrative coherence in one shot.

    Looks for opportunities to replace generic claims with sharper data,
    add actionable insight, strengthen the narrative thread, and score cohesion.

    Returns dict with:
        - "corrected_slides": the improved slides
        - "coherence_score": 1 to 10 rating of narrative flow
        - "arc_analysis": brief description of the narrative arc
        - "issues": list of any problems fixed
    """
    prompt = f"""You are a senior finance content editor. Your reader is: {audience}.

Topic: {topic}
Angle: {angle}
Hook: {hook}

Here are the current slides:
{json.dumps(slides, indent=2)}

Your job: ONE comprehensive pass that covers BOTH value AND narrative coherence.

PART 1, VALUE CHECK. For each slide ask:
- Does this slide teach something or just state the obvious?
- Is every slide earning its place, or would the deck be weaker without it?

PART 2, NARRATIVE COHERENCE. Check each transition:
- Hook to Re-hook: do they approach the topic from complementary angles?
- Re-hook to first data slide: is the transition smooth?
- Data slides: do they build from context to tension to insight to payoff?
  No two similar fact types should be adjacent.
  Each slide should feel like it NEEDS to come after the previous one.
- Data to Verdict: does the last data point set up the takeaway?
- Verdict to CTA: does the CTA feel earned? Does it drive comments?
- Does every slide connect back to the hook's promise?
- Is there a clear "so what?" moment where context becomes insight?
- Would a reader feel satisfied at the end?

If the narrative has problems, REORDER or REWRITE slides to fix the flow.
If a slide breaks the narrative, rewrite it to connect to its neighbors.

RULES:
- You MUST return EXACTLY {len(slides)} slides, same count as the input.
- Slide 1 title MUST remain EXACTLY as-is. Do NOT change it.
- Keep the same structure (title, body, footer) for each slide.
- Keep all text lowercase except tickers, numbers, proper nouns, and acronyms.
- Keep body under 15 words per slide.
- Footer: short source attribution only. No hashtags, no emojis.
- Do NOT add filler or fluff. Only improve, never dilute.
- If a slide is already strong, leave it unchanged.

FACTUAL INTEGRITY (overrides all other rules):
- NEVER replace a number with a different specific number from your own knowledge.
  Your training data may be outdated. Only use numbers that already exist in the slides.
- NEVER add new statistics, prices, or percentages that aren't already in the slides.
- If a slide uses vague language (e.g. "rising sharply"), do NOT replace it with a specific
  number. Improve the wording and framing instead.
- If you are unsure whether a specific price, percentage, or figure is current, leave it as-is.

Return your response as JSON:
{{
  "coherence_score": 8,
  "arc_analysis": "brief description of the narrative arc",
  "issues": ["any issue fixed"],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)
    # The LLM may return a plain list of slides instead of the expected dict.
    # Wrap it so callers always get a consistent dict.
    if isinstance(result, list):
        result = {"corrected_slides": result, "coherence_score": 0, "arc_analysis": "", "issues": []}
    return result


def fact_check_slides(slides: list[dict], topic: str, angle: str) -> dict:
    """Fact-check slide content and return corrected slides with a report."""
    prompt = f"""You are a rigorous financial fact-checker. Your job is to verify every claim in these slides.

Topic: {topic}
Angle: {angle}

Slides to fact-check:
{json.dumps(slides, indent=2)}

For EACH slide, do the following:
1. Identify every factual claim (numbers, percentages, dollar amounts, dates, company names, events).
2. Check if the claim is accurate based on your knowledge. If you are not confident a claim is accurate, flag it.
3. If a claim is wrong or unverifiable, correct it or replace it with a verifiable fact.

IMPORTANT:
- Keep the same slide structure (title, body, footer)
- Keep all text lowercase except ticker symbols, numbers, proper nouns, and acronyms
- Keep every slide under 15 words for body text
- Every slide must still contain a specific number, % or $
- Do NOT add filler. Maintain the punchy style
- If a slide is factually sound, keep it as-is

Return your response as JSON with this structure:
{{
  "fact_check_report": [
    {{
      "slide": 1,
      "status": "verified" or "corrected" or "flagged",
      "notes": "explanation of what was checked or changed"
    }}
  ],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 3: Layered Fact-Check ───────────────────────────────────────────────


def layered_fact_check(
    slides: list[dict],
    research_text: str,
    research_facts: list[dict],
    topic: str,
    angle: str,
) -> dict:
    """Two-layer fact-check: news-sourced claims vs supporting data claims.

    Layer A: Verify news_source claims against the original research text.
    Layer B: Verify supporting_data claims independently.

    Returns dict with:
        - "layer_a_report": list of verification results for news-sourced claims
        - "layer_b_report": list of verification results for supporting data
        - "corrected_slides": the slides with any corrections applied
    """
    prompt = f"""You are a rigorous two-layer financial fact-checker.

Topic: {topic}
Angle: {angle}

ORIGINAL RESEARCH TEXT (ground truth for Layer A):
{research_text}

EXTRACTED RESEARCH FACTS:
{json.dumps(research_facts, indent=2)}

SLIDES TO FACT-CHECK:
{json.dumps(slides, indent=2)}

Perform TWO layers of fact-checking:

LAYER A, NEWS SOURCED CLAIMS:
For each slide claim tagged as "news_source" (or any claim that references the research):
1. Find the matching fact in the original research text above.
2. Verify the slide's claim EXACTLY matches the source data (numbers, names, dates).
3. If the claim distorts, exaggerates, or misquotes the source, CORRECT it to match the research.
4. Status: "verified" (matches research), "corrected" (fixed to match), or "unverifiable" (not found in research).

LAYER B, SUPPORTING DATA CLAIMS:
For each slide claim tagged as "supporting_data" (or general knowledge claims):
1. Assess whether the claim is factually plausible based on your knowledge.
2. Check for outdated statistics, wrong magnitudes, or common misconceptions.
3. If inaccurate, replace with a verifiable alternative.
4. Status: "verified", "corrected", or "flagged" (uncertain, needs human review).

IMPORTANT:
- You MUST return EXACTLY the same number of slides as the input, never add or remove slides
- Slide 1 title MUST remain EXACTLY as-is. Do NOT change it
- Keep the same slide structure (title, body, footer, and claims if present)
- Keep all text lowercase except ticker symbols, numbers, proper nouns, and acronyms
- Keep every slide under 15 words for body text
- Every slide must still contain a specific number, % or $
- Maintain the punchy style

Return your response as JSON:
{{
  "layer_a_report": [
    {{
      "slide": 1,
      "claim": "the specific claim text",
      "source_match": "the matching text from research (or 'not found')",
      "status": "verified" or "corrected" or "unverifiable",
      "notes": "what was checked or changed"
    }}
  ],
  "layer_b_report": [
    {{
      "slide": 1,
      "claim": "the specific claim text",
      "status": "verified" or "corrected" or "flagged",
      "notes": "what was checked or changed"
    }}
  ],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "...", "claims": [...]}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 4: Conclusion Validation ────────────────────────────────────────────


def validate_conclusion(
    slides: list[dict],
    research_facts: list[dict],
    topic: str,
    angle: str,
) -> dict:
    """Check that the verdict or takeaway logically follows from the evidence presented.

    Returns dict with:
        - "verdict_slide": which slide number is the verdict
        - "evidence_used": list of facts the verdict relies on
        - "logic_valid": bool, whether the conclusion follows
        - "issues": list of logic gaps or unsupported leaps
        - "corrected_slides": slides with verdict fixed if needed
    """
    prompt = f"""You are a logic and reasoning auditor for financial content.

Topic: {topic}
Angle: {angle}

RESEARCH FACTS AVAILABLE:
{json.dumps(research_facts, indent=2)}

SLIDES:
{json.dumps(slides, indent=2)}

Your job: Validate that the VERDICT / TAKEAWAY slide logically follows from the evidence in the preceding slides.

Check the following:
1. IDENTIFY the verdict slide (usually second-to-last, the one with the main takeaway).
2. LIST every piece of evidence presented in the data slides (slides 3 to N-2).
3. CHECK: Does the verdict LOGICALLY follow from this evidence? Or does it make an unsupported leap?
4. CHECK: Is the verdict too strong for the evidence? (e.g., "X will definitely happen" when data only shows a trend)
5. CHECK: Does the verdict contradict any of the presented facts?
6. CHECK: Is there evidence in the slides that the verdict ignores?

If the conclusion has logic gaps:
- Soften overconfident claims (e.g., change "will" to "historically tends to")
- Add a qualifying fact if the verdict needs more support
- Ensure the verdict references the strongest evidence presented

RULES:
- You MUST return EXACTLY {len(slides)} slides, same count as the input
- Slide 1 title MUST remain EXACTLY as-is. Do NOT change it
- Keep all text lowercase except tickers, numbers, proper nouns, and acronyms
- Keep body under 15 words per slide
- Maintain the claims tags if present

Return your response as JSON:
{{
  "verdict_slide": 6,
  "evidence_used": ["fact 1 from slides", "fact 2 from slides"],
  "logic_valid": true or false,
  "issues": ["description of any logic gap or unsupported leap"],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 5: Narrative Coherence ──────────────────────────────────────────────


def check_narrative_coherence(
    slides: list[dict],
    topic: str,
    angle: str,
    hook: str,
) -> dict:
    """Ensure the story arc flows: news event, supporting context, insight, conclusion.

    Returns dict with:
        - "arc_analysis": assessment of the narrative flow
        - "coherence_score": 1-10 rating
        - "issues": list of flow problems
        - "corrected_slides": slides reordered/rewritten for better flow
    """
    prompt = f"""You are a narrative editor specializing in short-form financial storytelling.

Topic: {topic}
Angle: {angle}
Hook: {hook}

SLIDES:
{json.dumps(slides, indent=2)}

Your job: Ensure the slide deck tells a COHERENT STORY with the correct narrative arc.

CHECK EACH TRANSITION:
1. HOOK (Slide 1) to RE-HOOK (Slide 2): Do they approach the same topic from complementary angles? Does the re-hook work as a standalone entry point?
2. RE-HOOK to FIRST DATA SLIDE: Is the transition from hook to evidence smooth?
3. DATA SLIDES (3 to N-2): Do they follow the arc: news event, supporting context, deeper insight, payoff?
   - No two similar fact types should be adjacent (don't stack two comparisons or two trends)
   - Each slide should feel like it NEEDS to come after the previous one
   - The tension should BUILD. Start with context, escalate to the surprising or important data
4. DATA to VERDICT: Does the last data point naturally set up the takeaway?
5. VERDICT to CTA: Does the CTA feel earned by the story? Does it drive comments?

ALSO CHECK:
- Does every slide connect back to the hook's promise? (No tangential slides)
- Is there a clear "so what?" moment, the point where context becomes insight?
- Would a reader feel satisfied at the end, or like something is missing?

If the narrative has problems:
- REORDER slides if the arc is wrong (e.g., the most surprising fact should build toward the end)
- REWRITE transitions so each slide logically leads to the next
- REPLACE a weak slide's content, but you MUST keep the same total number of slides
- You MAY restructure the deck (change order, rewrite content) to improve cohesion
- You MUST NOT change the total number of slides. Always return the same count as the input

RULES:
- You MUST return EXACTLY {len(slides)} slides, same count as the input
- Keep all text lowercase except tickers, numbers, proper nouns, and acronyms
- Keep body under 15 words per slide
- Data slides should have a number, %, or $, but ONLY if verified. Use directional language if not.
- Footer: short source attribution only, no hashtags, no emojis
- Slide 1 title MUST remain the hook EXACTLY as written: "{hook}". Do NOT change it
- NEVER introduce new numbers, prices, or statistics that aren't already in the slides.
  You can reword, reorder, or replace content, but never add unverified data.

Return your response as JSON:
{{
  "arc_analysis": "brief description of the current narrative arc",
  "coherence_score": 8,
  "issues": ["issue 1", "issue 2"],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 6: Web-Search-Grounded Fact-Check ────────────────────────────────


def web_search_fact_check(
    slides: list[dict],
    topic: str,
    angle: str,
) -> dict:
    """Fact-check slide claims against live web search results.

    Extracts specific numerical claims from slides, searches for current data
    via Google News RSS, then asks the LLM to compare slide claims against
    the search results and correct any outdated or inaccurate figures.

    Returns dict with:
        - "search_report": list of claims checked with search evidence
        - "corrected_slides": slides with any corrections applied
    """
    from src.research.web_search import search_claim, format_search_results_for_prompt

    # Step 1: Extract claims that contain specific numbers
    extract_prompt = f"""Extract every specific numerical claim from these slides.
A numerical claim is any statement with a specific price, percentage, dollar amount, date, or count.

Slides:
{json.dumps(slides, indent=2)}

For each claim, also suggest a short web search query that would help verify it.

Return a JSON array:
[
  {{
    "slide": 1,
    "claim": "oil prices at $85 per barrel",
    "search_query": "crude oil price today"
  }}
]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": extract_prompt}],
    )
    claims_to_check = _parse_json(_extract_text(response))

    # Step 2: Web search for each claim
    search_context_parts = []
    for item in claims_to_check:
        query = item.get("search_query", item.get("claim", ""))
        results = search_claim(query, max_results=3)
        search_context_parts.append(
            format_search_results_for_prompt(item.get("claim", ""), results)
        )

    search_context = "\n".join(search_context_parts)

    # Step 3: Ask LLM to compare slides against search results
    verify_prompt = f"""You are a rigorous financial fact-checker with access to current search results.

Topic: {topic}
Angle: {angle}

SLIDES TO FACT-CHECK:
{json.dumps(slides, indent=2)}

CURRENT WEB SEARCH RESULTS FOR EACH CLAIM:
{search_context}

Compare every numerical claim in the slides against the search results above.
If a search result shows a different current value (e.g. slide says "$85" but search shows "$66"),
CORRECT the slide to match the current data.

IMPORTANT:
- Trust the search results over the slide content for current prices, levels, and percentages
- You MUST return EXACTLY the same number of slides as the input, never add or remove slides
- Slide 1 title MUST remain EXACTLY as-is. Do NOT change it
- Keep the same slide structure (title, body, footer)
- Keep all text lowercase except ticker symbols, numbers, proper nouns, and acronyms
- Keep every slide under 15 words for body text
- Every slide must still contain a specific number, % or $
- Maintain the punchy style
- If a claim could not be verified (no search results), flag it

Return your response as JSON:
{{
  "search_report": [
    {{
      "slide": 1,
      "claim": "the specific claim",
      "search_evidence": "what the search results show",
      "status": "verified" or "corrected" or "unverifiable",
      "notes": "explanation"
    }}
  ],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "..."}}
  ]
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": verify_prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Helpers ───────────────────────────────────────────────────────────────────


def enforce_hook_and_count(
    slides: list[dict],
    hook: str,
    expected_count: int,
) -> list[dict]:
    """Enforce that slide 1 uses the exact hook text and slide count matches.

    - Restores the hook as slide 1 title if any pass changed it.
    - If a pass returned fewer slides than expected, pads with placeholder slides.
    - If a pass returned more slides than expected, truncates (keeping CTA at end).
    """
    if not slides:
        return slides

    # Restore hook on slide 1
    slides[0]["title"] = hook

    # Fix slide count
    if len(slides) < expected_count:
        # Pad before the last slide (CTA) with context slides
        cta = slides[-1]
        while len(slides) < expected_count:
            slides.insert(-1, {
                "title": "more to consider",
                "body": "(this slide needs content, edit in step 6)",
                "footer": "",
            })
    elif len(slides) > expected_count:
        # Keep slide 1 (hook), trim middle, keep last slide (CTA)
        slides = slides[:expected_count - 1] + [slides[-1]]

    return slides


def strip_claim_tags(slides: list[dict]) -> list[dict]:
    """Remove the 'claims' metadata from slides for final output."""
    return [
        {"title": s["title"], "body": s["body"], "footer": s.get("footer", "")}
        for s in slides
    ]


def generate_image_search_queries(
    slides: list[dict],
    scripts: list[str],
    topic: str = "",
    angle: str = "",
) -> list[str]:
    """Generate web image search queries for each slide.

    Uses the video script and slide content to produce search queries
    optimized for finding relevant stock/editorial photos on the web.

    Returns a list of search query strings, one per slide.
    """
    slide_text = json.dumps(slides, indent=2)
    scripts_text = json.dumps(scripts, indent=2)
    prompt = f"""You are an expert visual researcher finding stock photos and editorial images for a short-form finance video.

Topic: {topic or 'finance'}
Angle: {angle or 'general'}

Here are the slides:
{slide_text}

Here are the voiceover scripts:
{scripts_text}

Generate ONE image search query per slide. Each query should:

1. Be optimized for stock photo search engines (Pexels, Unsplash, Google Images)
2. Describe a real-world scene or concept that matches the slide's content
3. Use concrete, visual terms — NOT abstract concepts
4. Be 3-6 words — short and specific
5. Focus on finance/business imagery: "stock trading floor", "bitcoin gold coin", "wall street building", "person checking stocks phone", "federal reserve building"
6. Vary across slides — don't repeat the same query
7. Avoid text-heavy queries — search for SCENES, not facts

Good examples: "stock market bull statue", "cryptocurrency mining farm", "person analyzing charts laptop", "wall street aerial view", "gold bars vault"
Bad examples: "S&P 500 up 5%", "earnings report Q3", "market crash 2024"

Return your response as a JSON array of strings, one query per slide:
["query for slide 1", "query for slide 2", ...]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)

    if isinstance(result, list):
        while len(result) < len(slides):
            result.append(f"{topic or 'finance'} business")
        return result[: len(slides)]

    # Fallback: generic queries
    return [f"{topic or 'finance'} stock photo" for _ in slides]


def generate_image_prompts(
    slides: list[dict],
    topic: str = "",
    angle: str = "",
) -> list[str]:
    """Generate cinematic image prompts for each slide.

    Returns a list of image generation prompts optimized for Flux,
    one per slide. Each prompt describes a visually striking,
    finance-themed scene that matches the slide content.
    """
    slide_text = json.dumps(slides, indent=2)
    prompt = f"""You are an expert visual director creating image prompts for AI image generation.

Topic: {topic or 'finance'}
Angle: {angle or 'general'}

Here are the slides:
{slide_text}

Generate ONE image prompt per slide. Each prompt should describe a cinematic realistic scene that:

1. Matches the slide's content and mood (bullish = green/gold/upward, bearish = red/dark/downward)
2. Uses finance-themed imagery: trading floors, stock tickers, cityscapes, charts, currency, corporate boardrooms, Federal Reserve building, Wall Street, tech offices
3. Is cinematic realistic — hyper-realistic photography with cinematic color grading, like a still from a Hollywood film or high-end documentary
4. Includes lighting direction (e.g., "dramatic golden hour lighting", "neon-lit", "cold blue fluorescent")
5. NEVER includes text, numbers, letters, logos, watermarks, UI elements, or any written content — this is CRITICAL
6. Is 20-40 words — specific enough for good results, not overly detailed
7. Varies across slides — don't repeat the same scene
8. Describes a concrete, real-world SCENE — NOT abstract concepts or data visualizations
9. Uses clean composition with visual depth — avoid cluttered or busy scenes
10. Favors wide or medium shots with shallow depth of field — the image will have text overlaid on top, so it needs breathing room and contrast

IMPORTANT — These images will be used as backgrounds in a video with text overlaid. They must:
- Be visually clean with areas of uniform tone (dark or blurred) where text can sit
- Show real-world scenes (people, places, objects) — NOT charts, graphs, or infographics
- Have NO text, labels, numbers, tickers, or any rendered characters in the image itself

Style keywords to use: "cinematic realistic", "hyper-realistic", "shallow depth of field", "dramatic lighting", "8k", "cinematic color grading", "film still"

IMPORTANT: Every prompt MUST start with "Cinematic realistic, " as a style prefix.

Return your response as a JSON array of strings, one prompt per slide:
["Cinematic realistic, prompt for slide 1", "Cinematic realistic, prompt for slide 2", ...]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)

    if isinstance(result, list):
        while len(result) < len(slides):
            result.append(result[-1] if result else "Cinematic realistic, finance scene, dramatic lighting, 8k")
        return result[: len(slides)]

    # Fallback: generate generic finance prompts
    return [
        f"Cinematic realistic, {topic or 'finance'} scene, dramatic lighting, hyper-realistic, 8k, cinematic color grading"
        for _ in slides
    ]


# ── Cinematic Overlay Styles ────────────────────────────────────────────────

OVERLAY_STYLE_DESCRIPTIONS = {
    "cinematic_bokeh": (
        "Shallow depth of field with beautiful bokeh orbs and soft light circles. "
        "Anamorphic oval bokeh shapes. Warm golden or cool blue tones. "
        "Shot on 85mm f/1.2 lens. Creamy background blur."
    ),
    "light_leak": (
        "Organic film light leak effect with warm amber, orange, and red tones "
        "bleeding from the edges. Authentic 35mm film exposure artifact. "
        "Subtle flare with natural light diffusion."
    ),
    "film_noir": (
        "High contrast dramatic lighting with deep shadows. "
        "Strong directional backlight creating silhouettes. "
        "Teal and orange color grading. Moody atmospheric haze."
    ),
    "volumetric_light": (
        "Volumetric god rays streaming through atmosphere. "
        "Dust particles catching light beams. Dramatic directional lighting. "
        "Cinematic fog or haze with visible light shafts."
    ),
    "neon_glow": (
        "Cyberpunk-inspired neon lighting with vivid electric blue, magenta, "
        "and cyan reflections. Wet surfaces reflecting neon. "
        "High-tech atmosphere with glowing light sources."
    ),
    "golden_hour": (
        "Warm golden hour sunlight with long shadows and rich amber tones. "
        "Lens flare from low sun. Soft warm rim lighting on subjects. "
        "Vermeer-style directional warm light."
    ),
    "film_grain": (
        "Authentic 35mm Kodak film grain texture with subtle color shifts. "
        "Analog photography aesthetic. Slightly desaturated with lifted blacks. "
        "Natural imperfections and organic texture."
    ),
    "auto": (
        "Automatically select the best cinematic style for each slide based on "
        "its content, mood, and position in the sequence."
    ),
}


def generate_overlay_prompts(
    slides: list[dict],
    topic: str = "",
    angle: str = "",
    overlay_style: str = "auto",
) -> list[list[str]]:
    """Generate cinematic overlay image prompts for each slide.

    Creates prompts for AI-generated foreground overlay images that add
    cinematic depth, atmosphere, and visual storytelling to slides.
    Multiple prompts per slide are linked to sentences so overlays
    transition as the narration progresses.

    Args:
        slides: List of slide dicts.
        topic: Content topic.
        angle: Content angle.
        overlay_style: One of the OVERLAY_STYLE_DESCRIPTIONS keys or 'auto'.

    Returns:
        List of lists — each inner list contains one overlay prompt per
        sentence in the slide body, so overlays transition with narration.
    """
    slide_text = json.dumps(slides, indent=2)

    style_context = ""
    if overlay_style != "auto" and overlay_style in OVERLAY_STYLE_DESCRIPTIONS:
        style_context = f"""
Use this specific cinematic style for ALL overlays:
Style: {overlay_style}
Description: {OVERLAY_STYLE_DESCRIPTIONS[overlay_style]}
"""
    else:
        style_context = f"""
Available cinematic styles (choose the best one per slide):
{json.dumps(OVERLAY_STYLE_DESCRIPTIONS, indent=2)}

Match the style to each slide's mood:
- Hook slides (slide 1): Use dramatic styles (volumetric_light, light_leak, neon_glow) for maximum visual impact in the first 0.3 seconds
- Context slides: Use subtle styles (cinematic_bokeh, film_grain, golden_hour) that enhance readability
- Payoff slides: Use bold styles (neon_glow, light_leak) for emotional peak
- CTA slides: Use clean styles (cinematic_bokeh, golden_hour) that keep focus on the call to action
"""

    prompt = f"""You are an award-winning cinematographer and visual effects artist creating overlay image prompts for premium TikTok/Instagram content.

These overlays are TRANSPARENT foreground elements layered ON TOP of slide backgrounds to add
cinematic depth, atmosphere, and professional polish. Think of them like the atmospheric effects
in high-budget film: the organic bokeh orbs in a Wong Kar-wai film, the golden light leaks of a
Terrence Malick sunrise, the volumetric haze in a Denis Villeneuve sci-fi scene.

Topic: {topic or 'finance'}
Angle: {angle or 'general'}

{style_context}

Here are the slides:
{slide_text}

Generate MULTIPLE overlay image prompts per slide — one for each sentence in the slide body.
This creates visual rhythm: as the narration reads each sentence, the overlay transitions to
match the sentence's content and mood. Each slide should have 2-4 overlay prompts.

CRITICAL DESIGN PRINCIPLES (follow strictly):

1. LESS IS MORE — One beautiful, well-placed effect beats three stacked effects. Each overlay
   should have ONE dominant visual element (e.g., bokeh orbs OR light leak, not both)
2. EDGE COMPOSITION — All visual elements must be concentrated at edges, corners, and borders.
   The center 60% of the frame must remain clear for text readability
3. DEPTH & DIMENSION — Create parallax-like depth: foreground out-of-focus elements in front,
   atmospheric haze behind. This gives the illusion of shooting through a real camera
4. NATURAL LIGHTING — Match real-world light behavior. Light leaks bleed from one direction.
   Bokeh orbs vary in size (larger = closer to camera). God rays converge toward a source
5. COLOR HARMONY — Use analogous colors (warm: amber/gold/peach, cool: teal/blue/indigo).
   Never use more than 2-3 colors per overlay. Match the sentence's emotional tone
6. ORGANIC TEXTURE — Include subtle imperfections: slight lens dust, micro film grain, natural
   flare artifacts. This sells the "shot on cinema camera" look
7. SMOOTH TRANSITIONS — Each overlay within a slide should feel like a camera movement
   (rack focus, dolly, pan) rather than a hard cut between effects

Each prompt must:
- Use real cinematography terms: bokeh, anamorphic, f-stop, rack focus, lens bloom, halation
- Specify light direction and color temperature (3200K warm / 5600K daylight / 6500K cool)
- Be 30-50 words — vivid and specific enough for photorealistic AI generation
- Never include text, logos, watermarks, or recognizable faces
- Describe the effect as if directing a camera operator ("soft golden bokeh orbs drifting
  from upper-left corner, out of focus at f/1.4, warm 3200K amber tones")

Return your response as a JSON array of arrays. Each inner array has one prompt per sentence:
[
  ["slide 1 sentence 1 overlay", "slide 1 sentence 2 overlay", "slide 1 sentence 3 overlay"],
  ["slide 2 sentence 1 overlay", "slide 2 sentence 2 overlay"],
  ...
]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)

    fallback_prompt = "cinematic bokeh overlay, warm light leak, 85mm f/1.2, transparent"

    if isinstance(result, list) and len(result) > 0:
        # Handle both old format (list[str]) and new format (list[list[str]])
        normalized: list[list[str]] = []
        for item in result:
            if isinstance(item, list):
                normalized.append(item if item else [fallback_prompt])
            elif isinstance(item, str):
                # Old flat format — wrap in a list
                normalized.append([item])
            else:
                normalized.append([fallback_prompt])

        while len(normalized) < len(slides):
            normalized.append(normalized[-1] if normalized else [fallback_prompt])
        return normalized[: len(slides)]

    # Fallback: generate style-appropriate overlay prompts
    fallback_style = OVERLAY_STYLE_DESCRIPTIONS.get(overlay_style, OVERLAY_STYLE_DESCRIPTIONS["cinematic_bokeh"])
    fb = f"Transparent cinematic overlay, {fallback_style[:80]}, 8k, photorealistic"
    return [[fb] for _ in slides]


def generate_video_script(
    slides: list[dict],
    topic: str = "",
    angle: str = "",
) -> list[str]:
    """Generate a natural voiceover script for each slide.

    Returns a list of strings, one narration per slide. The narration is
    conversational, 2-3 sentences, and designed to be read aloud by a TTS
    engine while the viewer looks at the slide.
    """
    slide_text = json.dumps(slides, indent=2)
    prompt = f"""You are a top-tier voiceover scriptwriter for short-form finance videos on TikTok and Instagram Reels.

Topic: {topic or 'finance'}
Angle: {angle or 'general'}

Here are the slides for the video:
{slide_text}

Write a compelling voiceover script for EACH slide. Follow these rules precisely:

STRUCTURE & PACING:
1. Each slide's narration: 2-4 sentences (20-40 words). Vary sentence length for rhythm — mix punchy 3-word statements with longer explanations.
2. Slide 1 (hook): Open with a provocative question, a surprising contrast, or a bold claim that makes scrollers stop. Never "hey guys" or "welcome to". Example patterns: "Most people think X... they're wrong.", "Here's what nobody tells you about X."
3. Middle slides: Each should deliver ONE clear insight. Build tension — set up "the problem" before "the answer". Use cause → effect structure.
4. Last slide (CTA): End with a specific, natural call to action. Don't just say "follow" — give them a reason. Example: "Follow for the next breakdown" or "Comment which one surprised you most."
5. The script must flow as one continuous story across all slides — each slide should feel like the next logical beat.

TONE & DELIVERY:
6. Sound like a sharp analyst briefing a friend, not a textbook or a hype influencer.
7. Use power words that create urgency: "quietly", "just happened", "most people missed", "here's the catch".
8. Include natural speech markers for TTS: brief pauses (use "..."), rhetorical questions, emphasis through repetition.
9. Vary your openings — don't start every slide the same way. Mix statements, questions, and "here's why" transitions.

FACTUAL ACCURACY (NON-NEGOTIABLE):
10. ONLY use numbers, percentages, prices, and statistics that appear in the slide content. NEVER invent or estimate data.
11. If a slide uses directional language ("rising sharply", "near highs"), keep it directional in the script. Do NOT convert vague language into specific numbers.
12. You may rephrase and add context, but the underlying facts must come exclusively from the slides. When in doubt, be vaguer, not more specific.
13. Add "why it matters" context — explain implications and connect dots — but never fabricate the supporting data.

Return your response as a JSON array of strings, one per slide:
["script for slide 1", "script for slide 2", ...]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)

    if isinstance(result, list):
        # Ensure we have exactly one script per slide
        while len(result) < len(slides):
            result.append(result[-1] if result else "")
        return result[: len(slides)]

    # Fallback: generate basic scripts from slide text
    return [
        f"{s.get('title', '')}. {s.get('body', '')}"
        for s in slides
    ]


def regenerate_video_script(
    slides: list[dict],
    current_scripts: list[str],
    feedback: str,
    topic: str = "",
    angle: str = "",
) -> list[str]:
    """Regenerate voiceover scripts incorporating user feedback.

    Takes the current scripts and user feedback, returns improved scripts.
    """
    slide_text = json.dumps(slides, indent=2)
    scripts_text = json.dumps(current_scripts, indent=2)
    prompt = f"""You are a top-tier voiceover scriptwriter for short-form finance videos on TikTok and Instagram Reels.

Topic: {topic or 'finance'}
Angle: {angle or 'general'}

Here are the slides for the video:
{slide_text}

Here are the current voiceover scripts:
{scripts_text}

The user has provided the following feedback on the scripts:
{feedback}

Rewrite the voiceover scripts incorporating the user's feedback. Follow these rules:

STRUCTURE & PACING:
1. Each slide's narration: 2-4 sentences (20-40 words). Vary sentence length for rhythm.
2. Slide 1 (hook): Open with a provocative question, surprising contrast, or bold claim.
3. Middle slides: Each should deliver ONE clear insight with cause → effect structure.
4. Last slide (CTA): End with a specific, natural call to action.
5. The script must flow as one continuous story across all slides.

TONE & DELIVERY:
6. Sound like a sharp analyst briefing a friend.
7. Use power words that create urgency.
8. Include natural speech markers for TTS: brief pauses (use "..."), rhetorical questions.
9. Vary your openings.

FACTUAL ACCURACY (NON-NEGOTIABLE):
10. ONLY use numbers, percentages, prices, and statistics that appear in the slide content.
11. If a slide uses directional language, keep it directional. Do NOT invent specific numbers.
12. You may rephrase and add context, but facts must come exclusively from the slides.

IMPORTANT: Address the user's feedback while maintaining quality and factual accuracy.

Return your response as a JSON array of strings, one per slide:
["script for slide 1", "script for slide 2", ...]

Return ONLY the JSON array, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    result = _parse_json(text)

    if isinstance(result, list):
        while len(result) < len(slides):
            result.append(result[-1] if result else "")
        return result[: len(slides)]

    # Fallback: return current scripts unchanged
    return current_scripts


def generate_tiktok_metadata(
    slides: list[dict],
    topic: str,
    angle: str,
    hook: str,
) -> dict:
    """Generate TikTok carousel title and description."""
    prompt = f"""You are a TikTok SEO expert for finance content.

Topic: {topic}
Angle: {angle}
Hook: {hook}

Slides:
{json.dumps(slides, indent=2)}

Generate a TikTok carousel post package:

1. VIDEO TITLE: A scroll-stopping title for the carousel post. Should be punchy, include a number, and create curiosity. Max 60 characters.

2. DESCRIPTION: A TikTok carousel description that:
   - Summarises the value of the carousel in 1 to 2 sentences
   - Adds extra insight or context not in the slides (bonus value for readers)
   - Includes a question that invites comments (e.g. "what's your take?" or "would you buy or sell?")
   - Includes relevant hashtags (5 to 8 hashtags)
   - Ends with this exact disclaimer: "views are my own, not my employer's. educational content only, not financial advice."
   - MUST be 200+ characters total (TikTok carousel SEO requirement)

Return your response as JSON:
{{
  "title": "the video title",
  "description": "the full description with hashtags and disclaimer"
}}

Return ONLY the JSON, no other text."""

    response = create_message(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def _extract_text(response) -> str:
    """Pull the first text block out of an Anthropic response."""
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""


def _parse_json(text: str):
    """Extract and parse JSON from a model response that may contain markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fix common LLM JSON issues: trailing commas before } or ]
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        # Fix unescaped newlines inside JSON string values
        cleaned = re.sub(
            r'(?<=": ")(.*?)(?=")',
            lambda m: m.group(0).replace("\n", "\\n"),
            cleaned,
        )
        return json.loads(cleaned)


def extract_entity_mentions(
    scripts: list[str],
) -> list[list[dict]]:
    """Extract company and person names from voiceover scripts.

    Uses Claude to identify entities worth showing as visual overlays
    (company logos, person headshots) during the video.

    Args:
        scripts: List of voiceover script strings, one per slide.

    Returns:
        List of lists (one per slide). Each inner list contains dicts:
        {
            "name": "Tesla",
            "type": "company" | "person",
            "search_query": "Tesla logo transparent png",
            "char_start": 45,   # character offset in the script
            "char_end": 50,
        }
    """
    combined = "\n---\n".join(
        f"SLIDE {i}: {s}" for i, s in enumerate(scripts)
    )

    prompt = f"""Analyze these voiceover scripts and extract every mention of a specific company, brand, or notable person that would benefit from showing their logo or photo as a visual overlay.

{combined}

Rules:
1. Only extract SPECIFIC named entities (e.g. "Tesla", "Warren Buffett", "Federal Reserve") — not generic terms like "the market" or "investors".
2. For each entity, provide:
   - "name": The entity name as it appears in the script
   - "type": "company" or "person"
   - "search_query": A search query to find their logo (for companies) or headshot (for people). Format: "CompanyName logo transparent png" or "PersonName headshot portrait".
   - "slide_index": Which slide (0-based) the mention is in
   - "char_start": Character offset where the name starts within that slide's script
   - "char_end": Character offset where the name ends
3. If the same entity appears in multiple slides, include each mention separately.
4. Only include entities where a visual (logo/photo) would genuinely enhance understanding.

Return a JSON array of objects. Return ONLY the JSON array, no other text.
If no entities found, return [].
"""

    try:
        response = create_message(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _extract_text(response)
        entities = _parse_json(text)
    except Exception:
        entities = []

    if not isinstance(entities, list):
        return [[] for _ in scripts]

    # Group by slide index
    result: list[list[dict]] = [[] for _ in scripts]
    for ent in entities:
        idx = ent.get("slide_index", 0)
        if 0 <= idx < len(scripts):
            result[idx].append(ent)

    return result
