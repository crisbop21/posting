"""Generate slide content from research using Claude."""

import json
import re

import anthropic

HOOK_FORMULAS = """
TOP 10 HOOK FORMULAS (ranked by proven performance — use these as templates):

1. DATA DIG — "[Stock/event] just [happened]. I analyzed [X] days/years of data to find out what happens every time."
   Best performer: 4,700 views.

2. COMPARISON SHOCK — "[Thing A] is up [X%]. [Thing B] is up [Y%]. You think [belief]. You're wrong."
   Best: 2,465 views.

3. HIDDEN GEM LIST — "I [research effort] to find [X] [things] that [remarkable outcome] since [time span]."
   Best: 1,390 views.

4. CONTRARIAN CALL — "Stop [common action]. This is what [topic] actually [truth/needs]."
   Best: 946 views.

5. COUNTDOWN TRIGGER — "Next week could [dramatic consequence]. These [X] events matter more than you think."
   Best: 939 views.

6. AUTHORITY INSIDER — "[Institution/Billionaire] just [dramatic action]. [Bold claim about implications]."
   Best: 832 views.

7. OBSCURE STORY — "[Country/Company] [dramatic event X months ago]. The [name] trade explained."
   Best: 813 views.

8. EXPERIENCE PLAY — "After [X] years in [field]. Here is what they don't tell you about [topic]."
   Best: 807 views.

9. SMART MONEY TRACKER — "[Billionaire/Fund] just put $[X]B into [stock] while everyone is selling."
   Best: 715 views.

10. PANIC REVERSAL — "[Stock/Market] is down [X%] you panic sell. Here is the [time] that save you."
    Best: 479 views.
"""

HOOK_RULES = """
HOOK RULES (based on pattern analysis):
- Hooks with specific numbers get 49% more views (777 avg vs 523 without)
- Hooks using "you/your" direct address get 26% more views (822 avg vs 652)
- Mid-length hooks (11-18 words) are the sweet spot (750 avg views)
- Data claim hooks ("analyzed", "data", "years", "%") average 733 views
- NEVER use vague hooks without specific numbers (avg 200 views or less)
- NEVER use generic hype language ("this stock went up 800%")
- ALWAYS create an "open loop" — promise a specific reveal
"""

SLIDE_RULES = """
SLIDE RULES (strict — follow exactly):
- Maximum 6-8 slides per carousel.
- Each slide: ONE idea, ONE sentence, under 15 words.
- Every slide MUST have a specific number, percentage, or dollar amount.
- Slide 1 = Hook. Slide 2 = Re-hook (works standalone for mid-scroll entry). Final slide = CTA.
- No filler slides. Every slide must make the viewer want to swipe.
- Text style: all lowercase except ticker symbols and numbers.

Slide structure:
- Slide 1: Hook — the headline event + open loop
- Slide 2: Re-hook — standalone entry point with a different angle
- Slides 3-N: Data — one key fact per slide, comparative or surprising
- Second-to-last slide: Verdict — the takeaway in one sentence
- Last slide: CTA — "save this", "comment below", "follow for update"
"""


def fact_check_news(news_text: str) -> list[dict]:
    """Fact-check news articles and return a verdict for each.

    Returns a list of dicts, each with:
        - "index": 1-based article number matching the input list
        - "title": original headline
        - "status": "verified" or "flagged"
        - "reason": brief explanation
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a rigorous financial news fact-checker. Review each numbered article below
and determine if it appears factually accurate and genuinely recent.

{news_text}

For EACH article:
1. Check if the headline and summary are consistent (no clickbait mismatch).
2. Check if any specific claims (numbers, %, $, company names, events) seem plausible.
3. Flag anything that looks like misinformation, outdated recycled news, or AI-generated spam.

Return your response as a JSON array — one object per article:
[
  {{
    "index": 1,
    "title": "original title",
    "status": "verified" or "flagged",
    "reason": "brief explanation"
  }}
]

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def suggest_topics(research_text: str, audience: str) -> list[dict]:
    """Analyse research data and return 10 potential topic ideas."""
    client = anthropic.Anthropic()

    prompt = f"""You are a finance content strategist. Your audience is: {audience}.

Based on the trending research below, suggest exactly 10 potential slide-deck topics.
Pick the most timely, engaging, and shareable angles from the data.
Prioritise topics that lend themselves to specific numbers, data points, and open-loop hooks.

{research_text}

Return your response as a JSON array of 10 objects, each with:
- "title": A concise topic title (max 10 words)
- "description": One sentence explaining what this deck would cover

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def generate_hooks(
    topic: str,
    angle: str,
    tone: str,
    audience: str,
) -> list[dict]:
    """Generate 10 hook options using the proven hook formulas."""
    client = anthropic.Anthropic()

    prompt = f"""You are a viral finance content creator. Your audience is: {audience}.
Tone: {tone}.

Topic: {topic}
Angle / key information: {angle}

{HOOK_FORMULAS}

{HOOK_RULES}

Generate exactly 10 different hook options for the opening slide of a slide deck on this topic.
Each hook MUST follow one of the 10 proven formulas above. Use a different formula for each hook.
Every hook MUST include a specific number, percentage, or dollar amount.
Every hook MUST be 11-18 words long (the sweet spot).
Every hook MUST use "you/your" direct address where possible.
Every hook MUST create an open loop (promise a reveal).

Return your response as a JSON array of 10 objects, each with:
- "hook": The hook text (11-18 words, must include a specific number)
- "style": The formula name used (e.g. "Data Dig", "Comparison Shock", "Hidden Gem List", etc.)

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
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
) -> list[dict]:
    """Generate structured slide content using strict slide rules."""
    client = anthropic.Anthropic()

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

STRICT REQUIREMENTS:
- Slide 1 title MUST be the hook above (copy it exactly)
- Slide 2 MUST be a re-hook — a standalone entry point with a different angle on the same topic
- Slides 3 to {slide_count - 2} MUST each contain one key data fact (number, %, or $)
- Slide {slide_count - 1} MUST be the verdict — one-sentence takeaway
- Slide {slide_count} MUST be a CTA ("save this", "comment below", "follow for more")
- ALL text must be lowercase except ticker symbols ($AAPL, $BTC) and numbers
- Each slide body: ONE sentence, under 15 words, must include a number/% or $
- No filler. Every slide must make the viewer want to swipe.

Return your response as a JSON array of slide objects. Each slide must have:
- "title": The headline (under 15 words, lowercase except tickers/numbers)
- "body": One sentence of content (under 15 words, must include a number)
- "footer": A small source note or hashtag

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def fact_check_slides(slides: list[dict], topic: str, angle: str) -> dict:
    """Fact-check slide content and return corrected slides with a report."""
    client = anthropic.Anthropic()

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
- Keep all text lowercase except ticker symbols and numbers
- Keep every slide under 15 words for body text
- Every slide must still contain a specific number, % or $
- Do NOT add filler — maintain the punchy style
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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def generate_tiktok_metadata(
    slides: list[dict],
    topic: str,
    angle: str,
    hook: str,
) -> dict:
    """Generate TikTok carousel title and description."""
    client = anthropic.Anthropic()

    prompt = f"""You are a TikTok SEO expert for finance content.

Topic: {topic}
Angle: {angle}
Hook: {hook}

Slides:
{json.dumps(slides, indent=2)}

Generate a TikTok carousel post package:

1. VIDEO TITLE: A scroll-stopping title for the carousel post. Should be punchy, include a number, and create curiosity. Max 60 characters.

2. DESCRIPTION: A TikTok carousel description that:
   - Summarises the value of the carousel in 1-2 sentences
   - Adds extra insight or context not in the slides (bonus value for readers)
   - Includes relevant hashtags (5-8 hashtags)
   - Ends with this exact disclaimer: "views are my own, not my employer's. educational content only — not financial advice."
   - MUST be 200+ characters total (TikTok carousel SEO requirement)

Return your response as JSON:
{{
  "title": "the video title",
  "description": "the full description with hashtags and disclaimer"
}}

Return ONLY the JSON, no other text."""

    response = client.messages.create(
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
    return json.loads(text)
