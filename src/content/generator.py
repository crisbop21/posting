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

STORYTELLING (critical):
- The slides MUST tell a cohesive story with a clear narrative arc.
- Each slide should logically lead to the next — the viewer should NEED to swipe.
- Structure the data slides as a build-up: context → tension → insight → payoff.
- Never repeat the same type of fact back-to-back; alternate between comparison, trend, and surprise.
- The verdict slide should feel like a satisfying conclusion to the story, not a random opinion.

FOOTER RULES:
- Footer is a SHORT source attribution (e.g. "source: bloomberg", "data: fed reserve", "per SEC filing").
- NO hashtags. NO emojis. NO self-promotion.
- Keep footers to 3-5 words max. Leave blank if no source needed.
"""


def fact_check_news(news_text: str) -> list[dict]:
    """Fact-check news articles and return corrected versions of any that are inaccurate.

    Returns a list of dicts (one per article), each with:
        - "index": 1-based article number matching the input list
        - "status": "verified" or "corrected"
        - "corrected_title": factually accurate title (same as original if verified)
        - "corrected_summary": factually accurate summary (same as original if verified)
        - "reason": brief explanation of what was checked or changed
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a rigorous financial news fact-checker. Review each numbered article below.

{news_text}

For EACH article:
1. Check if the headline and summary are consistent (no clickbait mismatch).
2. Check if any specific claims (numbers, %, $, company names, events) seem plausible.
3. If an article has inaccurate claims, misleading framing, or outdated info — CORRECT it.
   Rewrite the title and summary so they are factually accurate while keeping the same trending topic.
4. If the article is accurate, keep the original title and summary as-is.

Return your response as a JSON array — one object per article:
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

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 1: Extract structured facts from research ──────────────────────────

def extract_news_facts(research_text: str) -> list[dict]:
    """Extract structured facts from research text.

    Returns a list of dicts, each with:
        - "fact": the specific claim or data point
        - "source": where it came from (e.g. "Bloomberg", "r/wallstreetbets")
        - "date": when it was published/reported
        - "type": "news_event" | "data_point" | "market_move" | "opinion"
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a financial research analyst. Extract every specific, verifiable fact from the research below.

{research_text}

For each fact, capture:
1. The exact claim (with specific numbers, %, $, dates, company names).
2. The source it came from.
3. The date it was reported.
4. The type: "news_event" (something that happened), "data_point" (a statistic or metric),
   "market_move" (price/index change), or "opinion" (analyst view or prediction).

Be exhaustive — capture every number, percentage, dollar amount, date, and named entity.
Do NOT add facts that aren't in the research. Only extract what's actually there.

Return a JSON array:
[
  {{
    "fact": "S&P 500 dropped 3.2% on Feb 19, 2026",
    "source": "Bloomberg",
    "date": "2026-02-19",
    "type": "market_move"
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
    key_facts: list[dict] | None = None,
    research_text: str = "",
) -> list[dict]:
    """Generate structured slide content grounded in research facts."""
    client = anthropic.Anthropic()

    facts_block = ""
    if key_facts:
        facts_block = f"""
VERIFIED SOURCE FACTS (use these as your primary data):
{json.dumps(key_facts, indent=2)}

GROUNDING RULES:
- Data slides MUST use facts from the list above wherever possible.
- If you add a supporting claim not in the list, tag it as "claim_source": "supporting_data".
- Facts from the list should be tagged as "claim_source": "news_source".
- Do NOT invent numbers. Every number must come from the facts list or be a widely known statistic.
"""

    research_block = ""
    if research_text:
        research_block = f"""
ORIGINAL RESEARCH CONTEXT (for tone and narrative — facts above take priority):
{research_text}
"""

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
{facts_block}{research_block}
STRICT REQUIREMENTS:
- Slide 1 title MUST be the hook above (copy it exactly)
- Slide 2 MUST be a re-hook — a standalone entry point with a different angle on the same topic
- Slides 3 to {slide_count - 2} MUST each contain one key data fact (number, %, or $)
- Slide {slide_count - 1} MUST be the verdict — one-sentence takeaway
- Slide {slide_count} MUST be a CTA ("save this", "comment below", "follow for more")
- ALL text must be lowercase except ticker symbols ($AAPL, $BTC) and numbers
- Each slide body: ONE sentence, under 15 words, must include a number/% or $
- No filler. Every slide must make the viewer want to swipe.

STORYTELLING:
- The slides must tell ONE cohesive story — each slide builds on the previous.
- Data slides should follow: context → tension → insight → payoff.
- Never repeat the same type of fact back-to-back; alternate comparison, trend, and surprise.
- A reader who sees slides 1-N should feel like they followed a narrative, not read a list.

Return your response as a JSON array of slide objects. Each slide must have:
- "title": The headline (under 15 words, lowercase except tickers/numbers)
- "body": One sentence of content (under 15 words, must include a number)
- "footer": Short source attribution only (e.g. "source: bloomberg"). NO hashtags, NO emojis. Leave blank if no source.
- "claim_source": "news_source" if the fact comes from the research, "supporting_data" if it's additional context, or "none" for hook/CTA slides.

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


def add_value_pass(slides: list[dict], topic: str, angle: str, audience: str) -> list[dict]:
    """Final iteration: maximize reader value without adding clutter.

    Looks for opportunities to replace generic claims with sharper data,
    add actionable insight, or strengthen the narrative thread.
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a senior finance content editor. Your reader is: {audience}.

Topic: {topic}
Angle: {angle}

Here are the current slides:
{json.dumps(slides, indent=2)}

Your job: ONE final pass to maximize the value a reader gets from this deck.
Ask yourself for each slide:
- Could a vague claim be replaced with a sharper, more specific number?
- Is there an actionable insight missing that the reader could use TODAY?
- Does this slide teach something or just state the obvious?
- Does the narrative arc build tension and deliver a satisfying payoff?
- Is every slide earning its place — would the deck be weaker without it?

RULES:
- Keep the same number of slides, same structure (title, body, footer).
- Keep all text lowercase except tickers and numbers.
- Keep body under 15 words per slide. Every slide must have a number/% or $.
- Footer: short source attribution only. No hashtags, no emojis.
- Do NOT add filler or fluff. Only improve — never dilute.
- If a slide is already strong, leave it unchanged.

Return your response as a JSON array of the improved slides:
[
  {{"title": "...", "body": "...", "footer": "..."}}
]

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Layer 3A: Fact-check news-sourced claims against research ─────────────────

def fact_check_news_claims(
    slides: list[dict],
    key_facts: list[dict],
    research_text: str,
) -> dict:
    """Verify slides tagged as news_source against the original research.

    Returns dict with "fact_check_report" and "corrected_slides".
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a financial fact-checker. Your job: verify that every NEWS-SOURCED claim
in these slides accurately reflects the original research data.

ORIGINAL RESEARCH:
{research_text}

EXTRACTED FACTS:
{json.dumps(key_facts, indent=2)}

SLIDES TO CHECK:
{json.dumps(slides, indent=2)}

For each slide where "claim_source" is "news_source":
1. Find the matching fact in the research/extracted facts.
2. Verify the slide's numbers, dates, and entities match the source EXACTLY.
3. If the slide misquotes, exaggerates, or takes a fact out of context — correct it.
4. If the claim cannot be traced to the research, flag it.

For slides with "claim_source" = "supporting_data" or "none", mark as "skipped".

RULES: Keep same structure. Keep text lowercase except tickers. Body under 15 words.
Every slide must have a number/% or $. No hashtags in footer.

Return JSON:
{{
  "fact_check_report": [
    {{"slide": 1, "status": "verified" or "corrected" or "flagged", "notes": "..."}}
  ],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "...", "claim_source": "..."}}
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


# ── Layer 3B: Fact-check supporting data claims independently ─────────────────

def fact_check_supporting_claims(slides: list[dict], topic: str) -> dict:
    """Verify slides tagged as supporting_data using general knowledge.

    Returns dict with "fact_check_report" and "corrected_slides".
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a financial fact-checker. Your job: verify every SUPPORTING DATA claim
in these slides. These are contextual facts not from the original news — verify them
independently based on your knowledge.

Topic: {topic}

SLIDES TO CHECK:
{json.dumps(slides, indent=2)}

For each slide where "claim_source" is "supporting_data":
1. Identify every factual claim (numbers, %, $, dates, companies, events).
2. Verify accuracy. If uncertain, flag it.
3. If wrong or unverifiable, correct it with a verifiable fact.

For slides with "claim_source" = "news_source" or "none", mark as "skipped".

RULES: Keep same structure. Keep text lowercase except tickers. Body under 15 words.
Every slide must have a number/% or $. No hashtags in footer.

Return JSON:
{{
  "fact_check_report": [
    {{"slide": 1, "status": "verified" or "corrected" or "flagged" or "skipped", "notes": "..."}}
  ],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "...", "claim_source": "..."}}
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


# ── Layer 4: Validate conclusions ─────────────────────────────────────────────

def validate_conclusions(
    slides: list[dict],
    key_facts: list[dict],
    topic: str,
    hook: str,
) -> dict:
    """Check that the verdict/takeaway logically follows from the evidence.

    Returns dict with "valid" (bool), "issues" (list), and "corrected_slides".
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a financial logic checker. Your job: verify that the CONCLUSION slides
(verdict and any takeaway) logically follow from the evidence presented.

Topic: {topic}
Hook: {hook}

VERIFIED FACTS USED:
{json.dumps(key_facts, indent=2)}

SLIDES:
{json.dumps(slides, indent=2)}

Check:
1. Does the verdict slide (second-to-last) logically follow from the data slides?
2. Are there logical leaps — conclusions that the data doesn't support?
3. Does the hook's promise get fulfilled by the end?
4. Is there any correlation-vs-causation error?
5. Are predictions clearly framed as possibilities, not certainties?

If the conclusions are sound, return them unchanged.
If any conclusion is a logical stretch, rewrite it to be defensible.

RULES: Keep same structure. Keep text lowercase except tickers. Body under 15 words.

Return JSON:
{{
  "valid": true or false,
  "issues": ["list of any logical problems found"],
  "corrected_slides": [
    {{"title": "...", "body": "...", "footer": "...", "claim_source": "..."}}
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


# ── Layer 5: Narrative coherence check ────────────────────────────────────────

def check_narrative_coherence(
    slides: list[dict],
    hook: str,
    topic: str,
) -> list[dict]:
    """Ensure the story arc flows logically and every slide connects to the hook.

    Returns the corrected slide list.
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a senior story editor for finance content. Review this slide deck's NARRATIVE ARC.

Hook: {hook}
Topic: {topic}

SLIDES:
{json.dumps(slides, indent=2)}

Check the following:
1. Does each slide logically lead to the next? Is there a clear thread?
2. Is there a build-up: news event → context → tension → insight → payoff?
3. Does every slide feel connected to the hook's promise?
4. Are there any jumps where the reader would think "wait, where did this come from?"
5. Does the verdict feel like a satisfying conclusion to THIS specific story?

If the arc is broken or a slide feels disconnected:
- Rewrite it to bridge the gap while keeping the same factual claim.
- Do NOT change the facts — only improve the framing and transitions.

RULES: Keep same structure. Keep text lowercase except tickers. Body under 15 words.
Footer: short source attribution only. No hashtags. Keep "claim_source" tags intact.

Return your response as a JSON array of the slides (improved where needed):
[
  {{"title": "...", "body": "...", "footer": "...", "claim_source": "..."}}
]

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    text = _extract_text(response)
    return _parse_json(text)


# ── Legacy wrapper for backward compatibility ─────────────────────────────────

def fact_check_slides(slides: list[dict], topic: str, angle: str) -> dict:
    """Simple fact-check fallback (used when key_facts are not available)."""
    client = anthropic.Anthropic()

    prompt = f"""You are a rigorous financial fact-checker. Verify every claim in these slides.

Topic: {topic}
Angle: {angle}

Slides to fact-check:
{json.dumps(slides, indent=2)}

For EACH slide:
1. Identify every factual claim.
2. Check accuracy. If uncertain, flag it.
3. If wrong, correct it.

RULES: Keep same structure, lowercase except tickers, body under 15 words,
must have number/% or $. No hashtags in footer.

Return JSON:
{{
  "fact_check_report": [
    {{"slide": 1, "status": "verified" or "corrected" or "flagged", "notes": "..."}}
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
