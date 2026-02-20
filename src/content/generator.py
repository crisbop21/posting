"""Generate slide content from research using Claude."""

import json
import re

import anthropic


def suggest_topics(research_text: str, audience: str) -> list[dict]:
    """Analyse research data and return 10 potential topic ideas.

    Each topic has a short 'title' and a one-sentence 'description'.
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a finance content strategist. Your audience is: {audience}.

Based on the trending research below, suggest exactly 10 potential slide-deck topics.
Pick the most timely, engaging, and shareable angles from the data.

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
    """Generate 10 hook options for the first slide of a deck.

    Returns a list of dicts with 'hook' (the text) and 'style' (e.g. question,
    stat-based, bold claim, etc.).
    """
    client = anthropic.Anthropic()

    prompt = f"""You are a viral finance content creator. Your audience is: {audience}.
Tone: {tone}.

Topic: {topic}
Angle / key information: {angle}

Generate exactly 10 different hook options for the opening slide of a slide deck on this topic.
Vary the style — include question hooks, stat-based hooks, bold claims, curiosity gaps, contrarian takes, etc.

Return your response as a JSON array of 10 objects, each with:
- "hook": The hook text (one punchy sentence, max 12 words)
- "style": The hook style label (e.g. "Question", "Bold Claim", "Stat-Based", "Curiosity Gap", "Contrarian")

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
    """Generate structured slide content using a chosen topic, angle, and hook.

    Returns:
        List of slide dicts with 'title', 'body', and 'footer' keys.
    """
    client = anthropic.Anthropic()

    system_prompt = f"""You are an expert social media content creator specializing in finance.
You create slide decks that go viral on TikTok and Instagram.

Target audience: {audience}
Tone: {tone}

Style guidelines:
{style_notes}

IMPORTANT RULES:
- Each slide must stand alone as a compelling piece of content
- Use data points and numbers whenever possible
- The first slide MUST use the provided hook exactly as given
- Last slide should have a call to action or thought-provoking takeaway
- Keep text concise — slides are read in seconds, not minutes
- Avoid jargon unless your audience expects it"""

    user_prompt = f"""Create a {slide_count}-slide deck on the following:

Topic: {topic}
Angle / key information: {angle}
Opening hook (use this as the first slide title): {hook}

The deck must deliver real value on the topic using the angle provided.
The first slide title MUST be the hook above.

Return your response as a JSON array of slide objects. Each slide must have:
- "title": A short, attention-grabbing headline (max 8 words)
- "body": The main content text (2-3 short sentences max)
- "footer": A small note, source attribution, or hashtag line

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
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
