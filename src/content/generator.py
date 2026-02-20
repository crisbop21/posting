"""Generate slide content from research using Claude."""

import json

import anthropic


def generate_slide_content(
    research_text: str,
    slide_count: int,
    tone: str,
    audience: str,
    style_notes: str,
) -> list[dict]:
    """Generate structured slide content from research data using Claude.

    Args:
        research_text: Formatted research data from news/Reddit sources.
        slide_count: Number of slides to generate.
        tone: Desired tone (e.g., bold, casual, professional).
        audience: Target audience description.
        style_notes: Additional style guidance.

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
- First slide must be a strong hook that stops the scroll
- Last slide should have a call to action or thought-provoking takeaway
- Keep text concise — slides are read in seconds, not minutes
- Avoid jargon unless your audience expects it"""

    user_prompt = f"""Based on today's trending finance topics below, create a {slide_count}-slide deck.

Pick the most engaging and timely topic(s) from the research. You can combine related stories.

{research_text}

Return your response as a JSON array of slide objects. Each slide must have:
- "title": A short, attention-grabbing headline (max 8 words)
- "body": The main content text (2-3 short sentences max)
- "footer": A small note, source attribution, or hashtag line

Return ONLY the JSON array, no other text."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # Extract the text response
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
            break

    # Parse the JSON array from the response
    # Handle potential markdown code blocks wrapping
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # Remove first line (```json)
        text = text.rsplit("```", 1)[0]  # Remove last ```
        text = text.strip()

    slides = json.loads(text)
    return slides
