"""Review and iteratively improve slide content for engagement using Claude."""

import json
import re

import anthropic


def review_and_improve(
    slides: list[dict],
    tone: str,
    audience: str,
    iterations: int = 2,
) -> list[dict]:
    """Run multiple review iterations on slide content to maximize engagement.

    Each iteration asks Claude to score the slides, identify weaknesses,
    and produce an improved version.

    Args:
        slides: List of slide dicts with 'title', 'body', 'footer'.
        tone: Desired tone.
        audience: Target audience description.
        iterations: Number of review/improve cycles to run.

    Returns:
        The final improved list of slide dicts.
    """
    client = anthropic.Anthropic()
    current_slides = slides

    for i in range(iterations):
        print(f"  [review] Iteration {i + 1}/{iterations}...")

        review_prompt = f"""You are a viral content strategist reviewing a finance slide deck for {audience}.
The tone should be: {tone}.

Here are the current slides:

{json.dumps(current_slides, indent=2)}

Please do the following:

1. SCORE each slide from 1-10 on these criteria:
   - Hook power: Does it grab attention immediately?
   - Clarity: Is the message instantly understandable?
   - Engagement: Would someone share this or save it?
   - Visual text balance: Is the text concise enough for a slide?

2. IDENTIFY the weakest slide(s) and explain why.

3. PRODUCE an improved version of ALL slides. For each slide:
   - Make titles punchier and more curiosity-driven
   - Tighten body text — remove filler words
   - Ensure the first slide is the strongest hook
   - Add urgency or FOMO where appropriate (without being clickbait)
   - Make sure data points are highlighted

Return your response as JSON with this structure:
{{
  "review": {{
    "scores": [
      {{"slide": 1, "hook": 8, "clarity": 7, "engagement": 6, "balance": 8, "notes": "..."}},
      ...
    ],
    "weakest": "Slide 3 — the body text is too vague...",
    "overall_score": 7.2
  }},
  "improved_slides": [
    {{"title": "...", "body": "...", "footer": "..."}},
    ...
  ]
}}

Return ONLY the JSON, no other text."""

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": review_prompt}],
        )

        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
                break

        result = _parse_json(text)

        review = result.get("review", {})
        overall = review.get("overall_score", "N/A")
        print(f"  [review] Overall engagement score: {overall}")

        if "scores" in review:
            for score in review["scores"]:
                slide_num = score.get("slide", "?")
                avg = (
                    score.get("hook", 0)
                    + score.get("clarity", 0)
                    + score.get("engagement", 0)
                    + score.get("balance", 0)
                ) / 4
                print(f"    Slide {slide_num}: avg {avg:.1f} — {score.get('notes', '')}")

        if "weakest" in review:
            print(f"  [review] Weakest: {review['weakest'][:100]}")

        current_slides = result.get("improved_slides", current_slides)

    return current_slides


def _parse_json(text: str):
    """Extract and parse JSON from a model response that may contain markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    return json.loads(text)
