"""Review and iteratively improve slide content for engagement using Claude."""

import json
import re

from src.api import create_message


def review_and_improve(
    slides: list[dict],
    tone: str,
    audience: str,
    iterations: int = 2,
    hook: str = "",
) -> list[dict]:
    """Run multiple review iterations on slide content to maximize engagement.

    Args:
        slides: List of slide dicts with 'title', 'body', 'footer'.
        tone: Desired tone.
        audience: Target audience description.
        iterations: Number of review/improve cycles to run.
        hook: The opening hook text — slides must deliver on this promise.

    Returns:
        The final improved list of slide dicts.
    """
    current_slides = slides

    hook_context = ""
    if hook:
        hook_context = f"""
HOOK ALIGNMENT (critical):
The opening hook is: "{hook}"
- Every slide must feel connected to the promise made in this hook.
- The narrative must build toward delivering on the hook's open loop.
- Slide 1 title MUST remain the hook exactly as written above.
- If any slide feels disconnected from the hook's promise, rewrite it to stay on thread.
"""

    for i in range(iterations):
        print(f"  [review] Iteration {i + 1}/{iterations}...")

        review_prompt = f"""You are a viral content strategist reviewing a finance slide deck for {audience}.
The tone should be: {tone}.
{hook_context}
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
   - Ensure slides tell a cohesive STORY — each slide must logically lead to the next
   - Data slides should build: context → tension → insight → payoff
   - Footer: short source attribution only (e.g. "source: bloomberg"). NO hashtags, NO emojis

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

        response = create_message(
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
