"""Generate alternative slide design versions using the Canva Connect API.

Creates multiple design variations with distinct visual styles so the user
can compare them side-by-side and pick their favourite.  Each alternative
uses a different prompt (minimal, bold, editorial, data-forward) sent to
Canva's AI design generation endpoint.

The Canva MCP server (https://mcp.canva.com/mcp) uses OAuth 2.1 with PKCE
through the MCP protocol itself—separate from the Connect API OAuth flow.
For MCP-native clients (Claude Desktop, Cursor, etc.) use the .mcp.json
config with mcp-remote.  For the Streamlit app, we use the Connect API
directly since the user has already authenticated via the sidebar.
"""

from __future__ import annotations

import logging
from typing import Any

from src.slides.canva_builder import (
    generate_ai_design,
    create_from_ai_candidate,
    export_design,
    create_design,
    DIMENSIONS_9_16,
    DIMENSIONS_16_9,
)

logger = logging.getLogger(__name__)


def generate_alternative_slides(
    slides: list[dict],
    access_token: str,
    topic: str,
    aspect_ratio: str = "9:16",
    colors: dict | None = None,
    num_alternatives: int = 2,
) -> list[dict]:
    """Generate alternative Canva design versions of the given slides.

    Creates multiple design variations with different style prompts via
    Canva's AI design generation, then exports each as PNG.

    Args:
        slides: List of slide dicts with "title", "body", and optional "footer".
        access_token: Canva OAuth access token (from Connect API auth flow).
        topic: The topic / headline for the slides.
        aspect_ratio: "9:16" (vertical) or "16:9" (landscape).
        colors: Dict with "background", "accent", etc.
        num_alternatives: How many alternative versions to generate (default 2).

    Returns:
        List of dicts, each with:
            - "version": version label (e.g. "Alternative A")
            - "style": style description used
            - "design_id": Canva design ID
            - "edit_url": Canva editor link
            - "export_url": PNG download URL (or "" if export failed)
    """
    colors = colors or {}
    bg = colors.get("background", "#0D1117")
    accent = colors.get("accent", "#58A6FF")

    slides_text = "\n".join(
        f"Slide {i + 1}: Title: \"{s['title']}\" | Body: \"{s['body']}\" | "
        f"Footer: \"{s.get('footer', '')}\""
        for i, s in enumerate(slides)
    )

    dims = DIMENSIONS_9_16 if aspect_ratio == "9:16" else DIMENSIONS_16_9
    base_context = (
        f"{len(slides)}-page finance carousel for TikTok/Instagram. "
        f"Dimensions: {dims['width']}x{dims['height']}px. "
        f"Topic: {topic}. "
        f"Content for each slide:\n{slides_text}"
    )

    # Each variation gets a distinct visual style prompt
    style_variations = [
        {
            "label": "Alternative A",
            "style": "Clean Minimal",
            "prompt": (
                f"Create a clean, minimal {base_context} "
                f"Dark background ({bg}), accent ({accent}). "
                f"Large bold sans-serif headlines, generous whitespace, "
                f"subtle gradients. No clutter."
            ),
        },
        {
            "label": "Alternative B",
            "style": "Bold & Vibrant",
            "prompt": (
                f"Create a bold, high-contrast {base_context} "
                f"Dark background ({bg}), neon accent ({accent}). "
                f"Impactful typography with color highlights on key numbers. "
                f"Geometric shapes and strong visual hierarchy."
            ),
        },
        {
            "label": "Alternative C",
            "style": "Editorial",
            "prompt": (
                f"Create a sophisticated editorial-style {base_context} "
                f"Dark background ({bg}), muted accent ({accent}). "
                f"Mix of serif and sans-serif fonts, thin dividers, "
                f"newspaper/magazine layout feel."
            ),
        },
        {
            "label": "Alternative D",
            "style": "Data-Forward",
            "prompt": (
                f"Create a data-visualization-focused {base_context} "
                f"Dark background ({bg}), accent ({accent}). "
                f"Emphasize numbers with large counters, mini charts, "
                f"progress indicators. Clean tech aesthetic."
            ),
        },
    ]

    alternatives: list[dict] = []
    for variation in style_variations[:num_alternatives]:
        try:
            result = _create_one_alternative(
                access_token=access_token,
                prompt=variation["prompt"],
                title=f"{topic} — {variation['label']}",
                aspect_ratio=aspect_ratio,
            )
            alternatives.append({
                "version": variation["label"],
                "style": variation["style"],
                **result,
            })
        except Exception as exc:
            logger.error("Failed to create %s: %s", variation["label"], exc)
            alternatives.append({
                "version": variation["label"],
                "style": variation["style"],
                "design_id": "",
                "edit_url": "",
                "export_url": "",
                "error": str(exc),
            })

    return alternatives


def _create_one_alternative(
    access_token: str,
    prompt: str,
    title: str,
    aspect_ratio: str,
) -> dict:
    """Create a single alternative design and return its URLs.

    Calls Canva AI design generation, converts the first candidate into
    an editable design, and exports as PNG.
    """
    # Step 1: Generate AI design candidates
    gen_result = generate_ai_design(access_token, prompt)

    candidates = gen_result.get("candidates", [])
    if not candidates:
        # Fallback: create a blank design
        design = create_design(access_token, title, aspect_ratio)
        return {
            "design_id": design.get("id", ""),
            "edit_url": design.get("urls", {}).get("edit_url", ""),
            "export_url": "",
        }

    # Step 2: Convert first candidate to editable design
    candidate_id = candidates[0].get("id", "")
    design = create_from_ai_candidate(access_token, candidate_id, title)
    design_id = design.get("id", "")

    edit_url = design.get("urls", {}).get("edit_url", "")

    # Step 3: Export as PNG (best-effort)
    export_url = ""
    if design_id:
        try:
            export_url = export_design(access_token, design_id, format="png")
        except Exception as exc:
            logger.warning("Export failed for '%s': %s", title, exc)

    return {
        "design_id": design_id,
        "edit_url": edit_url,
        "export_url": export_url,
    }
