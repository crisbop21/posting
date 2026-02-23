"""Generate alternative slide versions via the Canva MCP (Model Context Protocol).

Uses the official Canva MCP server at https://mcp.canva.com/mcp to create
alternative design versions of slides.  The MCP server exposes four tools:

  - list_designs   – list designs in your Canva account
  - get_design     – retrieve design metadata
  - create_design  – create or duplicate a design (from template or scratch)
  - export_design  – export a design to PNG / PDF

This module wraps those tools via Streamable HTTP transport so the Streamlit
app can offer a "Generate Alternative (MCP)" button alongside the existing
Canva Connect API builder.

Setup:
  1. The project ships a .mcp.json that points to https://mcp.canva.com/mcp
  2. The user authenticates via OAuth when first connecting from the Streamlit
     sidebar (same credentials as the existing Canva integration).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

MCP_ENDPOINT = "https://mcp.canva.com/mcp"

# ── Low-level MCP helpers ────────────────────────────────────────────────────


def _mcp_request(
    method: str,
    params: dict[str, Any] | None = None,
    access_token: str | None = None,
    timeout: int = 60,
) -> dict:
    """Send a JSON-RPC 2.0 request to the Canva MCP server.

    The Canva MCP server uses Streamable HTTP transport, accepting standard
    JSON-RPC over POST with optional SSE for streaming responses.
    """
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    payload: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or {},
    }

    resp = requests.post(
        MCP_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()

    # Handle SSE vs plain JSON
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        return _parse_sse_response(resp.text)
    return resp.json()


def _parse_sse_response(text: str) -> dict:
    """Extract the final JSON-RPC result from an SSE stream."""
    result: dict = {}
    for line in text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            try:
                parsed = json.loads(data)
                if "result" in parsed or "error" in parsed:
                    result = parsed
            except json.JSONDecodeError:
                continue
    return result


def _extract_result(response: dict) -> Any:
    """Extract the result payload from a JSON-RPC response, raising on error."""
    if "error" in response:
        err = response["error"]
        raise RuntimeError(
            f"Canva MCP error ({err.get('code', '?')}): {err.get('message', 'Unknown')}"
        )
    return response.get("result", {})


# ── MCP tool wrappers ────────────────────────────────────────────────────────


def mcp_list_designs(
    access_token: str,
    query: str | None = None,
    limit: int = 10,
) -> list[dict]:
    """List designs in the user's Canva account, optionally filtered by query."""
    params: dict[str, Any] = {"name": "list_designs", "arguments": {}}
    if query:
        params["arguments"]["query"] = query
    params["arguments"]["limit"] = limit

    resp = _mcp_request("tools/call", params, access_token)
    result = _extract_result(resp)

    # The MCP result may contain content blocks; extract text
    contents = result.get("content", [])
    for block in contents:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return contents if isinstance(contents, list) else []


def mcp_create_design(
    access_token: str,
    title: str,
    description: str,
    design_type: str = "presentation",
    template_id: str | None = None,
) -> dict:
    """Create a new Canva design via MCP.

    Args:
        access_token: Canva OAuth access token.
        title: Title for the new design.
        description: Text prompt / description for the design content.
        design_type: Type of design (e.g. "presentation", "instagram_post").
        template_id: Optional template ID to duplicate from.

    Returns:
        Dict with design metadata (id, edit_url, view_url, etc.).
    """
    arguments: dict[str, Any] = {
        "title": title,
        "description": description,
        "design_type": design_type,
    }
    if template_id:
        arguments["template_id"] = template_id

    params = {"name": "create_design", "arguments": arguments}
    resp = _mcp_request("tools/call", params, access_token, timeout=90)
    result = _extract_result(resp)

    # Parse text content block
    contents = result.get("content", [])
    for block in contents:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return result


def mcp_export_design(
    access_token: str,
    design_id: str,
    format: str = "png",
) -> str:
    """Export a Canva design to the specified format and return the download URL."""
    params = {
        "name": "export_design",
        "arguments": {
            "design_id": design_id,
            "format": format,
        },
    }
    resp = _mcp_request("tools/call", params, access_token, timeout=90)
    result = _extract_result(resp)

    contents = result.get("content", [])
    for block in contents:
        if block.get("type") == "text":
            text = block["text"]
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    return data.get("url", data.get("download_url", text))
                return text
            except (json.JSONDecodeError, KeyError):
                # Might be a plain URL string
                if text.startswith("http"):
                    return text
    return ""


def mcp_get_design(access_token: str, design_id: str) -> dict:
    """Retrieve metadata for a specific design."""
    params = {
        "name": "get_design",
        "arguments": {"design_id": design_id},
    }
    resp = _mcp_request("tools/call", params, access_token, timeout=60)
    result = _extract_result(resp)

    contents = result.get("content", [])
    for block in contents:
        if block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                pass
    return result


# ── High-level: generate alternative slides ──────────────────────────────────


def generate_alternative_slides(
    slides: list[dict],
    access_token: str,
    topic: str,
    aspect_ratio: str = "9:16",
    colors: dict | None = None,
    num_alternatives: int = 2,
) -> list[dict]:
    """Generate alternative Canva design versions of the given slides via MCP.

    Creates multiple design variations with different style prompts so the
    user can compare and pick their favourite.

    Args:
        slides: List of slide dicts with "title", "body", and optional "footer".
        access_token: Canva OAuth access token.
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
        f"Slide {i + 1}: {s['title']} — {s['body']}"
        for i, s in enumerate(slides)
    )

    # Define distinct style variations for each alternative
    style_variations = [
        {
            "label": "Alternative A",
            "style": "Clean Minimal",
            "prompt": (
                f"Create a clean, minimal {len(slides)}-slide finance carousel. "
                f"Dark background ({bg}), accent ({accent}). "
                f"Large bold sans-serif headlines, generous whitespace, subtle gradients. "
                f"Aspect ratio: {aspect_ratio}. "
                f"Topic: {topic}.\n{slides_text}"
            ),
        },
        {
            "label": "Alternative B",
            "style": "Bold & Vibrant",
            "prompt": (
                f"Create a bold, high-contrast {len(slides)}-slide finance carousel. "
                f"Dark background ({bg}), neon accent ({accent}). "
                f"Impactful typography with color highlights on key numbers. "
                f"Use geometric shapes and strong visual hierarchy. "
                f"Aspect ratio: {aspect_ratio}. "
                f"Topic: {topic}.\n{slides_text}"
            ),
        },
        {
            "label": "Alternative C",
            "style": "Editorial",
            "prompt": (
                f"Create a sophisticated editorial-style {len(slides)}-slide finance carousel. "
                f"Dark background ({bg}), muted accent ({accent}). "
                f"Use a mix of serif and sans-serif fonts, thin dividers, "
                f"and a newspaper/magazine layout feel. "
                f"Aspect ratio: {aspect_ratio}. "
                f"Topic: {topic}.\n{slides_text}"
            ),
        },
        {
            "label": "Alternative D",
            "style": "Data-Forward",
            "prompt": (
                f"Create a data-visualization-focused {len(slides)}-slide finance carousel. "
                f"Dark background ({bg}), accent ({accent}). "
                f"Emphasize numbers and statistics with large counters, "
                f"mini charts, and progress indicators. Clean tech aesthetic. "
                f"Aspect ratio: {aspect_ratio}. "
                f"Topic: {topic}.\n{slides_text}"
            ),
        },
    ]

    alternatives: list[dict] = []
    for variation in style_variations[:num_alternatives]:
        try:
            design = mcp_create_design(
                access_token=access_token,
                title=f"{topic} — {variation['label']}",
                description=variation["prompt"],
            )

            design_id = design.get("id", "")
            edit_url = design.get("edit_url", "")
            if not edit_url:
                urls = design.get("urls", {})
                edit_url = urls.get("edit_url", "")

            # Attempt export
            export_url = ""
            if design_id:
                try:
                    export_url = mcp_export_design(access_token, design_id, "png")
                except Exception as exc:
                    logger.warning("Export failed for %s: %s", variation["label"], exc)

            alternatives.append({
                "version": variation["label"],
                "style": variation["style"],
                "design_id": design_id,
                "edit_url": edit_url,
                "export_url": export_url,
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
