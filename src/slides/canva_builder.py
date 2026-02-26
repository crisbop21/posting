"""Build slides via the Canva Connect API (same API the Canva AI Connector MCP wraps).

Provides two creation paths:
  1. AI-generated designs – Canva's AI creates a styled design from a prompt.
  2. Programmatic creation – create a blank design and add text pages.

Both paths produce a Canva design URL the user can edit, plus PNG exports for
side-by-side comparison with the local PNG/PPTX builders.

Setup:
  The user needs a Canva Connect API integration.  Go to
  https://www.canva.dev/console  →  Create integration  →  copy the Client ID
  and Client Secret.  In the Streamlit sidebar, paste these and click
  "Connect Canva" to run the OAuth flow.
"""

import json
import time
from urllib.parse import urlencode

import requests


CANVA_API_BASE = "https://api.canva.com/rest/v1"
CANVA_AUTH_URL = "https://www.canva.com/api/oauth/authorize"
CANVA_TOKEN_URL = "https://api.canva.com/rest/v1/oauth/token"

# Dimensions for TikTok/Instagram vertical carousels
DIMENSIONS_9_16 = {"width": 1080, "height": 1920}
DIMENSIONS_16_9 = {"width": 1920, "height": 1080}


# ── OAuth helpers ─────────────────────────────────────────────────────────────


def get_oauth_url(client_id: str, redirect_uri: str, state: str = "canva") -> str:
    """Return the Canva OAuth2 authorization URL."""
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": "design:content:read design:content:write design:meta:read asset:read asset:write",
        "state": state,
    }
    return f"{CANVA_AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict:
    """Exchange an OAuth authorization code for access + refresh tokens."""
    resp = requests.post(
        CANVA_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> dict:
    """Refresh an expired access token."""
    resp = requests.post(
        CANVA_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ── Canva Connect API wrappers ────────────────────────────────────────────────


def _headers(access_token: str) -> dict:
    return {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }


def create_design(
    access_token: str,
    title: str,
    aspect_ratio: str = "9:16",
) -> dict:
    """Create a blank Canva design with the given dimensions.

    Returns the full design object (id, urls.edit_url, urls.view_url, etc.).
    """
    dims = DIMENSIONS_9_16 if aspect_ratio == "9:16" else DIMENSIONS_16_9
    body = {
        "design_type": {
            "type": "custom",
            "width": dims["width"],
            "height": dims["height"],
        },
        "title": title,
    }
    resp = requests.post(
        f"{CANVA_API_BASE}/designs",
        headers=_headers(access_token),
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("design", resp.json())


def generate_ai_design(
    access_token: str,
    prompt: str,
) -> dict:
    """Use Canva's AI to generate a design from a text prompt.

    Returns a generation job with candidate IDs that can be converted to
    editable designs via ``create_from_ai_candidate``.
    """
    resp = requests.post(
        f"{CANVA_API_BASE}/ai/designs",
        headers=_headers(access_token),
        json={"query": prompt},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def create_from_ai_candidate(
    access_token: str,
    candidate_id: str,
    title: str = "Canva Slides",
) -> dict:
    """Convert an AI-generated candidate into a real editable Canva design."""
    resp = requests.post(
        f"{CANVA_API_BASE}/ai/designs/candidates",
        headers=_headers(access_token),
        json={"candidate_id": candidate_id, "title": title},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("design", resp.json())


def export_design(
    access_token: str,
    design_id: str,
    format: str = "png",
) -> str:
    """Start an export job and poll until complete.  Returns a download URL."""
    body = {
        "design_id": design_id,
        "format": {"type": format},
    }
    resp = requests.post(
        f"{CANVA_API_BASE}/exports",
        headers=_headers(access_token),
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    job = resp.json()
    job_id = job.get("job", {}).get("id") or job.get("id")

    # Poll for completion (max ~60 seconds)
    for _ in range(30):
        time.sleep(2)
        status_resp = requests.get(
            f"{CANVA_API_BASE}/exports/{job_id}",
            headers=_headers(access_token),
            timeout=30,
        )
        status_resp.raise_for_status()
        status = status_resp.json()
        job_data = status.get("job", status)
        if job_data.get("status") == "success":
            urls = job_data.get("urls", [])
            if urls:
                return urls[0] if isinstance(urls[0], str) else urls[0].get("url", "")
            return ""
        if job_data.get("status") == "failed":
            raise RuntimeError(f"Canva export failed: {job_data}")

    raise TimeoutError("Canva export timed out after 60 seconds")


# ── High-level builder ────────────────────────────────────────────────────────


def build_canva_slides(
    slides: list[dict],
    access_token: str,
    topic: str,
    aspect_ratio: str = "9:16",
    colors: dict | None = None,
) -> dict:
    """Generate a Canva design from slide data and return useful links.

    Uses Canva's AI design generation to create a professionally styled
    version of the slides, then exports as PNG.

    Returns dict with:
        - "edit_url": Canva editor link
        - "view_url": public view link
        - "export_url": PNG download URL (or empty if export failed)
        - "design_id": Canva design ID
    """
    colors = colors or {}
    bg = colors.get("background", "#0D0D15")
    accent = colors.get("accent", "#F7B731")

    # Build a detailed prompt for Canva AI
    slides_text = "\n".join(
        f"Slide {i + 1}: Title: \"{s['title']}\" | Body: \"{s['body']}\" | Footer: \"{s.get('footer', '')}\""
        for i, s in enumerate(slides)
    )

    dims = DIMENSIONS_9_16 if aspect_ratio == "9:16" else DIMENSIONS_16_9
    prompt = (
        f"Create a {len(slides)}-page finance carousel for TikTok/Instagram. "
        f"Dimensions: {dims['width']}x{dims['height']}px. "
        f"Dark background ({bg}), accent color ({accent}). "
        f"Modern, bold typography. Minimal design. "
        f"Topic: {topic}. "
        f"Content for each slide:\n{slides_text}"
    )

    # Step 1: Generate AI design
    gen_result = generate_ai_design(access_token, prompt)

    # Extract the first candidate
    candidates = gen_result.get("candidates", [])
    if not candidates:
        # Fallback: create a blank design
        design = create_design(access_token, f"Slides: {topic}", aspect_ratio)
        return {
            "edit_url": design.get("urls", {}).get("edit_url", ""),
            "view_url": design.get("urls", {}).get("view_url", ""),
            "export_url": "",
            "design_id": design.get("id", ""),
        }

    candidate_id = candidates[0].get("id", "")
    design = create_from_ai_candidate(access_token, candidate_id, f"Slides: {topic}")
    design_id = design.get("id", "")

    # Step 2: Export as PNG
    export_url = ""
    try:
        export_url = export_design(access_token, design_id, format="png")
    except Exception:
        pass  # Export is best-effort; user can still edit in Canva

    return {
        "edit_url": design.get("urls", {}).get("edit_url", ""),
        "view_url": design.get("urls", {}).get("view_url", ""),
        "export_url": export_url,
        "design_id": design_id,
    }
