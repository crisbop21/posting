"""Generate AI images for slides using Gemini Flash or OpenAI DALL-E 3.

Supports two providers (auto-detected from environment variables):
  - Google Gemini: Free via Google AI Studio API key (gemini-2.5-flash-image).
  - OpenAI DALL-E 3: Paid, requires OpenAI API key.

Each slide gets a cinematic, finance-themed image generated from its content.
The image is then composited with slide text overlay to create the final frame.

Visual compositing pipeline (5 layers, bottom to top):
  1. Background: blurred source image
  2. Gradient overlay: role-aware darkening + accent tint
  3. Foreground subject: transparent PNG cutout (optional)
  4. Branded frame: accent bars, counter pill, handle
  5. Text: role-specific layout (hook / context / payoff / cta)
"""

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.slides.png_builder import (
    _hex_to_tuple,
    _load_font,
    _wrap_text,
    _calc_title_font_size,
)


# ── Provider Detection ────────────────────────────────────────────────────────


def _get_provider_and_key() -> tuple[str, str]:
    """Determine image provider and API key from environment variables."""
    google_key = os.environ.get("GOOGLE_AI_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")

    if google_key:
        return "google", google_key
    if openai_key:
        return "openai", openai_key
    raise RuntimeError(
        "No image generation API key found. Set one of:\n"
        "  GOOGLE_AI_API_KEY — free at https://aistudio.google.com/apikey\n"
        "  OPENAI_API_KEY   — paid at https://platform.openai.com/api-keys"
    )


# ── Google Gemini Image Generation ────────────────────────────────────────────


def _generate_gemini(prompt: str, api_key: str) -> bytes:
    """Generate an image using Gemini's native image generation.

    Uses gemini-2.5-flash-image via the generateContent endpoint.
    Free with a Google AI Studio API key.  Retries on 429 rate limits
    with exponential backoff (up to 3 retries).
    Returns raw PNG bytes.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    max_retries = 3

    for attempt in range(max_retries + 1):
        resp = requests.post(
            url,
            headers={
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                },
            },
            timeout=120,
        )

        if resp.status_code == 429 and attempt < max_retries:
            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
            time.sleep(wait)
            continue

        if resp.status_code == 400:
            error_msg = resp.json().get("error", {}).get("message", resp.text)
            raise RuntimeError(
                f"Gemini rejected the prompt (safety filter or invalid request): {error_msg}"
            )
        resp.raise_for_status()
        break

    result = resp.json()

    # Extract inline image data from response
    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError(
            "Gemini returned no candidates. The prompt may have been filtered."
        )

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        inline_data = part.get("inlineData", {})
        if inline_data.get("mimeType", "").startswith("image/"):
            image_b64 = inline_data.get("data", "")
            if image_b64:
                return base64.b64decode(image_b64)

    raise RuntimeError(
        "Gemini response did not contain any image data. "
        "Try a different prompt."
    )


# ── OpenAI DALL-E 3 ──────────────────────────────────────────────────────────


def _generate_dalle(prompt: str, width: int, height: int, api_key: str) -> bytes:
    """Generate an image using OpenAI DALL-E 3.

    Retries on 429 rate limits with exponential backoff (up to 3 retries).
    Returns raw PNG bytes.
    """
    # DALL-E 3 supported sizes
    if width < height:
        size = "1024x1792"  # Portrait / stories
    elif width > height:
        size = "1792x1024"  # Landscape
    else:
        size = "1024x1024"  # Square

    # DALL-E 3 has a 4000-character prompt limit
    if len(prompt) > 4000:
        prompt = prompt[:4000]

    max_retries = 3

    for attempt in range(max_retries + 1):
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "n": 1,
                "size": size,
                "quality": "standard",
                "response_format": "b64_json",
            },
            timeout=90,
        )

        if resp.status_code == 429 and attempt < max_retries:
            wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
            time.sleep(wait)
            continue

        if resp.status_code == 400:
            try:
                error_data = resp.json().get("error", {})
                error_msg = error_data.get("message", resp.text)
            except Exception:
                error_msg = resp.text
            raise RuntimeError(
                f"DALL-E rejected the request: {error_msg}"
            )

        resp.raise_for_status()
        break

    result = resp.json()
    data = result.get("data", [])
    if not data:
        raise RuntimeError("DALL-E returned no image data.")

    image_b64 = data[0].get("b64_json", "")
    if not image_b64:
        raise RuntimeError("DALL-E response missing image data.")

    return base64.b64decode(image_b64)


# ── Unified API ──────────────────────────────────────────────────────────────


def generate_image(
    prompt: str,
    width: int = 1080,
    height: int = 1920,
    model: str = "",
) -> bytes:
    """Generate an image using the configured provider.

    Auto-detects provider from environment variables:
      - GOOGLE_AI_API_KEY → Gemini Flash image generation (free)
      - OPENAI_API_KEY    → OpenAI DALL-E 3 (paid)

    Returns raw image bytes (PNG).
    """
    provider, api_key = _get_provider_and_key()

    if provider == "google":
        # Enhance prompt with aspect ratio hint for Gemini
        if width < height:
            aspect_hint = "vertical portrait orientation (9:16 aspect ratio)"
        elif width > height:
            aspect_hint = "horizontal landscape orientation (16:9 aspect ratio)"
        else:
            aspect_hint = "square orientation (1:1 aspect ratio)"
        enhanced_prompt = f"{prompt}. Generate this image in {aspect_hint}."
        return _generate_gemini(enhanced_prompt, api_key)
    else:
        return _generate_dalle(prompt, width, height, api_key)


# ── Slide Roles ──────────────────────────────────────────────────────────────


def get_slide_role(index: int, total: int) -> str:
    """Determine the visual role of a slide based on its position.

    Returns one of: 'hook', 'context', 'payoff', 'cta'.
    """
    if index == 0:
        return "hook"
    if index == total - 1:
        return "cta"
    if index == total - 2 and total >= 4:
        return "payoff"
    return "context"


# ── Color Palette Presets ─────────────────────────────────────────────────────

PALETTE_PRESETS = {
    "premium_gold": {
        "background": "#0D0D15",
        "title": "#F0F0F0",
        "body": "#C0C0D0",
        "accent": "#F7B731",
        "highlight": "#FF5757",
    },
    "neon_finance": {
        "background": "#0A0A1A",
        "title": "#F0F0F0",
        "body": "#B8C0CC",
        "accent": "#00D4FF",
        "highlight": "#FF3B5C",
    },
    "money_green": {
        "background": "#0A1210",
        "title": "#F0F0F0",
        "body": "#B8C8C0",
        "accent": "#00FF87",
        "highlight": "#FFD700",
    },
    "electric": {
        "background": "#050510",
        "title": "#FFFFFF",
        "body": "#C8D0E0",
        "accent": "#25F4EE",
        "highlight": "#FE2C55",
    },
}


def get_palette(name: str = "premium_gold") -> dict:
    """Get a named color palette preset."""
    return PALETTE_PRESETS.get(name, PALETTE_PRESETS["premium_gold"]).copy()


# ── Layer 1: Background Treatment ────────────────────────────────────────────


def _blur_background(img: Image.Image, radius: int = 16) -> Image.Image:
    """Apply Gaussian blur to the background image."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


# ── Layer 2: Role-Aware Gradient Overlay ─────────────────────────────────────


def _gradient_overlay(
    img: Image.Image,
    accent_color: tuple,
    role: str = "context",
) -> Image.Image:
    """Apply a role-aware gradient overlay for text readability.

    Hook:    heavier at top (where the massive title sits)
    Context: balanced mid-range darkening
    Payoff:  accent-tinted gradient from bottom (visual shift)
    CTA:     heavy uniform overlay for clean text-forward look
    """
    w, h = img.size
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    ar = accent_color[0] * 0.12
    ag = accent_color[1] * 0.12
    ab = accent_color[2] * 0.12

    for y in range(h):
        t = y / h
        if role == "hook":
            # Heavy at top where the big title sits, lighter below for bg peek
            if t < 0.55:
                alpha = int(210 - t * 100)
            else:
                alpha = int(140 + (t - 0.55) * 180)
        elif role == "payoff":
            # Accent-heavy from bottom — visual shift
            alpha = int(100 + t * 120)
            ar = accent_color[0] * 0.20
            ag = accent_color[1] * 0.20
            ab = accent_color[2] * 0.20
        elif role == "cta":
            # Heavy uniform overlay for clean text-only look
            alpha = 190
        else:  # context
            # Balanced readability
            if t < 0.25:
                alpha = int(190 - t * 120)
            elif t > 0.75:
                alpha = int(130 + (t - 0.75) * 200)
            else:
                alpha = 130

        draw.line([(0, y), (w, y)], fill=(int(ar), int(ag), int(ab), alpha))

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return Image.alpha_composite(img, gradient)


# ── Layer 3: Foreground Subject Compositing ──────────────────────────────────


def _composite_foreground(
    img: Image.Image,
    cutout: Image.Image,
    role: str = "hook",
) -> Image.Image:
    """Layer a transparent PNG cutout onto the slide based on role.

    Hook:    large subject on the right (45% height)
    Context: smaller subject on the left (32% height)
    Payoff:  medium subject center-right (38% height)
    CTA:     no foreground — keep it clean
    """
    if role == "cta":
        return img

    w, h = img.size
    cw, ch = cutout.size

    if role == "hook":
        target_h = int(h * 0.45)
        scale = target_h / ch
        new_w = int(cw * scale)
        new_h = target_h
        # Right side, bottom-aligned above handle
        x = w - new_w - int(w * 0.02)
        y = h - new_h - int(h * 0.08)
    elif role == "payoff":
        target_h = int(h * 0.38)
        scale = target_h / ch
        new_w = int(cw * scale)
        new_h = target_h
        # Center-right, lower area
        x = w - new_w - int(w * 0.06)
        y = h - new_h - int(h * 0.12)
    else:  # context
        target_h = int(h * 0.32)
        scale = target_h / ch
        new_w = int(cw * scale)
        new_h = target_h
        # Left side, bottom area
        x = int(w * 0.02)
        y = h - new_h - int(h * 0.08)

    # Prevent overflow
    new_w = min(new_w, int(w * 0.55))
    resized = cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    if resized.mode != "RGBA":
        resized = resized.convert("RGBA")

    img.paste(resized, (x, y), resized)
    return img


# ── Layer 4: Branded Frame ───────────────────────────────────────────────────


def _draw_branded_frame(
    draw: ImageDraw.Draw,
    w: int,
    h: int,
    accent_c: tuple,
    role: str,
    slide_index: int,
    total_slides: int,
    handle: str,
):
    """Draw consistent branded frame: accent bars, counter pill, handle.

    Platform-safe placement (1080x1920):
      - Counter pill: top-left, within the 8-25% title zone
      - Handle: at ~70% from top (inside the caption zone, above platform UI)
      - Accent bars: top edge and at the 73% boundary
    """
    # Top accent bar (at top of safe zone, ~8% from top)
    bar_y = int(h * 0.078)
    bar_h = max(int(h * 0.003), 3)
    draw.rectangle([0, bar_y, w, bar_y + bar_h], fill=accent_c)

    # Lower accent bar at the content/caption boundary (~73% from top)
    lower_bar_y = int(h * 0.73)
    draw.rectangle([0, lower_bar_y, w, lower_bar_y + bar_h], fill=(*accent_c, 80))

    # Slide counter pill (top-left, within title zone)
    counter_font = _load_font("sans_bold", int(w * 0.032))
    counter_text = f"{slide_index + 1}/{total_slides}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_tw = counter_bbox[2] - counter_bbox[0]
    counter_th = counter_bbox[3] - counter_bbox[1]

    pill_pad_x = int(w * 0.025)
    pill_pad_y = int(h * 0.008)
    pill_x = w - int(w * 0.15) - counter_tw - pill_pad_x  # away from right buttons
    pill_y = int(h * 0.085)
    pill_w = counter_tw + pill_pad_x * 2
    pill_h = counter_th + pill_pad_y * 2

    # Pill background
    draw.rounded_rectangle(
        [pill_x, pill_y, pill_x + pill_w, pill_y + pill_h],
        radius=pill_h // 2,
        fill=(*accent_c, 180),
    )
    draw.text(
        (pill_x + pill_pad_x, pill_y + pill_pad_y - 2),
        counter_text,
        font=counter_font,
        fill=(255, 255, 255),
    )

    # Handle in the caption zone (~70% from top, above platform UI)
    handle_font = _load_font("sans", int(w * 0.028))
    handle_bbox = handle_font.getbbox(handle)
    handle_tw = handle_bbox[2] - handle_bbox[0]
    handle_x = (w - handle_tw) // 2
    handle_y = int(h * 0.71)
    draw.text((handle_x + 2, handle_y + 2), handle, font=handle_font, fill=(0, 0, 0))
    draw.text((handle_x, handle_y), handle, font=handle_font, fill=(200, 200, 200))


# ── Text Rendering Helpers ───────────────────────────────────────────────────

import re

# Pattern to find numbers, percentages, dollar amounts, multipliers
_NUMBER_RE = re.compile(
    r'(\$[\d,.]+(?:\s*(?:trillion|billion|million|thousand|[TBMK]))?\w*'
    r'|[\d,.]+%'
    r'|[\d,.]+x'
    r'|[\d,.]+(?:\s*(?:trillion|billion|million|thousand)))',
    re.IGNORECASE,
)


def _extract_key_number(text: str) -> str | None:
    """Pull the first big number/stat from text for standalone callout."""
    m = _NUMBER_RE.search(text)
    return m.group(0).strip() if m else None


def _draw_shadowed_text(
    draw: ImageDraw.Draw,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    shadow_offset: int = 3,
):
    """Draw text with a hard drop shadow for readability."""
    sx, sy = xy
    draw.text((sx + shadow_offset, sy + shadow_offset), text, font=font, fill=(0, 0, 0))
    draw.text((sx, sy), text, font=font, fill=fill)


def _draw_highlight_box(
    draw: ImageDraw.Draw,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    text_color: tuple,
    box_color: tuple,
    pad_x: int = 12,
    pad_y: int = 6,
):
    """Draw text with a colored background box behind it."""
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x, y = xy
    draw.rectangle(
        [x - pad_x, y - pad_y, x + tw + pad_x, y + th + pad_y],
        fill=box_color,
    )
    draw.text(xy, text, font=font, fill=text_color)


def _draw_line_with_highlights(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    default_color: tuple,
    highlight_color: tuple,
    shadow_offset: int = 2,
):
    """Draw a line of text, coloring numbers/stats in highlight_color."""
    cursor_x = x
    parts = _NUMBER_RE.split(text)
    for part in parts:
        if not part:
            continue
        is_number = bool(_NUMBER_RE.fullmatch(part))
        color = highlight_color if is_number else default_color
        # Shadow
        draw.text((cursor_x + shadow_offset, y + shadow_offset), part, font=font, fill=(0, 0, 0))
        draw.text((cursor_x, y), part, font=font, fill=color)
        bbox = font.getbbox(part)
        cursor_x += bbox[2] - bbox[0]


def _draw_big_number_callout(
    draw: ImageDraw.Draw,
    w: int,
    y: int,
    number_text: str,
    color: tuple,
    margin: int,
):
    """Render a key stat as a massive standalone callout."""
    # Size the number to fill ~80% of content width
    content_w = w - 2 * margin
    num_size = int(w * 0.18)
    num_font = _load_font("sans_bold", num_size)
    num_bbox = num_font.getbbox(number_text)
    num_tw = num_bbox[2] - num_bbox[0]
    # Shrink if too wide
    while num_tw > content_w * 0.85 and num_size > 80:
        num_size -= 10
        num_font = _load_font("sans_bold", num_size)
        num_bbox = num_font.getbbox(number_text)
        num_tw = num_bbox[2] - num_bbox[0]
    num_th = num_bbox[3] - num_bbox[1]

    # Center horizontally
    nx = (w - num_tw) // 2
    # Shadow
    draw.text((nx + 4, y + 4), number_text, font=num_font, fill=(0, 0, 0))
    draw.text((nx, y), number_text, font=num_font, fill=color)
    return y + num_th + int(w * 0.02)


def _draw_decorative_quotes(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    w: int,
    color: tuple,
):
    """Draw oversized decorative open-quote marks as a visual anchor."""
    quote_font = _load_font("serif", int(w * 0.15))
    draw.text((x, y), "\u201C", font=quote_font, fill=(*color, 80))


def _draw_vertical_accent_bar(
    draw: ImageDraw.Draw,
    x: int,
    y_start: int,
    y_end: int,
    color: tuple,
    bar_w: int = 5,
):
    """Draw a vertical accent bar along the left edge of content."""
    draw.rectangle([x, y_start, x + bar_w, y_end], fill=color)


# ── Layer 5: Role-Specific Text Layouts ──────────────────────────────────────


def _layout_hook(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Hook slide (slide 1): maximum impact, fills the frame.

    - Title with highlight box on first line
    - Key stat pulled out as a MASSIVE number
    - Body text with highlighted numbers
    - Decorative elements fill dead space
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    highlight_c = _hex_to_tuple(colors.get("highlight", "#FF5757"))

    margin = int(w * 0.07)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Extract key number from body only — title is already displayed, no need to repeat
    key_num = _extract_key_number(body)

    # ── Title (top portion) ──
    title_size = int(w * 0.115)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 72:
        title_size -= 8
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.10)
    line_h = int(title_size * 1.25)

    # First line: highlight box
    if title_lines:
        _draw_highlight_box(
            draw, (margin, title_y), title_lines[0], title_font,
            text_color=(255, 255, 255),
            box_color=(*highlight_c, 220),
            pad_x=int(w * 0.018), pad_y=int(h * 0.008),
        )
        title_y += line_h + int(h * 0.012)

    # Remaining title lines
    for line in title_lines[1:]:
        _draw_shadowed_text(draw, (margin, title_y), line, title_font, (255, 255, 255), 4)
        title_y += line_h

    # ── Key number callout (center of slide) ──
    if key_num:
        num_y = title_y + int(h * 0.04)
        # Decorative line above number
        line_w_px = int(content_w * 0.3)
        line_x = (w - line_w_px) // 2
        draw.rectangle([line_x, num_y, line_x + line_w_px, num_y + 3], fill=accent_c)
        num_y += int(h * 0.025)
        num_y = _draw_big_number_callout(draw, w, num_y, key_num, accent_c, margin)
        # Decorative line below number
        draw.rectangle([line_x, num_y, line_x + line_w_px, num_y + 3], fill=accent_c)
        body_start_y = num_y + int(h * 0.035)
    else:
        # Thick accent bar
        bar_y = title_y + int(h * 0.025)
        draw.rectangle(
            [margin, bar_y, margin + int(content_w * 0.4), bar_y + 7],
            fill=accent_c,
        )
        body_start_y = bar_y + int(h * 0.04)

    # ── Body text (lower portion) ──
    body_size = int(w * 0.054)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, content_w)
    body_line_h = int(body_size * 1.55)

    body_y = body_start_y
    for line in body_lines:
        _draw_line_with_highlights(
            draw, margin, body_y, line, body_font,
            default_color=(230, 230, 240), highlight_color=accent_c,
            shadow_offset=2,
        )
        body_y += body_line_h

    # ── Decorative bottom element ──
    deco_y = max(body_y + int(h * 0.04), int(h * 0.78))
    # Faded horizontal line
    draw.rectangle(
        [margin, deco_y, w - margin, deco_y + 1],
        fill=(*accent_c, 60),
    )
    # Swipe hint
    swipe_font = _load_font("sans", int(w * 0.030))
    swipe_text = "SWIPE \u25B6"
    swipe_bbox = swipe_font.getbbox(swipe_text)
    swipe_tw = swipe_bbox[2] - swipe_bbox[0]
    swipe_x = (w - swipe_tw) // 2
    draw.text(
        (swipe_x, deco_y + int(h * 0.015)),
        swipe_text, font=swipe_font, fill=(*accent_c, 160),
    )


def _layout_context(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Context slide: evidence-style with highlighted numbers.

    - Bigger title
    - Vertical accent bar along left edge of body
    - Numbers in body rendered in accent color
    - More vertical fill
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    highlight_c = _hex_to_tuple(colors.get("highlight", "#FF5757"))
    body_c = _hex_to_tuple(colors.get("body", "#C0C0D0"))

    margin = int(w * 0.08)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Extract key number for callout
    key_num = _extract_key_number(body)

    # ── Title ──
    title_size = int(w * 0.090)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 64:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.10)
    line_h = int(title_size * 1.25)

    for line in title_lines:
        _draw_shadowed_text(draw, (margin, title_y), line, title_font, (255, 255, 255), 3)
        title_y += line_h

    # Divider
    div_y = title_y + int(h * 0.015)
    draw.rectangle([margin, div_y, margin + content_w, div_y + 3], fill=accent_c)

    # ── Key number callout (if found) ──
    if key_num:
        num_y = div_y + int(h * 0.03)
        num_y = _draw_big_number_callout(draw, w, num_y, key_num, highlight_c, margin)
        body_start_y = num_y + int(h * 0.02)
    else:
        body_start_y = div_y + int(h * 0.03)

    # ── Body text with vertical accent bar ──
    body_indent = int(w * 0.04)
    body_content_w = content_w - body_indent
    body_size = int(w * 0.050)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, body_content_w)
    body_line_h = int(body_size * 1.55)

    body_y = body_start_y
    body_top = body_y
    for line in body_lines:
        _draw_line_with_highlights(
            draw, margin + body_indent, body_y, line, body_font,
            default_color=body_c, highlight_color=accent_c,
            shadow_offset=2,
        )
        body_y += body_line_h

    # Vertical accent bar alongside body text
    if body_lines:
        _draw_vertical_accent_bar(
            draw, margin + int(w * 0.01), body_top - 5, body_y - body_line_h + int(body_size * 1.2),
            accent_c, bar_w=4,
        )

    # ── Decorative bottom line ──
    deco_y = max(body_y + int(h * 0.04), int(h * 0.78))
    draw.rectangle([margin, deco_y, w - margin, deco_y + 1], fill=(*accent_c, 50))


def _layout_payoff(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Payoff slide: the reveal — visual shift, centered, fills the frame.

    - Decorative quote marks as texture
    - Title in highlight color, centered
    - Key number massive
    - Body centered with more presence
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    highlight_c = _hex_to_tuple(colors.get("highlight", "#FF5757"))
    body_c = _hex_to_tuple(colors.get("body", "#C0C0D0"))

    margin = int(w * 0.09)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    key_num = _extract_key_number(body) or _extract_key_number(title)

    # ── Decorative oversized quote mark (faded, as texture) ──
    _draw_decorative_quotes(draw, margin - int(w * 0.02), int(h * 0.05), w, highlight_c)

    # ── Title — centered, in highlight color, BIGGER ──
    title_size = int(w * 0.098)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 68:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.16)
    line_h = int(title_size * 1.3)

    for line in title_lines:
        line_bbox = title_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, title_y), line, title_font, highlight_c, 4)
        title_y += line_h

    # ── Key number callout ──
    if key_num:
        num_y = title_y + int(h * 0.03)
        num_y = _draw_big_number_callout(draw, w, num_y, key_num, (255, 255, 255), margin)
        body_start_y = num_y + int(h * 0.02)
    else:
        # Centered accent bar
        bar_w = int(content_w * 0.25)
        bar_x = (w - bar_w) // 2
        bar_y = title_y + int(h * 0.02)
        draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 6], fill=highlight_c)
        body_start_y = bar_y + int(h * 0.04)

    # ── Body — centered in a semi-transparent card ──
    body_size = int(w * 0.050)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, int(content_w * 0.9))
    body_line_h = int(body_size * 1.55)

    body_y = body_start_y
    body_block_h = len(body_lines) * body_line_h
    card_pad = int(w * 0.04)

    # Draw card background
    card_left = margin - card_pad
    card_top = body_y - card_pad
    card_right = w - margin + card_pad
    card_bottom = body_y + body_block_h + card_pad
    draw.rounded_rectangle(
        [card_left, card_top, card_right, card_bottom],
        radius=int(w * 0.02),
        fill=(0, 0, 0, 80),
    )
    # Card left accent edge
    draw.rectangle(
        [card_left, card_top, card_left + 5, card_bottom],
        fill=highlight_c,
    )

    for line in body_lines:
        line_bbox = body_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_line_with_highlights(
            draw, x, body_y, line, body_font,
            default_color=body_c, highlight_color=highlight_c,
            shadow_offset=2,
        )
        body_y += body_line_h


def _layout_cta(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """CTA slide (last slide): warm, inviting, personal.

    - Vertically centered with more presence
    - Bigger title
    - Emoji-style arrow pointers
    - Body text in warm tone
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    highlight_c = _hex_to_tuple(colors.get("highlight", "#FF5757"))

    margin = int(w * 0.09)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # ── Title sizing — bigger than before ──
    title_size = int(w * 0.088)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 60:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    # Body sizing — bigger
    body_size = int(w * 0.048)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, content_w)

    # Vertically center but slightly above middle (feels more natural)
    line_h = int(title_size * 1.35)
    body_line_h = int(body_size * 1.6)
    total_text_h = (
        len(title_lines) * line_h
        + int(h * 0.06)  # gap between title and body
        + len(body_lines) * body_line_h
    )
    start_y = int((h - total_text_h) * 0.42)  # slightly above center

    # ── Decorative top arrows (point down toward title) ──
    arrow_font = _load_font("sans_bold", int(w * 0.06))
    arrow_text = "\u25BC  \u25BC  \u25BC"
    arrow_bbox = arrow_font.getbbox(arrow_text)
    arrow_tw = arrow_bbox[2] - arrow_bbox[0]
    arrow_x = (w - arrow_tw) // 2
    draw.text(
        (arrow_x, start_y - int(h * 0.06)),
        arrow_text, font=arrow_font, fill=(*accent_c, 100),
    )

    # ── Title ──
    title_y = start_y
    for line in title_lines:
        line_bbox = title_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, title_y), line, title_font, (255, 255, 255), 3)
        title_y += line_h

    # ── Accent bar (wider, more presence) ──
    bar_w = int(content_w * 0.4)
    bar_x = (w - bar_w) // 2
    bar_y = title_y + int(h * 0.015)
    draw.rounded_rectangle(
        [bar_x, bar_y, bar_x + bar_w, bar_y + 6],
        radius=3,
        fill=highlight_c,
    )

    # ── Body — centered, warm color ──
    body_y = bar_y + int(h * 0.04)
    for line in body_lines:
        line_bbox = body_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, body_y), line, body_font, (220, 220, 230), 2)
        body_y += body_line_h

    # ── Bottom pointer arrows (point up toward body) ──
    up_arrow_text = "\u25B2  \u25B2  \u25B2"
    up_bbox = arrow_font.getbbox(up_arrow_text)
    up_tw = up_bbox[2] - up_bbox[0]
    up_x = (w - up_tw) // 2
    draw.text(
        (up_x, body_y + int(h * 0.03)),
        up_arrow_text, font=arrow_font, fill=(*highlight_c, 80),
    )


def _layout_title_only(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Minimal layout: just the title at the top. Used when external captions
    carry the script body, so only a short heading is needed on-screen.

    Platform-safe zones (1080x1920):
      - Top 8% (0-150px): status bar / platform header → start below this
      - 8-25% (150-480px): title zone → title must fit entirely here
      - Right 11% (x>960): platform action buttons → limit text width to 900px
    """
    draw = ImageDraw.Draw(img)
    highlight_c = _hex_to_tuple(colors.get("highlight", "#FF5757"))

    margin = int(w * 0.07)
    # Cap content width to ~900px to avoid platform action buttons on the right
    content_w = min(w - 2 * margin, int(w * 0.83))

    title = slide.get("title", "")
    if not title:
        return

    # Title must fit within 8-25% of frame height (150-480px on 1920)
    header_top = int(h * 0.08)
    header_bottom = int(h * 0.25)
    max_title_h = header_bottom - header_top

    title_size = int(w * 0.08)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    line_h = int(title_size * 1.3)

    # Shrink until title fits within the header zone
    while len(title_lines) * line_h > max_title_h and title_size > 48:
        title_size -= 4
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)
        line_h = int(title_size * 1.3)

    title_y = header_top
    for line in title_lines:
        _draw_shadowed_text(
            draw, (margin, title_y), line, title_font,
            highlight_c, shadow_offset=4,
        )
        title_y += line_h


# Layout dispatcher
_LAYOUT_FUNCS = {
    "hook": _layout_hook,
    "context": _layout_context,
    "payoff": _layout_payoff,
    "cta": _layout_cta,
}


# ── Legacy Compositing (kept for backward compatibility) ─────────────────────


def _add_dark_overlay(img: Image.Image, opacity: float = 0.55) -> Image.Image:
    """Add a semi-transparent dark overlay for text readability."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, int(255 * opacity)))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return Image.alpha_composite(img, overlay)


def _add_gradient_overlay(img: Image.Image, accent_color: tuple) -> Image.Image:
    """Add a gradient overlay from dark bottom to semi-transparent top."""
    w, h = img.size
    gradient = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)

    for y in range(h):
        t = y / h
        if t < 0.3:
            alpha = int(200 - (t / 0.3) * 80)
        elif t > 0.7:
            alpha = int(120 + ((t - 0.7) / 0.3) * 80)
        else:
            alpha = 120
        r = int(accent_color[0] * 0.15)
        g = int(accent_color[1] * 0.15)
        b = int(accent_color[2] * 0.15)
        draw.line([(0, y), (w, y)], fill=(r, g, b, alpha))

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return Image.alpha_composite(img, gradient)


# ── Main Compositing Pipeline ────────────────────────────────────────────────


def composite_slide(
    bg_image: Image.Image,
    slide: dict,
    slide_index: int,
    total_slides: int,
    colors: dict,
    handle: str = "@cristian.bojaca",
    foreground: Optional[Image.Image] = None,
    title_only: bool = False,
) -> Image.Image:
    """Composite a slide with the full 5-layer visual treatment.

    Layers (bottom to top):
      1. Background: blurred source image
      2. Gradient overlay: role-aware darkening + accent tint
      3. Foreground subject: transparent PNG cutout (optional)
      4. Branded frame: accent bars, counter pill, handle
      5. Text: role-specific layout (or title-only when captions carry the body)

    Args:
        bg_image: Source background image (will be blurred).
        slide: Slide dict with 'title', 'body', optional 'footer'.
        slide_index: 0-based position.
        total_slides: Total number of slides.
        colors: Color scheme dict.
        handle: Social media handle.
        foreground: Optional transparent PNG cutout to overlay.
        title_only: If True, render only the title at the top instead of
            the full role-specific layout.
    """
    w, h = bg_image.size
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    role = get_slide_role(slide_index, total_slides)

    # Layer 1: Blur background
    blur_radius = 18 if role == "hook" else 14
    img = _blur_background(bg_image, radius=blur_radius)

    # Layer 2: Role-aware gradient overlay
    img = _gradient_overlay(img, accent_c, role=role)

    # Layer 3: Foreground subject
    if foreground is not None:
        img = _composite_foreground(img, foreground, role=role)

    # Convert to RGB for drawing
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Layer 4: Branded frame
    _draw_branded_frame(draw, w, h, accent_c, role, slide_index, total_slides, handle)

    # Layer 5: Text layout (title-only when captions carry the script body)
    if title_only:
        _layout_title_only(img, slide, colors, w, h)
    else:
        layout_fn = _LAYOUT_FUNCS.get(role, _layout_context)
        layout_fn(img, slide, colors, w, h)

    return img


def treat_background(
    bg_image: Image.Image,
    target_w: int,
    target_h: int,
    accent_hex: str,
    role: str,
    bg_scale: float = 1.20,
) -> Image.Image:
    """Apply blur + gradient treatment and upscale for Ken Burns animation.

    Returns an oversized treated image (target_size * bg_scale).
    """
    accent_c = _hex_to_tuple(accent_hex)
    scaled_w = int(target_w * bg_scale)
    scaled_h = int(target_h * bg_scale)
    bg = bg_image.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
    blur_radius = 18 if role == "hook" else 14
    bg = _blur_background(bg, radius=blur_radius)
    bg = _gradient_overlay(bg, accent_c, role=role)
    return bg


def composite_slide_layers(
    bg_image: Image.Image,
    slide: dict,
    slide_index: int,
    total_slides: int,
    colors: dict,
    handle: str = "@cristian.bojaca",
    foreground: Optional[Image.Image] = None,
    bg_scale: float = 1.20,
    caption_safe_pct: float = 0.0,
    title_only: bool = False,
) -> tuple[Image.Image, Image.Image]:
    """Render slide as separate background + overlay for animated compositing.

    Unlike composite_slide() which returns a single flat image, this returns
    two layers that can be animated independently (Ken Burns on background
    while text overlay stays static).

    Args:
        bg_scale: Factor to upscale background (1.20 = 20% extra room for
            zoom/pan animation).
        caption_safe_pct: Fraction of frame height to reserve at the bottom
            for platform UI + external captions (e.g. 0.27 = bottom 27% clear).
        title_only: If True, render only the title at the top instead of
            the full role-specific layout. Use when external captions carry
            the script body.

    Returns:
        (treated_bg, text_overlay):
        - treated_bg: Blurred + gradient background at bg_scale x target size.
        - text_overlay: RGBA image at target size with foreground, frame,
          and text (title-only or full layout).
    """
    w, h = bg_image.size
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
    role = get_slide_role(slide_index, total_slides)

    # --- Layers 1-2: Treated background (oversized for Ken Burns room) ---
    bg_treated = treat_background(
        bg_image, w, h, colors.get("accent", "#F7B731"), role, bg_scale,
    )

    # --- Layers 3-5: Text overlay (target size, transparent canvas) ---
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    # Layer 3: Foreground subject
    if foreground is not None:
        overlay = _composite_foreground(overlay, foreground, role=role)

    # Layer 4: Branded frame
    draw = ImageDraw.Draw(overlay)
    _draw_branded_frame(draw, w, h, accent_c, role, slide_index, total_slides, handle)

    # Layer 5: Text layout (title-only when captions carry the script body)
    if title_only:
        _layout_title_only(overlay, slide, colors, w, h)
    else:
        layout_fn = _LAYOUT_FUNCS.get(role, _layout_context)
        layout_fn(overlay, slide, colors, w, h)

    # Clear caption safe zone at bottom (for external subtitle overlays)
    if caption_safe_pct > 0:
        zone_h = int(h * caption_safe_pct)
        clear_draw = ImageDraw.Draw(overlay)
        clear_draw.rectangle([0, h - zone_h, w, h], fill=(0, 0, 0, 0))

    return bg_treated, overlay


# ── Main Pipeline ────────────────────────────────────────────────────────────


def generate_slide_images(
    slides: list[dict],
    image_prompts: list[str],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
) -> list[str]:
    """Generate AI background images and composite with slide text.

    Args:
        slides: List of slide dicts.
        image_prompts: One image prompt per slide.
        colors: Color scheme.
        aspect_ratio: '9:16' or '16:9'.
        output_dir: Where to save the composited PNGs.
        handle: Social media handle.

    Returns:
        List of file paths to the composited PNG images.
    """
    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i, (slide, prompt) in enumerate(zip(slides, image_prompts)):
        # Small delay between requests to avoid rate limits
        if i > 0:
            time.sleep(2)

        # Generate AI background
        img_bytes = generate_image(
            prompt=prompt,
            width=img_w,
            height=img_h,
        )
        bg_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        bg_image = bg_image.resize((img_w, img_h), Image.Resampling.LANCZOS)

        # Composite slide text over AI background
        final = composite_slide(
            bg_image=bg_image,
            slide=slide,
            slide_index=i,
            total_slides=len(slides),
            colors=colors,
            handle=handle,
        )

        # Save
        out_path = os.path.join(output_dir, f"ai_slide_{i + 1:02d}.png")
        final.save(out_path, "PNG")
        paths.append(out_path)

    return paths
