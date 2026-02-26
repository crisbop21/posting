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
    """Draw consistent branded frame: accent bars, counter pill, handle."""
    # Top accent bar
    bar_h = max(int(h * 0.004), 3)
    draw.rectangle([0, 0, w, bar_h], fill=accent_c)

    # Bottom accent bar (thicker)
    bottom_bar_h = max(int(h * 0.005), 4)
    draw.rectangle([0, h - bottom_bar_h, w, h], fill=accent_c)

    # Slide counter pill (top-right)
    counter_font = _load_font("sans_bold", int(w * 0.032))
    counter_text = f"{slide_index + 1}/{total_slides}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_tw = counter_bbox[2] - counter_bbox[0]
    counter_th = counter_bbox[3] - counter_bbox[1]

    pill_pad_x = int(w * 0.025)
    pill_pad_y = int(h * 0.008)
    pill_x = w - int(w * 0.08) - counter_tw - pill_pad_x
    pill_y = int(h * 0.035)
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

    # Handle at bottom center
    handle_font = _load_font("sans", int(w * 0.028))
    handle_bbox = handle_font.getbbox(handle)
    handle_tw = handle_bbox[2] - handle_bbox[0]
    handle_x = (w - handle_tw) // 2
    handle_y = h - int(h * 0.045)
    draw.text((handle_x + 2, handle_y + 2), handle, font=handle_font, fill=(0, 0, 0))
    draw.text((handle_x, handle_y), handle, font=handle_font, fill=(200, 200, 200))


# ── Text Rendering Helpers ───────────────────────────────────────────────────


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


# ── Layer 5: Role-Specific Text Layouts ──────────────────────────────────────


def _layout_hook(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Hook slide (slide 1): massive bold text, high-impact, attention-grabbing.

    Large title top-left, accent highlight on first line,
    body text below with generous spacing.
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))
    highlight_c = _hex_to_tuple(colors.get("highlight", "#F0883E"))

    margin = int(w * 0.08)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Massive title — aim for 2-3 lines max
    title_size = int(w * 0.105)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 70:
        title_size -= 8
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.12)
    line_h = int(title_size * 1.3)

    # First line gets a highlight box
    if title_lines:
        _draw_highlight_box(
            draw,
            (margin, title_y),
            title_lines[0],
            title_font,
            text_color=(255, 255, 255),
            box_color=(*highlight_c, 220),
            pad_x=int(w * 0.015),
            pad_y=int(h * 0.006),
        )
        title_y += line_h + int(h * 0.01)

    # Remaining title lines — white with shadow
    for line in title_lines[1:]:
        _draw_shadowed_text(draw, (margin, title_y), line, title_font, (255, 255, 255), 4)
        title_y += line_h

    # Thick accent bar
    bar_y = title_y + int(h * 0.02)
    draw.rectangle(
        [margin, bar_y, margin + int(content_w * 0.35), bar_y + 6],
        fill=accent_c,
    )

    # Body text — larger than other roles
    body_y = bar_y + int(h * 0.035)
    body_size = int(w * 0.050)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, content_w)
    body_line_h = int(body_size * 1.5)

    for line in body_lines:
        _draw_shadowed_text(draw, (margin, body_y), line, body_font, (230, 230, 240), 2)
        body_y += body_line_h


def _layout_context(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Context slide (slides 2-3): clean, left-aligned, evidence-style.

    Medium title, thin divider, body text with breathing room.
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))
    body_c = _hex_to_tuple(colors.get("body", "#C9D1D9"))

    margin = int(w * 0.08)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Medium title
    title_size = int(w * 0.078)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 4 and title_size > 60:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.18)
    line_h = int(title_size * 1.3)

    for line in title_lines:
        _draw_shadowed_text(draw, (margin, title_y), line, title_font, (255, 255, 255), 3)
        title_y += line_h

    # Thin divider line
    div_y = title_y + int(h * 0.015)
    draw.rectangle([margin, div_y, margin + content_w, div_y + 2], fill=(*accent_c, 120))

    # Body text
    body_y = div_y + int(h * 0.025)
    body_size = int(w * 0.044)
    body_font = _load_font("serif", body_size)
    body_lines = _wrap_text(body, body_font, content_w)
    body_line_h = int(body_size * 1.55)

    for line in body_lines:
        _draw_shadowed_text(draw, (margin, body_y), line, body_font, body_c, 2)
        body_y += body_line_h


def _layout_payoff(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """Payoff slide: visual shift — accent color swap, centered layout.

    Uses the highlight color as dominant for the 'reveal' moment.
    """
    draw = ImageDraw.Draw(img)
    highlight_c = _hex_to_tuple(colors.get("highlight", "#F0883E"))
    body_c = _hex_to_tuple(colors.get("body", "#C9D1D9"))

    margin = int(w * 0.10)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Title — centered, in highlight color
    title_size = int(w * 0.088)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 4 and title_size > 64:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    title_y = int(h * 0.20)
    line_h = int(title_size * 1.3)

    for line in title_lines:
        line_bbox = title_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, title_y), line, title_font, highlight_c, 3)
        title_y += line_h

    # Centered accent bar
    bar_w = int(content_w * 0.2)
    bar_x = (w - bar_w) // 2
    bar_y = title_y + int(h * 0.02)
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + 5], fill=highlight_c)

    # Body — centered
    body_y = bar_y + int(h * 0.035)
    body_size = int(w * 0.044)
    body_font = _load_font("serif", body_size)
    body_lines = _wrap_text(body, body_font, content_w)
    body_line_h = int(body_size * 1.5)

    for line in body_lines:
        line_bbox = body_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, body_y), line, body_font, body_c, 2)
        body_y += body_line_h


def _layout_cta(
    img: Image.Image,
    slide: dict,
    colors: dict,
    w: int,
    h: int,
):
    """CTA slide (last slide): minimal, personal, direct.

    Heavy overlay for clean look. Text vertically centered.
    """
    draw = ImageDraw.Draw(img)
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))

    margin = int(w * 0.12)
    content_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")

    # Title sizing
    title_size = int(w * 0.072)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, content_w)
    while len(title_lines) > 3 and title_size > 56:
        title_size -= 6
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title, title_font, content_w)

    # Body sizing
    body_size = int(w * 0.040)
    body_font = _load_font("sans", body_size)
    body_lines = _wrap_text(body, body_font, content_w)

    # Vertically center the entire text block
    line_h = int(title_size * 1.35)
    body_line_h = int(body_size * 1.5)
    total_text_h = (
        len(title_lines) * line_h
        + int(h * 0.04)
        + len(body_lines) * body_line_h
    )
    start_y = (h - total_text_h) // 2

    title_y = start_y
    for line in title_lines:
        line_bbox = title_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, title_y), line, title_font, (255, 255, 255), 3)
        title_y += line_h

    # Three accent dots instead of a bar
    dot_y = title_y + int(h * 0.015)
    dot_r = int(w * 0.006)
    cx = w // 2
    for offset in [-20, 0, 20]:
        draw.ellipse(
            [cx + offset - dot_r, dot_y - dot_r, cx + offset + dot_r, dot_y + dot_r],
            fill=accent_c,
        )

    # Body — centered
    body_y = dot_y + int(h * 0.025)
    for line in body_lines:
        line_bbox = body_font.getbbox(line)
        line_w = line_bbox[2] - line_bbox[0]
        x = (w - line_w) // 2
        _draw_shadowed_text(draw, (x, body_y), line, body_font, (200, 200, 210), 2)
        body_y += body_line_h


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
) -> Image.Image:
    """Composite a slide with the full 5-layer visual treatment.

    Layers (bottom to top):
      1. Background: blurred source image
      2. Gradient overlay: role-aware darkening + accent tint
      3. Foreground subject: transparent PNG cutout (optional)
      4. Branded frame: accent bars, counter pill, handle
      5. Text: role-specific layout (hook / context / payoff / cta)

    Args:
        bg_image: Source background image (will be blurred).
        slide: Slide dict with 'title', 'body', optional 'footer'.
        slide_index: 0-based position.
        total_slides: Total number of slides.
        colors: Color scheme dict.
        handle: Social media handle.
        foreground: Optional transparent PNG cutout to overlay.
    """
    w, h = bg_image.size
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))
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

    # Layer 5: Role-specific text layout
    layout_fn = _LAYOUT_FUNCS.get(role, _layout_context)
    layout_fn(img, slide, colors, w, h)

    return img


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
