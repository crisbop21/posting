"""Generate AI images for slides using Gemini Flash or OpenAI DALL-E 3.

Supports two providers (auto-detected from environment variables):
  - Google Gemini: Free via Google AI Studio API key (gemini-2.5-flash-image).
  - OpenAI DALL-E 3: Paid, requires OpenAI API key.

Each slide gets a cinematic, finance-themed image generated from its content.
The image is then composited with slide text overlay to create the final frame.
"""

import base64
import io
import os
import time
from pathlib import Path

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


# ── Slide Image Compositing ──────────────────────────────────────────────────


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
        # Stronger darkness at top and bottom, lighter in middle
        t = y / h
        if t < 0.3:
            alpha = int(200 - (t / 0.3) * 80)
        elif t > 0.7:
            alpha = int(120 + ((t - 0.7) / 0.3) * 80)
        else:
            alpha = 120
        # Slight accent color tint
        r = int(accent_color[0] * 0.15)
        g = int(accent_color[1] * 0.15)
        b = int(accent_color[2] * 0.15)
        draw.line([(0, y), (w, y)], fill=(r, g, b, alpha))

    if img.mode != "RGBA":
        img = img.convert("RGBA")
    return Image.alpha_composite(img, gradient)


def composite_slide(
    bg_image: Image.Image,
    slide: dict,
    slide_index: int,
    total_slides: int,
    colors: dict,
    handle: str = "@cristian.bojaca",
) -> Image.Image:
    """Composite slide text over an AI-generated background image.

    Applies a dark gradient overlay for readability, then renders
    title, body, footer, and slide counter on top.
    """
    w, h = bg_image.size
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))

    # Apply overlay for text readability
    img = _add_gradient_overlay(bg_image, accent_c)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    margin = int(w * 0.08)
    usable_w = w - 2 * margin

    title = slide.get("title", "")
    body = slide.get("body", "")
    footer = slide.get("footer", "")

    title_c = (255, 255, 255)
    body_c = _hex_to_tuple(colors.get("body", "#C9D1D9"))

    # Slide counter
    counter_font = _load_font("sans", int(w * 0.035))
    counter_text = f"{slide_index + 1}/{total_slides}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_x = w - margin - (counter_bbox[2] - counter_bbox[0])
    counter_y = int(h * 0.04)
    draw.text(
        (counter_x + 2, counter_y + 2),
        counter_text,
        font=counter_font,
        fill=(0, 0, 0, 128),
    )
    draw.text(
        (counter_x, counter_y),
        counter_text,
        font=counter_font,
        fill=(200, 200, 200),
    )

    # Title
    base_title_size = int(w * 0.083)
    title_size = _calc_title_font_size(title, usable_w, base_title_size)
    title_font = _load_font("sans_bold", title_size)
    title_lines = _wrap_text(title, title_font, usable_w)

    title_y = int(h * 0.28)
    line_spacing = int(title_size * 1.25)

    for line in title_lines:
        # Text shadow
        draw.text(
            (margin + 3, title_y + 3),
            line,
            font=title_font,
            fill=(0, 0, 0),
        )
        draw.text((margin, title_y), line, font=title_font, fill=title_c)
        title_y += line_spacing

    # Accent bar under title
    bar_y = title_y + int(h * 0.015)
    draw.rectangle(
        [margin, bar_y, margin + int(usable_w * 0.25), bar_y + 4],
        fill=accent_c,
    )

    # Body
    body_y = bar_y + int(h * 0.04)
    body_font = _load_font("serif", int(w * 0.042))
    body_lines = _wrap_text(body, body_font, usable_w)
    body_line_spacing = int(w * 0.042 * 1.5)

    for line in body_lines:
        draw.text(
            (margin + 2, body_y + 2),
            line,
            font=body_font,
            fill=(0, 0, 0),
        )
        draw.text((margin, body_y), line, font=body_font, fill=body_c)
        body_y += body_line_spacing

    # Footer / handle
    footer_font = _load_font("sans", int(w * 0.032))
    footer_text = handle if not footer else f"{footer}  ·  {handle}"
    footer_bbox = footer_font.getbbox(footer_text)
    footer_w = footer_bbox[2] - footer_bbox[0]
    footer_x = (w - footer_w) // 2
    footer_y = h - int(h * 0.06)
    draw.text(
        (footer_x + 2, footer_y + 2),
        footer_text,
        font=footer_font,
        fill=(0, 0, 0),
    )
    draw.text(
        (footer_x, footer_y),
        footer_text,
        font=footer_font,
        fill=(180, 180, 180),
    )

    return img


# ── Main Pipeline ─────────────────────────────────────────────────────────────


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
