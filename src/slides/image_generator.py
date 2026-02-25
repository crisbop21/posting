"""Generate AI images for slides using Flux via Replicate.

Uses Flux Schnell (fast, ~2-4s per image) for slide background images.
Each slide gets a cinematic, finance-themed image generated from its content.
The image is then composited with slide text overlay to create the final frame.
"""

import io
import os
import tempfile
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.slides.png_builder import (
    _hex_to_tuple,
    _load_font,
    _wrap_text,
    _calc_title_font_size,
)


# ── Replicate API ─────────────────────────────────────────────────────────────


def _get_replicate_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN", "")
    if not token:
        raise RuntimeError(
            "REPLICATE_API_TOKEN environment variable is not set. "
            "Get your token at https://replicate.com/account/api-tokens"
        )
    return token


def generate_image(
    prompt: str,
    width: int = 1080,
    height: int = 1920,
    model: str = "black-forest-labs/flux-schnell",
) -> bytes:
    """Generate an image using Flux via Replicate API.

    Uses raw HTTP requests to avoid requiring the replicate package.

    Returns raw image bytes (WebP/PNG).
    """
    token = _get_replicate_token()

    # Create a prediction
    resp = requests.post(
        "https://api.replicate.com/v1/models/{}/predictions".format(model),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json={
            "input": {
                "prompt": prompt,
                "num_outputs": 1,
                "aspect_ratio": f"{width}:{height}",
                "output_format": "png",
                "output_quality": 90,
            }
        },
        timeout=30,
    )
    resp.raise_for_status()
    prediction = resp.json()

    # Poll for completion
    poll_url = prediction.get("urls", {}).get("get", "")
    if not poll_url:
        poll_url = f"https://api.replicate.com/v1/predictions/{prediction['id']}"

    import time

    for _ in range(60):  # max 60 polls × 2s = 2 minutes
        poll_resp = requests.get(
            poll_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        poll_resp.raise_for_status()
        result = poll_resp.json()

        status = result.get("status", "")
        if status == "succeeded":
            output = result.get("output", [])
            if output:
                # Download the image
                img_url = output[0] if isinstance(output, list) else output
                img_resp = requests.get(img_url, timeout=60)
                img_resp.raise_for_status()
                return img_resp.content
            raise RuntimeError("Flux prediction succeeded but returned no output")
        elif status == "failed":
            error = result.get("error", "Unknown error")
            raise RuntimeError(f"Flux image generation failed: {error}")
        elif status == "canceled":
            raise RuntimeError("Flux image generation was canceled")

        time.sleep(2)

    raise RuntimeError("Flux image generation timed out after 2 minutes")


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
