"""Render slides as PNG images using Pillow. Mirrors the PPTX layout.

Also provides ``build_style_alternatives`` for generating multiple local
design variations without any external API or login.
"""

import io
import math
import os
import random
import textwrap
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont


# Font paths: Liberation is metric compatible with Arial and Times New Roman.
# Fall back to DejaVu if Liberation isn't available.
_FONT_PATHS = {
    "sans": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "sans_bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "serif": "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
}
_FALLBACK_PATHS = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "sans_bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
}

# ── Style presets for alternative versions ──────────────────────────────────

STYLE_PRESETS: list[dict] = [
    {
        "name": "Neon Cyber",
        "colors": {
            "background": "#0A0A1A",
            "title": "#00FFAA",
            "body": "#E0E0FF",
            "accent": "#FF00FF",
        },
        "title_font": "sans_bold",
        "body_font": "sans",
        "accent_style": "top_bottom_bars",
        "body_size": 58,
    },
    {
        "name": "Warm Editorial",
        "colors": {
            "background": "#1A1210",
            "title": "#FFD6A5",
            "body": "#E8D5C4",
            "accent": "#FF6B35",
        },
        "title_font": "serif",
        "body_font": "serif",
        "accent_style": "bottom_bar",
        "body_size": 62,
    },
    {
        "name": "Ice Blue",
        "colors": {
            "background": "#0B1929",
            "title": "#FFFFFF",
            "body": "#B0D4F1",
            "accent": "#00BFFF",
        },
        "title_font": "sans_bold",
        "body_font": "serif",
        "accent_style": "left_gradient",
        "body_size": 60,
    },
    {
        "name": "Bold Red",
        "colors": {
            "background": "#1A0000",
            "title": "#FF4444",
            "body": "#F5D6D6",
            "accent": "#FF0044",
        },
        "title_font": "sans_bold",
        "body_font": "sans",
        "accent_style": "corner_brackets",
        "body_size": 60,
    },
    {
        "name": "Green Money",
        "colors": {
            "background": "#0D1A0D",
            "title": "#00FF66",
            "body": "#C8E6C9",
            "accent": "#00CC44",
        },
        "title_font": "sans_bold",
        "body_font": "sans",
        "accent_style": "left_bar",
        "body_size": 60,
    },
    {
        "name": "Soft Minimal",
        "colors": {
            "background": "#F5F0EB",
            "title": "#1A1A2E",
            "body": "#3D3D5C",
            "accent": "#6C63FF",
        },
        "title_font": "serif",
        "body_font": "serif",
        "accent_style": "thin_underline",
        "body_size": 60,
    },
]


def _load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    path = _FONT_PATHS.get(style, _FONT_PATHS["sans"])
    if not os.path.exists(path):
        path = _FALLBACK_PATHS.get(style, _FALLBACK_PATHS["sans"])
    return ImageFont.truetype(path, size)


def _hex_to_tuple(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _calc_title_font_size(title_text: str, max_width: int, base_size: int = 90) -> int:
    """Pick a title font size that keeps the title to max 4 lines."""
    for size in [base_size, 76, 64]:
        font = _load_font("sans_bold", size)
        lines = _wrap_text(title_text, font, max_width)
        if len(lines) <= 4:
            return size
    return 64


# ── Accent decorators ───────────────────────────────────────────────────────

def _draw_accent_left_bar(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    draw.rectangle([0, 0, bar_w, img_h], fill=accent)


def _draw_accent_top_bottom_bars(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    bar_h = max(int(img_h * 0.006), 4)
    draw.rectangle([0, 0, img_w, bar_h], fill=accent)
    draw.rectangle([0, img_h - bar_h, img_w, img_h], fill=accent)


def _draw_accent_bottom_bar(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    bar_h = max(int(img_h * 0.008), 6)
    draw.rectangle([0, img_h - bar_h, img_w, img_h], fill=accent)


def _draw_accent_left_gradient(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    grad_w = max(int(img_w * 0.03), 20)
    for x in range(grad_w):
        alpha = 1.0 - (x / grad_w)
        c = tuple(int(v * alpha) for v in accent)
        draw.line([(x, 0), (x, img_h)], fill=c)


def _draw_accent_corner_brackets(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    length = int(min(img_w, img_h) * 0.08)
    thickness = max(int(img_w * 0.005), 3)
    m = int(img_w * 0.06)
    # top-left
    draw.rectangle([m, m, m + length, m + thickness], fill=accent)
    draw.rectangle([m, m, m + thickness, m + length], fill=accent)
    # top-right
    draw.rectangle([img_w - m - length, m, img_w - m, m + thickness], fill=accent)
    draw.rectangle([img_w - m - thickness, m, img_w - m, m + length], fill=accent)
    # bottom-left
    draw.rectangle([m, img_h - m - thickness, m + length, img_h - m], fill=accent)
    draw.rectangle([m, img_h - m - length, m + thickness, img_h - m], fill=accent)
    # bottom-right
    draw.rectangle([img_w - m - length, img_h - m - thickness, img_w - m, img_h - m], fill=accent)
    draw.rectangle([img_w - m - thickness, img_h - m - length, img_w - m, img_h - m], fill=accent)


def _draw_accent_thin_underline(draw: ImageDraw.Draw, img_w: int, img_h: int, accent: tuple, bar_w: int):
    """Drawn later after the title — just a no-op here."""
    pass


_ACCENT_DRAWERS = {
    "left_bar": _draw_accent_left_bar,
    "top_bottom_bars": _draw_accent_top_bottom_bars,
    "bottom_bar": _draw_accent_bottom_bar,
    "left_gradient": _draw_accent_left_gradient,
    "corner_brackets": _draw_accent_corner_brackets,
    "thin_underline": _draw_accent_thin_underline,
}


def build_pngs(
    slides: list[dict],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
) -> list[str]:
    """Render each slide as a PNG image.

    Returns:
        List of file paths to the generated PNG images.
    """
    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    bg = _hex_to_tuple(colors.get("background", "#0D1117"))
    title_c = _hex_to_tuple(colors.get("title", "#FFFFFF"))
    body_c = _hex_to_tuple(colors.get("body", "#C9D1D9"))
    accent_c = _hex_to_tuple(colors.get("accent", "#58A6FF"))
    muted_c = (74, 85, 104)  # #4A5568

    margin = int(img_w * 0.12)  # ~12% margin
    content_w = img_w - 2 * margin
    accent_bar_w = max(int(img_w * 0.007), 4)  # ~5px at 1080w

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths: list[str] = []

    total = len(slides)
    for idx, slide_data in enumerate(slides):
        img = Image.new("RGB", (img_w, img_h), bg)
        draw = ImageDraw.Draw(img)

        # -- Left accent bar --
        draw.rectangle([0, 0, accent_bar_w, img_h], fill=accent_c)

        # -- Slide counter top-right --
        counter_font = _load_font("sans", 28)
        counter_text = f"{idx + 1}/{total}"
        counter_bbox = counter_font.getbbox(counter_text)
        counter_w = counter_bbox[2] - counter_bbox[0]
        draw.text(
            (img_w - margin - counter_w, int(img_h * 0.045)),
            counter_text,
            fill=accent_c,
            font=counter_font,
        )

        # -- Title --
        title_text = slide_data.get("title", "")
        title_size = _calc_title_font_size(title_text, content_w)
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title_text, title_font, content_w)

        title_y = int(img_h * 0.165)
        title_line_h = int(title_size * 1.35)
        for line in title_lines:
            draw.text((margin, title_y), line, fill=title_c, font=title_font)
            title_y += title_line_h

        # -- Divider --
        divider_y = title_y + int(img_h * 0.015)
        draw.rectangle(
            [margin, divider_y, img_w - margin, divider_y + 2],
            fill=muted_c,
        )

        # -- Body --
        body_text = slide_data.get("body", "")
        body_font = _load_font("serif", 64)
        body_lines = _wrap_text(body_text, body_font, content_w)
        body_y = divider_y + int(img_h * 0.02)
        body_line_h = int(64 * 1.45)
        for line in body_lines:
            draw.text((margin, body_y), line, fill=body_c, font=body_font)
            body_y += body_line_h

        # -- Handle at bottom --
        handle_font = _load_font("sans", 24)
        handle_bbox = handle_font.getbbox(handle)
        handle_w = handle_bbox[2] - handle_bbox[0]
        draw.text(
            ((img_w - handle_w) // 2, img_h - int(img_h * 0.06)),
            handle,
            fill=muted_c,
            font=handle_font,
        )

        # Save
        fname = f"slide_{idx + 1:02d}_{timestamp}.png"
        fpath = os.path.join(output_dir, fname)
        img.save(fpath, "PNG")
        paths.append(fpath)

    return paths


# ── Local alternative style generator ───────────────────────────────────────


def _render_styled_slide(
    slide_data: dict,
    idx: int,
    total: int,
    preset: dict,
    img_w: int,
    img_h: int,
    handle: str,
) -> Image.Image:
    """Render a single slide with the given style preset."""
    sc = preset["colors"]
    bg = _hex_to_tuple(sc["background"])
    title_c = _hex_to_tuple(sc["title"])
    body_c = _hex_to_tuple(sc["body"])
    accent_c = _hex_to_tuple(sc["accent"])
    muted_c = tuple(max(0, min(255, c + (40 if sum(bg) < 384 else -40))) for c in bg)

    margin = int(img_w * 0.12)
    content_w = img_w - 2 * margin
    accent_bar_w = max(int(img_w * 0.007), 4)

    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # -- Accent decoration --
    accent_style = preset.get("accent_style", "left_bar")
    drawer = _ACCENT_DRAWERS.get(accent_style, _draw_accent_left_bar)
    drawer(draw, img_w, img_h, accent_c, accent_bar_w)

    # -- Slide counter --
    counter_font = _load_font("sans", 28)
    counter_text = f"{idx + 1}/{total}"
    counter_bbox = counter_font.getbbox(counter_text)
    counter_w = counter_bbox[2] - counter_bbox[0]
    draw.text(
        (img_w - margin - counter_w, int(img_h * 0.045)),
        counter_text,
        fill=accent_c,
        font=counter_font,
    )

    # -- Title --
    title_text = slide_data.get("title", "")
    title_font_style = preset.get("title_font", "sans_bold")
    title_size = _calc_title_font_size(title_text, content_w)
    title_font = _load_font(title_font_style, title_size)
    title_lines = _wrap_text(title_text, title_font, content_w)

    title_y = int(img_h * 0.165)
    title_line_h = int(title_size * 1.35)
    for line in title_lines:
        draw.text((margin, title_y), line, fill=title_c, font=title_font)
        title_y += title_line_h

    # -- Divider / underline --
    divider_y = title_y + int(img_h * 0.015)
    if accent_style == "thin_underline":
        draw.rectangle(
            [margin, divider_y, margin + int(content_w * 0.3), divider_y + 3],
            fill=accent_c,
        )
    else:
        draw.rectangle(
            [margin, divider_y, img_w - margin, divider_y + 2],
            fill=muted_c,
        )

    # -- Body --
    body_text = slide_data.get("body", "")
    body_font_style = preset.get("body_font", "serif")
    body_size = preset.get("body_size", 64)
    body_font = _load_font(body_font_style, body_size)
    body_lines = _wrap_text(body_text, body_font, content_w)
    body_y = divider_y + int(img_h * 0.025)
    body_line_h = int(body_size * 1.45)
    for line in body_lines:
        draw.text((margin, body_y), line, fill=body_c, font=body_font)
        body_y += body_line_h

    # -- Handle --
    handle_font = _load_font("sans", 24)
    handle_bbox = handle_font.getbbox(handle)
    handle_w = handle_bbox[2] - handle_bbox[0]
    draw.text(
        ((img_w - handle_w) // 2, img_h - int(img_h * 0.06)),
        handle,
        fill=muted_c,
        font=handle_font,
    )

    return img


def build_style_alternatives(
    slides: list[dict],
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    num_alternatives: int = 4,
) -> list[dict]:
    """Generate multiple local design variations of the same slide content.

    Works entirely offline using Pillow — no API keys, no logins, no internet.

    Returns:
        List of dicts, each with:
          - version: str (e.g. "Alternative A")
          - style: str (preset name)
          - png_paths: list[str] (one PNG per slide)
    """
    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Pick presets (shuffle so results vary across runs)
    available = list(STYLE_PRESETS)
    random.shuffle(available)
    chosen = available[: min(num_alternatives, len(available))]

    total = len(slides)
    labels = ["A", "B", "C", "D", "E", "F"]
    results: list[dict] = []

    for alt_idx, preset in enumerate(chosen):
        label = labels[alt_idx] if alt_idx < len(labels) else str(alt_idx + 1)
        version_name = f"Alternative {label}"
        style_name = preset["name"]
        paths: list[str] = []

        for slide_idx, slide_data in enumerate(slides):
            img = _render_styled_slide(
                slide_data, slide_idx, total, preset, img_w, img_h, handle,
            )
            fname = f"alt{label}_slide_{slide_idx + 1:02d}_{timestamp}.png"
            fpath = os.path.join(output_dir, fname)
            img.save(fpath, "PNG")
            paths.append(fpath)

        results.append({
            "version": version_name,
            "style": style_name,
            "png_paths": paths,
        })

    return results
