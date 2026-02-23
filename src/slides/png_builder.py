"""Render slides as PNG images using Pillow. Mirrors the PPTX layout."""

import io
import math
import os
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
