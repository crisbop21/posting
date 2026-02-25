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

from PIL import Image, ImageDraw, ImageFilter, ImageFont


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


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Linearly interpolate between two RGB colors."""
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


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


# ── Generative background painters ─────────────────────────────────────────


def _bg_neon_cyber(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Dark bg with diagonal gradient, neon glow circles, and grid lines."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Diagonal gradient: bg -> slightly lighter
    bg2 = _lerp_color(bg, accent, 0.08)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Subtle grid lines
    grid_color = _lerp_color(bg, accent, 0.12)
    spacing = w // 8
    for x in range(0, w, spacing):
        draw.line([(x, 0), (x, h)], fill=grid_color, width=1)
    for y in range(0, h, spacing):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Neon glow circles (drawn on overlay, then blurred and composited)
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for _ in range(rng.randint(5, 9)):
        cx = rng.randint(0, w)
        cy = rng.randint(0, h)
        r = rng.randint(40, 160)
        alpha = rng.randint(20, 50)
        glow_draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(*accent, alpha),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=50))
    img.paste(Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), glow).convert("RGB"),
              mask=glow.split()[3])

    # Small bright dots (stars)
    for _ in range(rng.randint(30, 60)):
        dx = rng.randint(0, w)
        dy = rng.randint(0, h)
        dot_r = rng.randint(1, 3)
        dot_alpha = rng.randint(80, 180)
        dot_color = _lerp_color(accent, (255, 255, 255), 0.6)
        draw = ImageDraw.Draw(img)
        draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r], fill=dot_color)


def _bg_warm_editorial(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Warm radial gradient with soft bokeh circles."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Radial gradient from center
    center_color = _lerp_color(bg, accent, 0.15)
    cx, cy = w // 2, int(h * 0.4)
    max_dist = math.sqrt(cx ** 2 + cy ** 2) * 1.2
    for y in range(h):
        for x_step in range(0, w, 4):  # step by 4 for performance
            dist = math.sqrt((x_step - cx) ** 2 + (y - cy) ** 2)
            t = min(dist / max_dist, 1.0)
            c = _lerp_color(center_color, bg, t)
            draw.line([(x_step, y), (x_step + 3, y)], fill=c)

    # Soft bokeh circles
    bokeh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bokeh)
    warm_tones = [
        _lerp_color(accent, (255, 200, 100), 0.3),
        _lerp_color(accent, (255, 180, 80), 0.5),
        accent,
    ]
    for _ in range(rng.randint(8, 14)):
        bx = rng.randint(-100, w + 100)
        by = rng.randint(-100, h + 100)
        br = rng.randint(60, 200)
        bc = rng.choice(warm_tones)
        alpha = rng.randint(12, 30)
        bdraw.ellipse([bx - br, by - br, bx + br, by + br], fill=(*bc, alpha))
        # Ring outline
        bdraw.ellipse(
            [bx - br, by - br, bx + br, by + br],
            outline=(*bc, alpha + 10), width=2,
        )
    bokeh = bokeh.filter(ImageFilter.GaussianBlur(radius=25))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), bokeh).convert("RGB"),
        mask=bokeh.split()[3],
    )


def _bg_ice_blue(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Cool diagonal gradient with frost dots and light streaks."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Diagonal gradient (top-left dark -> bottom-right lighter)
    corner_color = _lerp_color(bg, accent, 0.18)
    for y in range(h):
        t = y / h
        row_start = _lerp_color(bg, _lerp_color(bg, corner_color, t * 0.5), t)
        row_end = _lerp_color(bg, corner_color, t)
        for x_step in range(0, w, 4):
            xt = x_step / w
            c = _lerp_color(row_start, row_end, xt)
            draw.line([(x_step, y), (x_step + 3, y)], fill=c)

    # Light streaks (diagonal lines)
    streak = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(streak)
    for _ in range(rng.randint(3, 6)):
        sx = rng.randint(-200, w)
        sy = rng.randint(-200, h)
        length = rng.randint(400, 900)
        alpha = rng.randint(8, 18)
        sdraw.line(
            [(sx, sy), (sx + length, sy + int(length * 0.6))],
            fill=(*accent, alpha), width=rng.randint(40, 100),
        )
    streak = streak.filter(ImageFilter.GaussianBlur(radius=40))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), streak).convert("RGB"),
        mask=streak.split()[3],
    )

    # Frost dots
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(40, 80)):
        fx = rng.randint(0, w)
        fy = rng.randint(0, h)
        fr = rng.randint(1, 4)
        fc = _lerp_color(accent, (255, 255, 255), rng.random() * 0.5)
        draw.ellipse([fx - fr, fy - fr, fx + fr, fy + fr], fill=fc)


def _bg_bold_red(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Dark radial gradient with angular geometric shapes."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Radial gradient from bottom-right corner
    cx, cy = int(w * 0.8), int(h * 0.7)
    hot_spot = _lerp_color(bg, accent, 0.20)
    max_dist = math.sqrt(w ** 2 + h ** 2)
    for y in range(h):
        dist_y = (y - cy) ** 2
        for x_step in range(0, w, 4):
            dist = math.sqrt((x_step - cx) ** 2 + dist_y)
            t = min(dist / max_dist, 1.0)
            c = _lerp_color(hot_spot, bg, t)
            draw.line([(x_step, y), (x_step + 3, y)], fill=c)

    # Angular geometric shapes (triangles / diamonds)
    shapes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shdraw = ImageDraw.Draw(shapes)
    for _ in range(rng.randint(4, 8)):
        cx_ = rng.randint(0, w)
        cy_ = rng.randint(0, h)
        size = rng.randint(80, 300)
        alpha = rng.randint(10, 25)
        # Diamond
        pts = [
            (cx_, cy_ - size),
            (cx_ + size, cy_),
            (cx_, cy_ + size),
            (cx_ - size, cy_),
        ]
        shdraw.polygon(pts, fill=(*accent, alpha))
        shdraw.polygon(pts, outline=(*accent, alpha + 15), width=2)
    shapes = shapes.filter(ImageFilter.GaussianBlur(radius=15))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), shapes).convert("RGB"),
        mask=shapes.split()[3],
    )


def _bg_green_money(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Dark bg with upward chart lines and floating currency circles."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Subtle vertical gradient
    bg2 = _lerp_color(bg, accent, 0.06)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg2, bg, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Chart-like upward trend lines
    chart = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(chart)
    for line_i in range(rng.randint(3, 5)):
        points = []
        y_base = rng.randint(int(h * 0.3), int(h * 0.8))
        for xi in range(0, w + 40, 40):
            # Trending up with noise
            trend = -int((xi / w) * h * rng.uniform(0.05, 0.15))
            noise = rng.randint(-30, 30)
            points.append((xi, y_base + trend + noise))
        alpha = rng.randint(15, 35)
        if len(points) >= 2:
            cdraw.line(points, fill=(*accent, alpha), width=rng.randint(2, 4))
            # Fill area below the line
            fill_points = points + [(w, h), (0, h)]
            cdraw.polygon(fill_points, fill=(*accent, alpha // 3))
    chart = chart.filter(ImageFilter.GaussianBlur(radius=8))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), chart).convert("RGB"),
        mask=chart.split()[3],
    )

    # Floating circles with $ sign vibe (just circles)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(8, 15)):
        cx_ = rng.randint(0, w)
        cy_ = rng.randint(0, h)
        cr = rng.randint(15, 50)
        ring_c = _lerp_color(bg, accent, rng.uniform(0.15, 0.35))
        draw.ellipse([cx_ - cr, cy_ - cr, cx_ + cr, cy_ + cr], outline=ring_c, width=2)


def _bg_soft_minimal(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Light gradient with subtle geometric circles and thin lines."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Soft vertical gradient (light -> slightly warmer)
    bg2 = _lerp_color(bg, (240, 235, 225), 0.3)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Large subtle circles
    shapes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    shdraw = ImageDraw.Draw(shapes)
    for _ in range(rng.randint(3, 6)):
        cx_ = rng.randint(-100, w + 100)
        cy_ = rng.randint(-100, h + 100)
        cr = rng.randint(150, 400)
        alpha = rng.randint(10, 20)
        shdraw.ellipse(
            [cx_ - cr, cy_ - cr, cx_ + cr, cy_ + cr],
            outline=(*accent, alpha + 15), width=2,
        )
        shdraw.ellipse(
            [cx_ - cr, cy_ - cr, cx_ + cr, cy_ + cr],
            fill=(*accent, alpha),
        )
    shapes = shapes.filter(ImageFilter.GaussianBlur(radius=20))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), shapes).convert("RGB"),
        mask=shapes.split()[3],
    )

    # Thin decorative lines
    draw = ImageDraw.Draw(img)
    line_color = _lerp_color(bg, accent, 0.12)
    for _ in range(rng.randint(2, 4)):
        lx = rng.randint(0, w)
        draw.line([(lx, 0), (lx + rng.randint(-100, 100), h)], fill=line_color, width=1)


# Map style name -> background painter
_BG_PAINTERS = {
    "Neon Cyber": _bg_neon_cyber,
    "Warm Editorial": _bg_warm_editorial,
    "Ice Blue": _bg_ice_blue,
    "Bold Red": _bg_bold_red,
    "Green Money": _bg_green_money,
    "Soft Minimal": _bg_soft_minimal,
}


# ── Text shadow helper ─────────────────────────────────────────────────────


def _draw_text_with_shadow(
    img: Image.Image,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    shadow_color: tuple = (0, 0, 0),
    shadow_offset: int = 3,
):
    """Draw text with a soft drop shadow for readability over busy backgrounds."""
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sx, sy = xy[0] + shadow_offset, xy[1] + shadow_offset
    sdraw.text((sx, sy), text, fill=(*shadow_color, 140), font=font)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=4))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 0)), shadow).convert("RGB"),
        mask=shadow.split()[3],
    )
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, fill=fill, font=font)


# ── Vignette helper ─────────────────────────────────────────────────────────


def _apply_vignette(img: Image.Image, intensity: float = 0.5):
    """Apply a dark vignette around the edges for text readability."""
    w, h = img.size
    vignette = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vignette)

    # Draw concentric rectangles getting darker toward edges
    steps = 30
    for i in range(steps):
        t = i / steps
        # Opacity goes from 0 in center to intensity*255 at edges
        alpha = int(t * t * intensity * 200)
        inset = int((1 - t) * min(w, h) * 0.5)
        vdraw.rectangle(
            [0, 0, w, inset], fill=(0, 0, 0, alpha),
        )
        vdraw.rectangle(
            [0, h - inset, w, h], fill=(0, 0, 0, alpha),
        )
    vignette = vignette.filter(ImageFilter.GaussianBlur(radius=60))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), vignette).convert("RGB"),
        mask=vignette.split()[3],
    )


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
    """Render a single slide with the given style preset and rich background."""
    sc = preset["colors"]
    bg = _hex_to_tuple(sc["background"])
    title_c = _hex_to_tuple(sc["title"])
    body_c = _hex_to_tuple(sc["body"])
    accent_c = _hex_to_tuple(sc["accent"])
    is_dark = sum(bg) < 384
    muted_c = tuple(max(0, min(255, c + (40 if is_dark else -40))) for c in bg)
    shadow_c = (0, 0, 0) if is_dark else (100, 100, 100)

    margin = int(img_w * 0.12)
    content_w = img_w - 2 * margin
    accent_bar_w = max(int(img_w * 0.007), 4)

    img = Image.new("RGB", (img_w, img_h), bg)

    # -- Rich generative background --
    style_name = preset["name"]
    bg_painter = _BG_PAINTERS.get(style_name)
    if bg_painter:
        # Use idx as part of seed so each slide has a slightly different bg
        bg_painter(img, bg, accent_c, seed=42 + idx)

    # -- Vignette for text readability --
    _apply_vignette(img, intensity=0.35 if is_dark else 0.15)

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
    _draw_text_with_shadow(
        img,
        (img_w - margin - counter_w, int(img_h * 0.045)),
        counter_text, counter_font, accent_c, shadow_c, 2,
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
        _draw_text_with_shadow(
            img, (margin, title_y), line, title_font, title_c, shadow_c, 3,
        )
        title_y += title_line_h

    # -- Divider / underline --
    draw = ImageDraw.Draw(img)
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
        _draw_text_with_shadow(
            img, (margin, body_y), line, body_font, body_c, shadow_c, 2,
        )
        body_y += body_line_h

    # -- Handle --
    handle_font = _load_font("sans", 24)
    handle_bbox = handle_font.getbbox(handle)
    handle_w = handle_bbox[2] - handle_bbox[0]
    _draw_text_with_shadow(
        img,
        ((img_w - handle_w) // 2, img_h - int(img_h * 0.06)),
        handle, handle_font, muted_c, shadow_c, 2,
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
