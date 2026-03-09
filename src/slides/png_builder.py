"""Render slides as PNG images using Pillow. Mirrors the PPTX layout.

Also provides ``build_style_alternatives`` for generating multiple local
design variations without any external API or login.
"""

import io
import math
import os
import random
import re
import textwrap
from datetime import datetime
from functools import lru_cache

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


@lru_cache(maxsize=64)
def _resolve_font_path(style: str) -> str | None:
    """Resolve a font style to a filesystem path (cached).

    Returns the path string, or None if only the PIL default is available.
    """
    # Try primary path
    path = _FONT_PATHS.get(style, _FONT_PATHS["sans"])
    if os.path.exists(path):
        return path

    # Try fallback path
    path = _FALLBACK_PATHS.get(style, _FALLBACK_PATHS["sans"])
    if os.path.exists(path):
        return path

    # Try common system font directories
    _bold = "bold" in style.lower()
    _serif = "serif" in style.lower() and "sans" not in style.lower()
    search_dirs = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/Library/Fonts"),
        "/Library/Fonts",
        "/System/Library/Fonts",
        "C:\\Windows\\Fonts",
    ]
    for font_dir in search_dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _dirs, files in os.walk(font_dir):
            for f in files:
                if not f.lower().endswith((".ttf", ".otf")):
                    continue
                fl = f.lower()
                if _bold and "bold" not in fl:
                    continue
                if not _bold and "bold" in fl:
                    continue
                if _serif and "sans" in fl:
                    continue
                full = os.path.join(root, f)
                try:
                    ImageFont.truetype(full, 12)  # validate
                    return full
                except (OSError, IOError):
                    continue
    return None


@lru_cache(maxsize=128)
def _load_font(style: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font with multi-level fallback (cached).

    Tries: Liberation → DejaVu → any system TTF → PIL default.
    Never raises — always returns a usable font object.
    """
    try:
        path = _resolve_font_path(style)
        if path:
            return ImageFont.truetype(path, size)
    except Exception:
        pass

    # Last resort: PIL default font (Pillow >=10 returns a scalable font)
    return ImageFont.load_default(size=size)


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
    """Dark bg with candlestick chart pattern and ticker tape — Wall Street trading floor vibe."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Subtle vertical gradient
    bg2 = _lerp_color(bg, accent, 0.06)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Faint price grid lines
    grid_color = _lerp_color(bg, accent, 0.08)
    spacing = h // 12
    for y in range(spacing, h, spacing):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Candlestick chart pattern in the lower half
    candle_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(candle_overlay)
    num_candles = rng.randint(18, 30)
    candle_w = max(w // (num_candles + 4), 8)
    gap = max(candle_w // 3, 2)
    base_y = int(h * 0.55)
    trend_slope = rng.uniform(-0.12, 0.12)
    green = _lerp_color(accent, (0, 200, 80), 0.6)
    red = _lerp_color(accent, (200, 50, 50), 0.5)

    for i in range(num_candles):
        x = int(w * 0.05) + i * (candle_w + gap)
        if x + candle_w > w:
            break
        trend_offset = int(trend_slope * i * 8)
        open_y = base_y + rng.randint(-60, 60) + trend_offset
        close_y = open_y + rng.randint(-50, 50)
        high_y = min(open_y, close_y) - rng.randint(10, 40)
        low_y = max(open_y, close_y) + rng.randint(10, 40)
        is_green = close_y < open_y
        color = green if is_green else red
        alpha = rng.randint(25, 50)
        mid_x = x + candle_w // 2
        # Wick
        cdraw.line([(mid_x, high_y), (mid_x, low_y)], fill=(*color, alpha), width=1)
        # Body
        body_top = min(open_y, close_y)
        body_bot = max(open_y, close_y)
        cdraw.rectangle([x, body_top, x + candle_w, body_bot], fill=(*color, alpha))

    candle_overlay = candle_overlay.filter(ImageFilter.GaussianBlur(radius=3))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), candle_overlay).convert("RGB"),
        mask=candle_overlay.split()[3],
    )

    # Ticker tape strip near the top
    ticker_y = int(h * 0.02)
    ticker_h = int(h * 0.025)
    ticker_color = _lerp_color(bg, accent, 0.10)
    draw.rectangle([0, ticker_y, w, ticker_y + ticker_h], fill=ticker_color)


def _bg_warm_editorial(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Warm bg with rising line chart and dollar sign watermarks — Wall Street editorial."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Warm vertical gradient
    bg2 = _lerp_color(bg, accent, 0.10)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t * 0.5)
        draw.line([(0, y), (w, y)], fill=c)

    # Rising stock line chart in background
    chart = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(chart)
    for line_i in range(rng.randint(2, 4)):
        points = []
        y_base = rng.randint(int(h * 0.4), int(h * 0.7))
        for xi in range(0, w + 30, 30):
            trend = -int((xi / w) * h * rng.uniform(0.08, 0.18))
            noise = rng.randint(-20, 20)
            points.append((xi, y_base + trend + noise))
        alpha = rng.randint(18, 35)
        line_color = _lerp_color(accent, (255, 200, 100), 0.3)
        if len(points) >= 2:
            cdraw.line(points, fill=(*line_color, alpha), width=rng.randint(2, 3))
            fill_points = points + [(w, h), (0, h)]
            cdraw.polygon(fill_points, fill=(*line_color, alpha // 4))
    chart = chart.filter(ImageFilter.GaussianBlur(radius=6))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), chart).convert("RGB"),
        mask=chart.split()[3],
    )

    # Faint dollar sign watermarks
    watermark = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wdraw = ImageDraw.Draw(watermark)
    try:
        dollar_font = _load_font("sans_bold", 120)
    except Exception:
        dollar_font = None
    if dollar_font:
        for _ in range(rng.randint(3, 6)):
            dx = rng.randint(-50, w)
            dy = rng.randint(0, h)
            alpha = rng.randint(8, 18)
            wdraw.text((dx, dy), "$", fill=(*accent, alpha), font=dollar_font)
    watermark = watermark.filter(ImageFilter.GaussianBlur(radius=15))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), watermark).convert("RGB"),
        mask=watermark.split()[3],
    )


def _bg_ice_blue(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Cool bg with stock ticker grid and bar chart silhouette — Bloomberg terminal vibe."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Clean diagonal gradient
    corner_color = _lerp_color(bg, accent, 0.10)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, corner_color, t * 0.6)
        draw.line([(0, y), (w, y)], fill=c)

    # Faint horizontal price grid
    grid_color = _lerp_color(bg, accent, 0.06)
    for y in range(0, h, h // 15):
        draw.line([(0, y), (w, y)], fill=grid_color, width=1)

    # Bar chart silhouette in bottom area
    bars = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bars)
    num_bars = rng.randint(20, 35)
    bar_w = max(w // (num_bars + 6), 6)
    bar_gap = max(bar_w // 4, 2)
    for i in range(num_bars):
        x = int(w * 0.05) + i * (bar_w + bar_gap)
        if x + bar_w > w:
            break
        bar_h = rng.randint(int(h * 0.03), int(h * 0.20))
        bar_y = h - int(h * 0.08) - bar_h
        alpha = rng.randint(15, 30)
        bar_color = _lerp_color(accent, (100, 200, 255), rng.random() * 0.4)
        bdraw.rectangle([x, bar_y, x + bar_w, h - int(h * 0.08)], fill=(*bar_color, alpha))
    bars = bars.filter(ImageFilter.GaussianBlur(radius=4))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), bars).convert("RGB"),
        mask=bars.split()[3],
    )


def _bg_bold_red(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Dark bg with downtrend arrow and warning-style diagonal stripes — market crash vibe."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Dark gradient from top
    bg2 = _lerp_color(bg, accent, 0.08)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t * 0.4)
        draw.line([(0, y), (w, y)], fill=c)

    # Downtrend line chart
    chart = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(chart)
    for line_i in range(rng.randint(2, 3)):
        points = []
        y_base = rng.randint(int(h * 0.25), int(h * 0.4))
        for xi in range(0, w + 30, 30):
            trend = int((xi / w) * h * rng.uniform(0.08, 0.18))
            noise = rng.randint(-25, 25)
            points.append((xi, y_base + trend + noise))
        alpha = rng.randint(20, 40)
        if len(points) >= 2:
            cdraw.line(points, fill=(*accent, alpha), width=2)
            fill_points = points + [(w, h), (0, h)]
            cdraw.polygon(fill_points, fill=(*accent, alpha // 5))
    chart = chart.filter(ImageFilter.GaussianBlur(radius=5))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), chart).convert("RGB"),
        mask=chart.split()[3],
    )

    # Subtle diagonal warning stripes in corner
    stripes = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(stripes)
    stripe_w = 20
    for i in range(-h, w, stripe_w * 3):
        alpha = rng.randint(6, 14)
        sdraw.line(
            [(i, h), (i + h, 0)],
            fill=(*accent, alpha), width=stripe_w,
        )
    stripes = stripes.filter(ImageFilter.GaussianBlur(radius=8))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), stripes).convert("RGB"),
        mask=stripes.split()[3],
    )


def _bg_green_money(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Dark bg with bullish trend lines, volume bars, and percentage watermarks — bull market vibe."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Subtle vertical gradient
    bg2 = _lerp_color(bg, accent, 0.06)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg2, bg, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Bullish trend line chart
    chart = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(chart)
    for line_i in range(rng.randint(2, 4)):
        points = []
        y_base = rng.randint(int(h * 0.4), int(h * 0.7))
        for xi in range(0, w + 40, 40):
            trend = -int((xi / w) * h * rng.uniform(0.06, 0.16))
            noise = rng.randint(-20, 20)
            points.append((xi, y_base + trend + noise))
        alpha = rng.randint(18, 38)
        if len(points) >= 2:
            cdraw.line(points, fill=(*accent, alpha), width=rng.randint(2, 3))
            fill_points = points + [(w, h), (0, h)]
            cdraw.polygon(fill_points, fill=(*accent, alpha // 4))
    chart = chart.filter(ImageFilter.GaussianBlur(radius=6))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), chart).convert("RGB"),
        mask=chart.split()[3],
    )

    # Volume bars along bottom
    vol = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    vdraw = ImageDraw.Draw(vol)
    num_bars = rng.randint(25, 40)
    bar_w = max(w // (num_bars + 4), 4)
    bar_gap = max(bar_w // 5, 1)
    for i in range(num_bars):
        x = int(w * 0.03) + i * (bar_w + bar_gap)
        if x + bar_w > w:
            break
        bar_h = rng.randint(int(h * 0.01), int(h * 0.08))
        alpha = rng.randint(12, 28)
        vdraw.rectangle(
            [x, h - int(h * 0.05) - bar_h, x + bar_w, h - int(h * 0.05)],
            fill=(*accent, alpha),
        )
    vol = vol.filter(ImageFilter.GaussianBlur(radius=3))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), vol).convert("RGB"),
        mask=vol.split()[3],
    )

    # Faint percentage watermarks
    watermark = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    wdraw = ImageDraw.Draw(watermark)
    try:
        pct_font = _load_font("sans_bold", 80)
    except Exception:
        pct_font = None
    if pct_font:
        symbols = ["+2.4%", "+5.1%", "+0.8%", "+3.7%"]
        for _ in range(rng.randint(2, 4)):
            dx = rng.randint(-30, w - 100)
            dy = rng.randint(0, h)
            alpha = rng.randint(6, 14)
            wdraw.text((dx, dy), rng.choice(symbols), fill=(*accent, alpha), font=pct_font)
    watermark = watermark.filter(ImageFilter.GaussianBlur(radius=12))
    img.paste(
        Image.alpha_composite(Image.new("RGBA", (w, h), (0, 0, 0, 0)), watermark).convert("RGB"),
        mask=watermark.split()[3],
    )


def _bg_soft_minimal(img: Image.Image, bg: tuple, accent: tuple, seed: int):
    """Clean light background with thin horizontal rules — professional finance report style."""
    w, h = img.size
    draw = ImageDraw.Draw(img)
    rng = random.Random(seed)

    # Clean vertical gradient (light -> slightly warmer)
    bg2 = _lerp_color(bg, (240, 235, 225), 0.15)
    for y in range(h):
        t = y / h
        c = _lerp_color(bg, bg2, t)
        draw.line([(0, y), (w, y)], fill=c)

    # Thin horizontal rules (like a financial report)
    rule_color = _lerp_color(bg, accent, 0.08)
    margin = int(w * 0.12)
    spacing = h // 18
    for y in range(spacing * 2, h - spacing, spacing):
        draw.line([(margin, y), (w - margin, y)], fill=rule_color, width=1)

    # Subtle corner accent (top-right and bottom-left)
    accent_muted = _lerp_color(bg, accent, 0.12)
    length = int(min(w, h) * 0.06)
    thickness = 2
    m = int(w * 0.08)
    # Top-right
    draw.line([(w - m - length, m), (w - m, m)], fill=accent_muted, width=thickness)
    draw.line([(w - m, m), (w - m, m + length)], fill=accent_muted, width=thickness)
    # Bottom-left
    draw.line([(m, h - m), (m + length, h - m)], fill=accent_muted, width=thickness)
    draw.line([(m, h - m - length), (m, h - m)], fill=accent_muted, width=thickness)


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


# ── Number highlighting helper ─────────────────────────────────────────────

# Regex matching numbers, percentages, dollar amounts, and tickers like $AAPL
_NUMBER_RE = re.compile(
    r'(?:\$[\d,.]+[BMKTbmkt]?%?'      # $123, $1.2B, $45K
    r'|[\d,.]+%'                        # 42%, 1,200%
    r'|\d[\d,.]*[xXBMKTbmkt])'         # 10x, 1.5B, 200M
)


def _draw_text_with_number_highlight(
    draw: ImageDraw.Draw,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    accent: tuple,
):
    """Draw text, rendering numbers/percentages in accent color for emphasis."""
    x, y = xy
    last_end = 0
    for m in _NUMBER_RE.finditer(text):
        # Draw text before the number in normal color
        prefix = text[last_end:m.start()]
        if prefix:
            draw.text((x, y), prefix, fill=fill, font=font)
            bbox = font.getbbox(prefix)
            x += bbox[2] - bbox[0]
        # Draw the number in accent color
        num_text = m.group()
        draw.text((x, y), num_text, fill=accent, font=font)
        bbox = font.getbbox(num_text)
        x += bbox[2] - bbox[0]
        last_end = m.end()
    # Draw remaining text
    suffix = text[last_end:]
    if suffix:
        draw.text((x, y), suffix, fill=fill, font=font)


def _draw_text_with_number_highlight_shadow(
    img: Image.Image,
    xy: tuple,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    accent: tuple,
    shadow_color: tuple = (0, 0, 0),
    shadow_offset: int = 2,
):
    """Draw text with shadow, rendering numbers in accent color."""
    x, y = xy
    last_end = 0
    for m in _NUMBER_RE.finditer(text):
        prefix = text[last_end:m.start()]
        if prefix:
            _draw_text_with_shadow(img, (x, y), prefix, font, fill, shadow_color, shadow_offset)
            bbox = font.getbbox(prefix)
            x += bbox[2] - bbox[0]
        num_text = m.group()
        _draw_text_with_shadow(img, (x, y), num_text, font, accent, shadow_color, shadow_offset)
        bbox = font.getbbox(num_text)
        x += bbox[2] - bbox[0]
        last_end = m.end()
    suffix = text[last_end:]
    if suffix:
        _draw_text_with_shadow(img, (x, y), suffix, font, fill, shadow_color, shadow_offset)


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
            "accent": "#00CC88",
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

    bg = _hex_to_tuple(colors.get("background", "#0D0D15"))
    title_c = _hex_to_tuple(colors.get("title", "#FFFFFF"))
    body_c = _hex_to_tuple(colors.get("body", "#C0C0D0"))
    accent_c = _hex_to_tuple(colors.get("accent", "#F7B731"))
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

        is_hook = idx == 0

        # -- Left accent bar --
        draw.rectangle([0, 0, accent_bar_w, img_h], fill=accent_c)

        # -- Slide counter top-right (above title zone) --
        counter_font = _load_font("sans", 24)
        counter_text = f"{idx + 1}/{total}"
        counter_bbox = counter_font.getbbox(counter_text)
        counter_w = counter_bbox[2] - counter_bbox[0]
        counter_x = img_w - margin - counter_w
        counter_y = int(img_h * 0.025)
        # Subtle muted counter — doesn't compete with title
        draw.text(
            (counter_x, counter_y),
            counter_text,
            fill=muted_c,
            font=counter_font,
        )

        # -- Title --
        title_text = slide_data.get("title", "")
        # Hook slide gets a larger title for scroll-stopping impact
        base_title_size = 100 if is_hook else 90
        title_size = _calc_title_font_size(title_text, content_w, base_size=base_title_size)
        title_font = _load_font("sans_bold", title_size)
        title_lines = _wrap_text(title_text, title_font, content_w)

        title_y = int(img_h * 0.14) if is_hook else int(img_h * 0.165)
        title_line_h = int(title_size * 1.35)

        # Hook slide: accent glow behind title for pattern interrupt
        if is_hook:
            glow = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow)
            glow_y = title_y
            for line in title_lines:
                glow_draw.text(
                    (margin, glow_y), line,
                    fill=(*accent_c, 50), font=title_font,
                )
                glow_y += title_line_h
            glow = glow.filter(ImageFilter.GaussianBlur(radius=18))
            img.paste(
                Image.alpha_composite(
                    Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0)), glow
                ).convert("RGB"),
                mask=glow.split()[3],
            )
            draw = ImageDraw.Draw(img)

        for line in title_lines:
            draw.text((margin, title_y), line, fill=title_c, font=title_font)
            title_y += title_line_h

        # -- Divider --
        divider_y = title_y + int(img_h * 0.015)
        draw.rectangle(
            [margin, divider_y, img_w - margin, divider_y + 2],
            fill=muted_c,
        )

        # -- Body (with number highlighting) --
        body_text = slide_data.get("body", "")
        body_font = _load_font("serif", 64)
        body_lines = _wrap_text(body_text, body_font, content_w)
        body_y = divider_y + int(img_h * 0.02)
        body_line_h = int(64 * 1.45)
        for line in body_lines:
            _draw_text_with_number_highlight(
                draw, (margin, body_y), line, body_font, body_c, accent_c,
            )
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

    is_hook = idx == 0
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
    # Hook slide gets a larger title for scroll-stopping impact
    base_title_size = 100 if is_hook else 90
    title_size = _calc_title_font_size(title_text, content_w, base_size=base_title_size)
    title_font = _load_font(title_font_style, title_size)
    title_lines = _wrap_text(title_text, title_font, content_w)

    title_y = int(img_h * 0.14) if is_hook else int(img_h * 0.165)
    title_line_h = int(title_size * 1.35)

    # Hook slide: accent glow behind title for pattern interrupt
    if is_hook:
        glow = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow)
        glow_y = title_y
        for line in title_lines:
            glow_draw.text(
                (margin, glow_y), line,
                fill=(*accent_c, 50), font=title_font,
            )
            glow_y += title_line_h
        glow = glow.filter(ImageFilter.GaussianBlur(radius=18))
        img.paste(
            Image.alpha_composite(
                Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0)), glow
            ).convert("RGB"),
            mask=glow.split()[3],
        )

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

    # -- Body (with number highlighting) --
    body_text = slide_data.get("body", "")
    body_font_style = preset.get("body_font", "serif")
    body_size = preset.get("body_size", 64)
    body_font = _load_font(body_font_style, body_size)
    body_lines = _wrap_text(body_text, body_font, content_w)
    body_y = divider_y + int(img_h * 0.025)
    body_line_h = int(body_size * 1.45)
    for line in body_lines:
        _draw_text_with_number_highlight_shadow(
            img, (margin, body_y), line, body_font, body_c, accent_c, shadow_c, 2,
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
