"""Build PPTX slide decks from structured content."""

import os
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR


def _hex_to_rgb(hex_color: str) -> RGBColor:
    """Convert a hex color string to an RGBColor."""
    h = hex_color.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def build_pptx(
    slides: list[dict],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
) -> str:
    """Build a PPTX file from slide content.

    Args:
        slides: List of slide dicts with 'title', 'body', 'footer'.
        colors: Dict with 'background', 'title', 'body', 'accent', 'highlight'.
        aspect_ratio: Either '9:16' (vertical/stories) or '16:9' (landscape).
        output_dir: Directory to save the output file.

    Returns:
        Path to the generated PPTX file.
    """
    prs = Presentation()

    # Set slide dimensions based on aspect ratio
    if aspect_ratio == "9:16":
        prs.slide_width = Inches(7.5)
        prs.slide_height = Inches(13.33)
    else:
        prs.slide_width = Inches(13.33)
        prs.slide_height = Inches(7.5)

    bg_color = _hex_to_rgb(colors.get("background", "#0D1117"))
    title_color = _hex_to_rgb(colors.get("title", "#FFFFFF"))
    body_color = _hex_to_rgb(colors.get("body", "#C9D1D9"))
    accent_color = _hex_to_rgb(colors.get("accent", "#58A6FF"))
    highlight_color = _hex_to_rgb(colors.get("highlight", "#F0883E"))

    slide_w = prs.slide_width
    slide_h = prs.slide_height
    margin = Inches(0.8)

    # Left accent bar width (consultant-style vertical bar)
    accent_bar_w = Inches(0.12)

    for idx, slide_data in enumerate(slides):
        slide_layout = prs.slide_layouts[6]  # Blank layout
        slide = prs.slides.add_slide(slide_layout)

        # -- Background --
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = bg_color

        # -- Left accent bar (consultant-style vertical stripe) --
        bar_color = accent_color if idx % 2 == 0 else highlight_color
        left_bar = slide.shapes.add_shape(
            1,  # Rectangle
            0,
            0,
            accent_bar_w,
            slide_h,
        )
        left_bar.fill.solid()
        left_bar.fill.fore_color.rgb = bar_color
        left_bar.line.fill.background()

        # -- Slide number (top-right, clean consultant format) --
        indicator_w = Inches(1.5)
        indicator_box = slide.shapes.add_textbox(
            slide_w - margin - indicator_w,
            Inches(0.5),
            indicator_w,
            Inches(0.4),
        )
        indicator_tf = indicator_box.text_frame
        indicator_tf.word_wrap = True
        indicator_p = indicator_tf.paragraphs[0]
        indicator_p.text = f"{idx + 1} / {len(slides)}"
        indicator_p.font.size = Pt(18)
        indicator_p.font.name = "Calibri"
        indicator_p.font.color.rgb = accent_color
        indicator_p.font.bold = False
        indicator_p.alignment = PP_ALIGN.RIGHT

        # -- Thin horizontal rule under slide number --
        rule_top = Inches(1.05)
        rule = slide.shapes.add_shape(
            1,  # Rectangle
            margin,
            rule_top,
            slide_w - 2 * margin,
            Inches(0.02),
        )
        rule.fill.solid()
        rule.fill.fore_color.rgb = accent_color
        rule.line.fill.background()

        # -- Title: Calibri Bold 45pt --
        title_top = Inches(1.4)
        title_height = Inches(3.5)
        title_box = slide.shapes.add_textbox(
            margin, title_top, slide_w - 2 * margin, title_height
        )
        title_tf = title_box.text_frame
        title_tf.word_wrap = True
        title_tf.auto_size = None
        title_p = title_tf.paragraphs[0]
        title_p.text = slide_data.get("title", "")
        title_p.font.size = Pt(45)
        title_p.font.name = "Calibri"
        title_p.font.bold = True
        title_p.font.color.rgb = title_color
        title_p.alignment = PP_ALIGN.LEFT
        title_p.space_after = Pt(12)

        # -- Body: Times New Roman 35pt --
        body_top = title_top + title_height + Inches(0.5)
        body_height = Inches(4.0)
        body_box = slide.shapes.add_textbox(
            margin, body_top, slide_w - 2 * margin, body_height
        )
        body_tf = body_box.text_frame
        body_tf.word_wrap = True
        body_tf.auto_size = None
        body_p = body_tf.paragraphs[0]
        body_p.text = slide_data.get("body", "")
        body_p.font.size = Pt(35)
        body_p.font.name = "Times New Roman"
        body_p.font.color.rgb = body_color
        body_p.alignment = PP_ALIGN.LEFT
        body_p.space_after = Pt(16)
        body_p.line_spacing = Pt(44)

        # -- Footer: Calibri Italic 22pt, anchored at bottom --
        footer_top = slide_h - Inches(1.2)
        footer_box = slide.shapes.add_textbox(
            margin, footer_top, slide_w - 2 * margin, Inches(0.6)
        )
        footer_tf = footer_box.text_frame
        footer_tf.word_wrap = True
        footer_p = footer_tf.paragraphs[0]
        footer_p.text = slide_data.get("footer", "")
        footer_p.font.size = Pt(22)
        footer_p.font.name = "Calibri"
        footer_p.font.color.rgb = accent_color
        footer_p.font.italic = True
        footer_p.alignment = PP_ALIGN.LEFT

        # -- Bottom accent line --
        bottom_rule = slide.shapes.add_shape(
            1,
            0,
            slide_h - Inches(0.08),
            slide_w,
            Inches(0.08),
        )
        bottom_rule.fill.solid()
        bottom_rule.fill.fore_color.rgb = bar_color
        bottom_rule.line.fill.background()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"finance_slides_{timestamp}.pptx"
    filepath = os.path.join(output_dir, filename)
    prs.save(filepath)

    return filepath
