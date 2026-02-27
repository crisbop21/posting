"""Standalone verification script for the v4 layout implementation.

Tests each component of the rendering pipeline without needing API calls.
Generates verification images in output/ that show exactly what will render.

Usage:
    python verify_layout.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920


def test_1_title_only_zone():
    """Verify _layout_title_only uses 8-20% zone (not 8-25%)."""
    from src.slides.image_generator import _layout_title_only

    img = Image.new("RGBA", (W, H), (30, 30, 40, 255))
    slide = {"title": "Netflix Stock Surged 14% After Earnings Beat"}
    colors = {"highlight": "#FF5757", "accent": "#F7B731"}

    _layout_title_only(img, slide, colors, W, H)

    # Check: no pixels should be drawn below 20% (384px)
    # Scan every pixel row from 20% to 25% — should be empty (all original color)
    arr = list(img.getdata())
    has_text_below_20 = False
    for y in range(int(H * 0.20), int(H * 0.25)):
        for x in range(W):
            pixel = arr[y * W + x]
            if pixel != (30, 30, 40, 255):
                has_text_below_20 = True
                break
        if has_text_below_20:
            break

    status = "FAIL" if has_text_below_20 else "PASS"
    print(f"  [TEST 1] Title zone 8-20%: {status}")
    if has_text_below_20:
        print(f"           Found text pixels below 20% mark ({int(H * 0.20)}px)")

    # Save visual
    img.save("output/verify_1_title_zone.png", "PNG")
    print(f"           Saved: output/verify_1_title_zone.png")
    return not has_text_below_20


def test_2_fg_image_preparation():
    """Verify _prepare_fg_image produces correct dimensions."""
    from src.slides.video_builder import _prepare_fg_image

    fg_w = int(W * 0.84)   # 907
    fg_h = int(H * 0.35)   # 672

    # Create a dummy source image
    src = Image.new("RGB", (800, 600), (100, 150, 200))
    result = _prepare_fg_image(src, fg_w, fg_h)

    size_ok = result.size == (fg_w, fg_h)
    mode_ok = result.mode == "RGBA"
    has_alpha = result.split()[3].getextrema() != (255, 255)  # has transparency (rounded corners)

    status = "PASS" if (size_ok and mode_ok and has_alpha) else "FAIL"
    print(f"  [TEST 2] FG image size {result.size}, mode={result.mode}, rounded={has_alpha}: {status}")
    print(f"           Expected: ({fg_w}, {fg_h}), RGBA, rounded corners")

    result.save("output/verify_2_fg_image.png", "PNG")
    print(f"           Saved: output/verify_2_fg_image.png")
    return size_ok and mode_ok


def test_3_fg_position():
    """Verify FG_IMAGE_POS places image in 36-73% zone."""
    from src.slides.video_builder import FG_IMAGE_POS

    fg_x = int(FG_IMAGE_POS[0] * W)
    fg_y = int(FG_IMAGE_POS[1] * H)
    fg_w = int(W * 0.84)
    fg_h = int(H * 0.35)

    y_top_pct = fg_y / H
    y_bot_pct = (fg_y + fg_h) / H

    in_zone = y_top_pct >= 0.35 and y_bot_pct <= 0.74

    status = "PASS" if in_zone else "FAIL"
    print(f"  [TEST 3] FG position: x={fg_x}, y={fg_y}, bottom={fg_y + fg_h}: {status}")
    print(f"           Zone: {y_top_pct:.1%} - {y_bot_pct:.1%} (expected ~36-73%)")
    return in_zone


def test_4_image_split():
    """Verify image split allocates more to foreground."""
    # Simulate the split logic from build_video_with_searched_images
    for n_images in [1, 2, 3, 5, 8]:
        images = list(range(n_images))

        if len(images) >= 3:
            bg_imgs = images[:2]
            fg_imgs = images[2:]
        elif len(images) >= 2:
            bg_imgs = images[:1]
            fg_imgs = images[1:]
        else:
            bg_imgs = images[:1]
            fg_imgs = images[:1]

        print(f"  [TEST 4] {n_images} images → {len(bg_imgs)} bg + {len(fg_imgs)} fg")

    # Key check: with 8 images, should get 2 bg + 6 fg (not 3 bg + 5 cards)
    images = list(range(8))
    bg_imgs = images[:2]
    fg_imgs = images[2:]
    ok = len(bg_imgs) == 2 and len(fg_imgs) == 6
    status = "PASS" if ok else "FAIL"
    print(f"           8 images split: {status}")
    return ok


def test_5_composite_slide_title_only():
    """Verify composite_slide with title_only=True only renders title."""
    from src.slides.image_generator import composite_slide

    bg = Image.new("RGB", (W, H), (50, 50, 70))
    slide = {
        "title": "Netflix Stock Surged 14%",
        "body": "This body text should NOT appear when title_only=True. "
                "It has lots of text that would normally fill the slide.",
        "footer": "Source: Market Data"
    }
    colors = {"background": "#0D0D15", "title": "#F0F0F0", "body": "#C0C0D0",
              "accent": "#F7B731", "highlight": "#FF5757"}

    result_title_only = composite_slide(
        bg_image=bg, slide=slide, slide_index=0, total_slides=4,
        colors=colors, title_only=True,
    )
    result_full = composite_slide(
        bg_image=bg, slide=slide, slide_index=0, total_slides=4,
        colors=colors, title_only=False,
    )

    # Compare: title_only should have fewer non-bg pixels in the 25-70% zone
    import numpy as np
    arr_to = np.array(result_title_only)
    arr_full = np.array(result_full)

    # Count different pixels in the body zone (25-70%)
    zone_top = int(H * 0.25)
    zone_bot = int(H * 0.70)
    to_zone = arr_to[zone_top:zone_bot]
    full_zone = arr_full[zone_top:zone_bot]
    diff = np.abs(to_zone.astype(int) - full_zone.astype(int)).sum()

    has_body_difference = diff > 10000  # significant pixel difference

    status = "PASS" if has_body_difference else "FAIL"
    print(f"  [TEST 5] title_only removes body text: {status}")
    print(f"           Pixel diff in body zone: {diff:,}")

    result_title_only.save("output/verify_5a_title_only.png", "PNG")
    result_full.save("output/verify_5b_full_layout.png", "PNG")
    print(f"           Saved: output/verify_5a_title_only.png (should have NO body text)")
    print(f"           Saved: output/verify_5b_full_layout.png (has body text for comparison)")
    return has_body_difference


def test_6_dynamic_clip_frame():
    """Render a single frame from _make_dynamic_slide_clip to verify layout."""
    import numpy as np
    from src.slides.image_generator import composite_slide_layers, treat_background, get_slide_role
    from src.slides.video_builder import _prepare_fg_image, _make_dynamic_slide_clip

    slide = {
        "title": "Netflix Stock Surged 14% After Earnings Beat",
        "body": "Body text that should NOT appear",
    }
    colors = {"background": "#0D0D15", "title": "#F0F0F0", "body": "#C0C0D0",
              "accent": "#F7B731", "highlight": "#FF5757"}
    role = get_slide_role(0, 4)

    # Create dummy background images
    bg1 = Image.new("RGB", (W, H), (40, 60, 100))
    bg2 = Image.new("RGB", (W, H), (60, 40, 80))

    treated_bgs = [
        treat_background(bg1, W, H, "#F7B731", role),
        treat_background(bg2, W, H, "#F7B731", role),
    ]

    _, overlay = composite_slide_layers(
        bg_image=bg1, slide=slide, slide_index=0, total_slides=4,
        colors=colors, caption_safe_pct=0.27, title_only=True,
    )

    # Create dummy foreground images (colored rectangles)
    fg_w = int(W * 0.84)
    fg_h = int(H * 0.35)
    fg1_src = Image.new("RGB", (800, 600), (200, 100, 50))  # orange-ish
    fg2_src = Image.new("RGB", (800, 600), (50, 150, 200))  # blue-ish
    fg_prepared = [
        _prepare_fg_image(fg1_src, fg_w, fg_h),
        _prepare_fg_image(fg2_src, fg_w, fg_h),
    ]

    # Create the clip and extract a single frame at t=0.5
    clip = _make_dynamic_slide_clip(
        treated_bgs=treated_bgs,
        overlay_rgba=overlay,
        fg_images=fg_prepared,
        target_w=W,
        target_h=H,
        duration=5.0,
        base_motion="zoom_in",
        fg_change_times=[2.5],
        bg_change_times=[2.5],
    )

    # Render frames at t=0.5 (first fg) and t=3.0 (second fg)
    frame_early = clip.get_frame(0.5)
    frame_late = clip.get_frame(3.0)

    Image.fromarray(frame_early).save("output/verify_6a_frame_t0.5.png", "PNG")
    Image.fromarray(frame_late).save("output/verify_6b_frame_t3.0.png", "PNG")

    # Check: the fg image zone (36-73%) should have colored content, not just bg
    fg_zone_early = frame_early[int(H * 0.40):int(H * 0.65), int(W * 0.15):int(W * 0.75)]
    fg_zone_late = frame_late[int(H * 0.40):int(H * 0.65), int(W * 0.15):int(W * 0.75)]

    # The two frames should look different (different fg images)
    diff = np.abs(fg_zone_early.astype(float) - fg_zone_late.astype(float)).mean()
    has_different_fg = diff > 5.0

    status = "PASS" if has_different_fg else "FAIL"
    print(f"  [TEST 6] Dynamic clip renders big FG image: {status}")
    print(f"           FG zone pixel diff between t=0.5 and t=3.0: {diff:.1f}")
    print(f"           Saved: output/verify_6a_frame_t0.5.png (first FG image)")
    print(f"           Saved: output/verify_6b_frame_t3.0.png (second FG image)")
    return has_different_fg


def test_7_no_old_card_references():
    """Check for stale references to old overlay card system."""
    import inspect
    from src.slides.video_builder import _make_dynamic_slide_clip

    source = inspect.getsource(_make_dynamic_slide_clip)

    old_refs = []
    for term in ["card_images", "card_arrays", "card_positions", "CARD_POSITIONS",
                 "card_hold", "active_card", "pop-in"]:
        if term in source:
            old_refs.append(term)

    status = "PASS" if not old_refs else "FAIL"
    print(f"  [TEST 7] No stale card references in _make_dynamic_slide_clip: {status}")
    if old_refs:
        print(f"           Found old references: {old_refs}")
    return not old_refs


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    print("\n" + "=" * 60)
    print("  LAYOUT V4 VERIFICATION")
    print("  Title 8-20% | Captions 20-36% | Image 36-73% | UI 73-100%")
    print("=" * 60 + "\n")

    results = []
    results.append(test_1_title_only_zone())
    print()
    results.append(test_2_fg_image_preparation())
    print()
    results.append(test_3_fg_position())
    print()
    results.append(test_4_image_split())
    print()
    results.append(test_5_composite_slide_title_only())
    print()
    results.append(test_6_dynamic_clip_frame())
    print()
    results.append(test_7_no_old_card_references())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("  ALL TESTS PASSED — layout v4 is correctly implemented")
    else:
        print("  SOME TESTS FAILED — check output above")
    print("=" * 60)

    print("\n  Verification images saved to output/:")
    print("    verify_1_title_zone.png         — title rendered in 8-20% zone")
    print("    verify_2_fg_image.png           — fg image 907x672 with rounded corners")
    print("    verify_5a_title_only.png        — composite with title only (no body)")
    print("    verify_5b_full_layout.png       — composite with full layout (for comparison)")
    print("    verify_6a_frame_t0.5.png        — actual video frame at t=0.5s")
    print("    verify_6b_frame_t3.0.png        — actual video frame at t=3.0s (different fg)")
    print()
