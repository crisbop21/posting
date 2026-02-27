"""Build narrated video from slides using ElevenLabs TTS and MoviePy.

Pipeline:
  1. Generate voiceover script per slide (via Claude)
  2. Synthesize audio per slide (via ElevenLabs)
  3. Render PNG slides as video frames with audio overlay
  4. Combine into a single MP4 with crossfade transitions

Dynamic slide mode (web images):
  - Rotating backgrounds: multiple web images crossfade under Ken Burns motion
  - Overlay cards: cropped images pop in every ~2s as picture-in-picture
  - Sound effects: subtle whoosh/pop synced to visual changes
  - Caption safe zone: bottom 15% kept clear for external subtitles
"""

import io
import math
import os
import tempfile
import wave
from pathlib import Path

import requests

from src.slides.png_builder import build_pngs


# ── ElevenLabs TTS ────────────────────────────────────────────────────────────


def _get_elevenlabs_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ELEVENLABS_API_KEY environment variable is not set. "
            "Get your key at https://elevenlabs.io/app/settings/api-keys"
        )
    return key


def synthesize_speech(
    text: str,
    voice_id: str = "pNInz6obpgDQGcFmaJgB",  # "Adam" – deep, clear male voice
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> bytes:
    """Convert text to speech using ElevenLabs API.

    Returns raw MP3 bytes.
    """
    api_key = _get_elevenlabs_key()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.content


def list_voices() -> list[dict]:
    """Fetch available ElevenLabs voices. Useful for letting the user pick."""
    api_key = _get_elevenlabs_key()
    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    voices = resp.json().get("voices", [])
    return [
        {
            "voice_id": v["voice_id"],
            "name": v["name"],
            "category": v.get("category", ""),
            "description": v.get("labels", {}).get("description", ""),
        }
        for v in voices
    ]


# ── Video Assembly ────────────────────────────────────────────────────────────


def _ensure_moviepy():
    """Import moviepy lazily so the rest of the module works without it."""
    try:
        import moviepy
        return moviepy
    except ImportError:
        raise ImportError(
            "moviepy is required for video generation. "
            "Install it with: pip install moviepy"
        )


# ── Ken Burns Motion ─────────────────────────────────────────────────────────

# Role → motion type mapping.
# Each slide role gets a distinct camera movement to match its emotional beat.
ROLE_MOTION = {
    "hook": "zoom_in",       # Draws viewer in, creates urgency
    "context": "pan_right",  # Exploration feel, reveals information
    "payoff": "zoom_out",    # Pull-back reveal moment
    "cta": "drift",          # Very subtle, keeps focus on text
}

# Motions cycled across background segments within a single slide.
SEGMENT_MOTIONS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]


def _kenburns_crop(bg_array, src_w, src_h, target_w, target_h, progress, motion):
    """Compute a single Ken Burns cropped frame from an oversized background.

    Args:
        progress: 0.0 → 1.0 through the segment, already eased.
    Returns:
        numpy array (target_h, target_w, 3), float32.
    """
    import numpy as np
    from PIL import Image

    p = progress

    if motion == "zoom_in":
        crop_w = int(src_w - (src_w - target_w) * p)
        crop_h = int(src_h - (src_h - target_h) * p)
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2
    elif motion == "zoom_out":
        crop_w = int(target_w + (src_w - target_w) * p)
        crop_h = int(target_h + (src_h - target_h) * p)
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2
    elif motion == "pan_right":
        crop_w = target_w
        crop_h = target_h
        x = int((src_w - crop_w) * p)
        y = (src_h - crop_h) // 2
    elif motion == "pan_left":
        crop_w = target_w
        crop_h = target_h
        x = int((src_w - crop_w) * (1 - p))
        y = (src_h - crop_h) // 2
    else:  # drift
        dp = p * 0.3
        crop_w = int(src_w - (src_w - target_w) * dp)
        crop_h = int(src_h - (src_h - target_h) * dp)
        x = (src_w - crop_w) // 2
        y = (src_h - crop_h) // 2

    x = max(0, min(x, src_w - crop_w))
    y = max(0, min(y, src_h - crop_h))
    crop_w = min(crop_w, src_w - x)
    crop_h = min(crop_h, src_h - y)

    cropped = bg_array[y:y + crop_h, x:x + crop_w]
    return np.array(
        Image.fromarray(cropped).resize(
            (target_w, target_h), Image.Resampling.BILINEAR,
        )
    ).astype(np.float32)


def _make_kenburns_clip(
    treated_bg,
    overlay_rgba,
    target_w: int,
    target_h: int,
    duration: float,
    motion: str = "zoom_in",
):
    """Create an animated video clip with Ken Burns motion on the background.

    Simple single-background version. For multi-image dynamic slides, use
    _make_dynamic_slide_clip instead.
    """
    import numpy as np
    from moviepy import VideoClip
    from PIL import Image

    bg_array = np.array(treated_bg.convert("RGB"))
    src_h, src_w = bg_array.shape[:2]

    ov_array = np.array(overlay_rgba.convert("RGBA"))
    alpha = ov_array[:, :, 3:4].astype(np.float32) / 255.0
    ov_rgb = ov_array[:, :, :3].astype(np.float32)

    def make_frame(t):
        p = t / duration if duration > 0 else 0
        p = 0.5 - 0.5 * math.cos(math.pi * p)
        bg_frame = _kenburns_crop(
            bg_array, src_w, src_h, target_w, target_h, p, motion,
        )
        result = bg_frame * (1.0 - alpha) + ov_rgb * alpha
        return result.astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(24)


# ── Sound Effects ────────────────────────────────────────────────────────────


def _write_wav(path, signal, sample_rate=44100):
    """Write a float32 numpy signal (values in -1..1) to a 16-bit WAV."""
    import numpy as np

    signal = np.clip(signal, -1.0, 1.0)
    pcm = (signal * 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _generate_pop_sfx(path, sample_rate=44100):
    """Generate a subtle 'pop' sound effect — short sine burst with decay."""
    import numpy as np

    duration = 0.12
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)
    signal = np.sin(2 * np.pi * 900 * t) * np.exp(-t * 35) * 0.18
    _write_wav(path, signal, sample_rate)


def _generate_whoosh_sfx(path, sample_rate=44100):
    """Generate a subtle 'whoosh' — filtered noise with bell envelope."""
    import numpy as np

    duration = 0.25
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)
    noise = np.random.RandomState(42).randn(n) * 0.12
    envelope = np.sin(np.pi * t / duration)
    signal = noise * envelope
    _write_wav(path, signal, sample_rate)


# ── Overlay Card Preparation ────────────────────────────────────────────────

# Positions for overlay cards (x_frac, y_frac) — top-left corner.
# Placed to avoid typical title (top 10-25%) and caption zone (bottom 15%).
CARD_POSITIONS = [
    (0.60, 0.28),   # Right, upper
    (0.04, 0.42),   # Left, middle
    (0.56, 0.54),   # Right, lower-middle
    (0.06, 0.34),   # Left, upper-middle
]


def _prepare_overlay_card(img, card_w, card_h, corner_radius=18, border=3):
    """Crop and style a web image as a rounded overlay card with border.

    Returns an RGBA PIL Image ready for compositing.
    """
    from PIL import Image, ImageDraw

    # Center-crop to card aspect ratio
    src_w, src_h = img.size
    target_ratio = card_w / card_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        x = (src_w - new_w) // 2
        img = img.crop((x, 0, x + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        y = (src_h - new_h) // 2
        img = img.crop((0, y, src_w, y + new_h))

    img = img.resize((card_w, card_h), Image.Resampling.LANCZOS).convert("RGBA")

    # Rounded-corner mask for the photo
    mask = Image.new("L", (card_w, card_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, card_w - 1, card_h - 1], radius=corner_radius, fill=255,
    )

    # Card with white border
    total_w = card_w + 2 * border
    total_h = card_h + 2 * border
    card = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 0))

    # Border mask (rounded, slightly larger)
    border_mask = Image.new("L", (total_w, total_h), 0)
    ImageDraw.Draw(border_mask).rounded_rectangle(
        [0, 0, total_w - 1, total_h - 1],
        radius=corner_radius + border, fill=255,
    )
    border_fill = Image.new("RGBA", (total_w, total_h), (255, 255, 255, 180))
    card = Image.composite(border_fill, card, border_mask)

    # Paste the photo inside the border
    img.putalpha(mask)
    card.paste(img, (border, border), img)

    return card


# ── Dynamic Slide Clip ───────────────────────────────────────────────────────


def _make_dynamic_slide_clip(
    treated_bgs,
    overlay_rgba,
    card_images,
    target_w,
    target_h,
    duration,
    base_motion="zoom_in",
    bg_interval=3.5,
    card_interval=2.0,
    bg_crossfade=0.4,
):
    """Create an animated slide clip with rotating backgrounds and overlay cards.

    This is the full-featured version of _make_kenburns_clip. It handles:
      - Multiple background images rotating with crossfade and Ken Burns motion
      - Overlay card images popping in every card_interval seconds
      - All composited with the static text overlay on top

    Args:
        treated_bgs: List of oversized PIL Images (blurred + gradient).
        overlay_rgba: RGBA PIL Image at target size (text + frame + fg).
        card_images: List of RGBA PIL Images (pre-processed overlay cards).
        target_w, target_h: Output frame dimensions.
        duration: Total clip duration in seconds.
        base_motion: Primary Ken Burns motion for the first segment.
        bg_interval: Seconds between background rotations.
        card_interval: Seconds between overlay card appearances.
        bg_crossfade: Duration of crossfade between backgrounds.
    """
    import numpy as np
    from moviepy import VideoClip
    from PIL import Image

    # Pre-compute all numpy arrays
    bg_arrays = []
    bg_dims = []
    for bg in treated_bgs:
        arr = np.array(bg.convert("RGB"))
        bg_arrays.append(arr)
        h, w = arr.shape[:2]
        bg_dims.append((w, h))

    # Text overlay
    ov = np.array(overlay_rgba.convert("RGBA"))
    text_alpha = ov[:, :, 3:4].astype(np.float32) / 255.0
    text_rgb = ov[:, :, :3].astype(np.float32)

    # Overlay cards — pre-compute arrays and positions
    card_arrays = []
    card_positions_px = []
    for i, card_img in enumerate(card_images):
        card_arr = np.array(card_img.convert("RGBA"))
        card_arrays.append(card_arr)
        pos_frac = CARD_POSITIONS[i % len(CARD_POSITIONS)]
        cx = int(pos_frac[0] * target_w)
        cy = int(pos_frac[1] * target_h)
        card_positions_px.append((cx, cy))

    n_bgs = len(bg_arrays)

    # Assign a motion per bg segment
    motions = [base_motion]
    for j in range(1, n_bgs):
        motions.append(SEGMENT_MOTIONS[j % len(SEGMENT_MOTIONS)])

    def make_frame(t):
        # ── 1. BACKGROUND LAYER (rotating + Ken Burns) ──
        if n_bgs == 1:
            # Single background — simple Ken Burns
            src_w, src_h = bg_dims[0]
            p = t / duration if duration > 0 else 0
            p = 0.5 - 0.5 * math.cos(math.pi * p)
            bg_frame = _kenburns_crop(
                bg_arrays[0], src_w, src_h, target_w, target_h, p, motions[0],
            )
        else:
            seg_idx = min(int(t / bg_interval), n_bgs - 1)
            t_in_seg = t - seg_idx * bg_interval
            seg_dur = bg_interval

            # Ken Burns progress within segment
            p = t_in_seg / seg_dur if seg_dur > 0 else 0
            p = 0.5 - 0.5 * math.cos(math.pi * p)
            src_w, src_h = bg_dims[seg_idx]
            bg_frame = _kenburns_crop(
                bg_arrays[seg_idx], src_w, src_h,
                target_w, target_h, p, motions[seg_idx],
            )

            # Crossfade to next background
            next_idx = seg_idx + 1
            if next_idx < n_bgs and t_in_seg > seg_dur - bg_crossfade:
                cf = (t_in_seg - (seg_dur - bg_crossfade)) / bg_crossfade
                nsrc_w, nsrc_h = bg_dims[next_idx]
                bg_next = _kenburns_crop(
                    bg_arrays[next_idx], nsrc_w, nsrc_h,
                    target_w, target_h, 0.0, motions[next_idx],
                )
                bg_frame = bg_frame * (1.0 - cf) + bg_next * cf

        # ── 2. OVERLAY CARD LAYER (pop-in images) ──
        frame = bg_frame.copy()

        if card_arrays:
            card_idx = int(t / card_interval)
            if card_idx < len(card_arrays):
                t_card = t - card_idx * card_interval

                # Fade in / hold / fade out
                fade_in = 0.2
                fade_out = 0.2
                if t_card < fade_in:
                    fade = t_card / fade_in
                elif t_card > card_interval - fade_out:
                    fade = max(0.0, (card_interval - t_card) / fade_out)
                else:
                    fade = 1.0

                if fade > 0.01:
                    card = card_arrays[card_idx]
                    cx, cy = card_positions_px[card_idx]
                    ch, cw = card.shape[:2]

                    # Clamp to frame bounds
                    y1 = max(0, cy)
                    x1 = max(0, cx)
                    y2 = min(target_h, cy + ch)
                    x2 = min(target_w, cx + cw)
                    sy1 = y1 - cy
                    sx1 = x1 - cx
                    sy2 = sy1 + (y2 - y1)
                    sx2 = sx1 + (x2 - x1)

                    card_slice = card[sy1:sy2, sx1:sx2]
                    card_a = card_slice[:, :, 3:4].astype(np.float32) / 255.0 * fade
                    card_rgb = card_slice[:, :, :3].astype(np.float32)

                    region = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = (
                        region * (1.0 - card_a) + card_rgb * card_a
                    )

        # ── 3. TEXT OVERLAY (always on top) ──
        result = frame * (1.0 - text_alpha) + text_rgb * text_alpha
        return result.astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(24)


# ── Static Video Builders ────────────────────────────────────────────────────


def build_video(
    slides: list[dict],
    scripts: list[str],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    crossfade: float = 0.3,
    min_duration: float = 4.0,
    padding: float = 0.8,
) -> str:
    """Build a narrated MP4 video from slides and voiceover scripts.

    Args:
        slides: List of slide dicts (title, body, footer).
        scripts: List of voiceover text strings, one per slide.
        colors: Color scheme dict.
        aspect_ratio: '9:16' or '16:9'.
        output_dir: Where to save the MP4.
        handle: Social media handle for slide watermark.
        voice_id: ElevenLabs voice ID.
        crossfade: Crossfade duration in seconds between slides.
        min_duration: Minimum seconds a slide is shown.
        padding: Extra seconds of silence after each voiceover ends.

    Returns:
        Path to the generated MP4 file.
    """
    mp = _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        CompositeVideoClip,
        ImageClip,
        concatenate_videoclips,
    )

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Render slide PNGs
    png_paths = build_pngs(
        slides=slides,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
        handle=handle,
    )

    # Step 2: Synthesize audio per slide
    with tempfile.TemporaryDirectory() as tmp_dir:
        slide_clips = []

        for i, (png_path, script) in enumerate(zip(png_paths, scripts)):
            # Generate audio
            audio_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            # Load audio to determine duration
            audio_clip = AudioFileClip(audio_path)
            duration = max(audio_clip.duration + padding, min_duration)

            # Create image clip with audio
            img_clip = (
                ImageClip(png_path)
                .with_duration(duration)
                .with_audio(audio_clip)
            )

            slide_clips.append(img_clip)

        # Step 3: Concatenate with crossfade
        if crossfade > 0 and len(slide_clips) > 1:
            final = concatenate_videoclips(
                slide_clips,
                method="compose",
                padding=-crossfade,
            )
        else:
            final = concatenate_videoclips(slide_clips, method="compose")

        # Step 4: Write MP4
        output_path = os.path.join(output_dir, "narrated_slides.mp4")
        final.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
        )

        # Cleanup
        for clip in slide_clips:
            clip.close()
        final.close()

    return output_path


def build_video_with_ai_images(
    slides: list[dict],
    scripts: list[str],
    image_prompts: list[str],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    crossfade: float = 0.3,
    min_duration: float = 4.0,
    padding: float = 0.8,
) -> str:
    """Build a narrated MP4 using AI-generated background images.

    Same as build_video() but uses AI-generated images composited
    with slide text instead of plain PNG slides.
    """
    mp = _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
    )
    from src.slides.image_generator import generate_slide_images

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Generate AI background images composited with slide text
    png_paths = generate_slide_images(
        slides=slides,
        image_prompts=image_prompts,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
        handle=handle,
    )

    # Step 2: Synthesize audio per slide and assemble
    with tempfile.TemporaryDirectory() as tmp_dir:
        slide_clips = []

        for i, (png_path, script) in enumerate(zip(png_paths, scripts)):
            audio_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            audio_clip = AudioFileClip(audio_path)
            duration = max(audio_clip.duration + padding, min_duration)

            img_clip = (
                ImageClip(png_path)
                .with_duration(duration)
                .with_audio(audio_clip)
            )
            slide_clips.append(img_clip)

        if crossfade > 0 and len(slide_clips) > 1:
            final = concatenate_videoclips(
                slide_clips, method="compose", padding=-crossfade,
            )
        else:
            final = concatenate_videoclips(slide_clips, method="compose")

        output_path = os.path.join(output_dir, "narrated_ai_slides.mp4")
        final.write_videofile(
            output_path, fps=24, codec="libx264",
            audio_codec="aac", logger="bar",
        )

        for clip in slide_clips:
            clip.close()
        final.close()

    return output_path


# ── Dynamic Video Builder (Ken Burns + Rotating BG + Overlay Cards + SFX) ────


def build_video_with_searched_images(
    slides: list[dict],
    scripts: list[str],
    search_queries: list[str],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    crossfade: float = 0.3,
    min_duration: float = 4.0,
    padding: float = 0.8,
    cutout_queries: list[str] | None = None,
    ken_burns: bool = True,
    caption_safe_pct: float = 0.15,
) -> dict:
    """Build a narrated MP4 using web-searched background images.

    Full dynamic pipeline:
      - Searches for multiple images per slide
      - Splits them into rotating backgrounds and overlay cards
      - Applies Ken Burns motion with crossfade between backgrounds
      - Pops in overlay cards every ~2s as picture-in-picture
      - Generates subtle SFX (pop/whoosh) at each visual change
      - Reserves bottom 15% of frame for external captions

    Falls back to standard PNG slides when no images are found.

    Args:
        cutout_queries: Optional search terms for transparent PNG foreground
            cutouts (one per slide). If None, skips cutout search.
        ken_burns: If True, animate backgrounds with Ken Burns motion.
        caption_safe_pct: Fraction of frame height reserved for captions
            at the bottom (0.15 = 15%). Set to 0 to use the full frame.

    Returns:
        Dict with 'video_path' and 'search_results' (query -> found/not found).
    """
    mp = _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        concatenate_videoclips,
    )
    from src.slides.image_search import (
        search_and_download_all_images,
        search_transparent_cutouts,
    )
    from src.slides.image_generator import (
        composite_slide,
        composite_slide_layers,
        get_slide_role,
        treat_background,
    )
    from src.slides.png_builder import build_pngs

    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Search and download ALL images per query (for bg rotation + cards)
    image_results = search_and_download_all_images(
        queries=search_queries,
        per_query=8,
        orientation="portrait" if aspect_ratio == "9:16" else "landscape",
        target_size=(img_w, img_h),
    )

    # Step 1b: Search for foreground cutouts (optional)
    cutout_images = [None] * len(slides)
    if cutout_queries:
        cutout_results = search_transparent_cutouts(
            queries=cutout_queries,
            per_query=8,
            target_height=int(img_h * 0.45),
        )
        cutout_images = [img for _, img in cutout_results]

    # Step 2: Prepare slide visuals
    search_report = {}

    # Build fallback standard PNGs
    fallback_pngs = build_pngs(
        slides=slides,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
        handle=handle,
    )

    # Each entry: ("dynamic", treated_bgs, overlay, cards, role)
    #          or ("static", png_path)
    slide_visuals = []

    # Overlay card dimensions
    card_w = int(img_w * 0.30)
    card_h = int(card_w * 0.75)  # 4:3 landscape

    accent_hex = colors.get("accent", "#F7B731")

    for i, (slide, (query, images)) in enumerate(zip(slides, image_results)):
        foreground = cutout_images[i] if i < len(cutout_images) else None
        role = get_slide_role(i, len(slides))

        if images and ken_burns:
            # Split images: first 2-3 as backgrounds, rest as overlay cards
            if len(images) >= 4:
                bg_imgs = images[:3]
                card_imgs = images[3:]
            elif len(images) >= 2:
                bg_imgs = images[:2]
                card_imgs = images[2:]
            else:
                bg_imgs = images[:1]
                card_imgs = []

            # Treat each background (blur + gradient + upscale)
            treated_bgs = [
                treat_background(img, img_w, img_h, accent_hex, role)
                for img in bg_imgs
            ]

            # Render text overlay (using the first bg for reference sizing)
            _, overlay = composite_slide_layers(
                bg_image=bg_imgs[0],
                slide=slide,
                slide_index=i,
                total_slides=len(slides),
                colors=colors,
                handle=handle,
                foreground=foreground,
                caption_safe_pct=caption_safe_pct,
            )

            # Prepare overlay cards
            cards = [
                _prepare_overlay_card(img, card_w, card_h)
                for img in card_imgs
            ]

            slide_visuals.append(("dynamic", treated_bgs, overlay, cards, role))

            # Save static preview for reference
            preview = composite_slide(
                bg_image=bg_imgs[0], slide=slide, slide_index=i,
                total_slides=len(slides), colors=colors,
                handle=handle, foreground=foreground,
            )
            preview.save(
                os.path.join(output_dir, f"web_slide_{i + 1:02d}.png"), "PNG",
            )
            search_report[query] = f"found ({len(images)} images)"

        elif images and not ken_burns:
            # Static composite (no animation)
            final = composite_slide(
                bg_image=images[0], slide=slide, slide_index=i,
                total_slides=len(slides), colors=colors,
                handle=handle, foreground=foreground,
            )
            out_path = os.path.join(output_dir, f"web_slide_{i + 1:02d}.png")
            final.save(out_path, "PNG")
            slide_visuals.append(("static", out_path))
            search_report[query] = "found (1 image, static)"
        else:
            slide_visuals.append(("static", fallback_pngs[i]))
            search_report[query] = "not_found"

    # Step 3: Synthesize audio, generate SFX, and assemble video
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pre-generate SFX files
        pop_sfx_path = os.path.join(tmp_dir, "sfx_pop.wav")
        whoosh_sfx_path = os.path.join(tmp_dir, "sfx_whoosh.wav")
        _generate_pop_sfx(pop_sfx_path)
        _generate_whoosh_sfx(whoosh_sfx_path)

        slide_clips = []

        for i, (visual, script) in enumerate(zip(slide_visuals, scripts)):
            # Synthesize voiceover
            vo_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(vo_path, "wb") as f:
                f.write(audio_bytes)

            vo_clip = AudioFileClip(vo_path)
            duration = max(vo_clip.duration + padding, min_duration)

            if visual[0] == "dynamic":
                _, treated_bgs, overlay, cards, role = visual
                motion = ROLE_MOTION.get(role, "drift")
                if role == "context" and i % 2 == 1:
                    motion = "pan_left"

                # Calculate intervals based on available content
                bg_interval = max(3.0, duration / max(len(treated_bgs), 1))
                card_interval = 2.0

                clip = _make_dynamic_slide_clip(
                    treated_bgs=treated_bgs,
                    overlay_rgba=overlay,
                    card_images=cards,
                    target_w=img_w,
                    target_h=img_h,
                    duration=duration,
                    base_motion=motion,
                    bg_interval=bg_interval,
                    card_interval=card_interval,
                )

                # Build audio: voiceover + SFX at card/bg transition points
                audio_parts = [vo_clip]

                # SFX for overlay card appearances
                sfx_paths = [pop_sfx_path, whoosh_sfx_path]
                for c_idx in range(len(cards)):
                    t_pop = c_idx * card_interval
                    if t_pop < duration - 0.3:
                        sfx_path = sfx_paths[c_idx % len(sfx_paths)]
                        sfx = AudioFileClip(sfx_path).with_start(t_pop)
                        audio_parts.append(sfx)

                # SFX for background transitions
                for b_idx in range(1, len(treated_bgs)):
                    t_bg = b_idx * bg_interval
                    if t_bg < duration - 0.3:
                        sfx = AudioFileClip(whoosh_sfx_path).with_start(t_bg)
                        audio_parts.append(sfx)

                if len(audio_parts) > 1:
                    mixed_audio = CompositeAudioClip(audio_parts)
                else:
                    mixed_audio = vo_clip

                clip = clip.with_audio(mixed_audio)
            else:
                _, png_path = visual
                clip = (
                    ImageClip(png_path)
                    .with_duration(duration)
                    .with_audio(vo_clip)
                )

            slide_clips.append(clip)

        if crossfade > 0 and len(slide_clips) > 1:
            final_video = concatenate_videoclips(
                slide_clips, method="compose", padding=-crossfade,
            )
        else:
            final_video = concatenate_videoclips(slide_clips, method="compose")

        output_path = os.path.join(output_dir, "narrated_web_slides.mp4")
        final_video.write_videofile(
            output_path, fps=24, codec="libx264",
            audio_codec="aac", logger="bar",
        )

        for clip in slide_clips:
            clip.close()
        final_video.close()

    return {"video_path": output_path, "search_results": search_report}


def build_video_from_slides(
    slides: list[dict],
    colors: dict,
    topic: str = "",
    angle: str = "",
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    use_ai_images: bool = False,
    use_web_images: bool = False,
    ken_burns: bool = True,
    caption_safe_pct: float = 0.15,
) -> dict:
    """Full pipeline: generate script → (optionally images) → audio → video.

    This is the main entry point that combines script generation with
    video assembly. Use this from the CLI or Streamlit UI.

    Args:
        use_ai_images: If True, generates AI background images per slide.
            Requires GOOGLE_AI_API_KEY or OPENAI_API_KEY to be set.
        use_web_images: If True, searches the web for relevant images
            per slide and uses them as backgrounds. Falls back to
            standard slides when no image is found.
        ken_burns: If True and use_web_images=True, animate backgrounds
            with Ken Burns zoom/pan motion per slide role.
        caption_safe_pct: Fraction of frame height reserved for captions.

    Returns:
        Dict with 'video_path', 'scripts', and optionally
        'image_prompts', 'search_queries', 'search_results' keys.
    """
    from src.content.generator import generate_video_script

    # Generate voiceover scripts
    scripts = generate_video_script(slides=slides, topic=topic, angle=angle)

    result = {"scripts": scripts}

    if use_web_images:
        from src.content.generator import generate_image_search_queries

        # Generate search queries from script + slides
        search_queries = generate_image_search_queries(
            slides=slides, scripts=scripts, topic=topic, angle=angle,
        )
        result["search_queries"] = search_queries

        # Build video with web-searched images (dynamic pipeline)
        web_result = build_video_with_searched_images(
            slides=slides,
            scripts=scripts,
            search_queries=search_queries,
            colors=colors,
            aspect_ratio=aspect_ratio,
            output_dir=output_dir,
            handle=handle,
            voice_id=voice_id,
            ken_burns=ken_burns,
            caption_safe_pct=caption_safe_pct,
        )
        result["video_path"] = web_result["video_path"]
        result["search_results"] = web_result["search_results"]

    elif use_ai_images:
        from src.content.generator import generate_image_prompts

        # Generate image prompts
        image_prompts = generate_image_prompts(
            slides=slides, topic=topic, angle=angle,
        )
        result["image_prompts"] = image_prompts

        # Build video with AI images
        video_path = build_video_with_ai_images(
            slides=slides,
            scripts=scripts,
            image_prompts=image_prompts,
            colors=colors,
            aspect_ratio=aspect_ratio,
            output_dir=output_dir,
            handle=handle,
            voice_id=voice_id,
        )
        result["video_path"] = video_path
    else:
        # Build video with standard PNG slides
        video_path = build_video(
            slides=slides,
            scripts=scripts,
            colors=colors,
            aspect_ratio=aspect_ratio,
            output_dir=output_dir,
            handle=handle,
            voice_id=voice_id,
        )
        result["video_path"] = video_path

    return result
