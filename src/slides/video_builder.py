"""Build narrated video from slides using ElevenLabs TTS and MoviePy.

Pipeline:
  1. Generate voiceover script per slide (via Claude)
  2. Synthesize audio per slide (via ElevenLabs)
  3. Render PNG slides as video frames with audio overlay
  4. Combine into a single MP4 with crossfade transitions

When web-searched images are used, the background is animated with
role-based Ken Burns motion (zoom/pan) to make the video feel produced.
"""

import io
import math
import os
import tempfile
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


def _make_kenburns_clip(
    treated_bg,
    overlay_rgba,
    target_w: int,
    target_h: int,
    duration: float,
    motion: str = "zoom_in",
):
    """Create an animated video clip with Ken Burns motion on the background.

    Composites a static text overlay on top of a slowly-moving background
    to make still images feel alive in video.

    Args:
        treated_bg: PIL Image — oversized background (blurred + gradient).
        overlay_rgba: PIL Image — RGBA overlay at target size (text, frame, fg).
        target_w: Output frame width.
        target_h: Output frame height.
        duration: Clip duration in seconds.
        motion: 'zoom_in', 'zoom_out', 'pan_right', 'pan_left', or 'drift'.
    """
    import numpy as np
    from moviepy import VideoClip
    from PIL import Image

    bg_array = np.array(treated_bg.convert("RGB"))
    src_h, src_w = bg_array.shape[:2]

    # Precompute overlay arrays for fast per-frame compositing
    ov_array = np.array(overlay_rgba.convert("RGBA"))
    alpha = ov_array[:, :, 3:4].astype(np.float32) / 255.0
    ov_rgb = ov_array[:, :, :3].astype(np.float32)

    def make_frame(t):
        p = t / duration if duration > 0 else 0
        # Smooth ease in-out via cosine interpolation
        p = 0.5 - 0.5 * math.cos(math.pi * p)

        if motion == "zoom_in":
            # Start wide (show full oversized bg), end tight (target crop)
            crop_w = int(src_w - (src_w - target_w) * p)
            crop_h = int(src_h - (src_h - target_h) * p)
            x = (src_w - crop_w) // 2
            y = (src_h - crop_h) // 2
        elif motion == "zoom_out":
            # Start tight (target crop), end wide (show full bg)
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
        else:  # drift — very subtle zoom for CTA
            drift_amount = 0.3  # use 30% of available zoom range
            dp = p * drift_amount
            crop_w = int(src_w - (src_w - target_w) * dp)
            crop_h = int(src_h - (src_h - target_h) * dp)
            x = (src_w - crop_w) // 2
            y = (src_h - crop_h) // 2

        # Clamp to valid bounds
        x = max(0, min(x, src_w - crop_w))
        y = max(0, min(y, src_h - crop_h))
        crop_w = min(crop_w, src_w - x)
        crop_h = min(crop_h, src_h - y)

        # Crop background region
        cropped = bg_array[y:y + crop_h, x:x + crop_w]

        # Resize to target dimensions (bilinear is fast for video frames)
        bg_frame = np.array(
            Image.fromarray(cropped).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
        ).astype(np.float32)

        # Composite text overlay on top of animated background
        result = bg_frame * (1.0 - alpha) + ov_rgb * alpha
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


# ── Animated Video Builder (Ken Burns) ───────────────────────────────────────


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
) -> dict:
    """Build a narrated MP4 using web-searched background images.

    Searches the web for images matching each query, composites them
    with slide text using the 5-layer visual pipeline (blur + gradient +
    foreground cutout + branded frame + role-specific text layout).
    Falls back to standard PNG slides when no image is found.

    When ken_burns=True (default), background images are animated with
    role-based Ken Burns motion (zoom/pan) while text stays static,
    making the video feel produced rather than a slideshow.

    Motion mapping per role:
      - hook:    zoom-in  (draws viewer in, urgency)
      - context: pan      (exploration, alternates L/R)
      - payoff:  zoom-out (pull-back reveal)
      - cta:     drift    (subtle, keeps focus on text)

    Args:
        cutout_queries: Optional search terms for transparent PNG foreground
            cutouts (one per slide). If None, skips cutout search.
        ken_burns: If True, animate backgrounds with Ken Burns motion.

    Returns:
        Dict with 'video_path' and 'search_results' (query -> found/not found).
    """
    mp = _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
    )
    from src.slides.image_search import search_and_download_images, search_transparent_cutouts
    from src.slides.image_generator import (
        composite_slide,
        composite_slide_layers,
        get_slide_role,
    )
    from src.slides.png_builder import build_pngs

    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Search and download background images
    image_results = search_and_download_images(
        queries=search_queries,
        per_query=5,
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

    # Build fallback standard PNGs for slides without images
    fallback_pngs = build_pngs(
        slides=slides,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
        handle=handle,
    )

    # Each entry is either ("animated", bg, overlay, role) or ("static", png_path)
    slide_visuals = []

    for i, (slide, (query, img)) in enumerate(zip(slides, image_results)):
        foreground = cutout_images[i] if i < len(cutout_images) else None
        if img is not None:
            if ken_burns:
                # Separate layers for animated compositing
                bg_treated, overlay = composite_slide_layers(
                    bg_image=img,
                    slide=slide,
                    slide_index=i,
                    total_slides=len(slides),
                    colors=colors,
                    handle=handle,
                    foreground=foreground,
                )
                role = get_slide_role(i, len(slides))
                slide_visuals.append(("animated", bg_treated, overlay, role))

                # Save static preview for reference
                preview = composite_slide(
                    bg_image=img, slide=slide, slide_index=i,
                    total_slides=len(slides), colors=colors,
                    handle=handle, foreground=foreground,
                )
                preview.save(
                    os.path.join(output_dir, f"web_slide_{i + 1:02d}.png"), "PNG",
                )
            else:
                # Static composite (no animation)
                final = composite_slide(
                    bg_image=img,
                    slide=slide,
                    slide_index=i,
                    total_slides=len(slides),
                    colors=colors,
                    handle=handle,
                    foreground=foreground,
                )
                out_path = os.path.join(output_dir, f"web_slide_{i + 1:02d}.png")
                final.save(out_path, "PNG")
                slide_visuals.append(("static", out_path))
            search_report[query] = "found"
        else:
            # Use fallback standard slide
            slide_visuals.append(("static", fallback_pngs[i]))
            search_report[query] = "not_found"

    # Step 3: Synthesize audio and assemble video
    with tempfile.TemporaryDirectory() as tmp_dir:
        slide_clips = []

        for i, (visual, script) in enumerate(zip(slide_visuals, scripts)):
            audio_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)

            audio_clip = AudioFileClip(audio_path)
            duration = max(audio_clip.duration + padding, min_duration)

            if visual[0] == "animated":
                _, bg, overlay, role = visual
                motion = ROLE_MOTION.get(role, "drift")
                # Alternate pan direction for consecutive context slides
                if role == "context" and i % 2 == 1:
                    motion = "pan_left"

                clip = _make_kenburns_clip(
                    treated_bg=bg,
                    overlay_rgba=overlay,
                    target_w=img_w,
                    target_h=img_h,
                    duration=duration,
                    motion=motion,
                )
                clip = clip.with_audio(audio_clip)
            else:
                _, png_path = visual
                clip = (
                    ImageClip(png_path)
                    .with_duration(duration)
                    .with_audio(audio_clip)
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

        # Build video with web-searched images (+ Ken Burns animation)
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
