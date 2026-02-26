"""Build narrated video from slides using ElevenLabs TTS and MoviePy.

Pipeline:
  1. Generate voiceover script per slide (via Claude)
  2. Synthesize audio per slide (via ElevenLabs)
  3. Render PNG slides as video frames with audio overlay
  4. Combine into a single MP4 with crossfade transitions
"""

import io
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
) -> dict:
    """Build a narrated MP4 using web-searched background images.

    Searches the web for images matching each query, composites them
    with slide text, and assembles the final video with voiceover.
    Falls back to standard PNG slides when no image is found.

    Returns:
        Dict with 'video_path' and 'search_results' (query -> found/not found).
    """
    mp = _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        ImageClip,
        concatenate_videoclips,
    )
    from src.slides.image_search import search_and_download_images
    from src.slides.image_generator import composite_slide
    from src.slides.png_builder import build_pngs

    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Search and download images for each slide
    image_results = search_and_download_images(
        queries=search_queries,
        per_query=5,
        orientation="portrait" if aspect_ratio == "9:16" else "landscape",
        target_size=(img_w, img_h),
    )

    # Step 2: Build slide PNGs — composite searched images with text,
    # fall back to standard PNG slides where no image was found
    search_report = {}
    png_paths = []

    # Build fallback standard PNGs for slides without images
    fallback_pngs = build_pngs(
        slides=slides,
        colors=colors,
        aspect_ratio=aspect_ratio,
        output_dir=output_dir,
        handle=handle,
    )

    for i, (slide, (query, img)) in enumerate(zip(slides, image_results)):
        if img is not None:
            # Composite slide text over the searched image
            final = composite_slide(
                bg_image=img,
                slide=slide,
                slide_index=i,
                total_slides=len(slides),
                colors=colors,
                handle=handle,
            )
            out_path = os.path.join(output_dir, f"web_slide_{i + 1:02d}.png")
            final.save(out_path, "PNG")
            png_paths.append(out_path)
            search_report[query] = "found"
        else:
            # Use fallback standard slide
            png_paths.append(fallback_pngs[i])
            search_report[query] = "not_found"

    # Step 3: Synthesize audio and assemble video
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

        # Build video with web-searched images
        web_result = build_video_with_searched_images(
            slides=slides,
            scripts=scripts,
            search_queries=search_queries,
            colors=colors,
            aspect_ratio=aspect_ratio,
            output_dir=output_dir,
            handle=handle,
            voice_id=voice_id,
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
