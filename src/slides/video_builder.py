"""Build narrated video from slides using ElevenLabs TTS and MoviePy.

Pipeline:
  1. Generate voiceover script per slide (via Claude)
  2. Synthesize audio per slide (via ElevenLabs, with word timestamps)
  3. Render PNG slides as video frames with audio overlay
  4. Combine into a single MP4 with crossfade transitions

Dynamic slide mode (web images):
  - Rotating backgrounds: multiple web images crossfade under Ken Burns motion
  - Big foreground image: always visible in 36-73% zone, crossfades on voice cues
  - Sound effects: subtle whoosh synced to visual changes
  - Layout: title (8-20%), captions (20-36%), image (36-73%), platform UI (73-100%)

Visual events are aligned to the voiceover rather than a fixed clock:
  - Stats/numbers in the script trigger foreground image transitions
  - Sentence boundaries trigger background transitions
  - Fallback to even spacing when timestamps are unavailable
"""

import base64
import io
import math
import os
import re
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


def synthesize_speech_with_timestamps(
    text: str,
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    model_id: str = "eleven_multilingual_v2",
    stability: float = 0.5,
    similarity_boost: float = 0.75,
) -> tuple[bytes, list[float], list[float]]:
    """TTS with character-level timing via ElevenLabs /with-timestamps endpoint.

    The alignment maps each character in the input text to its spoken time,
    which lets us sync visual events to specific words in the voiceover.

    Returns:
        (mp3_bytes, char_start_times, char_end_times).
        Falls back to (mp3_bytes, [], []) if timestamps unavailable.
    """
    api_key = _get_elevenlabs_key()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps"
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

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        audio_b64 = data.get("audio_base64", "")
        audio_bytes = base64.b64decode(audio_b64)

        alignment = data.get("alignment", {})
        char_starts = alignment.get("character_start_times_seconds", [])
        char_ends = alignment.get("character_end_times_seconds", [])

        return audio_bytes, char_starts, char_ends
    except Exception:
        # Fallback: regular TTS without timestamps
        audio_bytes = synthesize_speech(
            text=text, voice_id=voice_id, model_id=model_id,
            stability=stability, similarity_boost=similarity_boost,
        )
        return audio_bytes, [], []


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


# ── Voice-Synced Visual Cues ────────────────────────────────────────────────

# Pattern for numbers, percentages, dollar amounts, multipliers in script text.
_STAT_RE = re.compile(
    r'(\$[\d,.]+(?:\s*(?:trillion|billion|million|thousand|[TBMK]))?\w*'
    r'|[\d,.]+%'
    r'|[\d,.]+x'
    r'|[\d,.]+(?:\s*(?:trillion|billion|million|thousand)))',
    re.IGNORECASE,
)


def _extract_visual_cues(
    script: str,
    char_starts: list[float],
    char_ends: list[float],
    duration: float,
    n_cards: int,
    n_bgs: int,
    min_gap: float = 1.5,
) -> tuple[list[float] | None, list[float] | None]:
    """Derive visual event times from voiceover character timestamps.

    Scans the script for key stats/numbers and sentence boundaries, maps
    them to their spoken timestamps, and distributes overlay card pop-ins
    and background transitions at those moments.

    Args:
        script: Original voiceover text sent to TTS.
        char_starts: Per-character start times from ElevenLabs alignment.
        char_ends: Per-character end times from ElevenLabs alignment.
        duration: Total slide duration in seconds.
        n_cards: Number of overlay cards available.
        n_bgs: Number of background images available.
        min_gap: Minimum seconds between consecutive events.

    Returns:
        (card_times, bg_change_times) or (None, None) if no timestamps.
    """
    if not char_starts:
        return None, None

    # --- Collect candidate cue times ---

    # Stats/numbers → ideal moments for overlay cards
    stat_times = set()
    for m in _STAT_RE.finditer(script):
        pos = m.start()
        if pos < len(char_starts):
            stat_times.add(char_starts[pos])

    # Sentence boundaries → ideal moments for background transitions
    sentence_times = set()
    for m in re.finditer(r'[.!?;—]\s', script):
        pos = m.start()
        if pos < len(char_ends):
            sentence_times.add(char_ends[pos])

    # Merge, sort, and filter: enforce min gap, avoid extreme start/end
    all_cues = sorted(stat_times | sentence_times)
    filtered = []
    for t in all_cues:
        if t < 0.3 or t > duration - 0.5:
            continue
        if not filtered or t - filtered[-1] >= min_gap:
            filtered.append(t)

    # --- Assign cues to visual events ---

    card_times: list[float] = []
    bg_times: list[float] = []
    used: set[float] = set()

    # First pass: stat-aligned cues → cards
    for t in filtered:
        if t in stat_times and len(card_times) < n_cards:
            card_times.append(t)
            used.add(t)

    # Second pass: sentence-aligned cues → bg changes
    for t in filtered:
        if t not in used and t in sentence_times and len(bg_times) < n_bgs - 1:
            bg_times.append(t)
            used.add(t)

    # Third pass: fill remaining from any unused cues
    for t in filtered:
        if t in used:
            continue
        if len(card_times) < n_cards:
            card_times.append(t)
            used.add(t)
        elif len(bg_times) < n_bgs - 1:
            bg_times.append(t)
            used.add(t)

    # Fallback: fill with evenly-spaced times if not enough cues
    if len(card_times) < n_cards:
        interval = duration / (n_cards + 1)
        for j in range(len(card_times), n_cards):
            t = interval * (j + 1)
            if all(abs(t - ct) >= min_gap for ct in card_times + bg_times):
                card_times.append(t)

    if len(bg_times) < n_bgs - 1:
        interval = duration / n_bgs
        for j in range(len(bg_times), n_bgs - 1):
            t = interval * (j + 1)
            if all(abs(t - bt) >= min_gap for bt in bg_times + card_times):
                bg_times.append(t)

    card_times.sort()
    bg_times.sort()

    return card_times, bg_times


def _extract_sentence_times(
    script: str,
    char_ends: list[float],
    duration: float,
    n_overlays: int,
    min_gap: float = 1.0,
) -> list[float]:
    """Derive overlay transition times from sentence boundaries in the script.

    Returns N-1 times (transitions between N overlays).
    """
    if n_overlays <= 1:
        return []

    n_transitions = n_overlays - 1

    # Collect sentence boundary times from the voiceover
    sentence_times = []
    if char_ends:
        for m in re.finditer(r'[.!?;—]\s', script):
            pos = m.start()
            if pos < len(char_ends):
                t = char_ends[pos]
                if 0.3 < t < duration - 0.5:
                    if not sentence_times or t - sentence_times[-1] >= min_gap:
                        sentence_times.append(t)

    # If we got enough sentence cues, use them (evenly spaced subset)
    if len(sentence_times) >= n_transitions:
        # Pick evenly-spaced subset
        step = len(sentence_times) / n_transitions
        return [sentence_times[int(i * step)] for i in range(n_transitions)]

    # Fallback: evenly space through the duration
    interval = duration / n_overlays
    return [interval * (j + 1) for j in range(n_transitions) if interval * (j + 1) < duration - 0.3]


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


def _kenburns_crop(bg_array, src_w, src_h, target_w, target_h, progress, motion,
                   _out_buf=None, _pil_cache=None):
    """Compute a single Ken Burns cropped frame from an oversized background.

    Args:
        progress: 0.0 → 1.0 through the segment, already eased.
        _out_buf: Optional pre-allocated float32 output buffer to reuse.
        _pil_cache: Optional dict for caching the reusable PIL Image wrapper.
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

    # If crop matches target size exactly, skip the resize entirely
    cropped = bg_array[y:y + crop_h, x:x + crop_w]
    if crop_w == target_w and crop_h == target_h:
        if _out_buf is not None:
            np.copyto(_out_buf, cropped, casting='unsafe')
            return _out_buf
        return cropped.astype(np.float32)

    # Use PIL resize but write directly into the output buffer
    resized = np.array(
        Image.fromarray(cropped).resize(
            (target_w, target_h), Image.Resampling.BILINEAR,
        )
    )
    if _out_buf is not None:
        np.copyto(_out_buf, resized, casting='unsafe')
        return _out_buf
    return resized.astype(np.float32)


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

    bg_array = np.array(treated_bg.convert("RGB"))  # uint8
    src_h, src_w = bg_array.shape[:2]

    ov_array = np.array(overlay_rgba.convert("RGBA"))  # uint8
    ov_alpha_u8 = ov_array[:, :, 3]       # (H, W) uint8
    ov_rgb_u8 = ov_array[:, :, :3]        # (H, W, 3) uint8
    ov_mask = ov_alpha_u8 > 0

    _kb_buf = np.empty((target_h, target_w, 3), dtype=np.float32)

    def make_frame(t):
        p = t / duration if duration > 0 else 0
        p = 0.5 - 0.5 * math.cos(math.pi * p)
        frame = _kenburns_crop(
            bg_array, src_w, src_h, target_w, target_h, p, motion,
            _out_buf=_kb_buf,
        )
        # Only composite where overlay has content
        if ov_mask.any():
            a = ov_alpha_u8[ov_mask].astype(np.float32).reshape(-1, 1) * (1.0 / 255.0)
            rgb = ov_rgb_u8[ov_mask].astype(np.float32)
            frame[ov_mask] = frame[ov_mask] * (1.0 - a) + rgb * a
        return frame.astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(18)


# ── Entity Overlay Timing ────────────────────────────────────────────────────


def _build_entity_overlay_timing(
    slide_entities: list[dict],
    entity_arrays: dict,
    script: str,
    char_starts: list[float],
    char_ends: list[float],
    pre_roll: float = 0.0,
    linger: float = 0.5,
) -> list[dict]:
    """Build timed entity overlays from extracted entity mentions + TTS timestamps.

    For each entity in the slide, maps its character position in the script
    to the corresponding speech timestamps so the logo/headshot appears
    exactly when the name is spoken.

    Returns:
        List of dicts with keys: image (numpy), start (float), end (float), linger (float).
    """
    result = []
    for ent in slide_entities:
        name = ent.get("name", "")
        if name not in entity_arrays:
            continue

        char_start = ent.get("char_start", 0)
        char_end = ent.get("char_end", char_start + len(name))

        # Map character positions to speech timestamps
        if char_starts and char_start < len(char_starts):
            start_time = char_starts[min(char_start, len(char_starts) - 1)]
        else:
            # Fallback: estimate from character position ratio
            ratio = char_start / max(len(script), 1)
            start_time = ratio * (char_ends[-1] if char_ends else 3.0)

        if char_ends and char_end - 1 < len(char_ends):
            end_time = char_ends[min(char_end - 1, len(char_ends) - 1)]
        else:
            end_time = start_time + 0.5

        result.append({
            "image": entity_arrays[name],
            "start": start_time + pre_roll,
            "end": end_time + pre_roll,
            "linger": linger,
        })

    return result


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


def _generate_reveal_sfx(path, sample_rate=44100):
    """Generate an attention-grabbing 'reveal' impact — layered chime + sub bass.

    Combines a bright harmonic chime with a short sub-bass thump for a
    satisfying reveal moment when images/overlays appear.
    """
    import numpy as np

    duration = 0.35
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n)

    # Bright chime: two harmonically-related tones
    chime = (
        np.sin(2 * np.pi * 1200 * t) * 0.10
        + np.sin(2 * np.pi * 1800 * t) * 0.06
    )
    chime *= np.exp(-t * 12)  # Fast decay

    # Sub-bass thump for impact
    bass = np.sin(2 * np.pi * 80 * t) * 0.15 * np.exp(-t * 18)

    # Subtle noise transient at the start
    rng = np.random.RandomState(77)
    noise = rng.randn(n) * 0.04 * np.exp(-t * 40)

    signal = chime + bass + noise
    _write_wav(path, signal, sample_rate)


# ── Foreground Image Preparation ────────────────────────────────────────────

# Single centered foreground image position within the 36-73% image zone.
FG_IMAGE_POS = (0.08, 0.36)  # (x_frac, y_frac) — centered horizontally


def _fit_fg_image(img, fg_w, fg_h, corner_radius=24, bg_color=(13, 17, 23)):
    """Fit entire image within bounding box without cropping (letterboxed).

    Unlike _prepare_fg_image which center-crops, this preserves the full
    image — ideal for charts where cropping loses data.

    Returns an RGBA PIL Image ready for compositing.
    """
    from PIL import Image, ImageDraw

    src_w, src_h = img.size
    scale = min(fg_w / src_w, fg_h / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS).convert("RGBA")

    # Place on transparent canvas centered in fg_w x fg_h
    canvas = Image.new("RGBA", (fg_w, fg_h), (*bg_color, 0))
    x_off = (fg_w - new_w) // 2
    y_off = (fg_h - new_h) // 2
    canvas.paste(resized, (x_off, y_off), resized)
    resized.close()

    # Rounded-corner mask
    mask = Image.new("L", (fg_w, fg_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, fg_w - 1, fg_h - 1], radius=corner_radius, fill=255,
    )
    canvas.putalpha(mask)

    return canvas


def _prepare_fg_image(img, fg_w, fg_h, corner_radius=24):
    """Crop and style a web image as a big foreground image with rounded corners.

    Returns an RGBA PIL Image ready for compositing.
    """
    from PIL import Image, ImageDraw

    # Center-crop to target aspect ratio
    src_w, src_h = img.size
    target_ratio = fg_w / fg_h
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        new_w = int(src_h * target_ratio)
        x = (src_w - new_w) // 2
        img = img.crop((x, 0, x + new_w, src_h))
    else:
        new_h = int(src_w / target_ratio)
        y = (src_h - new_h) // 2
        img = img.crop((0, y, src_w, y + new_h))

    img = img.resize((fg_w, fg_h), Image.Resampling.LANCZOS).convert("RGBA")

    # Rounded-corner mask
    mask = Image.new("L", (fg_w, fg_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, fg_w - 1, fg_h - 1], radius=corner_radius, fill=255,
    )
    img.putalpha(mask)

    return img


# ── Dynamic Slide Clip ───────────────────────────────────────────────────────


def _make_dynamic_slide_clip(
    treated_bgs,
    overlay_rgba,
    fg_images,
    target_w,
    target_h,
    duration,
    base_motion="zoom_in",
    fg_change_times=None,
    bg_change_times=None,
    fg_crossfade=0.3,
    bg_crossfade=0.4,
    caption_safe_pct=0.27,
    cinematic_overlays=None,
    overlay_change_times=None,
    overlay_crossfade=0.5,
    entity_overlays=None,
):
    """Create an animated slide clip with rotating backgrounds and foreground images.

    Args (new):
        entity_overlays: List of dicts with keys:
            - "image": numpy array (H, W, 4) uint8 RGBA
            - "start": float, seconds when entity name starts being spoken
            - "end": float, seconds when entity name finishes being spoken
            - "linger": float, extra seconds to keep visible after speech ends

    Layout zones (1080x1920):
      - 8-20%: title (text overlay)
      - 20-36%: captions (external, kept clear)
      - 36-73%: big foreground image (crossfades on voice cues)
      - 73-100%: platform UI (kept clear)

    Args:
        treated_bgs: List of oversized PIL Images (blurred + gradient).
        overlay_rgba: RGBA PIL Image at target size (text + frame).
        fg_images: List of RGBA PIL Images (big foreground images).
        target_w, target_h: Output frame dimensions.
        duration: Total clip duration in seconds.
        base_motion: Primary Ken Burns motion for the first segment.
        fg_change_times: Explicit crossfade times between foreground images.
        bg_change_times: Explicit transition times between backgrounds.
        fg_crossfade: Duration of crossfade between foreground images.
        bg_crossfade: Duration of crossfade between backgrounds.
        cinematic_overlays: Optional list of RGBA PIL Images (cinematic
            overlay effects). Multiple overlays transition at sentence
            boundaries for visual rhythm synced to narration.
        overlay_change_times: Transition times between cinematic overlays.
        overlay_crossfade: Duration of crossfade between overlays.
    """
    import numpy as np
    from moviepy import VideoClip
    from PIL import Image

    # Fallback to even spacing if no explicit times
    if fg_change_times is None and len(fg_images) > 1:
        interval = max(2.0, duration / len(fg_images))
        fg_change_times = [
            i * interval for i in range(1, len(fg_images))
            if i * interval < duration - 0.5
        ]
    if fg_change_times is None:
        fg_change_times = []
    if bg_change_times is None:
        bg_interval = max(3.0, duration / max(len(treated_bgs), 1))
        bg_change_times = [
            i * bg_interval for i in range(1, len(treated_bgs))
            if i * bg_interval < duration - 0.5
        ]

    # Pre-compute all numpy arrays — keep as uint8 to save memory.
    # Conversion to float32 happens only in the small regions needed per frame.
    bg_arrays = []
    bg_dims = []
    for bg in treated_bgs:
        arr = np.array(bg.convert("RGB"))  # uint8
        bg_arrays.append(arr)
        h, w = arr.shape[:2]
        bg_dims.append((w, h))

    # Text overlay — store alpha as uint8, convert per-frame
    ov = np.array(overlay_rgba.convert("RGBA"))  # uint8
    text_alpha_u8 = ov[:, :, 3]   # (H, W) uint8
    text_rgb_u8 = ov[:, :, :3]    # (H, W, 3) uint8
    # Pre-compute mask of pixels that actually have text (skip transparent regions)
    text_mask = text_alpha_u8 > 0

    # Cinematic overlays — store as uint8 RGBA
    co_arrays = []
    if cinematic_overlays:
        for co_img in cinematic_overlays:
            co = co_img
            if co.size != (target_w, target_h):
                co = co.resize((target_w, target_h), Image.Resampling.LANCZOS)
            if co.mode != "RGBA":
                co = co.convert("RGBA")
            co_arrays.append(np.array(co))  # uint8

    if overlay_change_times is None:
        overlay_change_times = []

    # Foreground images — store as uint8 RGBA
    # Accept both PIL Images and pre-converted numpy arrays.
    fg_arrays = []
    for fg_img in fg_images:
        if isinstance(fg_img, np.ndarray):
            fg_arrays.append(fg_img)
        else:
            fg_arr = np.array(fg_img.convert("RGBA"))  # uint8
            fg_arrays.append(fg_arr)

    # Entity overlays — pre-processed list of (array, start, end) tuples
    ent_items = []
    if entity_overlays:
        for eo in entity_overlays:
            arr = eo["image"]  # already numpy uint8 RGBA
            ent_items.append((arr, eo["start"], eo["end"] + eo.get("linger", 0.5)))
    # Entity overlay position: top-right corner with padding
    ent_pad = int(target_w * 0.04)
    ent_y = int(target_h * 0.08)

    # Pre-allocate reusable frame buffer for Ken Burns crops
    _kb_buf = np.empty((target_h, target_w, 3), dtype=np.float32)

    # Foreground image position (centered in 36-73% zone)
    fg_x = int(FG_IMAGE_POS[0] * target_w)
    fg_y = int(FG_IMAGE_POS[1] * target_h)

    n_bgs = len(bg_arrays)
    n_cos = len(co_arrays)

    # Assign a Ken Burns motion per background segment
    motions = [base_motion]
    for j in range(1, n_bgs):
        motions.append(SEGMENT_MOTIONS[j % len(SEGMENT_MOTIONS)])

    def make_frame(t):
        # ── 1. BACKGROUND LAYER (rotating + Ken Burns) ──
        if n_bgs == 1:
            src_w, src_h = bg_dims[0]
            p = t / duration if duration > 0 else 0
            p = 0.5 - 0.5 * math.cos(math.pi * p)
            bg_frame = _kenburns_crop(
                bg_arrays[0], src_w, src_h, target_w, target_h, p, motions[0],
                _out_buf=_kb_buf,
            )
        else:
            seg_idx = sum(1 for bt in bg_change_times if t >= bt)
            seg_idx = min(seg_idx, n_bgs - 1)

            seg_start = bg_change_times[seg_idx - 1] if seg_idx > 0 else 0.0
            seg_end = (
                bg_change_times[seg_idx]
                if seg_idx < len(bg_change_times)
                else duration
            )
            seg_dur = max(seg_end - seg_start, 0.01)

            p = (t - seg_start) / seg_dur
            p = max(0.0, min(1.0, p))
            p = 0.5 - 0.5 * math.cos(math.pi * p)
            src_w, src_h = bg_dims[seg_idx]
            bg_frame = _kenburns_crop(
                bg_arrays[seg_idx], src_w, src_h,
                target_w, target_h, p, motions[seg_idx],
                _out_buf=_kb_buf,
            )

            # Crossfade approaching the next change time
            if seg_idx < len(bg_change_times):
                time_to_change = bg_change_times[seg_idx] - t
                next_idx = seg_idx + 1
                if 0 < time_to_change < bg_crossfade and next_idx < n_bgs:
                    cf = 1.0 - time_to_change / bg_crossfade
                    nsrc_w, nsrc_h = bg_dims[next_idx]
                    bg_next = _kenburns_crop(
                        bg_arrays[next_idx], nsrc_w, nsrc_h,
                        target_w, target_h, 0.0, motions[next_idx],
                    )
                    bg_frame = bg_frame * (1.0 - cf) + bg_next * cf

        # Work directly on bg_frame (avoid .copy() — it's already our buffer
        # or a freshly computed array from crossfade)
        frame = bg_frame

        # ── 2. CINEMATIC OVERLAY LAYER (sentence-synced transitions) ──
        if n_cos > 0:
            co_seg = sum(1 for ot in overlay_change_times if t >= ot)
            co_seg = min(co_seg, n_cos - 1)

            cur_co = co_arrays[co_seg]
            co_a = cur_co[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
            co_rgb = cur_co[:, :, :3].astype(np.float32)

            co_blend = 0.0
            next_co_seg = co_seg + 1
            if co_seg < len(overlay_change_times) and next_co_seg < n_cos:
                time_to_change = overlay_change_times[co_seg] - t
                if 0 < time_to_change < overlay_crossfade:
                    co_blend = 1.0 - time_to_change / overlay_crossfade

            if co_blend > 0.01 and next_co_seg < n_cos:
                next_co = co_arrays[next_co_seg]
                next_a = next_co[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
                next_rgb = next_co[:, :, :3].astype(np.float32)
                blended_a = co_a * (1.0 - co_blend) + next_a * co_blend
                blended_rgb = co_rgb * (1.0 - co_blend) + next_rgb * co_blend
                frame = frame * (1.0 - blended_a) + blended_rgb * blended_a
            else:
                frame = frame * (1.0 - co_a) + co_rgb * co_a

        # ── 3. FOREGROUND IMAGE LAYER (pop-in scale + crossfade rotation) ──
        if fg_arrays:
            n_fg = len(fg_arrays)
            seg = sum(1 for ft in fg_change_times if t >= ft)
            seg = min(seg, n_fg - 1)

            cur_fg = fg_arrays[seg]
            ch, cw = int(cur_fg.shape[0]), int(cur_fg.shape[1])

            blend = 0.0
            next_seg = seg + 1
            if seg < len(fg_change_times) and next_seg < n_fg:
                time_to_change = fg_change_times[seg] - t
                if 0 < time_to_change < fg_crossfade:
                    blend = 1.0 - time_to_change / fg_crossfade

            # Pop-in scale animation: 93% → 100% over 0.3s with ease-out
            pop_dur = 0.3
            seg_start = fg_change_times[seg - 1] if seg > 0 else 0.0
            time_in_seg = t - seg_start
            if time_in_seg < pop_dur:
                ease = 1.0 - (1.0 - time_in_seg / pop_dur) ** 3  # ease-out cubic
                scale = 0.93 + 0.07 * ease
            else:
                scale = 1.0

            if scale < 0.999:
                # Scale from center of the fg region
                scaled_w = int(cw * scale)
                scaled_h = int(ch * scale)
                crop_x = (cw - scaled_w) // 2
                crop_y = (ch - scaled_h) // 2
                fg_crop = cur_fg[crop_y:crop_y + scaled_h, crop_x:crop_x + scaled_w]
                from PIL import Image as _PILImg
                fg_resized = np.array(
                    _PILImg.fromarray(fg_crop).resize(
                        (cw, ch), _PILImg.Resampling.BILINEAR,
                    )
                )
                cur_fg_used = fg_resized
            else:
                cur_fg_used = cur_fg

            y1 = max(0, fg_y)
            x1 = max(0, fg_x)
            y2 = min(target_h, fg_y + ch)
            x2 = min(target_w, fg_x + cw)
            sy1 = y1 - fg_y
            sx1 = x1 - fg_x
            sy2 = sy1 + (y2 - y1)
            sx2 = sx1 + (x2 - x1)

            fg_slice = cur_fg_used[sy1:sy2, sx1:sx2].astype(np.float32)
            if blend > 0.01 and next_seg < n_fg:
                next_fg = fg_arrays[next_seg]
                next_slice = next_fg[sy1:sy2, sx1:sx2].astype(np.float32)
                fg_slice = fg_slice * (1.0 - blend) + next_slice * blend

            fg_a = fg_slice[:, :, 3:4] * (1.0 / 255.0)
            fg_rgb = fg_slice[:, :, :3]

            region = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = region * (1.0 - fg_a) + fg_rgb * fg_a

        # ── 4. ENTITY OVERLAY (logo/headshot synced to spoken name) ──
        if ent_items:
            fade = 0.15  # fade in/out duration
            for ent_arr, ent_start, ent_end in ent_items:
                if t < ent_start - fade or t > ent_end + fade:
                    continue
                # Compute opacity with fade-in / fade-out
                if t < ent_start:
                    opacity = (t - (ent_start - fade)) / fade
                elif t > ent_end:
                    opacity = 1.0 - (t - ent_end) / fade
                else:
                    opacity = 1.0
                opacity = max(0.0, min(1.0, opacity))
                if opacity < 0.01:
                    continue

                eh, ew = ent_arr.shape[:2]
                ex = target_w - ew - ent_pad
                ey = ent_y

                # Clip to frame bounds
                ey1 = max(0, ey)
                ex1 = max(0, ex)
                ey2 = min(target_h, ey + eh)
                ex2 = min(target_w, ex + ew)
                sy1 = ey1 - ey
                sx1 = ex1 - ex
                sy2 = sy1 + (ey2 - ey1)
                sx2 = sx1 + (ex2 - ex1)

                ent_slice = ent_arr[sy1:sy2, sx1:sx2].astype(np.float32)
                ent_a = ent_slice[:, :, 3:4] * (opacity / 255.0)
                ent_rgb = ent_slice[:, :, :3]

                region = frame[ey1:ey2, ex1:ex2]
                frame[ey1:ey2, ex1:ex2] = region * (1.0 - ent_a) + ent_rgb * ent_a

        # ── 5. TEXT OVERLAY (only where text exists — skip transparent pixels) ──
        if text_mask.any():
            t_a = text_alpha_u8[text_mask].astype(np.float32).reshape(-1, 1) * (1.0 / 255.0)
            t_rgb = text_rgb_u8[text_mask].astype(np.float32)
            frame[text_mask] = frame[text_mask] * (1.0 - t_a) + t_rgb * t_a

        return frame.astype(np.uint8)

    return VideoClip(make_frame, duration=duration).with_fps(18)


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

        import gc

        for i, (png_path, script) in enumerate(zip(png_paths, scripts)):
            # Generate audio
            audio_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            del audio_bytes

            # Load audio to determine duration
            audio_clip = AudioFileClip(audio_path)

            # First slide: add pre-roll silence so the voiceover doesn't
            # start at t=0 (prevents first words being lost to buffering)
            if i == 0:
                from moviepy import CompositeAudioClip
                pre_roll = 0.5
                audio_clip = CompositeAudioClip(
                    [audio_clip.with_start(pre_roll)]
                ).with_duration(audio_clip.duration + pre_roll)

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
            fps=18,
            codec="libx264",
            audio_codec="aac",
            logger="bar",
            ffmpeg_params=["-movflags", "+faststart"],
        )

        # Cleanup
        for clip in slide_clips:
            clip.close()
        final.close()
        del slide_clips, final
        gc.collect()

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
    overlay_prompts: list[list[str]] | list[str] | None = None,
    overlay_style: str = "auto",
) -> str:
    """Build a narrated MP4 using AI-generated background images.

    Same as build_video() but uses AI-generated images composited
    with slide text instead of plain PNG slides.

    Args:
        overlay_prompts: Optional cinematic overlay prompts per slide.
        overlay_style: Cinematic overlay style preset or 'auto'.
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
        overlay_prompts=overlay_prompts,
        overlay_style=overlay_style,
        show_counter=False,
    )

    # Step 2: Synthesize audio per slide and assemble
    import gc

    with tempfile.TemporaryDirectory() as tmp_dir:
        slide_clips = []

        for i, (png_path, script) in enumerate(zip(png_paths, scripts)):
            audio_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes = synthesize_speech(text=script, voice_id=voice_id)
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            del audio_bytes

            audio_clip = AudioFileClip(audio_path)

            # First slide: add pre-roll silence so the voiceover doesn't
            # start at t=0 (prevents first words being lost to buffering)
            if i == 0:
                from moviepy import CompositeAudioClip
                pre_roll = 0.5
                audio_clip = CompositeAudioClip(
                    [audio_clip.with_start(pre_roll)]
                ).with_duration(audio_clip.duration + pre_roll)

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
            output_path, fps=18, codec="libx264",
            audio_codec="aac", logger="bar",
            ffmpeg_params=["-movflags", "+faststart"],
        )

        for clip in slide_clips:
            clip.close()
        final.close()
        del slide_clips, final
        gc.collect()

    return output_path


# ── Predetermined Ken Burns Backgrounds ──────────────────────────────────────


def generate_predetermined_backgrounds(
    num_slides: int,
    target_w: int = 1080,
    target_h: int = 1920,
    accent_hex: str = "#F7B731",
    bg_hex: str = "#0D1117",
    bgs_per_slide: int = 2,
) -> list[list]:
    """Generate predetermined Ken Burns backgrounds using built-in painters.

    No API calls needed — uses the algorithmic background painters from
    png_builder (candlestick charts, editorial lines, geometric patterns).

    Args:
        num_slides: Number of slides to generate backgrounds for.
        target_w, target_h: Output frame dimensions.
        accent_hex, bg_hex: Color scheme.
        bgs_per_slide: Number of background variants per slide for rotation.

    Returns:
        List of lists, each containing PIL Images (oversized for Ken Burns).
    """
    from PIL import Image, ImageFilter
    from src.slides.png_builder import _hex_to_tuple, _BG_PAINTERS

    accent = _hex_to_tuple(accent_hex)
    bg_color = _hex_to_tuple(bg_hex)

    painter_names = list(_BG_PAINTERS.keys())
    bg_scale = 1.20
    scaled_w = int(target_w * bg_scale)
    scaled_h = int(target_h * bg_scale)

    all_bgs = []
    for slide_idx in range(num_slides):
        slide_bgs = []
        for bg_idx in range(bgs_per_slide):
            img = Image.new("RGB", (scaled_w, scaled_h), bg_color)
            # Cycle through painters and use different seeds
            painter_name = painter_names[(slide_idx + bg_idx) % len(painter_names)]
            painter = _BG_PAINTERS[painter_name]
            seed = slide_idx * 1000 + bg_idx * 100 + 42
            painter(img, bg_color, accent, seed)
            # Apply blur for Ken Burns readability
            img = img.filter(ImageFilter.GaussianBlur(radius=18))
            slide_bgs.append(img)
        all_bgs.append(slide_bgs)

    return all_bgs


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
    caption_safe_pct: float = 0.27,
    cinematic_overlays: list | None = None,
    image_source: str = "web",
) -> dict:
    """Build a narrated MP4 using web-searched or AI-generated background images.

    Full dynamic pipeline with voice-synced visual events:
      - Uses ElevenLabs /with-timestamps for character-level speech timing
      - Stats/numbers in the script trigger overlay card pop-ins
      - Sentence boundaries trigger background transitions
      - SFX (pop/whoosh) synced to each visual event
      - Falls back to even spacing when timestamps are unavailable

    Args:
        cutout_queries: Optional search terms for transparent PNG foreground
            cutouts (one per slide). If None, skips cutout search.
        ken_burns: If True, animate backgrounds with Ken Burns motion.
        cinematic_overlays: Optional list of RGBA PIL Images (cinematic
            overlays from generate_ai_overlay), one per slide.
        caption_safe_pct: Fraction of frame height reserved for captions
            at the bottom (0.15 = 15%). Set to 0 to use the full frame.
        image_source: 'web' to search for stock photos, 'ai' to generate
            images with AI. Both feed into the same compositing pipeline.

    Returns:
        Dict with 'video_path' and 'search_results' (query -> found/not found).
    """
    import gc

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
        generate_all_ai_images,
        get_slide_role,
        treat_background,
    )
    from src.slides.png_builder import build_pngs

    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Get images — either web search or AI generation
    if image_source == "ai":
        image_results = generate_all_ai_images(
            prompts=search_queries,
            per_prompt=3,
            target_size=(img_w, img_h),
        )
    else:
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

    # Foreground image dimensions — big image in the 36-73% zone
    fg_w = int(img_w * 0.84)    # ~907px on 1080
    fg_h = int(img_h * 0.35)    # ~672px on 1920 (covers 36-71%)

    accent_hex = colors.get("accent", "#F7B731")

    for i, (slide, (query, images)) in enumerate(zip(slides, image_results)):
        foreground = cutout_images[i] if i < len(cutout_images) else None
        # cinematic_overlays is now list[list[Image]] — multiple per slide
        slide_overlays = cinematic_overlays[i] if cinematic_overlays and i < len(cinematic_overlays) else []
        # Filter out None entries
        slide_overlays = [co for co in slide_overlays if co is not None]
        # First overlay used for static compositing (text layer + preview)
        c_overlay_first = slide_overlays[0] if slide_overlays else None
        role = get_slide_role(i, len(slides))

        if images and ken_burns:
            # Split images: first 2 as backgrounds, rest as foreground rotation
            if len(images) >= 3:
                bg_imgs = images[:2]
                fg_imgs = images[2:]
            elif len(images) >= 2:
                bg_imgs = images[:1]
                fg_imgs = images[1:]
            else:
                bg_imgs = images[:1]
                fg_imgs = images[:1]  # use same image as fallback

            # Treat each background (blur + gradient + upscale)
            treated_bgs = [
                treat_background(img, img_w, img_h, accent_hex, role)
                for img in bg_imgs
            ]

            # Render text overlay (frame + title only — captions carry the script body)
            # Use first cinematic overlay for the static text layer
            _, overlay = composite_slide_layers(
                bg_image=bg_imgs[0],
                slide=slide,
                slide_index=i,
                total_slides=len(slides),
                colors=colors,
                handle=handle,
                foreground=foreground,
                caption_safe_pct=caption_safe_pct,
                title_only=True,
                cinematic_overlay=None,  # overlays handled separately in make_frame
                show_counter=False,
            )

            # Prepare foreground images
            fg_prepared = [
                _prepare_fg_image(img, fg_w, fg_h)
                for img in fg_imgs
            ]

            slide_visuals.append(("dynamic", treated_bgs, overlay, fg_prepared, role, slide_overlays))

            # Save static preview for reference (title-only to match video)
            preview = composite_slide(
                bg_image=bg_imgs[0], slide=slide, slide_index=i,
                total_slides=len(slides), colors=colors,
                handle=handle, foreground=foreground,
                title_only=True,
                cinematic_overlay=c_overlay_first,
                show_counter=False,
            )
            preview.save(
                os.path.join(output_dir, f"web_slide_{i + 1:02d}.png"), "PNG",
            )
            preview.close()
            search_report[query] = f"found ({len(images)} images)"

        elif images and not ken_burns:
            # Static composite (no animation, title-only for captioned mode)
            final = composite_slide(
                bg_image=images[0], slide=slide, slide_index=i,
                total_slides=len(slides), colors=colors,
                handle=handle, foreground=foreground,
                title_only=True,
                cinematic_overlay=c_overlay_first,
                show_counter=False,
            )
            out_path = os.path.join(output_dir, f"web_slide_{i + 1:02d}.png")
            final.save(out_path, "PNG")
            final.close()
            slide_visuals.append(("static", out_path))
            search_report[query] = "found (1 image, static)"
        else:
            slide_visuals.append(("static", fallback_pngs[i]))
            search_report[query] = "not_found"

        # Free source images after preparing this slide's visuals.
        # The treated_bgs/overlay/fg_prepared are still needed for video,
        # but the raw downloaded images can go now.
        for img in images:
            img.close()
        # Don't close slide_overlays here — they're needed by _make_dynamic_slide_clip
        gc.collect()

    # Free cutout images — no longer needed after visual preparation
    for ci in cutout_images:
        if ci is not None:
            ci.close()
    del cutout_images, image_results
    gc.collect()

    # Step 2c: Extract entity mentions and fetch logos/headshots
    from src.content.generator import extract_entity_mentions
    from src.slides.image_search import search_entity_images
    import numpy as np

    try:
        entity_mentions = extract_entity_mentions(scripts)
        entity_images = search_entity_images(entity_mentions, target_size=int(img_w * 0.18))
        entity_arrays: dict[str, object] = {}
        for name, eimg in entity_images.items():
            if eimg is not None:
                entity_arrays[name] = np.array(eimg.convert("RGBA"))
                eimg.close()
        del entity_images
    except Exception as exc:
        print(f"[entity_overlay] Entity extraction failed (proceeding without): {exc}")
        entity_mentions = [[] for _ in scripts]
        entity_arrays = {}
    gc.collect()

    # Step 3: Synthesize audio (with timestamps), generate SFX, assemble video
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Pre-generate SFX files
        pop_sfx_path = os.path.join(tmp_dir, "sfx_pop.wav")
        whoosh_sfx_path = os.path.join(tmp_dir, "sfx_whoosh.wav")
        reveal_sfx_path = os.path.join(tmp_dir, "sfx_reveal.wav")
        _generate_pop_sfx(pop_sfx_path)
        _generate_whoosh_sfx(whoosh_sfx_path)
        _generate_reveal_sfx(reveal_sfx_path)

        slide_clips = []

        for i, (visual, script) in enumerate(zip(slide_visuals, scripts)):
            # Synthesize voiceover WITH character timestamps
            vo_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes, char_starts, char_ends = (
                synthesize_speech_with_timestamps(
                    text=script, voice_id=voice_id,
                )
            )
            with open(vo_path, "wb") as f:
                f.write(audio_bytes)
            del audio_bytes

            vo_clip = AudioFileClip(vo_path)

            # First slide: add pre-roll silence so the voiceover doesn't
            # start at t=0 (prevents first words being lost to buffering)
            pre_roll = 0.5 if i == 0 else 0.0
            duration = max(vo_clip.duration + pre_roll + padding, min_duration)

            if visual[0] == "dynamic":
                _, treated_bgs, overlay, fg_prepared, role = visual[:5]
                slide_overlays = visual[5] if len(visual) > 5 else []
                motion = ROLE_MOTION.get(role, "drift")
                if role == "context" and i % 2 == 1:
                    motion = "pan_left"

                # Extract voice-synced cue times (or None → even spacing)
                # n_cards = n_fg - 1 (number of transitions between images)
                n_fg_transitions = max(0, len(fg_prepared) - 1)
                fg_change_times, bg_change_times = _extract_visual_cues(
                    script=script,
                    char_starts=char_starts,
                    char_ends=char_ends,
                    duration=duration,
                    n_cards=n_fg_transitions,
                    n_bgs=len(treated_bgs),
                )

                # Extract sentence boundary times for overlay transitions
                overlay_change_times = None
                if len(slide_overlays) > 1 and char_starts:
                    overlay_change_times = _extract_sentence_times(
                        script=script,
                        char_ends=char_ends,
                        duration=duration,
                        n_overlays=len(slide_overlays),
                    )

                # Shift visual cue times by pre-roll offset
                if pre_roll > 0:
                    if fg_change_times:
                        fg_change_times = [t + pre_roll for t in fg_change_times]
                    if bg_change_times:
                        bg_change_times = [t + pre_roll for t in bg_change_times]
                    if overlay_change_times:
                        overlay_change_times = [t + pre_roll for t in overlay_change_times]

                # Build entity overlay timing for this slide
                slide_ent = entity_mentions[i] if i < len(entity_mentions) else []
                ent_overlays = _build_entity_overlay_timing(
                    slide_entities=slide_ent,
                    entity_arrays=entity_arrays,
                    script=script,
                    char_starts=char_starts,
                    char_ends=char_ends,
                    pre_roll=pre_roll,
                )

                clip = _make_dynamic_slide_clip(
                    treated_bgs=treated_bgs,
                    overlay_rgba=overlay,
                    fg_images=fg_prepared,
                    target_w=img_w,
                    target_h=img_h,
                    duration=duration,
                    base_motion=motion,
                    fg_change_times=fg_change_times,
                    bg_change_times=bg_change_times,
                    caption_safe_pct=caption_safe_pct,
                    cinematic_overlays=slide_overlays,
                    overlay_change_times=overlay_change_times,
                    entity_overlays=ent_overlays,
                )

                # Free PIL images now — numpy arrays are held in the clip closure
                for bg in treated_bgs:
                    bg.close()
                overlay.close()
                for fg in fg_prepared:
                    fg.close()
                for co in slide_overlays:
                    co.close()

                # Build audio: voiceover + SFX synced to visual events
                if pre_roll > 0:
                    vo_clip = vo_clip.with_start(pre_roll)
                audio_parts = [vo_clip]

                # Reveal SFX when foreground image first appears
                if fg_prepared:
                    reveal_t = pre_roll + 0.05
                    sfx = AudioFileClip(reveal_sfx_path).with_start(reveal_t)
                    audio_parts.append(sfx)

                # Resolve actual timing (may be voice-synced or fallback)
                actual_fg_times = fg_change_times or []
                actual_bg_times = bg_change_times or []

                # Reveal SFX on foreground image transitions
                for ft in actual_fg_times:
                    if ft < duration - 0.3:
                        sfx = AudioFileClip(reveal_sfx_path).with_start(ft)
                        audio_parts.append(sfx)

                # Whoosh on background transitions
                for bt in actual_bg_times:
                    if bt < duration - 0.3:
                        sfx = AudioFileClip(whoosh_sfx_path).with_start(bt)
                        audio_parts.append(sfx)

                # Reveal SFX on cinematic overlay transitions
                for ot in (overlay_change_times or []):
                    if ot < duration - 0.3:
                        sfx = AudioFileClip(reveal_sfx_path).with_start(ot)
                        audio_parts.append(sfx)

                # Pop SFX when entity overlays appear
                for eo in ent_overlays:
                    eo_t = eo["start"]
                    if 0 < eo_t < duration - 0.3:
                        sfx = AudioFileClip(pop_sfx_path).with_start(eo_t)
                        audio_parts.append(sfx)

                if len(audio_parts) > 1:
                    mixed_audio = CompositeAudioClip(audio_parts)
                else:
                    mixed_audio = vo_clip

                clip = clip.with_audio(mixed_audio)
            else:
                _, png_path = visual
                if pre_roll > 0:
                    delayed_vo = CompositeAudioClip(
                        [vo_clip.with_start(pre_roll)]
                    ).with_duration(duration)
                    clip = (
                        ImageClip(png_path)
                        .with_duration(duration)
                        .with_audio(delayed_vo)
                    )
                else:
                    clip = (
                        ImageClip(png_path)
                        .with_duration(duration)
                        .with_audio(vo_clip)
                    )

            slide_clips.append(clip)
            # Free visual data for this slide (clip holds its own refs)
            slide_visuals[i] = None
            gc.collect()

        # All visuals freed, clear the list
        del slide_visuals

        if crossfade > 0 and len(slide_clips) > 1:
            final_video = concatenate_videoclips(
                slide_clips, method="compose", padding=-crossfade,
            )
        else:
            final_video = concatenate_videoclips(slide_clips, method="compose")

        output_path = os.path.join(output_dir, "narrated_web_slides.mp4")
        final_video.write_videofile(
            output_path, fps=18, codec="libx264",
            audio_codec="aac", logger="bar",
            ffmpeg_params=["-movflags", "+faststart"],
        )

        for clip in slide_clips:
            clip.close()
        final_video.close()
        del slide_clips, final_video
        gc.collect()

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
    caption_safe_pct: float = 0.27,
    use_overlays: bool = False,
    overlay_style: str = "auto",
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
        use_overlays: If True, generates AI cinematic overlay images
            (bokeh, light leaks, film grain, etc.) for each slide.
        overlay_style: Cinematic overlay style preset or 'auto'.

    Returns:
        Dict with 'video_path', 'scripts', and optionally
        'image_prompts', 'overlay_prompts', 'search_queries',
        'search_results' keys.
    """
    from src.content.generator import generate_video_script

    # Generate voiceover scripts
    scripts = generate_video_script(slides=slides, topic=topic, angle=angle)

    result = {"scripts": scripts}

    # Generate cinematic overlay prompts if enabled
    # overlay_prompts is list[list[str]] — multiple prompts per slide
    overlay_prompts = None
    cinematic_overlays = None
    if use_overlays:
        from src.content.generator import generate_overlay_prompts
        from src.slides.image_generator import generate_ai_overlay, get_slide_role

        overlay_prompts = generate_overlay_prompts(
            slides=slides, topic=topic, angle=angle,
            overlay_style=overlay_style,
        )
        result["overlay_prompts"] = overlay_prompts

        # For web_images path, pre-generate overlay PIL images
        # Generate ALL overlays per slide (multiple per slide for sentence sync)
        if use_web_images:
            img_w = 1080 if aspect_ratio == "9:16" else 1920
            img_h = 1920 if aspect_ratio == "9:16" else 1080
            auto_styles = {
                "hook": "volumetric_light",
                "context": "cinematic_bokeh",
                "payoff": "light_leak",
                "cta": "golden_hour",
            }
            import gc
            cinematic_overlays = []  # list[list[Image]] — multiple per slide
            for i, slide_prompts in enumerate(overlay_prompts):
                role = get_slide_role(i, len(slides))
                eff_style = auto_styles.get(role, "cinematic_bokeh") if overlay_style == "auto" else overlay_style
                # Handle both list[str] and str (for backwards compat)
                prompts = slide_prompts if isinstance(slide_prompts, list) else [slide_prompts]
                slide_overlays = []
                for op in prompts:
                    try:
                        co = generate_ai_overlay(
                            prompt=op, width=img_w, height=img_h,
                            style=eff_style, role=role,
                        )
                        slide_overlays.append(co)
                    except Exception:
                        slide_overlays.append(None)
                    gc.collect()
                cinematic_overlays.append(slide_overlays)

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
            cinematic_overlays=cinematic_overlays,
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
            overlay_prompts=overlay_prompts,
            overlay_style=overlay_style,
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


def build_video_with_chart_overlays(
    slides: list[dict],
    scripts: list[str],
    chart_image_paths: list[str],
    chart_slide_mapping: list[int | None],
    colors: dict,
    aspect_ratio: str = "9:16",
    output_dir: str = "./output",
    handle: str = "@cristian.bojaca",
    voice_id: str = "pNInz6obpgDQGcFmaJgB",
    crossfade: float = 0.3,
    min_duration: float = 4.0,
    padding: float = 0.8,
    caption_safe_pct: float = 0.27,
    topic: str = "",
    angle: str = "",
) -> dict:
    """Build a narrated MP4 using chart overlays + AI images for non-chart slides.

    Slides with a mapped chart get the chart as foreground overlay.
    Slides without a chart get a web-searched image as foreground instead,
    ensuring every slide has a visual element.

    Args:
        chart_image_paths: Paths to user-uploaded chart images.
        chart_slide_mapping: List of length len(slides), each element is a
            chart index (0-based) or None if no chart for that slide.
        topic: Topic string for generating image search queries.
        angle: Angle string for generating image search queries.

    Returns:
        Dict with 'video_path'.
    """
    import gc

    _ensure_moviepy()
    from moviepy import (
        AudioFileClip,
        CompositeAudioClip,
        ImageClip,
        concatenate_videoclips,
    )
    from src.slides.image_generator import (
        composite_slide_layers,
        get_slide_role,
    )

    if aspect_ratio == "9:16":
        img_w, img_h = 1080, 1920
    else:
        img_w, img_h = 1920, 1080

    os.makedirs(output_dir, exist_ok=True)

    accent_hex = colors.get("accent", "#F7B731")
    bg_hex = colors.get("background", "#0D1117")

    # Step 1: Generate predetermined backgrounds
    # Use 1 bg per slide instead of 2 to reduce memory — chart mode
    # already has foreground chart images for visual interest.
    predet_bgs = generate_predetermined_backgrounds(
        num_slides=len(slides),
        target_w=img_w,
        target_h=img_h,
        accent_hex=accent_hex,
        bg_hex=bg_hex,
        bgs_per_slide=1,
    )

    # Step 2: Load chart images, prepare foreground overlays, then close originals
    from PIL import Image

    fg_w = int(img_w * 0.84)
    fg_h = int(img_h * 0.45)  # Taller zone for charts to avoid cropping

    # Pre-prepare chart foreground images and close originals immediately
    chart_fg_by_idx = {}
    for ci, path in enumerate(chart_image_paths):
        try:
            img = Image.open(path).convert("RGBA")
            chart_fg_by_idx[ci] = _fit_fg_image(img, fg_w, fg_h)
            img.close()
        except Exception:
            pass

    # Step 2b: Fetch web images for slides that don't have a chart
    non_chart_indices = [
        i for i in range(len(slides))
        if (i >= len(chart_slide_mapping) or chart_slide_mapping[i] is None)
    ]
    non_chart_fg = {}
    if non_chart_indices:
        from src.content.generator import generate_image_search_queries
        from src.slides.image_search import search_and_download_all_images

        nc_slides = [slides[i] for i in non_chart_indices]
        nc_scripts = [scripts[i] for i in non_chart_indices]
        nc_queries = generate_image_search_queries(
            slides=nc_slides, scripts=nc_scripts, topic=topic, angle=angle,
        )
        nc_results = search_and_download_all_images(
            queries=nc_queries,
            per_query=3,
            orientation="portrait" if aspect_ratio == "9:16" else "landscape",
            target_size=(img_w, img_h),
        )
        web_fg_w = int(img_w * 0.84)
        web_fg_h = int(img_h * 0.35)
        for idx, (_, images) in zip(non_chart_indices, nc_results):
            if images:
                best = images[0]
                non_chart_fg[idx] = _prepare_fg_image(best, web_fg_w, web_fg_h)
                best.close()
                for extra in images[1:]:
                    extra.close()
            else:
                for extra in images:
                    extra.close()
        del nc_results
        gc.collect()

    # Step 3: Build slide visuals one at a time, freeing backgrounds as we go
    import numpy as np
    slide_visuals = []
    for i, slide in enumerate(slides):
        role = get_slide_role(i, len(slides))
        chart_idx = chart_slide_mapping[i] if i < len(chart_slide_mapping) else None

        treated_bgs = predet_bgs[i]

        # Create a dummy small bg for composite_slide_layers
        dummy_bg = Image.new("RGB", (img_w, img_h), (13, 17, 23))

        _, overlay = composite_slide_layers(
            bg_image=dummy_bg,
            slide=slide,
            slide_index=i,
            total_slides=len(slides),
            colors=colors,
            handle=handle,
            caption_safe_pct=caption_safe_pct,
            title_only=True,
            cinematic_overlay=None,
            show_counter=False,
        )
        dummy_bg.close()

        fg_list = []
        if chart_idx is not None and chart_idx in chart_fg_by_idx:
            fg_img = chart_fg_by_idx[chart_idx].copy()
            fg_list = [np.array(fg_img.convert("RGBA"))]
            fg_img.close()
        elif i in non_chart_fg:
            fg_img = non_chart_fg[i]
            fg_list = [np.array(fg_img.convert("RGBA"))]
            fg_img.close()

        slide_visuals.append(("dynamic", treated_bgs, overlay, fg_list, role, []))

    # Free the predet_bgs list reference — slide_visuals now owns the images
    del predet_bgs
    del chart_fg_by_idx
    non_chart_fg.clear()
    gc.collect()

    # Step 3b: Extract entity mentions and fetch logos/headshots
    from src.content.generator import extract_entity_mentions
    from src.slides.image_search import search_entity_images
    import numpy as np

    try:
        entity_mentions = extract_entity_mentions(scripts)
        entity_images = search_entity_images(entity_mentions, target_size=int(img_w * 0.18))
        # Convert PIL images to numpy arrays and close originals
        entity_arrays: dict[str, object] = {}
        for name, img in entity_images.items():
            if img is not None:
                entity_arrays[name] = np.array(img.convert("RGBA"))
                img.close()
        del entity_images
    except Exception as exc:
        print(f"[entity_overlay] Entity extraction failed (proceeding without): {exc}")
        entity_mentions = [[] for _ in scripts]
        entity_arrays = {}

    gc.collect()

    # Step 4: Synthesize audio, assemble video
    with tempfile.TemporaryDirectory() as tmp_dir:
        pop_sfx_path = os.path.join(tmp_dir, "sfx_pop.wav")
        whoosh_sfx_path = os.path.join(tmp_dir, "sfx_whoosh.wav")
        reveal_sfx_path = os.path.join(tmp_dir, "sfx_reveal.wav")
        _generate_pop_sfx(pop_sfx_path)
        _generate_whoosh_sfx(whoosh_sfx_path)
        _generate_reveal_sfx(reveal_sfx_path)

        slide_clips = []

        for i, (visual, script) in enumerate(zip(slide_visuals, scripts)):
            vo_path = os.path.join(tmp_dir, f"slide_{i:02d}.mp3")
            audio_bytes, char_starts, char_ends = (
                synthesize_speech_with_timestamps(text=script, voice_id=voice_id)
            )
            with open(vo_path, "wb") as f:
                f.write(audio_bytes)
            del audio_bytes

            vo_clip = AudioFileClip(vo_path)
            pre_roll = 0.5 if i == 0 else 0.0
            duration = max(vo_clip.duration + pre_roll + padding, min_duration)

            _, treated_bgs, overlay, fg_prepared, role = visual[:5]
            slide_overlays = visual[5] if len(visual) > 5 else []
            motion = ROLE_MOTION.get(role, "drift")

            # Voice-synced cue times
            n_fg_transitions = max(0, len(fg_prepared) - 1)
            fg_change_times, bg_change_times = _extract_visual_cues(
                script=script,
                char_starts=char_starts,
                char_ends=char_ends,
                duration=duration,
                n_cards=n_fg_transitions,
                n_bgs=len(treated_bgs),
            )

            if pre_roll > 0:
                if fg_change_times:
                    fg_change_times = [t + pre_roll for t in fg_change_times]
                if bg_change_times:
                    bg_change_times = [t + pre_roll for t in bg_change_times]

            # Build entity overlay timing for this slide
            slide_ent = entity_mentions[i] if i < len(entity_mentions) else []
            ent_overlays = _build_entity_overlay_timing(
                slide_entities=slide_ent,
                entity_arrays=entity_arrays,
                script=script,
                char_starts=char_starts,
                char_ends=char_ends,
                pre_roll=pre_roll,
            )

            clip = _make_dynamic_slide_clip(
                treated_bgs=treated_bgs,
                overlay_rgba=overlay,
                fg_images=fg_prepared,
                target_w=img_w,
                target_h=img_h,
                duration=duration,
                base_motion=motion,
                fg_change_times=fg_change_times,
                bg_change_times=bg_change_times,
                caption_safe_pct=caption_safe_pct,
                cinematic_overlays=slide_overlays,
                entity_overlays=ent_overlays,
            )

            # Free PIL images (fg_prepared are already numpy arrays)
            for bg in treated_bgs:
                bg.close()
            overlay.close()

            # Build audio
            if pre_roll > 0:
                vo_clip = vo_clip.with_start(pre_roll)
            audio_parts = [vo_clip]

            # Reveal SFX when foreground image first appears (at slide start)
            if fg_prepared:
                reveal_t = pre_roll + 0.05  # Slightly after slide starts
                sfx = AudioFileClip(reveal_sfx_path).with_start(reveal_t)
                audio_parts.append(sfx)

            # Whoosh on foreground image transitions
            for ft in (fg_change_times or []):
                if ft < duration - 0.3:
                    sfx = AudioFileClip(reveal_sfx_path).with_start(ft)
                    audio_parts.append(sfx)

            for bt in (bg_change_times or []):
                if bt < duration - 0.3:
                    sfx = AudioFileClip(whoosh_sfx_path).with_start(bt)
                    audio_parts.append(sfx)

            # Pop SFX when entity overlays appear
            for eo in ent_overlays:
                eo_t = eo["start"]
                if 0 < eo_t < duration - 0.3:
                    sfx = AudioFileClip(pop_sfx_path).with_start(eo_t)
                    audio_parts.append(sfx)

            if len(audio_parts) > 1:
                mixed_audio = CompositeAudioClip(audio_parts)
            else:
                mixed_audio = vo_clip

            clip = clip.with_audio(mixed_audio)
            slide_clips.append(clip)
            slide_visuals[i] = None
            gc.collect()

        del slide_visuals

        if crossfade > 0 and len(slide_clips) > 1:
            final_video = concatenate_videoclips(
                slide_clips, method="compose", padding=-crossfade,
            )
        else:
            final_video = concatenate_videoclips(slide_clips, method="compose")

        output_path = os.path.join(output_dir, "narrated_chart_slides.mp4")
        final_video.write_videofile(
            output_path, fps=18, codec="libx264",
            audio_codec="aac", logger="bar",
            ffmpeg_params=["-movflags", "+faststart"],
        )

        for clip in slide_clips:
            clip.close()
        final_video.close()
        del slide_clips, final_video
        gc.collect()

    return {"video_path": output_path}
