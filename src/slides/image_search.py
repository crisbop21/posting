"""Search the web for images related to video script content.

Supports three providers (auto-detected from environment variables):
  1. Pexels API: High-quality stock photos. Free key at https://www.pexels.com/api/
  2. Pixabay API: High-quality stock photos. Free key at https://pixabay.com/api/docs/
  3. Google Images: No API key needed — works out of the box.

Usage:
    from src.slides.image_search import search_and_download_images

    images = search_and_download_images(
        queries=["stock market trading floor", "bitcoin crypto chart"],
        per_query=3,
    )
    # Returns list of (query, PIL.Image) tuples
"""

import io
import os
import re
import time
from typing import Optional

import requests
from PIL import Image


_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ── Provider Detection ────────────────────────────────────────────────────────


def _get_pexels_key() -> Optional[str]:
    """Return Pexels API key if set, else None."""
    return os.environ.get("PEXELS_API_KEY", "").strip() or None


def _get_pixabay_key() -> Optional[str]:
    """Return Pixabay API key if set, else None."""
    return os.environ.get("PIXABAY_API_KEY", "").strip() or None


def get_available_provider() -> str:
    """Return the name of the best available image search provider.

    Always returns a provider — Google Images works without any key.
    """
    if _get_pexels_key():
        return "Pexels"
    if _get_pixabay_key():
        return "Pixabay"
    return "Google"


# ── Pexels API ────────────────────────────────────────────────────────────────


def _search_pexels(
    query: str,
    api_key: str,
    count: int = 5,
    orientation: str = "portrait",
) -> list[dict]:
    """Search Pexels for photos matching the query."""
    resp = requests.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": api_key},
        params={
            "query": query,
            "per_page": count,
            "orientation": orientation,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for photo in data.get("photos", []):
        src = photo.get("src", {})
        results.append({
            "url": src.get("large2x") or src.get("large") or src.get("original", ""),
            "width": photo.get("width", 0),
            "height": photo.get("height", 0),
            "photographer": photo.get("photographer", "Unknown"),
            "source": "pexels",
        })
    return results


# ── Pixabay API ───────────────────────────────────────────────────────────────


def _search_pixabay(
    query: str,
    api_key: str,
    count: int = 5,
    orientation: str = "vertical",
) -> list[dict]:
    """Search Pixabay for photos matching the query."""
    orientation_map = {"portrait": "vertical", "landscape": "horizontal", "square": "all"}
    pixabay_orientation = orientation_map.get(orientation, orientation)

    resp = requests.get(
        "https://pixabay.com/api/",
        params={
            "key": api_key,
            "q": query,
            "per_page": count,
            "image_type": "photo",
            "orientation": pixabay_orientation,
            "safesearch": "true",
            "min_width": 800,
            "min_height": 600,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for hit in data.get("hits", []):
        results.append({
            "url": hit.get("largeImageURL") or hit.get("webformatURL", ""),
            "width": hit.get("imageWidth", 0),
            "height": hit.get("imageHeight", 0),
            "source": "pixabay",
        })
    return results


# ── Google Images (no API key) ────────────────────────────────────────────────


def _search_google_images(
    query: str,
    count: int = 10,
) -> list[dict]:
    """Search Google Images by parsing the search results page.

    No API key required.

    Args:
        query: Search term.
        count: Number of results to return.

    Returns:
        List of dicts with keys: url, width, height, source.
    """
    resp = requests.get(
        "https://www.google.com/search",
        params={"q": query, "tbm": "isch", "ijn": "0"},
        headers={"User-Agent": _BROWSER_UA},
        timeout=15,
    )
    if resp.status_code != 200:
        return []

    # Google embeds full-size image URLs in JSON arrays inside the page.
    # Pattern matches: ["https://example.com/photo.jpg", ...]
    raw_urls = re.findall(
        r'\["(https?://[^"]+\.(?:jpg|jpeg|png|webp)(?:\?[^"]*)?)"',
        resp.text,
    )

    # Deduplicate while preserving order; unescape unicode
    seen = set()
    results = []
    for url in raw_urls:
        url = url.replace("\\u003d", "=").replace("\\u0026", "&")
        if url in seen:
            continue
        seen.add(url)
        # Skip tiny thumbnails (Google's encrypted-tbn URLs are thumbnails)
        if "encrypted-tbn" in url:
            continue
        results.append({
            "url": url,
            "width": 0,
            "height": 0,
            "source": "google",
        })
        if len(results) >= count:
            break

    return results


# ── Image Download ────────────────────────────────────────────────────────────


def download_image(url: str, timeout: int = 20) -> Optional[Image.Image]:
    """Download an image from a URL and return as a PIL Image.

    Returns None if the download fails.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _BROWSER_UA},
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()  # Force full decode to catch corrupt images
        return img
    except Exception:
        return None


# ── Unified Search ────────────────────────────────────────────────────────────


def search_images(
    query: str,
    count: int = 5,
    orientation: str = "portrait",
) -> list[dict]:
    """Search for images using the best available provider.

    Priority: Pexels > Pixabay > Google Images.
    Google Images always works as the final fallback (no key needed).

    Args:
        query: Search term.
        count: Number of results to request.
        orientation: 'portrait', 'landscape', or 'square' (Pexels/Pixabay only).

    Returns:
        List of dicts with keys: url, width, height, source.
    """
    pexels_key = _get_pexels_key()
    pixabay_key = _get_pixabay_key()

    if pexels_key:
        try:
            results = _search_pexels(query, pexels_key, count, orientation)
            if results:
                return results
        except Exception as e:
            print(f"[image_search] Pexels search failed: {e}")

    if pixabay_key:
        try:
            results = _search_pixabay(query, pixabay_key, count, orientation)
            if results:
                return results
        except Exception as e:
            print(f"[image_search] Pixabay search failed: {e}")

    # Google Images — always available, no key needed
    try:
        return _search_google_images(query, count)
    except Exception as e:
        print(f"[image_search] Google Images search failed: {e}")
        return []


def search_and_download_images(
    queries: list[str],
    per_query: int = 5,
    orientation: str = "portrait",
    target_size: tuple[int, int] = (1080, 1920),
) -> list[tuple[str, Optional[Image.Image]]]:
    """Search and download the best image for each query.

    For each query, searches for images and downloads the first one
    that successfully loads. Images are resized to the target size.

    Args:
        queries: List of search queries, one per slide.
        per_query: Number of search results to try per query.
        orientation: Image orientation preference.
        target_size: (width, height) to resize images to.

    Returns:
        List of (query, Image or None) tuples. One per query.
    """
    provider = get_available_provider()
    print(f"[image_search] Using {provider} for image search")

    results = []
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(1)  # Rate limiting between queries

        search_results = search_images(query, count=per_query, orientation=orientation)

        downloaded = None
        for result in search_results:
            img = download_image(result["url"])
            if img is not None:
                # Resize to target dimensions
                img = img.convert("RGBA")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                downloaded = img
                print(f"[image_search] Slide {i + 1}: found image for '{query}'")
                break

        if downloaded is None:
            print(f"[image_search] Slide {i + 1}: no image found for '{query}'")

        results.append((query, downloaded))

    return results


def search_and_download_all_images(
    queries: list[str],
    per_query: int = 8,
    orientation: str = "portrait",
    target_size: tuple[int, int] = (1080, 1920),
) -> list[tuple[str, list[Image.Image]]]:
    """Search and download ALL available images for each query.

    Unlike search_and_download_images which returns only the first match,
    this returns every successfully downloaded image per query. Used for
    rotating backgrounds and overlay cards in dynamic video slides.

    Returns:
        List of (query, [images]) tuples. The images list may be empty.
    """
    provider = get_available_provider()
    print(f"[image_search] Using {provider} for bulk image search")

    results = []
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(1)

        search_results = search_images(query, count=per_query, orientation=orientation)

        images = []
        for result in search_results:
            img = download_image(result["url"])
            if img is not None:
                img = img.convert("RGBA")
                img = img.resize(target_size, Image.Resampling.LANCZOS)
                images.append(img)

        print(f"[image_search] Slide {i + 1}: downloaded {len(images)} images for '{query}'")
        results.append((query, images))

    return results


# ── Transparent PNG Cutout Search ────────────────────────────────────────────


def _has_transparency(img: Image.Image) -> bool:
    """Check if an image has meaningful transparency (alpha channel with variation)."""
    if img.mode != "RGBA":
        return False
    alpha = img.getchannel("A")
    extrema = alpha.getextrema()
    # If alpha ranges from near-0 to near-255, it has real transparency
    return extrema[0] < 200 and extrema[1] > 50


def search_transparent_cutouts(
    queries: list[str],
    per_query: int = 8,
    target_height: int = 900,
) -> list[tuple[str, Optional[Image.Image]]]:
    """Search for transparent PNG cutouts of people/objects.

    Appends 'transparent png cutout no background' to each query
    to bias results toward images with alpha channels.

    Args:
        queries: List of search terms describing the subject.
        per_query: Number of search results to try per query.
        target_height: Approximate height to scale cutouts to.

    Returns:
        List of (query, Image or None) tuples. Images are RGBA with
        transparency preserved (not resized to fixed dimensions).
    """
    provider = get_available_provider()
    print(f"[cutout_search] Using {provider} for cutout search")

    results = []
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(1)

        # Bias search toward transparent PNGs
        cutout_query = f"{query} transparent png cutout no background"
        search_results = search_images(cutout_query, count=per_query, orientation="portrait")

        downloaded = None
        for result in search_results:
            img = download_image(result["url"])
            if img is None:
                continue

            img = img.convert("RGBA")

            # Check for actual transparency
            if _has_transparency(img):
                # Scale to target height, preserving aspect ratio
                orig_w, orig_h = img.size
                scale = target_height / orig_h
                new_w = int(orig_w * scale)
                img = img.resize((new_w, target_height), Image.Resampling.LANCZOS)
                downloaded = img
                print(f"[cutout_search] Slide {i + 1}: found cutout for '{query}'")
                break

        if downloaded is None:
            print(f"[cutout_search] Slide {i + 1}: no cutout found for '{query}'")

        results.append((query, downloaded))

    return results
