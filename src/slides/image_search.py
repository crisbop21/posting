"""Search the web for images related to video script content.

Supports two free providers (auto-detected from environment variables):
  - Pexels API: Free, high-quality stock photos. Get a key at https://www.pexels.com/api/
  - Pixabay API: Free, high-quality stock photos. Get a key at https://pixabay.com/api/docs/

Both are genuinely free (no credit card). Set at least one key.

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
import time
from typing import Optional

import requests
from PIL import Image


# ── Provider Detection ────────────────────────────────────────────────────────


def _get_pexels_key() -> Optional[str]:
    """Return Pexels API key if set, else None."""
    return os.environ.get("PEXELS_API_KEY", "").strip() or None


def _get_pixabay_key() -> Optional[str]:
    """Return Pixabay API key if set, else None."""
    return os.environ.get("PIXABAY_API_KEY", "").strip() or None


def get_available_provider() -> Optional[str]:
    """Return the name of the first available image search provider, or None.

    Used by the UI to check if web image search is configured.
    """
    if _get_pexels_key():
        return "Pexels"
    if _get_pixabay_key():
        return "Pixabay"
    return None


# ── Pexels API ────────────────────────────────────────────────────────────────


def _search_pexels(
    query: str,
    api_key: str,
    count: int = 5,
    orientation: str = "portrait",
) -> list[dict]:
    """Search Pexels for photos matching the query.

    Args:
        query: Search term.
        api_key: Pexels API key.
        count: Number of results to request.
        orientation: 'portrait', 'landscape', or 'square'.

    Returns:
        List of dicts with keys: url, width, height, photographer, source.
    """
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
    """Search Pixabay for photos matching the query.

    Args:
        query: Search term.
        api_key: Pixabay API key.
        count: Number of results to request.
        orientation: 'vertical', 'horizontal', or 'all'.

    Returns:
        List of dicts with keys: url, width, height, source.
    """
    # Map Pexels-style orientation to Pixabay-style
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


# ── Image Download ────────────────────────────────────────────────────────────


def download_image(url: str, timeout: int = 20) -> Optional[Image.Image]:
    """Download an image from a URL and return as a PIL Image.

    Returns None if the download fails.
    """
    try:
        resp = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            },
        )
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type and not url.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()  # Force decode to catch corrupt images
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

    Tries Pexels first, then Pixabay.

    Args:
        query: Search term.
        count: Number of results to request.
        orientation: 'portrait', 'landscape', or 'square'.

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

    return []


def search_and_download_images(
    queries: list[str],
    per_query: int = 3,
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
    if not provider:
        print("[image_search] No image search provider configured. "
              "Set PEXELS_API_KEY or PIXABAY_API_KEY.")
        return [(q, None) for q in queries]

    print(f"[image_search] Using {provider} for image search")

    results = []
    for i, query in enumerate(queries):
        if i > 0:
            time.sleep(0.5)  # Rate limiting between queries

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
