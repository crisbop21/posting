"""Search the web for images related to video script content.

Supports two providers (auto-detected from environment variables):
  - Pexels API: Free, high-quality stock photos. Get a key at https://www.pexels.com/api/
  - DuckDuckGo: No API key needed, fallback option.

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


# ── Provider Detection ────────────────────────────────────────────────────────


def _get_pexels_key() -> Optional[str]:
    """Return Pexels API key if set, else None."""
    return os.environ.get("PEXELS_API_KEY", "").strip() or None


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
        List of dicts with keys: url, width, height, photographer, src.
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


# ── DuckDuckGo Image Search ──────────────────────────────────────────────────


def _search_duckduckgo(
    query: str,
    count: int = 5,
) -> list[dict]:
    """Search DuckDuckGo for images matching the query.

    Uses the DDG image search API (no key required).

    Returns:
        List of dicts with keys: url, width, height, source.
    """
    # First get the vqd token
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    })

    token_resp = session.get(
        "https://duckduckgo.com/",
        params={"q": query},
        timeout=10,
    )
    vqd_match = re.search(r"vqd=['\"]([^'\"]+)['\"]", token_resp.text)
    if not vqd_match:
        # Try alternate pattern
        vqd_match = re.search(r"vqd=(\d+-\d+)", token_resp.text)
    if not vqd_match:
        return []

    vqd = vqd_match.group(1)

    # Search for images
    resp = session.get(
        "https://duckduckgo.com/i.js",
        params={
            "l": "us-en",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,,,",
            "p": "1",
        },
        timeout=15,
    )

    if resp.status_code != 200:
        return []

    data = resp.json()
    results = []
    for item in data.get("results", [])[:count]:
        results.append({
            "url": item.get("image", ""),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
            "source": "duckduckgo",
        })
    return [r for r in results if r["url"]]


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

    Tries Pexels first (if PEXELS_API_KEY is set), then falls back
    to DuckDuckGo image search (no key needed).

    Args:
        query: Search term.
        count: Number of results to request.
        orientation: 'portrait', 'landscape', or 'square' (Pexels only).

    Returns:
        List of dicts with keys: url, width, height, source.
    """
    pexels_key = _get_pexels_key()

    if pexels_key:
        try:
            results = _search_pexels(query, pexels_key, count, orientation)
            if results:
                return results
        except Exception as e:
            print(f"[image_search] Pexels search failed: {e}")

    # Fallback to DuckDuckGo
    try:
        return _search_duckduckgo(query, count)
    except Exception as e:
        print(f"[image_search] DuckDuckGo search failed: {e}")
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
                break

        results.append((query, downloaded))

    return results
