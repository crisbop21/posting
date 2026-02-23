"""Shared text sanitization for external data sources (RSS feeds, Reddit, etc.).

Replaces common non-ASCII characters with safe ASCII equivalents to prevent
UnicodeEncodeError when text is later sent to the Anthropic API on systems
where the default encoding is not UTF-8.
"""

import unicodedata

# Common Unicode to ASCII replacements
_REPLACEMENTS = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en-dash
    "\u2014": "--",  # em-dash
    "\u2026": "...", # ellipsis
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\u200e": "",    # left-to-right mark
    "\u200f": "",    # right-to-left mark
    "\ufeff": "",    # BOM / zero-width no-break space
    "\u00b7": "-",   # middle dot
    "\u2022": "-",   # bullet
    "\u2032": "'",   # prime
    "\u2033": '"',   # double prime
    "\u00ab": '"',   # left guillemet
    "\u00bb": '"',   # right guillemet
}


def sanitize_text(text: str) -> str:
    """Normalize Unicode text to ASCII-safe equivalents.

    1. Applies explicit replacements for common typographic characters.
    2. Uses NFKD normalization to decompose accented characters.
    3. Drops any remaining non-ASCII characters that can't be mapped.
    """
    if not text:
        return text

    # Apply known replacements
    for orig, repl in _REPLACEMENTS.items():
        text = text.replace(orig, repl)

    # NFKD normalize: decomposes accented characters (e.g. e with accent), then we strip accents
    text = unicodedata.normalize("NFKD", text)

    # Encode to ASCII, ignoring characters that still can't be represented
    text = text.encode("ascii", errors="ignore").decode("ascii")

    return text
