"""Thin wrapper around the Anthropic Messages API.

Creates a fresh client for every request to avoid any internal state
accumulation (e.g. duplicate tool_use IDs across calls), and retries
on transient errors with exponential backoff.
"""

import copy
import random
import time
import uuid

import anthropic


# Retryable error codes / substrings
_RETRYABLE_MESSAGES = [
    "tool_use ids must be unique",
    "overloaded",
    "rate_limit",
]

_MAX_RETRIES = 5
_BASE_DELAY = 2  # seconds


def create_message(*, model: str, max_tokens: int, messages: list, **kwargs):
    """Call Anthropic Messages API with a fresh client and retry logic.

    All keyword arguments are forwarded to ``client.messages.create``.

    Defensive measures against duplicate tool_use ID errors:
    - Deep-copies messages so no caller state is mutated.
    - Scans for and regenerates any duplicate tool_use IDs before each attempt.
    - Explicitly closes the HTTP client after each attempt.
    """
    last_error = None

    for attempt in range(_MAX_RETRIES):
        # Deep-copy to avoid mutating the caller's list and to isolate
        # each attempt from any SDK-side in-place modifications.
        clean_messages = copy.deepcopy(messages)
        _ensure_unique_tool_use_ids(clean_messages)

        # Fresh client per attempt to avoid any leaked internal state.
        client = anthropic.Anthropic()

        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=clean_messages,
                **kwargs,
            )
        except anthropic.BadRequestError as exc:
            last_error = exc
            if _is_retryable(exc):
                _backoff(attempt)
                continue
            raise
        except (anthropic.RateLimitError, anthropic.InternalServerError) as exc:
            last_error = exc
            _backoff(attempt)
            continue
        finally:
            # Explicitly close the underlying HTTP transport so connections
            # and any cached state are released immediately.
            client.close()

    # All retries exhausted, raise the last error.
    raise last_error  # type: ignore[misc]


def _ensure_unique_tool_use_ids(messages: list) -> None:
    """Scan message content blocks and regenerate any duplicate tool_use IDs.

    The Anthropic API requires every ``tool_use`` block across the entire
    ``messages`` array to carry a globally unique ``id``.  If the same
    conversation history is reused, or the SDK leaks serialised state,
    IDs can collide.  This function detects duplicates and replaces them
    with fresh UUIDs in the ``toolu_`` format the API expects.
    """
    seen: set[str] = set()

    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use" or "id" not in block:
                continue

            tool_id = block["id"]
            if tool_id in seen:
                block["id"] = _fresh_tool_id()
            seen.add(block["id"])


def _fresh_tool_id() -> str:
    """Generate a unique tool_use ID in the ``toolu_`` format."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


def _is_retryable(exc: anthropic.BadRequestError) -> bool:
    msg = str(exc).lower()
    return any(r in msg for r in _RETRYABLE_MESSAGES)


def _backoff(attempt: int) -> None:
    """Exponential backoff with jitter."""
    delay = _BASE_DELAY * (2 ** attempt) + random.uniform(0, 1)
    time.sleep(delay)
