"""Thin wrapper around the Anthropic Messages API.

Creates a fresh client for every request to avoid any internal state
accumulation (e.g. duplicate tool_use IDs across calls), and retries
on transient errors with exponential backoff.
"""

import time

import anthropic


# Retryable error codes / substrings
_RETRYABLE_MESSAGES = [
    "tool_use ids must be unique",
    "overloaded",
    "rate_limit",
]

_MAX_RETRIES = 3
_BASE_DELAY = 2  # seconds


def create_message(*, model: str, max_tokens: int, messages: list, **kwargs):
    """Call Anthropic Messages API with a fresh client and retry logic.

    All keyword arguments are forwarded to ``client.messages.create``.
    """
    last_error = None

    for attempt in range(_MAX_RETRIES):
        # Fresh client per attempt to avoid any leaked internal state.
        client = anthropic.Anthropic()

        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
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

    # All retries exhausted, raise the last error.
    raise last_error  # type: ignore[misc]


def _is_retryable(exc: anthropic.BadRequestError) -> bool:
    msg = str(exc).lower()
    return any(r in msg for r in _RETRYABLE_MESSAGES)


def _backoff(attempt: int) -> None:
    delay = _BASE_DELAY * (2 ** attempt)
    time.sleep(delay)
