"""Persist topics with their linked verified data to disk.

Saves each topic as a JSON file under ``saved_topics/`` so the user can
resume research on any topic across sessions.
"""

import hashlib
import json
import os
import time
from pathlib import Path

STORE_DIR = Path("saved_topics")


def _ensure_dir() -> None:
    STORE_DIR.mkdir(exist_ok=True)


def _topic_id(title: str) -> str:
    """Deterministic short id from the topic title."""
    return hashlib.sha256(title.encode()).hexdigest()[:12]


def _path_for(topic_id: str) -> Path:
    return STORE_DIR / f"{topic_id}.json"


# ── Public API ────────────────────────────────────────────────────────────────


def save_topic(
    topic: dict,
    *,
    verified_bullets: list[dict] | None = None,
    research_text: str = "",
    research_facts: list[dict] | None = None,
    angle: str = "",
    user_facts: str = "",
) -> str:
    """Save (or update) a topic with its associated data.  Returns the topic id."""
    _ensure_dir()
    tid = _topic_id(topic["title"])
    path = _path_for(tid)

    # Merge with existing data so partial saves don't erase earlier fields.
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    record = {
        "topic_id": tid,
        "topic": topic,
        "verified_bullets": verified_bullets if verified_bullets is not None else existing.get("verified_bullets", []),
        "research_text": research_text or existing.get("research_text", ""),
        "research_facts": research_facts if research_facts is not None else existing.get("research_facts", []),
        "angle": angle or existing.get("angle", ""),
        "user_facts": user_facts or existing.get("user_facts", ""),
        "updated_at": time.time(),
        "created_at": existing.get("created_at", time.time()),
    }

    path.write_text(json.dumps(record, indent=2))
    return tid


def load_topic(topic_id: str) -> dict | None:
    """Load a saved topic record by id.  Returns ``None`` if not found."""
    path = _path_for(topic_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_topics() -> list[dict]:
    """Return all saved topic records, newest first."""
    _ensure_dir()
    records = []
    for f in STORE_DIR.glob("*.json"):
        try:
            records.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    records.sort(key=lambda r: r.get("updated_at", 0), reverse=True)
    return records


def delete_topic(topic_id: str) -> bool:
    """Delete a saved topic.  Returns True if it existed."""
    path = _path_for(topic_id)
    if path.exists():
        path.unlink()
        return True
    return False
