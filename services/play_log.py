"""Persistent Play Log — appends AI play results to plays_log.json."""

import json
import os
from datetime import datetime

_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "plays_log.json")


def append_play(feature: str, engine: str, data: dict, meta: dict | None = None) -> None:
    """Append one play entry to plays_log.json. Silent on failure."""
    try:
        entry = {
            "id":        datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:18],
            "timestamp": datetime.now().isoformat(),
            "feature":   feature,
            "engine":    engine,
            "data":      data,
            "meta":      meta or {},
        }
        log = _load_raw()
        log.append(entry)
        with open(_LOG_PATH, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except Exception:
        pass  # Never crash the app for logging failures


def load_plays() -> list[dict]:
    """Return all logged plays, newest first."""
    return list(reversed(_load_raw()))


def clear_plays() -> None:
    """Delete all log entries."""
    try:
        with open(_LOG_PATH, "w") as f:
            json.dump([], f)
    except Exception:
        pass


def _load_raw() -> list:
    if not os.path.exists(_LOG_PATH):
        return []
    try:
        with open(_LOG_PATH) as f:
            return json.load(f)
    except Exception:
        return []
