"""Tactical score history — persists daily scores to data/tactical_score_history.json.

Provides:
  log_score(score, label)         → append today's score (one entry per day)
  load_history()                  → list of dicts [{date, score, label}, ...]
  get_trajectory(n_days=5)        → {delta, trend, score_avg, direction}
  get_percentile_thresholds()     → {p10, p35, p65, is_dynamic, n_samples}
"""
import json
import os
from datetime import date, datetime, timedelta

import numpy as np

_HISTORY_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "tactical_score_history.json")
)
_MAX_DAYS = 365 * 3          # keep up to 3 years
_MIN_SAMPLES_FOR_DYNAMIC = 30  # need at least 30 days before switching to percentile thresholds

# Static fallback thresholds (existing hardcoded values)
_STATIC_P10 = 38
_STATIC_P35 = 52
_STATIC_P65 = 65


# ── Persistence ───────────────────────────────────────────────────────────────

def load_history() -> list[dict]:
    """Return history list newest-first: [{date, score, label}, ...]."""
    path = _HISTORY_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        # Sort newest first
        return sorted(data, key=lambda x: x.get("date", ""), reverse=True)
    except Exception:
        return []


def log_score(score: int, label: str) -> None:
    """Append today's tactical score. Only one entry per calendar day (overwrites same-day entry)."""
    path = _HISTORY_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)

    today = str(date.today())
    existing: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []

    # Remove any existing entry for today
    existing = [e for e in existing if e.get("date") != today]

    existing.append({
        "date":      today,
        "score":     int(score),
        "label":     label,
        "logged_at": datetime.utcnow().isoformat(),
    })

    # Trim to _MAX_DAYS keeping most recent
    cutoff = str(date.today() - timedelta(days=_MAX_DAYS))
    existing = [e for e in existing if e.get("date", "") >= cutoff]
    existing.sort(key=lambda x: x.get("date", ""))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_trajectory(n_days: int = 5) -> dict:
    """Compute the first derivative of the tactical score.

    Returns:
        delta       — score change vs yesterday (int, positive = improving)
        delta_5d    — score change vs n_days ago
        trend       — "Rising" | "Falling" | "Stable"
        score_avg   — n_day moving average of score
        direction   — "↑" | "↓" | "→"
    """
    history = load_history()  # newest first
    scores = [e["score"] for e in history if isinstance(e.get("score"), (int, float))]

    if not scores:
        return {"delta": 0, "delta_5d": 0, "trend": "Stable", "score_avg": None, "direction": "→"}

    current = scores[0]
    delta = int(scores[0] - scores[1]) if len(scores) >= 2 else 0
    delta_5d = int(scores[0] - scores[n_days]) if len(scores) > n_days else delta
    score_avg = round(float(np.mean(scores[:n_days])), 1) if len(scores) >= n_days else round(float(np.mean(scores)), 1)

    if delta_5d >= 4:
        trend, direction = "Rising", "↑"
    elif delta_5d <= -4:
        trend, direction = "Falling", "↓"
    else:
        trend, direction = "Stable", "→"

    return {
        "delta":     delta,
        "delta_5d":  delta_5d,
        "trend":     trend,
        "score_avg": score_avg,
        "direction": direction,
        "current":   current,
    }


def get_percentile_thresholds() -> dict:
    """Return dynamic thresholds based on rolling percentiles of score history.

    Falls back to static values (38/52/65) if fewer than _MIN_SAMPLES_FOR_DYNAMIC entries.

    Returns:
        p10          — bottom 10th percentile (risk-off threshold)
        p35          — 35th percentile (caution threshold)
        p65          — 65th percentile (favorable entry threshold)
        is_dynamic   — True if percentile-based, False if static fallback
        n_samples    — number of historical data points used
    """
    history = load_history()
    scores = [e["score"] for e in history if isinstance(e.get("score"), (int, float))]
    n = len(scores)

    if n < _MIN_SAMPLES_FOR_DYNAMIC:
        return {
            "p10":        _STATIC_P10,
            "p35":        _STATIC_P35,
            "p65":        _STATIC_P65,
            "is_dynamic": False,
            "n_samples":  n,
        }

    arr = np.array(scores, dtype=float)
    return {
        "p10":        int(round(np.percentile(arr, 10))),
        "p35":        int(round(np.percentile(arr, 35))),
        "p65":        int(round(np.percentile(arr, 65))),
        "is_dynamic": True,
        "n_samples":  n,
    }
