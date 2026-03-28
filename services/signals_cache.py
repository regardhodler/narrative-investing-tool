"""Persistent cache for cross-module AI signals.

Saves all generated AI signal keys to a GitHub Gist (primary, survives Streamlit
Cloud deploys) and data/signals_cache.json (local fallback). Datetime objects are
serialized as ISO strings.
"""

import json
import os
from datetime import datetime

import streamlit as st

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "signals_cache.json")

# Gist-based persistence (set SIGNALS_GIST_ID + GIST_TOKEN in .env / Streamlit secrets)
_SIGNALS_GIST_ID  = os.getenv("SIGNALS_GIST_ID", "")
_SIGNALS_GIST_RAW = os.getenv("SIGNALS_GIST_RAW_URL", "")
_GIST_TOKEN       = (os.getenv("GIST_TOKEN") or os.getenv("GITHUB_GIST_TOKEN") or "").strip()
_GIST_FILENAME    = "signals_cache.json"

# All session_state keys that should survive a page refresh / redeploy
_SIGNAL_KEYS = [
    # Regime context
    "_regime_context",
    "_regime_context_ts",
    # Regime Plays
    "_rp_plays_result",
    "_rp_plays_last_tier",
    # Fed / Rate Path
    "_fed_funds_rate",
    "_dominant_rate_path",
    "_rate_path_probs",
    "_rate_path_probs_ts",
    # Fed Rate-Path Plays
    "_fed_plays_result",
    "_fed_plays_result_ts",
    "_fed_plays_engine",
    "_fed_plays_tier",
    # Policy Transmission
    "_chain_narration",
    "_chain_narration_engine",
    # Black Swans
    "_custom_swans",
    "_custom_swans_ts",
    # Doom Briefing
    "_doom_briefing",
    "_doom_briefing_ts",
    "_doom_briefing_engine",
    # Whale Summary
    "_whale_summary",
    "_whale_summary_ts",
    "_whale_summary_engine",
    # Discovery Plays
    "_plays_result",
    "_plays_engine",
    # Macro Fit
    "_macro_fit_results",
    # Bayesian calibrated rate probabilities
    "_calibrated_rate_probs",
    # Portfolio Analysis
    "_portfolio_analysis",
    "_portfolio_analysis_ts",
    "_portfolio_analysis_engine",
    # Current Events
    "_current_events_digest",
    "_current_events_digest_ts",
    "_current_events_engine",
    # Factor Analysis
    "_factor_analysis",
    "_factor_analysis_ts",
    "_factor_analysis_engine",
    # Trending Narratives (auto-discovery via Google Trends + news)
    "_trending_narratives",
    "_trending_narratives_ts",
    "_trending_narratives_tf",
]


def _serialize(v):
    """Convert a value to JSON-safe form."""
    if isinstance(v, datetime):
        return {"__datetime__": v.isoformat()}
    if isinstance(v, dict):
        return {k: _serialize(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_serialize(item) for item in v]
    return v


def _deserialize(v):
    """Restore datetime objects from serialized form."""
    if isinstance(v, dict):
        if "__datetime__" in v:
            return datetime.fromisoformat(v["__datetime__"])
        return {k: _deserialize(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialize(item) for item in v]
    return v


def load_signals() -> None:
    """Load signal cache into session_state. Tries Gist first, falls back to local file.
    Only sets keys not already present (non-destructive)."""
    import requests as _req
    payload = None

    # 1. Try Gist
    if _SIGNALS_GIST_RAW:
        try:
            resp = _req.get(
                _SIGNALS_GIST_RAW, timeout=8,
                headers={"User-Agent": "NarrativeInvestingTool/1.0",
                         "Cache-Control": "no-cache"},
            )
            if resp.ok:
                payload = resp.json()
        except Exception:
            pass

    # 2. Fall back to local file
    if payload is None and os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            pass

    if payload is None:
        return

    for key, val in payload.items():
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = _deserialize(val)


def save_signals() -> None:
    """Write all known signal keys from session_state to local file and Gist.
    Local file is written every call. Gist is written at most every 5 minutes."""
    import requests as _req
    try:
        payload = {}
        for key in _SIGNAL_KEYS:
            val = st.session_state.get(key)
            if val is not None:
                payload[key] = _serialize(val)

        payload_str = json.dumps(payload, indent=2)

        # Always write local file (fast, no network)
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(payload_str)

        # Write Gist at most every 5 minutes (debounce to avoid API rate limits)
        if _SIGNALS_GIST_ID and _GIST_TOKEN:
            last = st.session_state.get("_signals_gist_saved_at")
            now = datetime.now()
            if last is None or (now - last).total_seconds() > 300:
                try:
                    _req.patch(
                        f"https://api.github.com/gists/{_SIGNALS_GIST_ID}",
                        json={"files": {_GIST_FILENAME: {"content": payload_str}}},
                        headers={
                            "Authorization": f"Bearer {_GIST_TOKEN}",
                            "Accept": "application/vnd.github+json",
                        },
                        timeout=10,
                    )
                    st.session_state["_signals_gist_saved_at"] = now
                except Exception:
                    pass
    except Exception:
        pass
