"""Persistent cache for cross-module AI signals.

Saves all generated AI signal keys to data/signals_cache.json so they survive
page refreshes. Datetime objects are serialized as ISO strings.
"""

import json
import os
from datetime import datetime

import streamlit as st

_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "signals_cache.json")

# All session_state keys that should survive a page refresh
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


def save_signals() -> None:
    """Write all known signal keys from session_state to disk. Silent on failure."""
    try:
        payload = {}
        for key in _SIGNAL_KEYS:
            val = st.session_state.get(key)
            if val is not None:
                payload[key] = _serialize(val)
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def load_signals() -> None:
    """Load signal cache into session_state. Only sets keys not already present."""
    if not os.path.exists(_CACHE_FILE):
        return
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            payload = json.load(f)
        for key, val in payload.items():
            if key not in st.session_state or st.session_state[key] is None:
                st.session_state[key] = _deserialize(val)
    except Exception:
        pass
