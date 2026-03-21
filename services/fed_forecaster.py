"""
Fed Policy Forecasting Machine — data layer.

Functions:
  fetch_zq_probabilities()         — ZQ futures → 4-scenario probabilities
  fetch_fed_communications()       — Fed RSS feeds → tone items
  score_fed_tone()                 — Groq tone scoring of Fed comms
  adjust_probabilities()           — Apply tone adjustment to base probs
  build_fed_context()              — Package regime signals for Groq prompt
  generate_forecast()              — Groq causal chain + asset matrix
  get_next_fomc()                  — Days to next FOMC meeting
"""

import json
import os
import hashlib
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from services.market_data import fetch_fred_series_safe


# ─────────────────────────────────────────────────────────────────────────────
# FOMC CALENDAR  (update each January from federalreserve.gov/monetarypolicy/fomccalendars.htm)
# ─────────────────────────────────────────────────────────────────────────────

_FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 4, 29),
    date(2026, 6, 10),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


def get_next_fomc() -> dict:
    """Return the next upcoming FOMC meeting date and days away."""
    today = date.today()
    future = [d for d in _FOMC_DATES_2026 if d >= today]
    if not future:
        last = _FOMC_DATES_2026[-1]
        return {"date": last.strftime("%b %d, %Y"), "days_away": 0}
    nxt = future[0]
    return {
        "date": nxt.strftime("%b %d, %Y"),
        "days_away": (nxt - today).days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL SCENARIO KEYS
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO_KEYS = ["hold", "cut_25", "cut_50", "hike_25"]
SCENARIO_LABELS = {
    "hold":    "Fed Holds",
    "cut_25":  "Cut 25bp",
    "cut_50":  "Cut 50bp",
    "hike_25": "Hike 25bp",
}
_SCENARIO_DELTAS = {
    "cut_50":  -0.50,
    "cut_25":  -0.25,
    "hold":     0.00,
    "hike_25": +0.25,
}


# ─────────────────────────────────────────────────────────────────────────────
# ZQ PROBABILITY DERIVATION (pure, no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _derive_probabilities_from_implied_rate(implied_rate: float, current_rate: float) -> list[dict]:
    """
    Distribute probability across 4 scenarios using a normal distribution
    centred on (implied_rate - current_rate) with σ=0.15.
    """
    delta = implied_rate - current_rate
    sigma = 0.15
    scenario_deltas = [_SCENARIO_DELTAS[k] for k in SCENARIO_KEYS]
    raw = np.array([
        np.exp(-0.5 * ((delta - sd) / sigma) ** 2)
        for sd in scenario_deltas
    ])
    probs = raw / raw.sum()
    return [
        {
            "scenario": key,
            "prob": float(probs[i]),
            "implied_rate": implied_rate,
            "source": "yfinance",
        }
        for i, key in enumerate(SCENARIO_KEYS)
    ]


def _equal_weight_fallback() -> list[dict]:
    """Return equal 25% probability for all 4 scenarios."""
    return [
        {
            "scenario": key,
            "prob": 0.25,
            "implied_rate": None,
            "source": "fallback",
            "data_unavailable": True,
        }
        for key in SCENARIO_KEYS
    ]


# ─────────────────────────────────────────────────────────────────────────────
# FETCH ZQ PROBABILITIES  (cached, tiered)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=14400)
def fetch_zq_probabilities() -> list[dict]:
    """
    Derive 4-scenario Fed policy probabilities from Fed Funds Futures.

    Tier 1: yfinance ZQ=F (front-month generic)
    Tier 2: yfinance named contracts ZQH26, ZQK26, ZQM26
    Tier 3: equal-weight fallback (data_unavailable=True)
    """
    fedfunds_series = fetch_fred_series_safe("FEDFUNDS")
    if fedfunds_series is None or fedfunds_series.empty:
        return _equal_weight_fallback()
    current_rate = float(fedfunds_series.dropna().iloc[-1])

    # Tier 1 — generic front-month
    try:
        df = yf.download("ZQ=F", period="5d", interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        if df is not None and not df.empty and "Close" in df.columns:
            price = float(df["Close"].dropna().iloc[-1])
            implied = 100.0 - price
            return _derive_probabilities_from_implied_rate(implied, current_rate)
    except Exception:
        pass

    # Tier 2 — named contracts
    for ticker in ("ZQH26", "ZQK26", "ZQM26"):
        try:
            df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel("Ticker", axis=1)
            if df is not None and not df.empty and "Close" in df.columns:
                price = float(df["Close"].dropna().iloc[-1])
                implied = 100.0 - price
                result = _derive_probabilities_from_implied_rate(implied, current_rate)
                for r in result:
                    r["source"] = "yfinance"
                return result
        except Exception:
            continue

    # Tier 3 — fallback
    return _equal_weight_fallback()


# ─────────────────────────────────────────────────────────────────────────────
# FED RSS COMMUNICATIONS
# ─────────────────────────────────────────────────────────────────────────────

_FED_RSS_FEEDS = {
    "release": "https://www.federalreserve.gov/rss/releases.xml",
    "speech":  "https://www.federalreserve.gov/rss/speeches.xml",
}


def _parse_rss_feed(xml_text: str, source: str) -> list[dict]:
    """Parse Federal Reserve RSS XML text into a list of communication dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()

        if not title:
            continue

        # Parse date to sortable datetime
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub_date)
            date_str = dt.strftime("%Y-%m-%d")
            sort_key = dt.timestamp()
        except Exception:
            date_str = pub_date
            sort_key = 0.0

        items.append({
            "title": title,
            "date": date_str,
            "url": link,
            "source": source,
            "raw_text": description,
            "_sort_key": sort_key,
        })

    # Most recent first
    items.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in items:
        item.pop("_sort_key", None)
    return items


@st.cache_data(ttl=3600)
def fetch_fed_communications(max_items: int = 5) -> list[dict]:
    """
    Fetch and merge Fed press releases and speeches from official RSS feeds.
    Returns up to max_items most recent items, sorted by date descending.
    Falls back to [] on any error.
    """
    all_items = []
    for source, url in _FED_RSS_FEEDS.items():
        try:
            resp = requests.get(
                url,
                timeout=8,
                headers={"User-Agent": "NarrativeInvestingTool/1.0"},
            )
            resp.raise_for_status()
            all_items.extend(_parse_rss_feed(resp.text, source=source))
        except Exception:
            continue

    # Sort merged list by date descending, return top max_items
    all_items.sort(key=lambda x: x["date"], reverse=True)
    return all_items[:max_items]
