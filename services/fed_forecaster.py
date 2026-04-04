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
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from services.market_data import fetch_fred_series_safe


def _groq_post_with_retry(
    headers: dict,
    payload: dict,
    timeout: int = 60,
    max_retries: int = 4,
) -> requests.Response:
    """POST to GROQ_API_URL with exponential backoff on 429 / 5xx.

    Waits 2, 4, 8, 16 seconds between retries (doubles each time).
    Raises on final failure so callers can catch and log cleanly.
    """
    import time
    delay = 2
    for attempt in range(max_retries):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()  # final raise if all retries exhausted
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# FOMC / CPI / NFP CALENDARS  — auto-fetched, hardcoded 2026 as fallback
# ─────────────────────────────────────────────────────────────────────────────

# Fallback dates (used when live fetch fails)
_FOMC_DATES_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 10),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]
_CPI_DATES_2026 = [
    date(2026, 1, 15), date(2026, 2, 12), date(2026, 3, 12), date(2026, 4, 10),
    date(2026, 5, 13), date(2026, 6, 11), date(2026, 7, 15), date(2026, 8, 12),
    date(2026, 9, 10), date(2026, 10, 13), date(2026, 11, 12), date(2026, 12, 10),
]
_NFP_DATES_2026 = [
    date(2026, 1, 9), date(2026, 2, 6), date(2026, 3, 6), date(2026, 4, 3),
    date(2026, 5, 8), date(2026, 6, 5), date(2026, 7, 10), date(2026, 8, 7),
    date(2026, 9, 4), date(2026, 10, 2), date(2026, 11, 6), date(2026, 12, 4),
]


@st.cache_data(ttl=86400)
def _fetch_fomc_dates() -> list:
    """Fetch FOMC meeting dates from the Fed's iCal feed. Falls back to hardcoded 2026 list."""
    try:
        resp = requests.get(
            "https://www.federalreserve.gov/apps/fomccalendar/fomccalendar.ics",
            timeout=8, headers={"User-Agent": "NarrativeInvestingTool/1.0"},
        )
        if not resp.ok:
            return _FOMC_DATES_2026
        dates = []
        for line in resp.text.splitlines():
            line = line.strip()
            # iCal DTSTART lines: DTSTART;VALUE=DATE:20260128 or DTSTART:20260128
            if line.startswith("DTSTART"):
                val = line.split(":")[-1].strip()
                if len(val) == 8 and val.isdigit():
                    try:
                        dates.append(date(int(val[:4]), int(val[4:6]), int(val[6:8])))
                    except ValueError:
                        pass
        dates = sorted(set(dates))
        return dates if dates else _FOMC_DATES_2026
    except Exception:
        return _FOMC_DATES_2026


def _calc_cpi_dates(months_ahead: int = 18) -> list:
    """Calculate CPI release dates algorithmically (second Wednesday of each month).
    BLS typically releases CPI on the second or third Wednesday; second Wednesday
    is the most common pattern. Uses 2026 verified dates for current year."""
    today = date.today()
    # Use verified 2026 dates for this year, calculate forward for future years
    verified = {(d.year, d.month): d for d in _CPI_DATES_2026}
    result = []
    for i in range(months_ahead):
        month = (today.month - 1 + i) % 12 + 1
        year = today.year + (today.month - 1 + i) // 12
        if (year, month) in verified:
            result.append(verified[(year, month)])
        else:
            # Second Wednesday of the month
            first_day = date(year, month, 1)
            first_wed = first_day + __import__("datetime").timedelta(days=(2 - first_day.weekday()) % 7)
            second_wed = first_wed + __import__("datetime").timedelta(weeks=1)
            result.append(second_wed)
    return sorted(set(result))


def _calc_nfp_dates(months_ahead: int = 18) -> list:
    """Calculate NFP release dates algorithmically (first Friday of each month).
    Uses 2026 verified dates for current year, calculates forward for future years."""
    today = date.today()
    verified = {(d.year, d.month): d for d in _NFP_DATES_2026}
    result = []
    for i in range(months_ahead):
        month = (today.month - 1 + i) % 12 + 1
        year = today.year + (today.month - 1 + i) // 12
        if (year, month) in verified:
            result.append(verified[(year, month)])
        else:
            # First Friday of the month
            first_day = date(year, month, 1)
            first_fri = first_day + __import__("datetime").timedelta(days=(4 - first_day.weekday()) % 7)
            result.append(first_fri)
    return sorted(set(result))


def _next_event(dates: list) -> dict:
    """Return next upcoming date from a list and days away."""
    today = date.today()
    future = [d for d in dates if d >= today]
    if not future:
        last = max(dates)
        return {"date": last.strftime("%b %d, %Y"), "days_away": 0}
    nxt = future[0]
    return {"date": nxt.strftime("%b %d, %Y"), "days_away": (nxt - today).days}


def get_next_fomc() -> dict:
    """Return the next upcoming FOMC meeting date and days away (live from Fed iCal)."""
    return _next_event(_fetch_fomc_dates())


def get_next_cpi() -> dict:
    """Return the next CPI release date and days away."""
    return _next_event(_calc_cpi_dates())


def get_next_nfp() -> dict:
    """Return the next NFP (jobs) release date and days away."""
    return _next_event(_calc_nfp_dates())


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

# Asset groupings for expanded forecast
ASSET_GROUPS = {
    "us_equities":   ["spy", "qqq", "iwm", "dji"],
    "bonds":         ["bonds_long", "bonds_short"],
    "commodities":   ["oil", "natgas", "gold", "silver"],
    "international": ["china", "india", "japan", "europe"],
    "usd":           ["usd"],
}

ASSET_LABELS = {
    "spy":        "SPY (S&P 500)",
    "qqq":        "QQQ (Nasdaq)",
    "iwm":        "IWM (Russell 2K)",
    "dji":        "DJI (Dow Jones)",
    "bonds_long": "TLT (30Y Long End)",
    "bonds_short": "SHY (2Y Short End)",
    "usd":        "DXY (Dollar Index)",
    "oil":        "WTI Crude Oil",
    "natgas":     "Natural Gas",
    "gold":       "Gold",
    "silver":     "Silver",
    "china":      "FXI (China)",
    "india":      "INDA (India)",
    "japan":      "EWJ (Japan)",
    "europe":     "VGK (Europe)",
}

BLACK_SWAN_EVENTS = {
    "war_escalation": "Major War Escalation (NATO/Russia or Taiwan Strait)",
    "hormuz_closure": "Strait of Hormuz Closure (oil supply shock)",
    "nuclear_event":  "Nuclear Event (detonation or credible use threat)",
    "hyperinflation": "Hyperinflation (US CPI > 20% annualized)",
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

def _parse_cme_fedwatch_json(data: dict, current_rate: float) -> list[dict] | None:
    """Parse CME FedWatch JSON response into scenario probability list."""
    try:
        # CME returns a list of meeting objects; pick the nearest upcoming meeting
        meetings = data.get("meetings") or data.get("MeetingDates") or []
        if not meetings:
            return None
        # Find first upcoming meeting
        from datetime import date as _date
        today_str = _date.today().isoformat()
        upcoming = [m for m in meetings if m.get("meetingDate", "") >= today_str]
        if not upcoming:
            upcoming = meetings
        meeting = upcoming[0]
        probs_raw = meeting.get("probabilities") or meeting.get("Probabilities") or {}
        if not probs_raw:
            return None
        scenario_map = {
            "hold":    ["Hold", "HOLD", "hold", "noChange", "no_change"],
            "cut_25":  ["Cut 25", "CUT_25", "cut25", "minus25", "Cut25"],
            "cut_50":  ["Cut 50", "CUT_50", "cut50", "minus50", "Cut50"],
            "hike_25": ["Hike 25", "HIKE_25", "hike25", "plus25", "Hike25"],
        }
        result = []
        for scenario, aliases in scenario_map.items():
            prob = 0.0
            for alias in aliases:
                if alias in probs_raw:
                    prob = float(probs_raw[alias])
                    if prob > 1.0:
                        prob /= 100.0
                    break
            result.append({"scenario": scenario, "prob": prob, "source": "CME FedWatch"})
        # Only return if probabilities sum to something reasonable
        total = sum(r["prob"] for r in result)
        if total < 0.5:
            return None
        # Normalize
        for r in result:
            r["prob"] = r["prob"] / total
        return result
    except Exception:
        return None


@st.cache_data(ttl=14400)
def fetch_zq_probabilities() -> list[dict]:
    """
    Derive 4-scenario Fed policy probabilities from Fed Funds Futures.

    Tier 0: CME FedWatch public JSON endpoint
    Tier 1: yfinance ZQ=F (front-month generic)
    Tier 2: yfinance named contracts ZQH26, ZQK26, ZQM26
    Tier 3: equal-weight fallback (data_unavailable=True)
    """
    fedfunds_series = fetch_fred_series_safe("FEDFUNDS")
    if fedfunds_series is None or fedfunds_series.empty:
        return _equal_weight_fallback()
    current_rate = float(fedfunds_series.dropna().iloc[-1])

    # Tier 0 — CME FedWatch direct
    _CME_URLS = [
        "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/V2/FedWatch.json",
        "https://www.cmegroup.com/CmeWS/mvc/ProductCalendar/FedWatch.json",
    ]
    for _cme_url in _CME_URLS:
        try:
            import requests as _req
            _resp = _req.get(_cme_url, timeout=4,
                             headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
            if _resp.status_code == 200:
                _parsed = _parse_cme_fedwatch_json(_resp.json(), current_rate)
                if _parsed:
                    return _parsed
        except Exception:
            pass

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

    # Tier 2 — named contracts (dynamically built so they don't expire)
    _ZQ_MONTH_CODES = "FGHJKMNQUVXZ"  # CME month codes Jan–Dec
    _today = date.today()
    _zq_tickers = []
    _yr, _mo = _today.year, _today.month
    for _ in range(4):
        _mo += 1
        if _mo > 12:
            _mo = 1
            _yr += 1
        _zq_tickers.append(f"ZQ{_ZQ_MONTH_CODES[_mo - 1]}{str(_yr)[-2:]}")

    for ticker in _zq_tickers:
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

    # Most recent first within this feed
    items.sort(key=lambda x: x["_sort_key"], reverse=True)
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

    # Sort merged list by numeric timestamp descending, strip internal key, truncate
    all_items.sort(key=lambda x: x["_sort_key"], reverse=True)
    for item in all_items:
        item.pop("_sort_key", None)
    return all_items[:max_items]


# ─────────────────────────────────────────────────────────────────────────────
# ADJUST PROBABILITIES (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _regime_probability_bias(macro: dict) -> dict[str, float]:
    """Return additive probability deltas (ideally sum near 0) based on macro regime.

    Logic:
    - Stagflation: Fed constrained — reduce cuts, increase hold/hike
    - Deflation: Growth + inflation both falling — cuts more likely
    - Goldilocks: Minimal adjustment (futures well-calibrated)
    - Severe risk-off stress (high VIX + wide credit): boost cut probability
    """
    quadrant = macro.get("quadrant", "Unknown")
    vix_z = macro.get("vix_z") or 0.0
    credit_z = macro.get("credit_z") or 0.0

    deltas: dict[str, float] = {"hold": 0.0, "cut_25": 0.0, "cut_50": 0.0, "hike_25": 0.0}

    if quadrant == "Stagflation":
        deltas["hold"]    += 0.08
        deltas["hike_25"] += 0.04
        deltas["cut_25"]  -= 0.07
        deltas["cut_50"]  -= 0.05
    elif quadrant == "Deflation":
        deltas["cut_25"]  += 0.05
        deltas["cut_50"]  += 0.03
        deltas["hold"]    -= 0.05
        deltas["hike_25"] -= 0.03
    # Goldilocks: no bias — futures already calibrated

    # Severe market stress → Fed more likely to cut emergency-style
    stress = max(float(vix_z), float(credit_z))
    if stress > 0.7:
        boost = min(0.10, stress * 0.12)
        deltas["cut_50"]  += boost
        deltas["cut_25"]  += boost * 0.5
        deltas["hold"]    -= boost * 1.0
        deltas["hike_25"] -= boost * 0.5

    return deltas


def adjust_probabilities(
    base_probs: list[dict],
    tone_result: dict,
    macro: dict | None = None,
) -> list[dict]:
    """
    Apply tone-derived and regime-derived probability adjustments to base ZQ probabilities.
    Clamps to [0, 1] then re-normalises to sum to 1.0.
    Adds a signed `delta` field to each item (tone-only delta for display).
    """
    adjustments = tone_result.get("prob_adjustments", {})
    regime_deltas = _regime_probability_bias(macro) if macro else {}

    adjusted = []
    for item in base_probs:
        key = item["scenario"]
        tone_adj = adjustments.get(key, 0.0)
        regime_adj = regime_deltas.get(key, 0.0)
        new_prob = max(0.0, min(1.0, item["prob"] + tone_adj + regime_adj))
        adjusted.append({**item, "prob": new_prob, "delta": tone_adj})

    # Re-normalise
    total = sum(r["prob"] for r in adjusted)
    if total > 0:
        for r in adjusted:
            r["prob"] = r["prob"] / total

    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# BAYESIAN PROBABILITY CALIBRATION (pure)
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_probabilities(
    market_implied_probs: list[dict],
    adj_probs: list[dict],
    context: dict,
) -> dict:
    """
    Bayesian ensemble calibration of Fed rate-path probabilities.

    Ensemble weights:
      40% market-implied (ZQ futures)
      40% structural (macro signal likelihood multipliers)
      20% narrative (tone-adjusted probs from adj_probs)

    Returns a dict with:
      posteriors: list of {scenario, market_pct, structural_pct, posterior_pct, band_pct, rationale}
      signals_used: dict of signal values
    """
    # ── Extract macro signals from context ───────────────────────────────────
    pce         = context.get("core_pce") or 0.0
    unemployment = context.get("unemployment") or 0.0
    vix_z       = context.get("vix_z") or 0.0        # normalised [-1, 1]
    credit_z    = context.get("credit_z") or 0.0
    regime_score = context.get("macro_score") or 50   # 0-100, 50 = neutral
    quadrant    = context.get("quadrant", "Unknown")

    # ── Build structural prior using likelihood multipliers ──────────────────
    # Start from market-implied as structural base, then apply multipliers
    structural = {r["scenario"]: r["prob"] for r in market_implied_probs}

    multipliers: dict[str, dict[str, float]] = {k: {} for k in SCENARIO_KEYS}

    # PCE signal
    if pce > 3.0:
        multipliers["hike_25"]["pce"] = 1.30
        multipliers["hold"]["pce"]    = 1.10
        multipliers["cut_25"]["pce"]  = 0.75
        multipliers["cut_50"]["pce"]  = 0.60
    elif pce > 2.5:
        multipliers["hike_25"]["pce"] = 1.15
        multipliers["cut_25"]["pce"]  = 0.90
        multipliers["cut_50"]["pce"]  = 0.80
    elif pce < 2.0:
        multipliers["cut_25"]["pce"]  = 1.20
        multipliers["cut_50"]["pce"]  = 1.30
        multipliers["hike_25"]["pce"] = 0.70

    # Unemployment signal
    if unemployment > 4.5:
        multipliers["cut_50"]["unemp"] = 1.35
        multipliers["cut_25"]["unemp"] = 1.20
        multipliers["hike_25"]["unemp"] = 0.60
    elif unemployment > 4.2:
        multipliers["cut_25"]["unemp"] = 1.15
        multipliers["cut_50"]["unemp"] = 1.10
    elif unemployment < 3.8:
        multipliers["hike_25"]["unemp"] = 1.15
        multipliers["cut_50"]["unemp"]  = 0.80

    # VIX z-score signal (positive = elevated stress)
    if vix_z > 0.5:
        multipliers["cut_25"]["vix"]  = 1.15
        multipliers["cut_50"]["vix"]  = 1.20
        multipliers["hike_25"]["vix"] = 0.70
    elif vix_z < -0.3:
        multipliers["hold"]["vix"]    = 1.10
        multipliers["hike_25"]["vix"] = 1.05

    # Credit spread z-score
    if credit_z > 0.5:
        multipliers["cut_50"]["cred"] = 1.20
        multipliers["cut_25"]["cred"] = 1.10
        multipliers["hike_25"]["cred"] = 0.65

    # Quadrant bias
    if quadrant == "Stagflation":
        multipliers["hold"]["quad"]    = 1.10
        multipliers["hike_25"]["quad"] = 1.08
        multipliers["cut_50"]["quad"]  = 0.75
    elif quadrant == "Deflation":
        multipliers["cut_50"]["quad"]  = 1.20
        multipliers["cut_25"]["quad"]  = 1.10
        multipliers["hike_25"]["quad"] = 0.70

    # Apply all multipliers
    for key in SCENARIO_KEYS:
        m = 1.0
        for v in multipliers[key].values():
            m *= v
        structural[key] = max(0.001, structural.get(key, 0.25) * m)

    # Normalise structural
    s_total = sum(structural.values())
    structural = {k: v / s_total for k, v in structural.items()}

    # ── Ensemble blend ────────────────────────────────────────────────────────
    mi_map  = {r["scenario"]: r["prob"] for r in market_implied_probs}
    adj_map = {r["scenario"]: r["prob"] for r in adj_probs}

    posterior = {}
    for key in SCENARIO_KEYS:
        mi  = mi_map.get(key, 0.25)
        st  = structural.get(key, 0.25)
        adj = adj_map.get(key, 0.25)
        posterior[key] = 0.40 * mi + 0.40 * st + 0.20 * adj

    # Normalise posterior
    p_total = sum(posterior.values())
    posterior = {k: v / p_total for k, v in posterior.items()}

    # ── Confidence bands ──────────────────────────────────────────────────────
    # Band width reflects disagreement between market-implied and structural
    bands = {}
    for key in SCENARIO_KEYS:
        disagreement = abs(mi_map.get(key, 0.25) - structural.get(key, 0.25))
        bands[key] = round(min(0.15, max(0.03, disagreement * 1.5)), 2)

    # ── Rationale per scenario ────────────────────────────────────────────────
    _rationale_map = {
        "hold":    _hold_rationale(pce, unemployment, vix_z, quadrant),
        "cut_25":  _cut_rationale(unemployment, credit_z, vix_z, False),
        "cut_50":  _cut_rationale(unemployment, credit_z, vix_z, True),
        "hike_25": _hike_rationale(pce, quadrant),
    }

    # ── Assemble output ───────────────────────────────────────────────────────
    posteriors = []
    for key in SCENARIO_KEYS:
        posteriors.append({
            "scenario":       key,
            "label":          SCENARIO_LABELS[key],
            "market_pct":     round(mi_map.get(key, 0.25) * 100, 1),
            "structural_pct": round(structural.get(key, 0.25) * 100, 1),
            "posterior_pct":  round(posterior[key] * 100, 1),
            "posterior":      posterior[key],
            "band_pct":       round(bands[key] * 100, 1),
            "rationale":      _rationale_map[key],
        })

    return {
        "posteriors": posteriors,
        "signals_used": {
            "core_pce":    round(pce, 2) if pce else None,
            "unemployment": round(unemployment, 1) if unemployment else None,
            "vix_z":       round(vix_z, 2),
            "credit_z":    round(credit_z, 2),
            "quadrant":    quadrant,
        },
    }


def _hold_rationale(pce: float, unemp: float, vix_z: float, quadrant: str) -> str:
    if pce > 2.8 and unemp < 4.5:
        return f"PCE sticky at {pce:.1f}% with contained unemployment limits Fed's ability to cut."
    if quadrant == "Stagflation":
        return "Stagflation quadrant constrains policy — cutting risks inflation, hiking risks recession."
    if vix_z > 0.3:
        return "Elevated market stress favours patience; Fed likely to hold and monitor conditions."
    return "Balanced signals support a data-dependent hold at the current meeting."


def _cut_rationale(unemp: float, credit_z: float, vix_z: float, large: bool) -> str:
    size = "50bp emergency" if large else "25bp"
    if unemp > 4.5:
        return f"Rising unemployment ({unemp:.1f}%) raises recession risk, supporting a {size} cut."
    if credit_z > 0.5:
        return f"Widening credit spreads signal financial stress, raising probability of a {size} cut."
    if vix_z > 0.6:
        return f"Elevated volatility and risk-off regime conditions increase odds of a {size} cut."
    return f"Softening growth indicators provide basis for a {size} cut if inflation cooperates."


def _hike_rationale(pce: float, quadrant: str) -> str:
    if pce > 3.0:
        return f"Core PCE at {pce:.1f}% — well above target — keeps re-acceleration risk on the table."
    if quadrant == "Stagflation":
        return "Stagflation quadrant: supply-side inflation could force tightening despite weak growth."
    return "Resilient labour market and sticky inflation leave a residual hike tail risk."


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FED CONTEXT (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_last(series) -> float | None:
    """Extract last non-null float from a pandas Series, or None."""
    if series is None:
        return None
    try:
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])
    except Exception:
        return None


def build_fed_context(macro: dict, fred_data: dict) -> dict:
    """
    Package current macro regime signals into a serialisable dict for the Groq prompt.
    Falls back gracefully when series are None. If fred_data["fedfunds"] is None,
    attempts a last-resort fetch via fetch_fred_series_safe("FEDFUNDS").
    """
    fedfunds_series = fred_data.get("fedfunds")
    fed_rate = _safe_last(fedfunds_series)
    if fed_rate is None:
        # Last-resort: try disk cache directly
        fallback = fetch_fred_series_safe("FEDFUNDS")
        fed_rate = _safe_last(fallback)

    # Extract z-scores from pre-computed top_signals (set by _build_macro_dashboard)
    top_signals = macro.get("top_signals", [])

    def _find_z(keyword: str) -> float | None:
        for s in top_signals:
            if keyword.lower() in s["name"].lower():
                return s["score"]
        return None

    return {
        "fed_funds_rate":    fed_rate,
        "core_pce":          _safe_last(fred_data.get("core_pce")),
        "unemployment":      _safe_last(fred_data.get("unrate")),
        "yield_curve":       _safe_last(fred_data.get("yield_curve")),
        "credit_spread":     _safe_last(fred_data.get("credit_spread")),
        "quadrant":          macro.get("quadrant", "Unknown"),
        "macro_score":       macro.get("macro_score", 50),
        "regime":            macro.get("macro_regime", "Unknown"),
        # Enriched regime signals (z-scores normalized [-1, 1])
        "vix_z":             _find_z("VIX"),
        "credit_z":          _find_z("Credit Spreads"),
        "equity_momentum_z": _find_z("Equity Trend"),
        "dollar_z":          _find_z("Dollar"),
        "commodity_z":       _find_z("Commodity Trend"),
        "leading_index_z":   _find_z("Leading Economic"),
        "hyg_lqd_z":         _find_z("HYG/LQD"),
        "copper_gold_z":     _find_z("Copper/Gold"),
        "top_signals":       {s["name"]: s["score"] for s in top_signals},
    }

# ─────────────────────────────────────────────────────────────────────────────
# GROQ HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"
XAI_API_URL  = "https://api.x.ai/v1/chat/completions"


def _groq_headers() -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _strip_fences(text: str) -> str:
    """Extract JSON from LLM response, handling fences and preamble text."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n", 1)
        text = lines[1] if len(lines) > 1 else ""
        text = text.rstrip()
        if text.endswith("```"):
            text = text[:-3]
    text = text.strip()
    # If the model added preamble text, find the first { or [ and slice from there
    for char in ("{", "["):
        idx = text.find(char)
        if idx > 0:
            candidate = text[idx:]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    return text


def _safe_json_parse(raw: str) -> dict | None:
    """Parse LLM JSON output with multiple fallback strategies.

    Handles: markdown fences, preamble text, unescaped special chars in strings,
    truncated output. Returns None only if all strategies fail.
    """
    import re as _re

    if not raw or not raw.strip():
        return None

    # Strategy 1: strip fences and parse directly
    cleaned = _strip_fences(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract outermost {...} block and try again
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3: use regex to find all key-value pairs for simple flat schemas
    # Useful when the narrative string has unescaped quotes
    prob_match = _re.search(r'"probability_pct"\s*:\s*([\d.]+)', raw)
    narr_match = _re.search(r'"narrative"\s*:\s*"(.*?)"(?=\s*,\s*"|\s*})', raw, _re.DOTALL)
    impacts_match = _re.search(r'"asset_impacts"\s*:\s*(\{[^}]+\})', raw, _re.DOTALL)

    if prob_match and impacts_match:
        try:
            prob = float(prob_match.group(1))
            narrative = narr_match.group(1).strip() if narr_match else "See macro context."
            impacts = json.loads(impacts_match.group(1))
            return {"probability_pct": prob, "narrative": narrative, "asset_impacts": impacts}
        except Exception:
            pass

    return None


def _neutral_tone_fallback() -> dict:
    return {
        "items": [],
        "aggregate_bias": "neutral",
        "prob_adjustments": {k: 0.0 for k in SCENARIO_KEYS},
    }


# ─────────────────────────────────────────────────────────────────────────────
# FED TONE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_tone(communications: list[dict]) -> dict:
    """
    Internal: call Groq to score Fed communication tone.
    Not cached — called by the cached score_fed_tone wrapper.
    """
    if not communications:
        return _neutral_tone_fallback()

    items_text = "\n\n".join(
        f"[{i+1}] {c['title']} ({c['date']})\n{c['raw_text']}"
        for i, c in enumerate(communications)
    )

    prompt = f"""You are a Federal Reserve communication analyst. Score the following Fed statements for monetary policy tone.

Statements:
{items_text}

Return ONLY valid JSON (no markdown fences) matching this schema:
{{
  "items": [
    {{
      "title": "<title>",
      "hawkish_prob": <0.0-1.0>,
      "neutral_prob": <0.0-1.0>,
      "dovish_prob": <0.0-1.0>,
      "adjustment_confidence": <0.0-1.0>
    }}
  ],
  "aggregate_bias": "hawkish" | "neutral" | "dovish",
  "prob_adjustments": {{
    "hold": <-0.15 to +0.15>,
    "cut_25": <-0.15 to +0.15>,
    "cut_50": <-0.15 to +0.15>,
    "hike_25": <-0.15 to +0.15>
  }}
}}

Rules:
- hawkish_prob + neutral_prob + dovish_prob must sum to 1.0 per item
- prob_adjustments must sum to 0 (they are redistributions)
- Hawkish = signals higher rates or holding; dovish = signals cuts
- adjustment_confidence is how confident you are in the adjustment (0=none, 1=certain)"""

    try:
        resp = _groq_post_with_retry(
            headers=_groq_headers(),
            payload={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = _strip_fences(resp.json()["choices"][0]["message"]["content"])
        return json.loads(text)
    except Exception:
        return _neutral_tone_fallback()


@st.cache_data(ttl=3600)
def score_fed_tone(comm_key: str, _communications: list[dict]) -> dict:
    """
    Score Fed communications tone via Groq.
    comm_key: stable hash of [(title, date)] — used as cache discriminator.
    _communications: leading underscore = Streamlit skips hashing this arg.
    """
    return _call_groq_tone(_communications)


# ─────────────────────────────────────────────────────────────────────────────
# FORECAST GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _call_groq_forecast(context_json: str, scenarios_json: str) -> dict | None:
    """
    Internal: single Groq call covering all 4 scenarios, both time horizons.
    Not cached — called by the cached generate_forecast wrapper.
    Returns parsed dict or None on failure.
    """
    try:
        context = json.loads(context_json)
        scenarios = json.loads(scenarios_json)

        scenarios_text = "\n".join(
            f"- {SCENARIO_LABELS[s['scenario']]}: {s['prob']*100:.0f}%"
            for s in scenarios
        )

        prompt = f"""You are a senior macro strategist. Given the current economic regime and Fed policy scenarios below, provide a probability-weighted forecast for 4 asset classes.

CURRENT REGIME:
- Fed Funds Rate: {context.get('fed_funds_rate', 'N/A')}%
- Core PCE Inflation: {context.get('core_pce', 'N/A')}%
- Unemployment: {context.get('unemployment', 'N/A')}%
- Yield Curve (10Y-2Y): {context.get('yield_curve', 'N/A')}%
- Credit Spread (HY): {context.get('credit_spread', 'N/A')}%
- Dalio Quadrant: {context.get('quadrant', 'N/A')}
- Macro Score: {context.get('macro_score', 'N/A')}/100 ({context.get('regime', 'N/A')})

FED POLICY SCENARIOS (market-implied probabilities):
{scenarios_text}

Return ONLY valid JSON (no markdown fences) with this EXACT structure for all 4 scenarios (hold, cut_25, cut_50, hike_25):

{{
  "near_term": {{
    "<scenario_key>": {{
      "equities":    {{"direction": "up|down|flat", "magnitude_low": <float>, "magnitude_high": <float>, "direction_prob": <0-1>, "magnitude_confidence": <0-1>, "chain": [{{"step": "<text>", "confidence": <0-1>}}]}},
      "bonds":       {{...same structure...}},
      "commodities": {{...same structure...}},
      "usd":         {{...same structure...}}
    }}
  }},
  "medium_term": {{
    "<scenario_key>": {{
      "equities":    {{"monthly_p25": [<12 floats>], "monthly_p50": [<12 floats>], "monthly_p75": [<12 floats>], "narrative": "<1-2 sentences>"}},
      "bonds":       {{...same structure...}},
      "commodities": {{...same structure...}},
      "usd":         {{...same structure...}}
    }}
  }},
  "causal_chains": {{
    "<scenario_key>": [{{"step": "<text>", "confidence": <0-1 cumulative decay>}}]
  }}
}}

Rules:
- magnitude values are percentage returns (e.g. -8.0 means -8%)
- chain confidence values are CUMULATIVE (each hop lower than the previous)
- monthly arrays have EXACTLY 12 values (months 1-12 from today), cumulative % returns
- p25 < p50 < p75 for each month
- Use scenario keys: hold, cut_25, cut_50, hike_25"""

        resp = _groq_post_with_retry(
            headers=_groq_headers(),
            payload={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        text = _strip_fences(resp.json()["choices"][0]["message"]["content"])
        return json.loads(text)
    except Exception:
        return None


def _call_groq_core_forecast(context_json: str, scenarios_json: str) -> dict:
    """Call Groq for US equities, bonds, USD across all 3 time horizons + causal chains.

    Returns dict keyed by scenario (hold/cut_25/cut_50/hike_25), each containing:
      - spy, qqq, iwm, dji, bonds_long, bonds_short, usd: each with
          near_term: list of 7 floats (daily % change, days 1-7)
          medium_term: list of 12 floats (monthly % change, months 1-12)
          long_term: list of 8 floats (quarterly % change, Q1-Q8, 2-year horizon)
      - causal_chain: list of strings (≥2 after post-processing)
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    # Build regime signal block from enriched context
    _ctx = json.loads(context_json)
    _fmt = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) and v is not None else "n/a"
    _regime_block = (
        f"CURRENT MACRO REGIME (z-scores normalized -1 to +1, where ±1 = historically extreme):\n"
        f"- Quadrant: {_ctx.get('quadrant', 'Unknown')} | Regime: {_ctx.get('regime', 'Unknown')} | Score: {_ctx.get('macro_score', 50)}/100\n"
        f"- VIX z-score: {_fmt(_ctx.get('vix_z'))}  (positive = elevated fear/risk-off, negative = complacency)\n"
        f"- Credit spread z-score: {_fmt(_ctx.get('credit_z'))}  (positive = stress/widening, negative = tight)\n"
        f"- Equity momentum z-score: {_fmt(_ctx.get('equity_momentum_z'))}  (positive = uptrend, negative = downtrend)\n"
        f"- Dollar z-score: {_fmt(_ctx.get('dollar_z'))}  (positive = strong USD)\n"
        f"- Commodity trend z-score: {_fmt(_ctx.get('commodity_z'))}\n"
        f"- Leading indicators z-score: {_fmt(_ctx.get('leading_index_z'))}  (positive = expansion, negative = contraction)\n"
        f"- HYG/LQD credit appetite z-score: {_fmt(_ctx.get('hyg_lqd_z'))}\n"
        f"- Copper/Gold ratio z-score: {_fmt(_ctx.get('copper_gold_z'))}  (positive = growth bias, negative = safety bias)\n"
        "Use these signals to calibrate the DIRECTION and MAGNITUDE of each asset's response per scenario.\n"
        "Example: If VIX z-score is +0.8 (elevated fear) and credit spreads are +0.6 (widening),\n"
        "  a rate cut will be more stimulative to equities than in calm conditions.\n"
        "If quadrant is Stagflation (falling growth, rising inflation), bonds may not rally on cuts.\n\n"
    )

    prompt = (
        "You are a macro-economist. Return ONLY valid json (no commentary).\n\n"
        f"Given this macro context:\n{context_json}\n\n"
        f"{_regime_block}"
        f"And these FOMC scenarios:\n{scenarios_json}\n\n"
        "Return a JSON object with keys: hold, cut_25, cut_50, hike_25.\n"
        "Each scenario maps to an object with keys: spy, qqq, iwm, dji, bonds_long, bonds_short, usd, causal_chain.\n\n"
        "Each asset (except causal_chain) has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N (day 1-7)\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N (month 1-12)\n'
        '  "long_term": array of exactly 8 floats — CUMULATIVE % change from today through quarter N (Q1-Q8, 2 years)\n\n'
        "IMPORTANT magnitude guidance:\n"
        "- near_term (7 days): typical equity moves are -2% to +2% cumulative\n"
        "- medium_term (12 months): typical equity moves are -15% to +25% cumulative\n"
        "- long_term (8 quarters): typical equity moves are -30% to +40% cumulative\n"
        "- Bonds move less than equities. USD moves less than bonds.\n"
        "- A 25bp cut is meaningful — SPY might gain +3 to +8% over 6 months.\n"
        "- A 50bp cut is very stimulative — SPY might gain +8 to +15% over 6 months.\n"
        "- A hike is contractionary — SPY might lose -5 to -15% over 6 months.\n\n"
        '"causal_chain" is an array of AT LEAST 5 strings describing the Fed policy transmission mechanism.\n'
        "Example causal_chain for cut_25:\n"
        '["Fed cuts 25bp → fed funds target drops","Lower short rates reduce borrowing costs",'
        '"Credit conditions loosen → business investment rises",'
        '"Consumer spending picks up on lower mortgage/card rates",'
        '"Corporate earnings expand → equity multiples re-rate higher"]\n\n'
        "bonds_long = 30-year Treasury / TLT proxy\n"
        "bonds_short = 2-year Treasury / SHY proxy\n"
    )

    headers = _groq_headers()
    payload = {
        "model": GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a macro-economist. Return only valid json."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
    }
    resp = _groq_post_with_retry(headers=headers, payload=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(_strip_fences(raw))

    # Post-process: ensure causal chains have ≥2 steps
    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        chain = data.get(scenario_key, {}).get("causal_chain", [])
        if not chain:
            delta = _SCENARIO_DELTAS.get(scenario_key, 0.0)
            data.setdefault(scenario_key, {})["causal_chain"] = [
                f"Fed {scenario_label} → policy rate shifts {delta:+.2f}%",
                "Rate change transmits to credit markets over 3–6 months",
            ]
    return data


def _call_claude_core_forecast(context_json: str, scenarios_json: str, model: str = "grok-4-1-fast-reasoning") -> dict:
    """Use Claude for higher-quality causal chains and asset impact reasoning.

    Same schema as _call_groq_core_forecast. Requires ANTHROPIC_API_KEY in env.
    model: grok-4-1-fast (fast/cheap) or claude-sonnet-4-6 (most accurate).
    """
    _is_grok = model and model.startswith("grok-")
    if _is_grok:
        api_key = os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("XAI_API_KEY not set")
    else:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    # Reuse the same enriched prompt logic as Groq version
    _ctx = json.loads(context_json)
    _fmt = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) and v is not None else "n/a"
    _regime_block = (
        f"CURRENT MACRO REGIME (z-scores normalized -1 to +1, where ±1 = historically extreme):\n"
        f"- Quadrant: {_ctx.get('quadrant', 'Unknown')} | Regime: {_ctx.get('regime', 'Unknown')} | Score: {_ctx.get('macro_score', 50)}/100\n"
        f"- VIX z-score: {_fmt(_ctx.get('vix_z'))}  (positive = elevated fear/risk-off, negative = complacency)\n"
        f"- Credit spread z-score: {_fmt(_ctx.get('credit_z'))}  (positive = stress/widening, negative = tight)\n"
        f"- Equity momentum z-score: {_fmt(_ctx.get('equity_momentum_z'))}  (positive = uptrend, negative = downtrend)\n"
        f"- Dollar z-score: {_fmt(_ctx.get('dollar_z'))}  (positive = strong USD)\n"
        f"- Commodity trend z-score: {_fmt(_ctx.get('commodity_z'))}\n"
        f"- Leading indicators z-score: {_fmt(_ctx.get('leading_index_z'))}  (positive = expansion, negative = contraction)\n"
        f"- HYG/LQD credit appetite z-score: {_fmt(_ctx.get('hyg_lqd_z'))}\n"
        f"- Copper/Gold ratio z-score: {_fmt(_ctx.get('copper_gold_z'))}  (positive = growth bias, negative = safety bias)\n"
        "Use these signals to calibrate the DIRECTION and MAGNITUDE of each asset's response per scenario.\n\n"
    )

    prompt = (
        "You are a senior macro-economist and portfolio strategist. Return ONLY valid JSON.\n\n"
        f"Given this macro context:\n{context_json}\n\n"
        f"{_regime_block}"
        f"And these FOMC scenarios:\n{scenarios_json}\n\n"
        "Return a JSON object with keys: hold, cut_25, cut_50, hike_25.\n"
        "Each scenario maps to an object with keys: spy, qqq, iwm, dji, bonds_long, bonds_short, usd, causal_chain.\n\n"
        "Each asset (except causal_chain) has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N (day 1-7)\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N (month 1-12)\n'
        '  "long_term": array of exactly 8 floats — CUMULATIVE % change from today through quarter N (Q1-Q8, 2 years)\n\n'
        "Magnitude guidance:\n"
        "- near_term (7 days): typical equity moves -2% to +2% cumulative\n"
        "- medium_term (12 months): typical equity moves -15% to +25% cumulative\n"
        "- long_term (8 quarters): typical equity moves -30% to +40% cumulative\n"
        "- Bonds move less than equities. USD moves less than bonds.\n"
        "- A 25bp cut is meaningful — SPY might gain +3 to +8% over 6 months.\n"
        "- A 50bp cut is very stimulative — SPY might gain +8 to +15% over 6 months.\n"
        "- A hike is contractionary — SPY might lose -5 to -15% over 6 months.\n\n"
        '"causal_chain" is an array of AT LEAST 7 strings describing the full Fed policy transmission mechanism.\n'
        "Be specific and nuanced. Reference the current macro regime in your reasoning.\n"
        "bonds_long = 30-year Treasury / TLT proxy\n"
        "bonds_short = 2-year Treasury / SHY proxy\n"
    )

    if _is_grok:
        from services.claude_client import _call_xai as _xai
        raw = _xai([{"role": "user", "content": prompt}], model, max_tokens=8192, temperature=0.3)
    else:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
    data = json.loads(_strip_fences(raw))

    # Post-process: ensure causal chains have ≥2 steps
    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        chain = data.get(scenario_key, {}).get("causal_chain", [])
        if not chain:
            delta = _SCENARIO_DELTAS.get(scenario_key, 0.0)
            data.setdefault(scenario_key, {})["causal_chain"] = [
                f"Fed {scenario_label} → policy rate shifts {delta:+.2f}%",
                "Rate change transmits to credit markets over 3–6 months",
            ]
    return data


def _call_groq_commodities_intl_forecast(context_json: str, scenarios_json: str) -> dict:
    """Call Groq for commodities and international equities (near + medium term).

    Returns dict keyed by scenario, each containing:
      Commodities (oil, natgas, gold, silver):
        near_term: list of 7 floats, medium_term: list of 12 floats
      International (china, india, japan, europe):
        near_term: list of 7 floats, medium_term: list of 12 floats
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    prompt = (
        "You are a macro-economist. Return ONLY valid json (no commentary).\n\n"
        f"Given this macro context:\n{context_json}\n\n"
        f"And these FOMC scenarios:\n{scenarios_json}\n\n"
        "Return a JSON object with keys: hold, cut_25, cut_50, hike_25.\n"
        "Each scenario maps to an object with these asset keys and structures:\n\n"
        "COMMODITIES (oil, natgas, gold, silver) — each has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N\n\n'
        "INTERNATIONAL EQUITIES (china, india, japan, europe) — each has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N\n\n'
        "IMPORTANT magnitude guidance:\n"
        "- near_term (7 days): typical moves are -3% to +3% cumulative\n"
        "- medium_term (12 months): commodities can move -25% to +35%, intl equities -20% to +30%\n"
        "- Oil and natgas are volatile. Gold is a safe haven (benefits from cuts). Silver follows gold.\n"
        "- International equities correlate with US but react to USD strength (strong USD hurts EM).\n\n"
        "Asset notes:\n"
        "- china=FXI, india=INDA, japan=EWJ, europe=VGK\n"
    )

    headers = _groq_headers()
    payload = {
        "model": GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a macro-economist. Return only valid json."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
    }
    resp = _groq_post_with_retry(headers=headers, payload=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    data = json.loads(_strip_fences(raw))

    # Post-process: flatten category wrappers like "COMMODITIES" / "INTERNATIONAL EQUITIES"
    # Groq sometimes nests assets under category keys instead of putting them flat
    _EXPECTED_ASSETS = {"oil", "natgas", "gold", "silver", "china", "india", "japan", "europe"}
    for scenario_key in list(data.keys()):
        sc = data[scenario_key]
        if not isinstance(sc, dict):
            continue
        # Check if assets are wrapped in category keys
        if not any(k in sc for k in _EXPECTED_ASSETS):
            # Flatten: merge all sub-dicts into the scenario level
            flattened = {}
            for category_key, category_val in list(sc.items()):
                if isinstance(category_val, dict):
                    flattened.update(category_val)
            if flattened:
                data[scenario_key] = flattened

    return data


def _call_claude_commodities_intl_forecast(context_json: str, scenarios_json: str, model: str = "grok-4-1-fast-reasoning") -> dict:
    """Use Claude for commodities and international equities forecast.

    Same schema as _call_groq_commodities_intl_forecast.
    model: grok-4-1-fast or claude-sonnet-4-6.
    """
    import anthropic

    _is_grok = model and model.startswith("grok-")
    if _is_grok:
        api_key = os.getenv("XAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("XAI_API_KEY not set")
    else:
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")

    _ctx = json.loads(context_json)
    _fmt = lambda v: f"{v:+.2f}" if isinstance(v, (int, float)) and v is not None else "n/a"
    _regime_block = (
        f"CURRENT MACRO REGIME:\n"
        f"- Quadrant: {_ctx.get('quadrant', 'Unknown')} | Regime: {_ctx.get('regime', 'Unknown')} | Score: {_ctx.get('macro_score', 50)}/100\n"
        f"- Dollar z-score: {_fmt(_ctx.get('dollar_z'))}  (positive = strong USD, hurts EM and commodities)\n"
        f"- Commodity trend z-score: {_fmt(_ctx.get('commodity_z'))}\n"
        f"- VIX z-score: {_fmt(_ctx.get('vix_z'))}  (positive = fear/risk-off → gold bullish)\n"
        f"- Credit spread z-score: {_fmt(_ctx.get('credit_z'))}\n"
        f"- Copper/Gold ratio z-score: {_fmt(_ctx.get('copper_gold_z'))}  (positive = growth, negative = safety)\n\n"
    )

    prompt = (
        "You are a senior macro-economist. Return ONLY valid JSON.\n\n"
        f"Given this macro context:\n{context_json}\n\n"
        f"{_regime_block}"
        f"And these FOMC scenarios:\n{scenarios_json}\n\n"
        "Return a JSON object with keys: hold, cut_25, cut_50, hike_25.\n"
        "Each scenario maps to an object with these asset keys:\n\n"
        "COMMODITIES (oil, natgas, gold, silver) — each has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N\n\n'
        "INTERNATIONAL EQUITIES (china, india, japan, europe) — each has:\n"
        '  "near_term": array of exactly 7 floats — CUMULATIVE % change from today through day N\n'
        '  "medium_term": array of exactly 12 floats — CUMULATIVE % change from today through month N\n\n'
        "Magnitude guidance:\n"
        "- near_term (7 days): typical moves -3% to +3% cumulative\n"
        "- medium_term (12 months): commodities -25% to +35%, intl equities -20% to +30%\n"
        "- Oil and natgas are volatile. Gold is a safe haven (benefits from cuts and risk-off).\n"
        "- Strong USD (positive dollar_z) hurts EM equities and commodity prices.\n"
        "- china=FXI, india=INDA, japan=EWJ, europe=VGK\n"
        "Use the macro regime context to calibrate direction and magnitude precisely.\n"
    )

    if _is_grok:
        from services.claude_client import _call_xai as _xai
        raw = _xai([{"role": "user", "content": prompt}], model, max_tokens=4096, temperature=0.3)
    else:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text
    data = json.loads(_strip_fences(raw))

    # Same flattening post-processor as Groq version
    _EXPECTED_ASSETS = {"oil", "natgas", "gold", "silver", "china", "india", "japan", "europe"}
    for scenario_key in list(data.keys()):
        sc = data[scenario_key]
        if not isinstance(sc, dict):
            continue
        if not any(k in sc for k in _EXPECTED_ASSETS):
            flattened = {}
            for category_key, category_val in list(sc.items()):
                if isinstance(category_val, dict):
                    flattened.update(category_val)
            if flattened:
                data[scenario_key] = flattened

    return data


def _call_groq_black_swan_forecast(context_json: str) -> dict:
    """Call Groq to estimate black swan event probabilities and asset impacts.

    Returns dict keyed by event name (war_escalation/hormuz_closure/nuclear_event/hyperinflation),
    each containing:
      probability_pct: float (0-100) — estimated annual probability
      asset_impacts: dict of asset_key → qualitative label
        Keys: spy, qqq, iwm, bonds_long, bonds_short, gold, oil, usd
        Values: one of "strongly bullish"/"bullish"/"neutral"/"bearish"/"strongly bearish"
      narrative: str — 1-2 sentences on transmission mechanism
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    events_desc = "\n".join(
        f"- {k}: {v}" for k, v in BLACK_SWAN_EVENTS.items()
    )
    prompt = (
        "You are a macro risk analyst. Return ONLY valid json (no commentary).\n\n"
        f"Macro context:\n{context_json}\n\n"
        "For each of these tail-risk events, estimate:\n"
        "1. probability_pct: estimated annual probability (float 0-100)\n"
        "2. asset_impacts: dict mapping each of "
        "[spy, qqq, iwm, bonds_long, bonds_short, gold, oil, usd] to one of: "
        '"strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"\n'
        "3. narrative: 1-2 sentences on transmission mechanism\n\n"
        f"Events:\n{events_desc}\n\n"
        "Return JSON with exactly these top-level keys: "
        "war_escalation, hormuz_closure, nuclear_event, hyperinflation"
    )

    headers = _groq_headers()
    payload = {
        "model": GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a macro risk analyst. Return only valid json."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0.3,
    }
    resp = _groq_post_with_retry(headers=headers, payload=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    result = _safe_json_parse(raw)
    if result is None:
        raise ValueError(f"Could not parse black swan JSON response: {raw[:200]}")
    return result


def _call_claude_black_swan_forecast(context_json: str, model: str = "grok-4-1-fast-reasoning") -> dict:
    """xAI/Claude version of black swan forecast. Same return schema as Groq version."""
    from services.claude_client import _call_xai as _xai

    _is_grok = model and model.startswith("grok-")

    events_desc = "\n".join(f"- {k}: {v}" for k, v in BLACK_SWAN_EVENTS.items())
    prompt = (
        "You are a macro risk analyst. Return ONLY valid JSON (no commentary).\n\n"
        f"Macro context:\n{context_json}\n\n"
        "For each of these tail-risk events, estimate:\n"
        "1. probability_pct: estimated annual probability (float 0-100)\n"
        "2. asset_impacts: dict mapping each of "
        "[spy, qqq, iwm, bonds_long, bonds_short, gold, oil, usd] to one of: "
        '"strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"\n'
        "3. narrative: 1-2 sentences on transmission mechanism\n\n"
        f"Events:\n{events_desc}\n\n"
        "Return JSON with exactly these top-level keys: "
        "war_escalation, hormuz_closure, nuclear_event, hyperinflation"
    )

    if _is_grok:
        # Use shared _call_xai — handles reasoning timeout (120s) and skips temperature
        raw = _xai([{"role": "user", "content": prompt}], model, max_tokens=2048, temperature=0.3)
    else:
        import anthropic as _ant
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = _ant.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model, max_tokens=2048, temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()

    result = _safe_json_parse(raw)
    if result is None:
        raise ValueError(f"Could not parse black swan JSON response: {raw[:200]}")
    return result


def _call_groq_custom_event_forecast(event_label: str, context_json: str) -> dict:
    """Forecast probability and asset impacts for a single user-defined black swan event.

    Returns dict with: probability_pct, narrative, asset_impacts (same schema as built-in events).
    Raises on API failure or unparseable response.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    prompt = (
        "You are a macro tail-risk analyst. Return ONLY valid json (no commentary).\n\n"
        f"Macro context:\n{context_json}\n\n"
        f"Custom black swan event: {event_label}\n\n"
        "Estimate:\n"
        "1. probability_pct: estimated annual probability (float 0-100)\n"
        "2. asset_impacts: dict mapping each of "
        "[spy, qqq, iwm, bonds_long, bonds_short, gold, oil, usd] to one of: "
        '"strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"\n'
        "3. narrative: 1-2 sentences on transmission mechanism and how it impacts markets\n\n"
        'Return JSON with exactly these keys: "probability_pct", "narrative", "asset_impacts"'
    )

    headers = _groq_headers()
    payload = {
        "model": GROQ_MODEL,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "You are a macro tail-risk analyst. Return only valid json."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.3,
    }
    resp = _groq_post_with_retry(headers=headers, payload=payload, timeout=60)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    result = _safe_json_parse(raw)
    if result is None:
        raise ValueError(f"Could not parse custom event JSON: {raw[:300]}")
    return result


def _call_claude_custom_event_forecast(event_label: str, context_json: str, model: str) -> dict:
    """xAI/Claude version of custom black swan forecast. Same return schema as Groq version.
    Raises on API failure or unparseable response.
    """
    from services.claude_client import _call_xai as _xai

    _is_grok = model and model.startswith("grok-")
    prompt = (
        "You are a macro tail-risk analyst. Return ONLY valid JSON (no commentary).\n\n"
        f"Macro context:\n{context_json}\n\n"
        f"Custom black swan event: {event_label}\n\n"
        "Estimate:\n"
        "1. probability_pct: estimated annual probability (float 0-100)\n"
        "2. asset_impacts: dict mapping each of "
        "[spy, qqq, iwm, bonds_long, bonds_short, gold, oil, usd] to one of: "
        '"strongly bullish", "bullish", "neutral", "bearish", "strongly bearish"\n'
        "3. narrative: 2-3 sentences on transmission mechanism and how it impacts markets\n\n"
        'Return JSON with exactly these keys: "probability_pct", "narrative", "asset_impacts"'
    )
    if _is_grok:
        # Use shared _call_xai — handles reasoning timeout (120s) and skips temperature
        raw = _xai([{"role": "user", "content": prompt}], model, max_tokens=1024, temperature=0.3)
    else:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
    result = _safe_json_parse(raw)
    if result is None:
        raise ValueError(f"Could not parse custom event JSON: {raw[:300]}")
    return result


@st.cache_data(ttl=14400)
def generate_forecast(context_json: str, scenarios_json: str) -> dict | None:
    """
    Generate probability-weighted Fed policy forecast via Groq.
    Both args are JSON strings (hashable by st.cache_data).
    Returns parsed forecast dict or None on failure.
    """
    return _call_groq_forecast(context_json, scenarios_json)


# ─────────────────────────────────────────────────────────────────────────────
# EXPANDED FORECAST ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

_CLAUDE_MODEL_MAP = {
    "grok":   "grok-4-1-fast-reasoning",   # 🧠 Regard Mode → xAI Grok 4.1
    "haiku":  "grok-4-1-fast-reasoning",   # legacy alias
    "sonnet": "claude-sonnet-4-6",         # 👑 Highly Regarded Mode → Anthropic Sonnet
}


@st.cache_data(ttl=14400)
def generate_matrix_forecast(context_json: str, scenarios_json: str, model_tier: str = "groq") -> dict:
    """Run the 2 calls needed for the asset impact matrix + medium-term fan charts.

    model_tier: "groq" (free/fast), "grok" (Grok 4.1), "sonnet" (Claude Sonnet, most accurate).

    Returns: near_term, medium_term, long_term, _call_status, _core_engine
    """
    result: dict = {
        "near_term": {},
        "medium_term": {},
        "long_term": {},
        "_call_status": {"core": "ok", "commodities_intl": "ok"},
        "_core_engine": model_tier,
    }

    _CORE_ASSETS = ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]
    _COMM_ASSETS = ["oil", "natgas", "gold", "silver"]
    _INTL_ASSETS = ["china", "india", "japan", "europe"]
    _claude_model = _CLAUDE_MODEL_MAP.get(model_tier)

    try:
        if _claude_model and (os.getenv("XAI_API_KEY") if (_claude_model and _claude_model.startswith("grok-")) else os.getenv("ANTHROPIC_API_KEY")):
            core = _call_claude_core_forecast(context_json, scenarios_json, model=_claude_model)
        else:
            core = _call_groq_core_forecast(context_json, scenarios_json)
        for scenario in SCENARIO_KEYS:
            sc = core.get(scenario, {})
            result["near_term"].setdefault(scenario, {}).update(
                {k: sc[k]["near_term"] for k in _CORE_ASSETS if k in sc and "near_term" in sc[k]}
            )
            result["medium_term"].setdefault(scenario, {}).update(
                {k: sc[k]["medium_term"] for k in _CORE_ASSETS if k in sc and "medium_term" in sc[k]}
            )
            result["long_term"].setdefault(scenario, {}).update(
                {k: sc[k]["long_term"] for k in _CORE_ASSETS if k in sc and "long_term" in sc[k]}
            )
    except Exception as exc:
        result["_call_status"]["core"] = f"error: {exc}"

    # Call 2: Commodities + International — always Groq (not worth Grok quota for % estimates)
    try:
        comm = _call_groq_commodities_intl_forecast(context_json, scenarios_json)
        for scenario in SCENARIO_KEYS:
            sc = comm.get(scenario, {})
            result["near_term"].setdefault(scenario, {}).update(
                {k: sc[k]["near_term"] for k in _COMM_ASSETS + _INTL_ASSETS if k in sc and "near_term" in sc[k]}
            )
            result["medium_term"].setdefault(scenario, {}).update(
                {k: sc[k]["medium_term"] for k in _COMM_ASSETS + _INTL_ASSETS if k in sc and "medium_term" in sc[k]}
            )
    except Exception as exc:
        result["_call_status"]["commodities_intl"] = f"error: {exc}"

    return result


@st.cache_data(ttl=3600)
def generate_expanded_forecast(context_json: str, scenarios_json: str, model_tier: str = "groq") -> dict:
    """Orchestrate 3 calls and merge into unified expanded forecast dict.

    model_tier: "groq" (free/fast), "grok" (Grok 4.1), "sonnet" (Claude Sonnet, most accurate).

    Returns:
      near_term: dict[scenario][asset] = list of 7 floats
      medium_term: dict[scenario][asset] = list of 12 floats
      long_term: dict[scenario][asset] = list of 8 floats
      causal_chains: dict[scenario] = list of strings
      black_swans: dict[event_key] = {probability_pct, asset_impacts, narrative}
      _call_status: dict with "core", "commodities_intl", "black_swans" → "ok" or "error: ..."
      _core_engine: "claude" or "groq" — which LLM was used for core forecast
    """
    result: dict = {
        "near_term": {},
        "medium_term": {},
        "long_term": {},
        "causal_chains": {},
        "black_swans": {},
        "_call_status": {"core": "ok", "commodities_intl": "ok", "black_swans": "ok"},
        "_core_engine": model_tier,
    }

    _CORE_ASSETS = ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]
    _COMM_ASSETS = ["oil", "natgas", "gold", "silver"]
    _INTL_ASSETS = ["china", "india", "japan", "europe"]
    _claude_model = _CLAUDE_MODEL_MAP.get(model_tier)

    # Call 1: Core US assets (all 3 horizons + causal chains)
    try:
        if _claude_model and (os.getenv("XAI_API_KEY") if (_claude_model and _claude_model.startswith("grok-")) else os.getenv("ANTHROPIC_API_KEY")):
            core = _call_claude_core_forecast(context_json, scenarios_json, model=_claude_model)
        else:
            core = _call_groq_core_forecast(context_json, scenarios_json)
        for scenario in SCENARIO_KEYS:
            sc = core.get(scenario, {})
            result["near_term"].setdefault(scenario, {}).update(
                {k: sc[k]["near_term"] for k in _CORE_ASSETS if k in sc and "near_term" in sc[k]}
            )
            result["medium_term"].setdefault(scenario, {}).update(
                {k: sc[k]["medium_term"] for k in _CORE_ASSETS if k in sc and "medium_term" in sc[k]}
            )
            result["long_term"].setdefault(scenario, {}).update(
                {k: sc[k]["long_term"] for k in _CORE_ASSETS if k in sc and "long_term" in sc[k]}
            )
            result["causal_chains"][scenario] = sc.get("causal_chain", [])
    except Exception as exc:
        result["_call_status"]["core"] = f"error: {exc}"

    # Call 2: Commodities + International — always Groq (low-value % estimates, not worth burning Grok quota)
    try:
        comm = _call_groq_commodities_intl_forecast(context_json, scenarios_json)
        for scenario in SCENARIO_KEYS:
            sc = comm.get(scenario, {})
            result["near_term"].setdefault(scenario, {}).update(
                {k: sc[k]["near_term"] for k in _COMM_ASSETS + _INTL_ASSETS if k in sc and "near_term" in sc[k]}
            )
            result["medium_term"].setdefault(scenario, {}).update(
                {k: sc[k]["medium_term"] for k in _COMM_ASSETS + _INTL_ASSETS if k in sc and "medium_term" in sc[k]}
            )
    except Exception as exc:
        result["_call_status"]["commodities_intl"] = f"error: {exc}"

    # Call 3: Black Swans — route to xAI/Claude when model_tier is set
    try:
        if _claude_model and (os.getenv("XAI_API_KEY") if (_claude_model and _claude_model.startswith("grok-")) else os.getenv("ANTHROPIC_API_KEY")):
            result["black_swans"] = _call_claude_black_swan_forecast(context_json, model=_claude_model)
        else:
            result["black_swans"] = _call_groq_black_swan_forecast(context_json)
    except Exception as exc:
        result["_call_status"]["black_swans"] = f"error: {exc}"

    # Store context for custom event analysis
    result["_context_json"] = context_json

    return result
