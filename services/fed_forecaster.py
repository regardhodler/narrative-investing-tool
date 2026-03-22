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
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"


def _groq_headers() -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _strip_fences(text: str) -> str:
    """Extract JSON from Groq response, handling fences and preamble text."""
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
            # Try parsing from that position
            candidate = text[idx:]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass
    return text


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
        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            },
            timeout=20,
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

        resp = requests.post(
            GROQ_API_URL,
            headers=_groq_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
                "response_format": {"type": "json_object"},
            },
            timeout=45,
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
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
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


def _call_claude_core_forecast(context_json: str, scenarios_json: str) -> dict:
    """Use Claude Haiku for higher-quality causal chains and asset impact reasoning.

    Same schema as _call_groq_core_forecast. Requires ANTHROPIC_API_KEY in env.
    Only used when the user explicitly enables Claude mode via session state.
    """
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

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
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


def _call_claude_commodities_intl_forecast(context_json: str, scenarios_json: str) -> dict:
    """Use Claude Haiku for commodities and international equities forecast.

    Same schema as _call_groq_commodities_intl_forecast.
    """
    import anthropic

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

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    return json.loads(_strip_fences(raw))


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

@st.cache_data(ttl=14400)
def generate_matrix_forecast(context_json: str, scenarios_json: str, use_claude: bool = False) -> dict:
    """Run the 2 calls needed for the asset impact matrix + medium-term fan charts.

    When use_claude=True and ANTHROPIC_API_KEY is set, uses Claude Haiku for
    both calls — better calibrated numbers from richer macro reasoning.

    Returns: near_term, medium_term, long_term, _call_status, _core_engine
    """
    result: dict = {
        "near_term": {},
        "medium_term": {},
        "long_term": {},
        "_call_status": {"core": "ok", "commodities_intl": "ok"},
        "_core_engine": "groq",
    }

    _CORE_ASSETS = ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]
    _COMM_ASSETS = ["oil", "natgas", "gold", "silver"]
    _INTL_ASSETS = ["china", "india", "japan", "europe"]

    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
            core = _call_claude_core_forecast(context_json, scenarios_json)
            result["_core_engine"] = "claude"
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

    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
            comm = _call_claude_commodities_intl_forecast(context_json, scenarios_json)
        else:
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


@st.cache_data(ttl=14400)
def generate_expanded_forecast(context_json: str, scenarios_json: str, use_claude: bool = False) -> dict:
    """Orchestrate 3 calls and merge into unified expanded forecast dict.

    When use_claude=True and ANTHROPIC_API_KEY is set, uses Claude Haiku for
    the core forecast (better causal chains + reasoning). Falls back to Groq.

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
        "_core_engine": "groq",
    }

    _CORE_ASSETS = ["spy", "qqq", "iwm", "dji", "bonds_long", "bonds_short", "usd"]
    _COMM_ASSETS = ["oil", "natgas", "gold", "silver"]
    _INTL_ASSETS = ["china", "india", "japan", "europe"]

    # Call 1: Core US assets (all 3 horizons + causal chains)
    try:
        if use_claude and os.getenv("ANTHROPIC_API_KEY"):
            core = _call_claude_core_forecast(context_json, scenarios_json)
            result["_core_engine"] = "claude"
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

    # Call 2: Commodities + International
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

    # Call 3: Black Swans
    try:
        result["black_swans"] = _call_groq_black_swan_forecast(context_json)
    except Exception as exc:
        result["_call_status"]["black_swans"] = f"error: {exc}"

    return result
