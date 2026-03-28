"""
Module 0: Macro Dashboard

Daily macro regime indicator using 21 cross-asset signals:
- FRED macro series (yield curve, credit spreads, ISM, FCI, jobless claims, LEI, etc.)
- ETF proxies (equities, commodities, FX, volatility, credit ratios)
- SPY options chain (dealer gamma positioning)

Output:
- Risk-On / Neutral / Risk-Off verdict with Macro Score (0-100)
- Ray Dalio quadrant (Goldilocks / Reflation / Stagflation / Deflation)
- Valuation (CAPE)
- Cycle stage (CAPEX vs Liquidity)
- Portfolio bias by asset class
- SPY gamma sentiment (zone, flip, call wall, put wall)

Architecture:
- 21-indicator scoring engine (_build_macro_dashboard)
- Daily regime history persistence (JSON snapshots)
- Shared data layer via services/market_data.py
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from services.market_data import (
    fetch_batch_safe, AssetSnapshot,
    fetch_fred_series_safe, fetch_options_chain_snapshot_safe,
    warm_fred_cache,
)
from utils.theme import COLORS, apply_dark_layout, bloomberg_metric

# ─────────────────────────────────────────────
# HISTORY PERSISTENCE
# ─────────────────────────────────────────────

_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "regime_history.json")

# Gist-based persistence (set REGIME_GIST_ID + GIST_TOKEN in .env / Streamlit secrets)
_REGIME_GIST_ID  = os.getenv("REGIME_GIST_ID", "")
_REGIME_GIST_RAW = os.getenv("REGIME_GIST_RAW_URL", "")   # optional fast-read URL
_GIST_TOKEN      = (os.getenv("GIST_TOKEN") or os.getenv("GITHUB_GIST_TOKEN") or "").strip()
_GIST_FILENAME   = "regime_history.json"


def _gist_raw_url() -> str:
    """Derive the raw read URL from env or construct it from REGIME_GIST_ID."""
    if _REGIME_GIST_RAW:
        return _REGIME_GIST_RAW
    if _REGIME_GIST_ID:
        return f"https://gist.githubusercontent.com/raw/{_REGIME_GIST_ID}/{_GIST_FILENAME}"
    return ""


@st.cache_data(ttl=60)
def _load_history() -> list[dict]:
    """Load regime history — Gist first, local file as fallback."""
    import requests as _req

    # 1. Try Gist
    raw_url = _gist_raw_url()
    if raw_url:
        try:
            resp = _req.get(raw_url, timeout=8, headers={"User-Agent": "NarrativeInvestingTool/1.0",
                                                          "Cache-Control": "no-cache"})
            if resp.ok:
                return resp.json()
        except Exception:
            pass

    # 2. Local file fallback
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_snapshot(macro: dict):
    """Persist today's regime snapshot — writes to Gist (if configured) and local file."""
    import requests as _req

    today = datetime.now().strftime("%Y-%m-%d")
    score_normalized = (macro["macro_score"] - 50) / 50.0

    history = _load_history()
    history = [h for h in history if h.get("date") != today]
    history.append({
        "date": today,
        "score": round(score_normalized, 3),
        "regime": macro["macro_regime"],
        "signal_count": len(macro["signals"]),
        "macro_score": macro["macro_score"],
        "quadrant": macro["quadrant"],
        "growth_signal": macro.get("growth_signal"),
        "inflation_signal": macro.get("inflation_signal"),
        "signals_summary": {s["Indicator"]: s["Score"] for s in macro["signals"]},
    })

    # Keep last 730 days max (2 years)
    history = sorted(history, key=lambda x: x["date"])[-730:]
    payload_str = json.dumps(history, indent=2)

    # 1. Write to Gist (primary — persists across Streamlit Cloud deploys)
    if _REGIME_GIST_ID and _GIST_TOKEN:
        try:
            _req.patch(
                f"https://api.github.com/gists/{_REGIME_GIST_ID}",
                json={"files": {_GIST_FILENAME: {"content": payload_str}}},
                headers={
                    "Authorization": f"Bearer {_GIST_TOKEN}",
                    "Accept": "application/vnd.github+json",
                },
                timeout=10,
            )
        except Exception:
            pass

    # 2. Write local file (fallback / local dev)
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    with open(_HISTORY_FILE, "w") as f:
        f.write(payload_str)

    _load_history.clear()


# ─────────────────────────────────────────────
# TICKER UNIVERSE
# ─────────────────────────────────────────────

CORE_TICKERS = {
    # Used in _build_macro_dashboard signal calculations
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones ETF",
    "USO": "Oil (WTI)",
    "CPER": "Copper",
    "UUP": "USD Bull ETF",
    "^VIX": "VIX",
    "HYG": "High Yield Corp",
    "LQD": "Inv Grade Corp",
    # Display-only (ticker bar)
    "IWM": "Russell 2000",
    "GLD": "Gold",
    "SLV": "Silver",
    "TLT": "20Y+ Treasury",
}


# ─────────────────────────────────────────────
# SCORING UTILITIES
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _label_from_score(score: float) -> str:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "Risk-On"
    elif macro_score <= 40:
        return "Risk-Off"
    return _neutral_lean_label(macro_score)


def _score_to_bucket(score: float) -> tuple[str, str]:
    macro_score = int(round((score + 1.0) * 50))
    if macro_score >= 60:
        return "🟢", "Risk-On"
    if macro_score <= 40:
        return "🔴", "Risk-Off"
    return "🟡", _neutral_lean_label(macro_score)


def _safe_latest(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    return float(series.dropna().iloc[-1]) if len(series.dropna()) else None


def _series_trend(series: pd.Series | None, lookback: int = 22) -> float | None:
    """Return the change (latest - lookback-ago) for a FRED series, or None."""
    if series is None:
        return None
    clean = series.dropna()
    if len(clean) < lookback + 1:
        return None
    return float(clean.iloc[-1] - clean.iloc[-lookback - 1])


def _yoy_latest(series: pd.Series | None, periods: int = 12) -> float | None:
    if series is None or len(series.dropna()) <= periods:
        return None
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    base = clean.iloc[-periods - 1]
    if base == 0:
        return None
    return float((clean.iloc[-1] / base - 1) * 100)


def _clamp_score(value: float, scale: float) -> float:
    return _clamp(value / max(scale, 1e-6))


def _zscore_score(series: pd.Series | None, invert: bool = False, lookback: int = 252) -> float:
    """Convert a FRED/market series to a [-1, 1] score via z-score normalization.

    Uses the series' own historical distribution (lookback window) rather than
    hardcoded thresholds. Z-scores are divided by 2 so ±2σ maps to ±1.
    """
    if series is None:
        return 0.0
    clean = series.dropna()
    n = min(lookback, len(clean))
    if n < 20:
        return 0.0
    window = clean.iloc[-n:]
    std = float(window.std())
    if std < 1e-9:
        return 0.0
    z = (float(clean.iloc[-1]) - float(window.mean())) / std
    if invert:
        z = -z
    return _clamp(z / 2.0)


def _yoy_series(series: pd.Series | None, periods: int = 12) -> pd.Series | None:
    """Compute rolling YoY % change series."""
    if series is None:
        return None
    clean = series.dropna()
    if len(clean) <= periods:
        return None
    return clean.pct_change(periods=periods).dropna() * 100


def _age_days(series: pd.Series | None) -> float | None:
    if series is None or series.empty:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    last_idx = clean.index[-1]
    try:
        last_ts = pd.Timestamp(last_idx)
        now = pd.Timestamp.utcnow().tz_localize(None)
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert(None)
        return max(0.0, float((now - last_ts).days))
    except Exception:
        return None


def _confidence_label(score: int) -> str:
    if score >= 75:
        return "High"
    if score >= 50:
        return "Medium"
    return "Low"


def _neutral_lean_label(score: int) -> str:
    """Three-tier neutral label based on macro score (0-100 scale)."""
    if score >= 53:
        return "Neutral — Leaning Risk-On"
    elif score <= 47:
        return "Neutral — Leaning Risk-Off"
    return "True Neutral"


def _confidence_from_age(series: pd.Series | None, expected_days: int, fallback: int = 35) -> int:
    days = _age_days(series)
    if days is None:
        return fallback
    ratio = min(days / max(expected_days, 1), 2.0)
    conf = int(round(100 - (ratio * 45)))
    return max(25, min(95, conf))


def _confidence_from_snap(*tickers: str, snaps: dict[str, AssetSnapshot]) -> int:
    vals = []
    for t in tickers:
        snap = snaps.get(t)
        if snap is None or snap.latest_price is None:
            vals.append(35)
        else:
            vals.append(60 if snap.stale else 90)
    return int(round(np.mean(vals))) if vals else 35


def _interpret_valuation(cape: float | None) -> str:
    if cape is None:
        return "Valuation data unavailable."
    if cape > 25:
        return "S&P 500 P/E is elevated versus long-run norms, implying lower forward return expectations."
    if cape < 18:
        return "S&P 500 P/E is below long-run highs, valuation risk appears more moderate."
    return "S&P 500 P/E is near historical midrange; valuation neither clearly cheap nor deeply stretched."


def _portfolio_bias(regime: str) -> dict[str, str]:
    if regime == "Risk-On":
        return {
            "Equities": "Overweight cyclical and broad beta exposure",
            "Bonds": "Underweight duration; prefer short/intermediate credit",
            "Commodities": "Moderate overweight (energy/industrial metals)",
            "Defensive": "Underweight cash/defensive sectors",
        }
    if regime == "Risk-Off":
        return {
            "Equities": "Underweight high beta; tilt quality and low volatility",
            "Bonds": "Overweight high-quality duration",
            "Commodities": "Selective exposure, prefer gold over cyclicals",
            "Defensive": "Overweight cash, defensives, and hedges",
        }
    return {
        "Equities": "Neutral with barbell (quality + selective cyclicals)",
        "Bonds": "Neutral duration with balanced IG exposure",
        "Commodities": "Neutral, tactical allocations only",
        "Defensive": "Moderate buffer via cash/defensive assets",
    }


def _sector_rotation_recs(quadrant: str, regime: str, snaps: dict[str, AssetSnapshot]) -> list[dict]:
    """Sector rotation recommendations based on macro quadrant and regime."""
    quadrant_map = {
        "Goldilocks": {
            "Favor": [("QQQ", "Nasdaq 100"), ("SPY", "S&P 500"), ("IWM", "Russell 2000"), ("EEM", "Emerging Markets")],
            "Avoid": [("GLD", "Gold"), ("TLT", "20Y+ Treasuries")],
        },
        "Reflation": {
            "Favor": [("XLE", "Energy"), ("CPER", "Copper"), ("EEM", "Emerging Markets"), ("USO", "Oil")],
            "Avoid": [("TLT", "20Y+ Treasuries"), ("SHY", "Short Treasuries")],
        },
        "Stagflation": {
            "Favor": [("GLD", "Gold"), ("TIP", "TIPS"), ("XLE", "Energy"), ("UUP", "US Dollar")],
            "Avoid": [("QQQ", "Nasdaq 100"), ("IWM", "Russell 2000"), ("JNK", "High Yield")],
        },
        "Deflation": {
            "Favor": [("TLT", "20Y+ Treasuries"), ("IEF", "10Y Treasuries"), ("GLD", "Gold"), ("LQD", "IG Bonds")],
            "Avoid": [("XLE", "Energy"), ("USO", "Oil"), ("JNK", "High Yield"), ("EEM", "Emerging Markets")],
        },
    }

    mapping = quadrant_map.get(quadrant, quadrant_map["Goldilocks"])
    recs = []

    # Risk-Off override: prepend GLD, TLT to Favor if not already present
    if regime == "Risk-Off":
        favor_tickers = [t for t, _ in mapping["Favor"]]
        prepend = []
        if "GLD" not in favor_tickers:
            prepend.append(("GLD", "Gold"))
        if "TLT" not in favor_tickers:
            prepend.append(("TLT", "20Y+ Treasuries"))
        mapping["Favor"] = prepend + mapping["Favor"]

    for action in ("Favor", "Avoid"):
        for ticker, label in mapping[action]:
            snap = snaps.get(ticker)
            momentum = snap.pct_change_30d if snap and snap.pct_change_30d is not None else None
            reason = f"{quadrant} regime favors this asset" if action == "Favor" else f"{quadrant} regime suggests caution"
            if regime == "Risk-Off" and action == "Favor" and ticker in ("GLD", "TLT"):
                reason = "Risk-Off safe haven"
            recs.append({
                "action": action,
                "ticker": ticker,
                "label": label,
                "momentum_30d": round(momentum, 2) if momentum is not None else None,
                "reason": reason,
            })

    return recs


def _risk_management_alerts(macro: dict, snaps: dict[str, AssetSnapshot]) -> list[str]:
    """Generate risk management alerts based on current conditions."""
    alerts = []
    score_map = {s["Indicator"]: s["Score"] for s in macro["signals"]}

    # VIX checks
    vix_snap = snaps.get("^VIX")
    vix_price = vix_snap.latest_price if vix_snap else None
    if vix_price is not None:
        if vix_price > 35:
            alerts.append(f"VIX at {vix_price:.1f} — crisis-level volatility. Consider hedging or reducing gross exposure.")
        elif vix_price > 25:
            alerts.append(f"VIX at {vix_price:.1f} — elevated volatility. Widen stops and reduce position sizes.")

    # Gamma checks
    gamma = macro.get("gamma")
    if gamma:
        spot_gamma_score = score_map.get("Gamma Exposure (Dealer Positioning)", 0)
        if spot_gamma_score < 0:
            alerts.append("Negative gamma zone — dealer hedging amplifies moves in both directions.")

        spy_price = gamma.get("price")
        put_wall = gamma.get("put_wall")
        call_wall = gamma.get("call_wall")
        if spy_price is not None and put_wall is not None and spy_price < put_wall:
            alerts.append(f"SPY ({spy_price:.2f}) below put wall ({put_wall:.2f}) — broken support, expect elevated downside volatility.")
        if spy_price is not None and call_wall is not None and spy_price > call_wall:
            alerts.append(f"SPY ({spy_price:.2f}) above call wall ({call_wall:.2f}) — resistance zone, upside may stall.")

    # Valuation + Risk-On combo
    cape = macro.get("cape")
    if cape is not None and cape > 25 and macro["macro_regime"] == "Risk-On":
        alerts.append(f"P/E at {cape:.1f}x with Risk-On regime — elevated valuations reduce margin of safety on long positions.")

    # Credit stress
    cs_score = score_map.get("Credit Spreads (HY vs Treasuries)", 0)
    if cs_score < -0.3:
        alerts.append("Credit spreads widening — stress in high-yield markets signals deteriorating risk appetite.")

    # Stagflation
    if macro["quadrant"] == "Stagflation":
        alerts.append("Stagflation quadrant — worst environment for traditional balanced portfolios. Consider real assets and cash.")

    # Dollar surge
    uup_snap = snaps.get("UUP")
    uup_30d = uup_snap.pct_change_30d if uup_snap else None
    if uup_30d is not None and uup_30d > 3:
        alerts.append(f"USD surging ({uup_30d:.1f}% 30d) — headwind for EM equities and commodities priced in dollars.")

    if not alerts:
        alerts.append("No elevated risk signals detected.")

    return alerts


def _tactical_opportunities(macro: dict, snaps: dict[str, AssetSnapshot]) -> list[dict]:
    """Identify tactical opportunities from cross-signal analysis."""
    score_map = {s["Indicator"]: s["Score"] for s in macro["signals"]}
    opps = []

    cs_score = score_map.get("Credit Spreads (HY vs Treasuries)", 0)
    ism_score = score_map.get("Industrial Production", 0)
    eq_score = score_map.get("Equity Trend (S&P, Nasdaq, Dow)", 0)
    gamma_score = score_map.get("Gamma Exposure (Dealer Positioning)", 0)
    dxy_score = score_map.get("US Dollar Index (DXY proxy)", 0)
    commodity_score = score_map.get("Commodity Trend (Oil + Copper)", 0)
    infl_score = score_map.get("Core Inflation (PCE)", 0)
    liquidity_score = score_map.get("Global Liquidity (M2 proxy)", 0)

    # Gold check
    gld_snap = snaps.get("GLD")
    gld_30d = gld_snap.pct_change_30d if gld_snap else None
    spy_snap = snaps.get("SPY")
    spy_30d = spy_snap.pct_change_30d if spy_snap else None

    if cs_score > 0.2 and ism_score > 0.2:
        opps.append({"signal": "Credit tight + ISM rising", "opportunity": "Favor high-yield over investment-grade bonds", "tickers": ["JNK"]})

    if cs_score < -0.2 and ism_score < -0.2:
        opps.append({"signal": "Credit widening + ISM falling", "opportunity": "Rotate to Treasuries for safety", "tickers": ["TLT", "IEF"]})

    if eq_score > 0.2 and gamma_score > 0:
        opps.append({"signal": "Strong equity trend + positive gamma", "opportunity": "Stay long with tight trailing stops", "tickers": ["SPY", "QQQ"]})

    if eq_score < -0.2 and gamma_score < 0:
        opps.append({"signal": "Weak equity + negative gamma", "opportunity": "Consider bear hedges or put spreads", "tickers": ["SPY", "QQQ"]})

    if dxy_score > 0.2 and macro["macro_regime"] == "Risk-On":
        opps.append({"signal": "Dollar weakness + equity risk-on", "opportunity": "EM likely to outperform US", "tickers": ["EEM", "FXI"]})

    if commodity_score > 0.3 and infl_score < -0.1:
        opps.append({"signal": "Commodity surge + rising inflation", "opportunity": "Overweight real assets as inflation hedge", "tickers": ["GLD", "TIP", "XLE"]})

    if gld_30d is not None and spy_30d is not None and gld_30d > 2 and spy_30d < -2:
        opps.append({"signal": "Gold rallying + equities declining", "opportunity": "Classic risk-off rotation underway — favor gold and Treasuries", "tickers": ["GLD", "TLT"]})

    if liquidity_score > 0.2:
        opps.append({"signal": "M2 liquidity expanding", "opportunity": "Bullish for risk assets with lag — accumulate on dips", "tickers": ["SPY", "QQQ", "IBIT"]})

    return opps


def _classify_yield_curve(spread_series: pd.Series | None, ten_year_series: pd.Series | None, lookback: int = 22) -> dict:
    """
    Classify yield curve regime into one of 4 states:
    - Bull Steepening: spread widening + 10Y falling (Fed easing, growth expected)
    - Bear Steepening: spread widening + 10Y rising (inflation fears)
    - Bull Flattening: spread narrowing + 10Y falling (flight to safety)
    - Bear Flattening: spread narrowing + 10Y rising (Fed tightening)
    """
    result = {"regime": "Unknown", "spread_change": None, "rate_direction": None,
              "spread_now": None, "ten_year_now": None, "inverted": False}

    if spread_series is None or ten_year_series is None:
        return result

    spread_clean = spread_series.dropna()
    ten_year_clean = ten_year_series.dropna()

    if len(spread_clean) < lookback + 1 or len(ten_year_clean) < lookback + 1:
        return result

    spread_now = float(spread_clean.iloc[-1])
    spread_prev = float(spread_clean.iloc[-lookback])
    spread_change = spread_now - spread_prev

    ten_year_now = float(ten_year_clean.iloc[-1])
    ten_year_prev = float(ten_year_clean.iloc[-lookback])
    rate_change = ten_year_now - ten_year_prev

    steepening = spread_change > 0
    rates_rising = rate_change > 0

    if steepening and not rates_rising:
        regime = "Bull Steepening"
    elif steepening and rates_rising:
        regime = "Bear Steepening"
    elif not steepening and not rates_rising:
        regime = "Bull Flattening"
    else:
        regime = "Bear Flattening"

    return {
        "regime": regime,
        "spread_change": round(spread_change, 3),
        "rate_direction": "Rising" if rates_rising else "Falling",
        "spread_now": round(spread_now, 3),
        "ten_year_now": round(ten_year_now, 3),
        "inverted": spread_now < 0,
    }


def _key_levels(macro: dict, snaps: dict[str, AssetSnapshot]) -> list[dict]:
    """Compute key technical levels for SPY and QQQ plus gamma levels."""
    levels = []

    for ticker in ("SPY", "QQQ"):
        snap = snaps.get(ticker)
        if snap is None or snap.latest_price is None or snap.series is None:
            continue
        current = snap.latest_price
        s = snap.series.dropna()

        # 120d MA (uses 6mo data window — ~126 trading days available)
        if len(s) >= 120:
            ma120 = float(s.tail(120).mean())
            pct = (current / ma120 - 1) * 100
            note = "above" if pct > 0 else "below"
            levels.append({"asset": ticker, "level_type": "120d MA", "price": round(ma120, 2), "current": round(current, 2), "pct_away": round(pct, 2), "note": f"{note} 120d trend"})

        # 50d MA
        if len(s) >= 50:
            ma50 = float(s.tail(50).mean())
            pct = (current / ma50 - 1) * 100
            note = "above" if pct > 0 else "below"
            levels.append({"asset": ticker, "level_type": "50d MA", "price": round(ma50, 2), "current": round(current, 2), "pct_away": round(pct, 2), "note": f"{note} 50d trend"})

        # Period high/low (uses all available data — ~6mo)
        hi = float(s.max())
        lo = float(s.min())
        pct_hi = (current / hi - 1) * 100
        pct_lo = (current / lo - 1) * 100
        levels.append({"asset": ticker, "level_type": "Period High", "price": round(hi, 2), "current": round(current, 2), "pct_away": round(pct_hi, 2), "note": "from period high"})
        levels.append({"asset": ticker, "level_type": "Period Low", "price": round(lo, 2), "current": round(current, 2), "pct_away": round(pct_lo, 2), "note": "from period low"})

    # SPY gamma levels
    gamma = macro.get("gamma")
    if gamma:
        spy_current = gamma.get("price", 0)
        for level_name, key in [("Call Wall", "call_wall"), ("Put Wall", "put_wall")]:
            val = gamma.get(key)
            if val is not None and spy_current:
                pct = (spy_current / val - 1) * 100
                levels.append({"asset": "SPY", "level_type": level_name, "price": round(val, 2), "current": round(spy_current, 2), "pct_away": round(pct, 2), "note": "options-derived level"})

    return levels


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_core_data() -> dict[str, AssetSnapshot]:
    """Fetch core tickers for signals + ticker bar display."""
    return fetch_batch_safe(CORE_TICKERS, period="6mo", interval="1d")


def run_quick_regime(use_claude: bool = False, model: str | None = None) -> bool:
    """
    Background helper for Quick Intel Run.
    Fetches market + FRED data, computes regime, generates rate-path plays.
    Stores _regime_context and _rp_plays_result to session_state.
    """
    import streamlit as st
    from concurrent.futures import ThreadPoolExecutor
    from services.market_data import fetch_fred_series_safe, warm_fred_cache
    from services.claude_client import suggest_regime_plays
    import datetime as _dt

    _FRED_IDS = [
        "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
        "PCEPILFE", "PNFI", "THREEFYTP10", "INDPRO", "NFCI", "DGS10",
        "ICSA", "USSLIND", "UMCSENT", "PERMIT", "FEDFUNDS",
    ]
    fred_key_map = {
        "yield_curve": "T10Y2Y", "credit_spread": "BAMLH0A0HYM2",
        "m2": "M2SL", "sahm": "SAHMREALTIME", "unrate": "UNRATE",
        "core_pce": "PCEPILFE", "capex": "PNFI", "icsa": "ICSA",
        "lei": "USSLIND", "term_premium": "THREEFYTP10", "ism": "INDPRO",
        "fci": "NFCI", "dgs10": "DGS10", "umcsent": "UMCSENT",
        "permit": "PERMIT", "fedfunds": "FEDFUNDS",
    }

    with ThreadPoolExecutor(max_workers=3) as ex:
        fred_fut = ex.submit(warm_fred_cache, _FRED_IDS)
        core_fut = ex.submit(fetch_core_data)
        gamma_fut = ex.submit(_compute_spy_gamma_mode_with_retry, 1)
        fred_fut.result()
        core_snaps = core_fut.result()
        gamma = gamma_fut.result()

    fred_data = {k: fetch_fred_series_safe(v) for k, v in fred_key_map.items()}
    macro = _build_macro_dashboard(core_snaps, gamma_data=gamma, fred_data=fred_data)
    macro["sector_rotation"] = _sector_rotation_recs(macro["quadrant"], macro["macro_regime"], core_snaps)
    macro["tactical_opps"] = _tactical_opportunities(macro, core_snaps)
    macro["snaps"] = core_snaps

    top_sigs = macro.get("top_signals", [])[:8]
    sig_lines = [f"- {s['name']}: z={s['score']:+.2f} ({s.get('label', '')})" for s in top_sigs]
    signal_summary = "\n".join(sig_lines)
    norm_score = (macro.get("macro_score", 50) - 50) / 50

    _prev_regime = (st.session_state.get("_regime_context") or {}).get("regime", "")
    _new_regime = macro["macro_regime"]
    st.session_state["_regime_context"] = {
        "regime": _new_regime,
        "score": norm_score,
        "signal_summary": signal_summary,
        "quadrant": macro["quadrant"],
    }
    st.session_state["_regime_context_ts"] = _dt.datetime.now()

    # Telegram alert on regime flip
    if _prev_regime and _prev_regime != _new_regime:
        try:
            from services.telegram_client import send_alert as _tg_alert
            _quad = macro["quadrant"]
            _tg_alert(
                f"⚠️ <b>REGIME FLIP</b>\n"
                f"{_prev_regime} → <b>{_new_regime}</b>\n"
                f"Score: {norm_score:+.2f} | {_quad}"
            )
        except Exception:
            pass

    _plays = suggest_regime_plays(
        macro["macro_regime"], norm_score, signal_summary,
        use_claude=use_claude, model=model,
    )
    _tier = "👑 Highly Regarded Mode" if (use_claude and model == "claude-sonnet-4-6") else ("🧠 Regard Mode" if use_claude else "⚡ Freeloader Mode")
    st.session_state["_rp_plays_result"] = _plays
    st.session_state["_rp_plays_last_tier"] = _tier
    return macro, fred_data


# ─────────────────────────────────────────────
# SPY GAMMA MODE
# ─────────────────────────────────────────────

def _compute_spy_gamma_mode_with_retry(max_expiries: int = 2) -> dict | None:
    """Try gamma computation, retry with fewer expiries on failure."""
    result = _compute_spy_gamma_mode(max_expiries=max_expiries)
    if result is None and max_expiries > 1:
        result = _compute_spy_gamma_mode(max_expiries=1)
    return result


@st.cache_data(ttl=14400, show_spinner=False)
def _compute_spy_gamma_mode(max_expiries: int = 2) -> dict | None:
    snap = fetch_options_chain_snapshot_safe("SPY", max_expiries=max_expiries)
    if not snap:
        return None

    strikes = np.array(snap.get("strikes", []), dtype=float)
    call_oi = np.array(snap.get("call_oi", []), dtype=float)
    put_oi = np.array(snap.get("put_oi", []), dtype=float)
    net_gamma = np.array(snap.get("net_gamma_proxy", []), dtype=float)
    if len(strikes) == 0:
        return None

    price = float(snap["price"])
    nearest_idx = int(np.argmin(np.abs(strikes - price)))
    spot_gamma = float(net_gamma[nearest_idx]) if len(net_gamma) else 0.0
    zone = "Positive Gamma Zone (Stable / low volatility)" if spot_gamma >= 0 else "Negative Gamma Zone (Volatile / trending)"

    cumulative = np.cumsum(net_gamma)
    gamma_flip = None
    for i in range(1, len(cumulative)):
        if (cumulative[i - 1] <= 0 < cumulative[i]) or (cumulative[i - 1] >= 0 > cumulative[i]):
            gamma_flip = float(strikes[i])
            break

    call_wall = float(strikes[int(np.argmax(call_oi))]) if len(call_oi) else None
    put_wall = float(strikes[int(np.argmax(put_oi))]) if len(put_oi) else None

    return {
        "price": price,
        "zone": zone,
        "gamma_flip": gamma_flip,
        "call_wall": call_wall,
        "put_wall": put_wall,
        "strikes": strikes,
        "net_gamma": net_gamma,
        "asof": snap.get("asof"),
    }


# ─────────────────────────────────────────────
# MACRO DASHBOARD ENGINE (15 indicators)
# ─────────────────────────────────────────────

class _SpyPeFetchError(Exception):
    """Raised when SPY P/E fetch fails, preventing st.cache_data from caching None."""
    pass


_SPY_PE_FALLBACK = 24.0  # Recent historical average; CAPE/Buffett from FRED are more reliable


@st.cache_data(ttl=14400)
def _fetch_spy_pe() -> float:
    """Fetch SPY trailing P/E with 2s timeout, falling back to hardcoded value."""
    import concurrent.futures
    def _do_fetch():
        info = yf.Ticker("SPY").info
        val = info.get("trailingPE") or info.get("forwardPE")
        if val is not None:
            return float(val)
        return None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_fetch)
            result = fut.result(timeout=2)
            if result is not None:
                return result
    except Exception:
        pass
    return _SPY_PE_FALLBACK


def _fetch_spy_pe_safe() -> float | None:
    """Wrapper around _fetch_spy_pe — always returns a value (fallback-safe)."""
    try:
        return _fetch_spy_pe()
    except Exception:
        return _SPY_PE_FALLBACK


@st.cache_data(ttl=14400, show_spinner=False)
def _build_macro_dashboard(snaps: dict[str, AssetSnapshot], gamma_data: dict | None = None, fred_data: dict | None = None) -> dict:
    fred_ids = {
        "yield_curve": "T10Y2Y",
        "credit_spread": "BAMLH0A0HYM2",
        "m2": "M2SL",
        "sahm": "SAHMREALTIME",
        "unrate": "UNRATE",
        "core_pce": "PCEPILFE",
        "capex": "PNFI",  # Private Nonresidential Fixed Investment (quarterly BEA)
        "icsa": "ICSA",  # Initial Jobless Claims (weekly)
        "lei": "USSLIND",  # Leading Economic Index (monthly, Philly Fed)
        "term_premium": "THREEFYTP10",
        "ism": "INDPRO",  # Industrial Production Index (monthly, replaces discontinued NAPM)
        "fci": "NFCI",
        "dgs10": "DGS10",  # 10-Year Treasury yield (for yield curve regime classification)
        "umcsent": "UMCSENT",  # Consumer Sentiment (University of Michigan)
        "permit": "PERMIT",  # Building Permits (housing leading indicator)
    }
    if fred_data is not None:
        # Use pre-fetched FRED data from warm cache — avoid duplicate requests
        fred = fred_data
        spy_pe = _fetch_spy_pe_safe()
    else:
        with ThreadPoolExecutor(max_workers=10) as executor:
            fred_futures = {k: executor.submit(fetch_fred_series_safe, v) for k, v in fred_ids.items()}
            spy_pe_future = executor.submit(_fetch_spy_pe_safe)
            fred = {k: f.result() for k, f in fred_futures.items()}
            spy_pe = spy_pe_future.result()

    indicators = []

    yc = _safe_latest(fred["yield_curve"])
    yc_score = _zscore_score(fred["yield_curve"])
    yc_chg = _series_trend(fred["yield_curve"], lookback=22)
    dgs10_chg = _series_trend(fred["dgs10"], lookback=22)
    yc_dir = ""
    if yc_chg is not None and dgs10_chg is not None:
        steepening = yc_chg > 0
        rates_rising = dgs10_chg > 0
        if steepening and not rates_rising:
            yc_dir = " — Bull Steepening"
        elif steepening and rates_rising:
            yc_dir = " — Bear Steepening"
        elif not steepening and not rates_rising:
            yc_dir = " — Bull Flattening"
        else:
            yc_dir = " — Bear Flattening"
        if yc is not None and yc < 0:
            yc_dir += " (Inverted)"
    indicators.append(("Yield Curve (10Y-2Y)", yc, f"bps{yc_dir}", yc_score, _confidence_from_age(fred["yield_curve"], expected_days=14)))

    cs = _safe_latest(fred["credit_spread"])
    cs_score = _zscore_score(fred["credit_spread"], invert=True)
    cs_chg = _series_trend(fred["credit_spread"], lookback=22)
    cs_dir = ""
    if cs_chg is not None:
        cs_dir = " ▲ Widening" if cs_chg > 0 else " ▼ Narrowing"
    indicators.append(("Credit Spreads (HY vs Treasuries)", cs, f"%{cs_dir}", cs_score, _confidence_from_age(fred["credit_spread"], expected_days=7)))

    def _blend_pct(snap):
        """Blend 1d/5d/30d pct changes (50%/30%/20%) for a snap."""
        if snap is None:
            return None
        components = [(snap.pct_change_1d, 0.5), (snap.pct_change_5d, 0.3), (snap.pct_change_30d, 0.2)]
        vals = [(v * w) for v, w in components if v is not None]
        weights = [w for v, w in components if v is not None]
        return sum(vals) / sum(weights) if weights else None

    oil = _blend_pct(snaps.get("USO"))
    copper = _blend_pct(snaps.get("CPER"))
    commodity_trend = np.nanmean([oil if oil is not None else np.nan, copper if copper is not None else np.nan])
    commodity_trend = None if np.isnan(commodity_trend) else float(commodity_trend)
    commodity_score = _clamp_score((commodity_trend or 0.0), 5.0)
    indicators.append(("Commodity Trend (Oil + Copper)", commodity_trend, "% blend", commodity_score, _confidence_from_snap("USO", "CPER", snaps=snaps)))

    uup = snaps.get("UUP")
    dxy_1d = uup.pct_change_1d if uup else None
    dxy_5d = uup.pct_change_5d if uup else None
    dxy_30d = uup.pct_change_30d if uup else None
    # Blend: 50% daily, 30% weekly, 20% monthly — responsive to short-term moves
    dxy_components = [(dxy_1d, 0.5), (dxy_5d, 0.3), (dxy_30d, 0.2)]
    dxy_vals = [(v * w) for v, w in dxy_components if v is not None]
    dxy_weights = [w for v, w in dxy_components if v is not None]
    dxy = sum(dxy_vals) / sum(dxy_weights) if dxy_weights else None
    if dxy is None:
        dxy_score = 0.0
    elif dxy > 0:
        dxy_score = -_clamp_score(abs(dxy), 3.0)
    elif dxy < 0:
        dxy_score = _clamp_score(abs(dxy), 3.0)
    else:
        dxy_score = 0.0
    indicators.append(("US Dollar Index (DXY proxy)", dxy, "% blend", dxy_score, _confidence_from_snap("UUP", snaps=snaps)))

    m2_yoy = _yoy_latest(fred["m2"], periods=12)
    m2_yoy_full = _yoy_series(fred["m2"], periods=12)
    liquidity_score = _zscore_score(m2_yoy_full) if m2_yoy_full is not None else _clamp_score(((m2_yoy or 0.0) - 2.0), 4.0)
    indicators.append(("Global Liquidity (M2 proxy)", m2_yoy, "% YoY", liquidity_score, _confidence_from_age(fred["m2"], expected_days=45)))

    sahm = _safe_latest(fred["sahm"])
    if sahm is None:
        unrate = fred["unrate"].dropna() if fred["unrate"] is not None else None
        if unrate is not None and len(unrate) >= 12:
            sahm = float(unrate.iloc[-1] - unrate.iloc[-12:].min())
    unemp_score = _zscore_score(fred["sahm"], invert=True) if fred["sahm"] is not None else _clamp_score((0.4 - (sahm or 0.0)), 0.4)
    indicators.append(("Unemployment Trend (Sahm context)", sahm, "delta", unemp_score, _confidence_from_age(fred["sahm"] if fred["sahm"] is not None else fred["unrate"], expected_days=45)))

    core_yoy = _yoy_latest(fred["core_pce"], periods=12)
    core_yoy_full = _yoy_series(fred["core_pce"], periods=12)
    core_infl_score = _zscore_score(core_yoy_full, invert=True) if core_yoy_full is not None else _clamp_score((2.4 - (core_yoy or 2.4)), 1.5)
    indicators.append(("Core Inflation (PCE)", core_yoy, "% YoY", core_infl_score, _confidence_from_age(fred["core_pce"], expected_days=45)))

    eq_components = []
    for ticker in ("SPY", "QQQ", "DIA"):
        s = snaps.get(ticker).series if snaps.get(ticker) else None
        if s is not None and len(s) >= 20:
            ma_len = min(120, len(s))
            ma = s.tail(ma_len).mean()
            if ma and ma != 0:
                eq_components.append(float((s.iloc[-1] / ma - 1) * 100))
    eq_trend = float(np.mean(eq_components)) if eq_components else None
    equity_score = _clamp_score((eq_trend or 0.0), 5.0)
    indicators.append(("Equity Trend (S&P, Nasdaq, Dow)", eq_trend, "% vs 120d MA", equity_score, _confidence_from_snap("SPY", "QQQ", "DIA", snaps=snaps)))

    cape = float(spy_pe) if spy_pe is not None else None
    cape_score = _clamp_score((25.0 - (cape or 25.0)), 10.0)
    indicators.append(("S&P 500 P/E (CAPE proxy)", cape, "x", cape_score, 85 if cape is not None else 0))

    capex_yoy = _yoy_latest(fred["capex"], periods=4)
    capex_level = _safe_latest(fred["capex"])   # PNFI in $B
    m2_level = _safe_latest(fred["m2"])          # M2 in $B
    capex_vs_liquidity = (capex_yoy - m2_yoy) if (capex_yoy is not None and m2_yoy is not None) else None
    capliq_score = _clamp_score((capex_vs_liquidity or 0.0), 5.0)
    indicators.append(("Corporate CAPEX vs Liquidity", capex_vs_liquidity, "pp", capliq_score, int(round(np.mean([
        _confidence_from_age(fred["capex"], expected_days=120),
        _confidence_from_age(fred["m2"], expected_days=45),
    ])))))

    gamma_score = 0.0
    if gamma_data and len(gamma_data["net_gamma"]) > 0:
        nearest = int(np.argmin(np.abs(gamma_data["strikes"] - gamma_data["price"])))
        gamma_score = _clamp_score(float(gamma_data["net_gamma"][nearest]), 10000.0)
    gamma_conf = 85 if gamma_data else 35
    indicators.append(("Gamma Exposure (Dealer Positioning)", gamma_score, "score", gamma_score, gamma_conf))

    term = _safe_latest(fred["term_premium"])
    term_score = _zscore_score(fred["term_premium"])
    indicators.append(("Term Premium", term, "%", term_score, _confidence_from_age(fred["term_premium"], expected_days=14)))

    indpro_yoy = _yoy_latest(fred["ism"], periods=12)
    indpro_yoy_full = _yoy_series(fred["ism"], periods=12)
    ism_score = _zscore_score(indpro_yoy_full) if indpro_yoy_full is not None else _clamp_score((indpro_yoy or 0.0), 5.0)
    indicators.append(("Industrial Production", indpro_yoy, "% YoY", ism_score, _confidence_from_age(fred["ism"], expected_days=45)))

    fci = _safe_latest(fred["fci"])
    fci_score = _zscore_score(fred["fci"], invert=True)
    indicators.append(("Financial Conditions Index", fci, "index", fci_score, _confidence_from_age(fred["fci"], expected_days=14)))

    vix_snap = snaps.get("^VIX")
    vix = vix_snap.latest_price if vix_snap else None
    vix_series = vix_snap.series if vix_snap else None
    vix_score = _zscore_score(vix_series, invert=True) if vix_series is not None else _clamp_score((20.0 - (vix or 20.0)), 8.0)
    indicators.append(("VIX (Equity Volatility)", vix, "level", vix_score, _confidence_from_snap("^VIX", snaps=snaps)))

    # --- Initial Jobless Claims ---
    icsa = _safe_latest(fred["icsa"])
    icsa_score = _zscore_score(fred["icsa"], invert=True)  # higher claims = risk-off
    indicators.append(("Initial Jobless Claims", icsa, "K", icsa_score, _confidence_from_age(fred["icsa"], expected_days=14)))

    # --- HYG/LQD Ratio (high-yield vs investment-grade credit) ---
    hyg_snap = snaps.get("HYG")
    lqd_snap = snaps.get("LQD")
    hyg_lqd_val = None
    hyg_lqd_score = 0.0
    hyg_lqd_series = None
    if hyg_snap and lqd_snap and hyg_snap.series is not None and lqd_snap.series is not None:
        aligned = pd.DataFrame({"hyg": hyg_snap.series, "lqd": lqd_snap.series}).dropna()
        if len(aligned) > 20:
            hyg_lqd_series = aligned["hyg"] / aligned["lqd"]
            hyg_lqd_val = float(hyg_lqd_series.iloc[-1])
            hyg_lqd_score = _zscore_score(hyg_lqd_series)  # higher ratio = risk-on
    indicators.append(("HYG/LQD Ratio (Credit Risk Appetite)", hyg_lqd_val, "ratio", hyg_lqd_score, _confidence_from_snap("HYG", "LQD", snaps=snaps)))

    # --- Copper/Gold Ratio (CPER/GLD) ---
    cper_snap = snaps.get("CPER")
    gld_snap = snaps.get("GLD")
    cu_au_val = None
    cu_au_score = 0.0
    if cper_snap and gld_snap and cper_snap.series is not None and gld_snap.series is not None:
        aligned_cg = pd.DataFrame({"cper": cper_snap.series, "gld": gld_snap.series}).dropna()
        if len(aligned_cg) > 20:
            cu_au_series = aligned_cg["cper"] / aligned_cg["gld"]
            cu_au_val = float(cu_au_series.iloc[-1])
            cu_au_score = _zscore_score(cu_au_series)  # higher ratio = risk-on
    indicators.append(("Copper/Gold Ratio (Growth vs Safety)", cu_au_val, "ratio", cu_au_score, _confidence_from_snap("CPER", "GLD", snaps=snaps)))

    # --- Leading Economic Index (Philly Fed) ---
    lei = _safe_latest(fred["lei"])
    lei_score = _zscore_score(fred["lei"])  # higher = risk-on
    indicators.append(("Leading Economic Index", lei, "index", lei_score, _confidence_from_age(fred["lei"], expected_days=45)))

    # --- Consumer Sentiment (Michigan) ---
    umcsent = _safe_latest(fred["umcsent"])
    umcsent_score = _zscore_score(fred["umcsent"])  # higher sentiment = risk-on
    indicators.append(("Consumer Sentiment (Michigan)", umcsent, "index", umcsent_score, _confidence_from_age(fred["umcsent"], expected_days=30)))

    # --- Building Permits ---
    permit = _safe_latest(fred["permit"])
    permit_score = _zscore_score(fred["permit"])  # higher permits = risk-on
    indicators.append(("Building Permits", permit, "K", permit_score, _confidence_from_age(fred["permit"], expected_days=30)))

    SIGNAL_CATEGORIES = {
        "Yield Curve (10Y-2Y)": "Rates",
        "Credit Spreads (HY vs Treasuries)": "Credit",
        "VIX (Equity Volatility)": "Volatility",
        "Commodity Trend (Oil + Copper)": "Commodities",
        "US Dollar Index (DXY proxy)": "FX",
        "Global Liquidity (M2 proxy)": "Liquidity",
        "Unemployment Trend (Sahm context)": "Labor",
        "Core Inflation (PCE)": "Inflation",
        "Equity Trend (S&P, Nasdaq, Dow)": "Equities",
        "S&P 500 P/E (CAPE proxy)": "Valuation",
        "Corporate CAPEX vs Liquidity": "Growth",
        "Initial Jobless Claims": "Labor",
        "HYG/LQD Ratio (Credit Risk Appetite)": "Credit",
        "Copper/Gold Ratio (Growth vs Safety)": "Commodities",
        "Leading Economic Index": "Growth",
        "Gamma Exposure (Dealer Positioning)": "Positioning",
        "Term Premium": "Rates",
        "Industrial Production": "Growth",
        "Financial Conditions Index": "Credit",
        "Consumer Sentiment (Michigan)": "Sentiment",
        "Building Permits": "Housing",
    }

    SIGNAL_WEIGHTS = {
        # Tier 1 — Leading/highly predictive of regime shifts
        "Credit Spreads (HY vs Treasuries)": 2.0,
        "Yield Curve (10Y-2Y)": 2.0,
        "VIX (Equity Volatility)": 2.0,
        "Financial Conditions Index": 2.0,
        # Tier 2 — Strong regime signals
        "Equity Trend (S&P, Nasdaq, Dow)": 1.5,
        "Unemployment Trend (Sahm context)": 1.5,
        "Global Liquidity (M2 proxy)": 1.5,
        "Initial Jobless Claims": 1.5,
        "HYG/LQD Ratio (Credit Risk Appetite)": 1.5,
        # Tier 3 — Standard
        "Commodity Trend (Oil + Copper)": 1.0,
        "US Dollar Index (DXY proxy)": 1.0,
        "Industrial Production": 1.0,
        "Core Inflation (PCE)": 1.0,
        "Term Premium": 1.0,
        "Gamma Exposure (Dealer Positioning)": 1.0,
        "Copper/Gold Ratio (Growth vs Safety)": 1.0,
        "Consumer Sentiment (Michigan)": 1.0,
        "Building Permits": 1.0,
        # Tier 1 — Leading
        "Leading Economic Index": 2.0,
        # Tier 4 — Slow-moving / noisy
        "S&P 500 P/E (CAPE proxy)": 0.5,
        "Corporate CAPEX vs Liquidity": 0.5,
    }

    signal_rows = []
    scores = []
    confidence_scores = []
    for name, value, unit, score, confidence in indicators:
        emoji, verdict = _score_to_bucket(score)
        scores.append(score)
        confidence_scores.append(confidence)
        display_value = "N/A" if value is None else f"{value:.2f} {unit}".strip()
        signal_rows.append({
            "Category": SIGNAL_CATEGORIES.get(name, "Other"),
            "Indicator": name,
            "Signal": f"{emoji} {verdict}",
            "Direction": verdict,
            "Value": display_value,
            "Score": round(float(score), 3),
            "Confidence": confidence,
        })

    # Confidence-weighted scoring: effective_weight = tier_weight * (confidence / 100)
    weights = [
        SIGNAL_WEIGHTS.get(row["Indicator"], 1.0) * (row["Confidence"] / 100.0)
        for row in signal_rows
    ]
    aggregate = float(np.average(scores, weights=weights)) if scores else 0.0
    macro_score = int(round((aggregate + 1.0) * 50))

    if macro_score >= 60:
        macro_regime = "Risk-On"
    elif macro_score <= 40:
        macro_regime = "Risk-Off"
    else:
        macro_regime = _neutral_lean_label(macro_score)

    growth_signal = np.mean([yc_score, equity_score, ism_score, unemp_score])
    core_series = fred["core_pce"].dropna() if fred["core_pce"] is not None else None
    core_3m_change = None
    if core_series is not None and len(core_series) >= 4:
        core_3m_change = float(core_series.iloc[-1] - core_series.iloc[-4])
    inflation_direction_value = np.mean([
        1.0 if (core_3m_change is not None and core_3m_change > 0) else -1.0,
        1.0 if (commodity_trend is not None and commodity_trend > 0) else -1.0,
    ])

    growth_dir = "Rising" if growth_signal >= 0 else "Falling"
    inflation_dir = "Rising" if inflation_direction_value >= 0 else "Falling"
    if growth_dir == "Rising" and inflation_dir == "Rising":
        quadrant = "Reflation"
    elif growth_dir == "Rising" and inflation_dir == "Falling":
        quadrant = "Goldilocks"
    elif growth_dir == "Falling" and inflation_dir == "Rising":
        quadrant = "Stagflation"
    else:
        quadrant = "Deflation"

    if capex_vs_liquidity is None:
        cycle_stage = "Cycle signal unavailable"
    elif capex_vs_liquidity > 2:
        cycle_stage = "Capex-led expansion"
    elif capex_vs_liquidity < -2:
        cycle_stage = "Liquidity-led / capex slowdown"
    else:
        cycle_stage = "Balanced mid-cycle"

    valuation_text = _interpret_valuation(cape)

    ranked = sorted(signal_rows, key=lambda x: abs(x["Score"]), reverse=True)
    summary = [
        f"Macro score is {macro_score}/100 ({macro_regime}).",
        f"Dalio quadrant currently points to {quadrant} ({growth_dir.lower()} growth, {inflation_dir.lower()} inflation).",
        f"Valuation read: {valuation_text}",
    ]
    if ranked:
        summary.append(f"Strongest signal: {ranked[0]['Indicator']} at {ranked[0]['Signal']}.")
    if len(ranked) > 1:
        summary.append(f"Second strongest: {ranked[1]['Indicator']} at {ranked[1]['Signal']}.")

    # Compact signal summary for LLM context (top 10 most extreme z-scores)
    top_signals = [
        {
            "name": row["Indicator"],
            "score": round(row["Score"], 2),
            "value": row.get("Value", "n/a"),
            "confidence": row.get("Confidence", 50),
        }
        for row in signal_rows
        if abs(row.get("Score", 0.0)) >= 0.3
    ]
    top_signals.sort(key=lambda x: abs(x["score"]), reverse=True)

    result = {
        "signals": signal_rows,
        "top_signals": top_signals[:10],
        "macro_score": macro_score,
        "avg_confidence": int(round(float(np.mean(confidence_scores)))) if confidence_scores else 0,
        "macro_regime": macro_regime,
        "quadrant": quadrant,
        "growth_dir": growth_dir,
        "inflation_dir": inflation_dir,
        "growth_signal": round(float(growth_signal), 3),
        "inflation_signal": round(float(inflation_direction_value), 3),
        "valuation": valuation_text,
        "cape": cape,
        "cycle_stage": cycle_stage,
        "capex_vs_liquidity": capex_vs_liquidity,
        "capex_level": capex_level,
        "m2_level": m2_level,
        "capex_yoy": capex_yoy,
        "m2_yoy": m2_yoy,
        "summary": summary[:5],
        "portfolio_bias": _portfolio_bias(macro_regime),
        "gamma": gamma_data,
    }

    result["risk_alerts"] = _risk_management_alerts(result, snaps)
    result["key_levels"] = _key_levels(result, snaps)
    result["yield_curve_regime"] = _classify_yield_curve(fred["yield_curve"], fred["dgs10"])
    result["snaps"] = snaps

    # Persist daily snapshot
    try:
        _save_snapshot(result)
    except Exception:
        pass

    return result


# ─────────────────────────────────────────────
# PUBLIC API (consumed by other modules)
# ─────────────────────────────────────────────

def get_current_regime() -> dict:
    """Public accessor for other modules to consume regime data."""
    snaps = fetch_core_data()
    macro = _build_macro_dashboard(snaps)
    return {
        "regime": macro["macro_regime"],
        "score": macro["macro_score"],
        "quadrant": macro["quadrant"],
        "signals": macro["signals"],
    }


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

def _make_gauge(macro_score: int, regime: str, color: str) -> go.Figure:
    """Gauge chart for the 0-100 macro score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=macro_score,
        title={"text": f"Macro Score ({regime})", "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": " / 100", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_dim"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": COLORS["surface"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "#2d1b1b"},
                {"range": [40, 60], "color": "#1e1e10"},
                {"range": [60, 100], "color": "#1b2d1b"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": macro_score,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    apply_dark_layout(fig)
    return fig


def _make_regime_history() -> go.Figure | None:
    """Time-series chart of historical regime scores."""
    history = _load_history()
    if len(history) < 2:
        return None

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # Use macro_score (0-100) if available, else fall back to legacy score (-1..+1)
    if "macro_score" in df.columns:
        y_vals = df["macro_score"]
        y_range = [-5, 105]
        y_title = "Macro Score (0-100)"
        thresh_hi = 60
        thresh_lo = 40
        colors = [
            COLORS["green"] if s >= 60 else (COLORS["red"] if s <= 40 else COLORS["yellow"])
            for s in y_vals
        ]
    else:
        y_vals = df["score"]
        y_range = [-1.1, 1.1]
        y_title = "Regime Score"
        thresh_hi = 0.35
        thresh_lo = -0.35
        colors = [
            COLORS["green"] if s >= 0.35 else (COLORS["red"] if s <= -0.35 else COLORS["yellow"])
            for s in y_vals
        ]

    fig = go.Figure()

    # Background bands for regime zones
    fig.add_hrect(y0=thresh_hi, y1=y_range[1], fillcolor="#1b2d1b", opacity=0.5, line_width=0)
    fig.add_hrect(y0=thresh_lo, y1=thresh_hi, fillcolor="#1e1e10", opacity=0.5, line_width=0)
    fig.add_hrect(y0=y_range[0], y1=thresh_lo, fillcolor="#2d1b1b", opacity=0.5, line_width=0)

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=y_vals,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(color=colors, size=6),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y}<extra></extra>",
    ))

    fig.add_hline(y=thresh_hi, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
    fig.add_hline(y=thresh_lo, line_dash="dash", line_color=COLORS["red"], opacity=0.5)

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=30),
        yaxis=dict(range=y_range, title=y_title),
        xaxis=dict(title=""),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Regime Score History")
    return fig


def _growth_proxy_from_signals(s: dict) -> float:
    vals = [s.get("Yield Curve (10Y-2Y)", 0), s.get("Equity Trend (S&P, Nasdaq, Dow)", 0),
            s.get("Industrial Production", 0), s.get("Unemployment Trend (Sahm context)", 0)]
    active = [v for v in vals if v != 0]
    return sum(active) / len(active) if active else 0.0


def _inflation_proxy_from_signals(s: dict) -> float:
    vals = [s.get("Core Inflation (PCE)", 0), s.get("Commodity Trend (Oil + Copper)", 0)]
    active = [v for v in vals if v != 0]
    return sum(active) / len(active) if active else 0.0


def _make_quadrant_scatter() -> go.Figure | None:
    """Dalio quadrant scatter: growth (x) vs inflation (y) for each history entry."""
    history = _load_history()
    if not history:
        return None

    xs, ys, dates, regimes, quadrants = [], [], [], [], []
    for h in sorted(history, key=lambda x: x["date"]):
        sigs = h.get("signals_summary", {})
        gx = h["growth_signal"] if h.get("growth_signal") is not None else _growth_proxy_from_signals(sigs)
        iy = h["inflation_signal"] if h.get("inflation_signal") is not None else _inflation_proxy_from_signals(sigs)
        xs.append(gx)
        ys.append(iy)
        dates.append(h["date"])
        regimes.append(h.get("regime", "Neutral"))
        quadrants.append(h.get("quadrant", ""))

    if not xs:
        return None

    dot_colors = [
        COLORS["green"] if "Risk-On" in r else (COLORS["red"] if "Risk-Off" in r else COLORS["yellow"])
        for r in regimes
    ]

    fig = go.Figure()

    # Quadrant background zones (low opacity)
    _zone_cfg = [
        (0, 1, 0, 1,  "#22c55e", "Goldilocks"),    # growth+, inflation-  (right, bottom)
        (0, 1, 0, 1,  "#f59e0b", "Reflation"),     # growth+, inflation+  (right, top)
        (-1, 0, 0, 1, "#ef4444", "Stagflation"),   # growth-, inflation+  (left, top)
        (-1, 0, -1, 0,"#3b82f6", "Deflation"),     # growth-, inflation-  (left, bottom)
    ]
    quadrant_zones = [
        # (x0, x1, y0, y1, color, label)
        (0,  1,  -1,  0,  "#22c55e", "Goldilocks"),
        (0,  1,   0,  1,  "#f59e0b", "Reflation"),
        (-1, 0,   0,  1,  "#ef4444", "Stagflation"),
        (-1, 0,  -1,  0,  "#3b82f6", "Deflation"),
    ]
    for x0, x1, y0, y1, color, label in quadrant_zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                      fillcolor=color, opacity=0.10, line_width=0)
        fig.add_annotation(
            x=(x0 + x1) / 2, y=(y0 + y1) / 2,
            text=label, showarrow=False,
            font=dict(size=11, color=color, family="JetBrains Mono, Consolas, monospace"),
            opacity=0.55,
        )

    # Historical trail (all but last)
    if len(xs) > 1:
        fig.add_trace(go.Scatter(
            x=xs[:-1], y=ys[:-1],
            mode="markers",
            marker=dict(color=dot_colors[:-1], size=7, opacity=0.7,
                        line=dict(width=0)),
            hovertemplate="<b>%{customdata[0]}</b><br>Growth: %{x:.2f} | Inflation: %{y:.2f}<br>%{customdata[1]}<extra></extra>",
            customdata=list(zip(dates[:-1], quadrants[:-1])),
            name="History",
        ))

    # Current dot (last entry)
    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode="markers+text",
        marker=dict(color=dot_colors[-1], size=14, symbol="circle",
                    line=dict(color="white", width=2)),
        text=[dates[-1]],
        textposition="top center",
        textfont=dict(size=10, color="#C8D8E8"),
        hovertemplate=f"<b>{dates[-1]}</b><br>Growth: {xs[-1]:.2f} | Inflation: {ys[-1]:.2f}<br>{quadrants[-1]}<extra></extra>",
        name="Current",
    ))

    # Zero-line guides
    fig.add_vline(x=0, line_dash="dash", line_color="#334155", opacity=0.7)
    fig.add_hline(y=0, line_dash="dash", line_color="#334155", opacity=0.7)

    fig.update_layout(
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(title="Growth →", range=[-1.1, 1.1], zeroline=False, showgrid=False),
        yaxis=dict(title="Inflation →", range=[-1.1, 1.1], zeroline=False, showgrid=False),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Dalio Quadrant — Growth vs Inflation")
    return fig


def _make_category_radar(signals: list[dict]) -> go.Figure | None:
    """Radar chart showing average score per signal category."""
    df = pd.DataFrame(signals)
    if "Category" not in df.columns or len(df) < 3:
        return None
    cat_scores = df.groupby("Category")["Score"].mean().reset_index()
    categories = cat_scores["Category"].tolist()
    scores = cat_scores["Score"].tolist()
    # Close the polygon
    categories += [categories[0]]
    scores += [scores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=scores, theta=categories, fill="toself",
        fillcolor="rgba(75, 159, 255, 0.15)",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(color=COLORS["blue"], size=6),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(range=[-1, 1], gridcolor=COLORS["grid"], tickfont=dict(size=9)),
            angularaxis=dict(gridcolor=COLORS["grid"], tickfont=dict(size=10, color=COLORS["text"])),
        ),
        height=380, margin=dict(l=40, r=40, t=30, b=30), showlegend=False,
    )
    apply_dark_layout(fig)
    return fig


# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────

def _section_header(title: str, badge: str = ""):
    """Render a Bloomberg-styled section header with optional engine badge."""
    badge_html = (
        f' <span style="font-size:10px;background:{COLORS["bloomberg_orange"]};'
        f'color:#000;padding:1px 6px;border-radius:3px;vertical-align:middle;'
        f'font-weight:700;letter-spacing:0.04em;">{badge}</span>'
        if badge else ""
    )
    st.markdown(
        f'<div style="border-left:3px solid {COLORS["bloomberg_orange"]};'
        f'background:{COLORS["surface"]};padding:8px 14px;margin:20px 0 10px 0;'
        f'font-family:\'JetBrains Mono\',Consolas,monospace;font-size:14px;'
        f'font-weight:600;color:{COLORS["bloomberg_orange"]};letter-spacing:0.08em;'
        f'text-transform:uppercase;">{title}{badge_html}</div>',
        unsafe_allow_html=True,
    )


def _render_signals_table(signals: list[dict]):
    """Render core signals as a styled Bloomberg-terminal dataframe."""
    import pandas as pd
    df = pd.DataFrame(signals)
    display_cols = ["Indicator", "Value", "Score", "Direction", "Confidence"]
    df = df[[c for c in display_cols if c in df.columns]]
    df = df.rename(columns={"Indicator": "Signal"})

    def _color_score(val):
        try:
            v = float(val)
        except (ValueError, TypeError):
            return ""
        if v > 0:
            return f"color: {COLORS['green']}"
        elif v < 0:
            return f"color: {COLORS['red']}"
        return f"color: {COLORS['text_dim']}"

    def _color_direction(val):
        if val == "Risk-On":
            return f"color: {COLORS['green']}"
        elif val == "Risk-Off":
            return f"color: {COLORS['red']}"
        return f"color: {COLORS['yellow']}"

    styled = (
        df.style
        .map(_color_score, subset=["Score"] if "Score" in df.columns else [])
        .map(_color_direction, subset=["Direction"] if "Direction" in df.columns else [])
        .set_properties(**{
            "font-family": "'JetBrains Mono', Consolas, monospace",
            "font-size": "12px",
        })
    )

    # ── Confidence legend ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:11px;color:#888;line-height:1.7;margin-bottom:6px;">'
        f'<b style="color:#ccc;">Confidence</b> measures how fresh and reliable each signal\'s underlying data is. '
        f'It is computed from the <i>age of the last data point</i> relative to its expected update frequency:<br>'
        f'&nbsp;&nbsp;<b style="color:#4caf50;">● High (≥75)</b> — data is current (updated within its normal release window). Weight this signal fully.<br>'
        f'&nbsp;&nbsp;<b style="color:#ff9800;">● Medium (50–74)</b> — data is slightly stale (e.g. FRED series released monthly and overdue by a few days). '
        f'Signal is still valid but treat with mild caution — it may not reflect the most recent shift.<br>'
        f'&nbsp;&nbsp;<b style="color:#f44336;">● Low (&lt;50)</b> — data is significantly lagged or unavailable. '
        f'This signal\'s vote is down-weighted in the macro score automatically.</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False)
    st.download_button("Export CSV", csv, "risk_regime_signals.csv", "text/csv", key="dl_risk_signals")


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────

# ── Sector Rotation Monitor ───────────────────────────────────────────────────

_SECTOR_ETFS = {
    "XLK":  "Technology",
    "XLV":  "Health Care",
    "XLE":  "Energy",
    "XLF":  "Financials",
    "XLI":  "Industrials",
    "XLC":  "Communication",
    "XLP":  "Consumer Staples",
    "XLY":  "Consumer Discr.",
    "XLU":  "Utilities",
    "XLB":  "Materials",
    "XLRE": "Real Estate",
}

# Sectors historically favored in each Dalio quadrant
_QUADRANT_SECTOR_ALIGNMENT = {
    "Goldilocks":  ["XLK", "XLC", "XLY", "XLI", "XLF"],
    "Reflation":   ["XLE", "XLB", "XLF", "XLI", "XLY"],
    "Stagflation": ["XLE", "XLB", "XLU", "XLP", "XLV"],
    "Deflation":   ["XLU", "XLV", "XLP", "XLRE"],
}


@st.cache_data(ttl=3600)
def _fetch_sector_momentum() -> list[dict]:
    """Download 6-month weekly closes for all 11 SPDR sector ETFs.
    Returns list sorted by 4W momentum descending."""
    tickers = list(_SECTOR_ETFS.keys())
    try:
        raw = yf.download(tickers, period="6mo", interval="1wk",
                          progress=False, auto_adjust=True)
        if raw is None or raw.empty:
            return []
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
        results = []
        for t in tickers:
            if t not in close.columns:
                continue
            s = close[t].dropna()
            if len(s) < 5:
                continue
            price = float(s.iloc[-1])
            ret_4w  = (price / float(s.iloc[-5])  - 1) * 100 if len(s) >= 5  else None
            ret_12w = (price / float(s.iloc[-13]) - 1) * 100 if len(s) >= 13 else None
            ret_26w = (price / float(s.iloc[0])   - 1) * 100
            results.append({
                "ticker": t,
                "name": _SECTOR_ETFS[t],
                "price": round(price, 2),
                "ret_4w":  round(ret_4w,  2) if ret_4w  is not None else None,
                "ret_12w": round(ret_12w, 2) if ret_12w is not None else None,
                "ret_26w": round(ret_26w, 2),
            })
        # Rank by 4W and 12W (1 = best)
        v4  = sorted([r for r in results if r["ret_4w"]  is not None], key=lambda x: x["ret_4w"],  reverse=True)
        v12 = sorted([r for r in results if r["ret_12w"] is not None], key=lambda x: x["ret_12w"], reverse=True)
        r4m  = {r["ticker"]: i + 1 for i, r in enumerate(v4)}
        r12m = {r["ticker"]: i + 1 for i, r in enumerate(v12)}
        for r in results:
            r["rank_4w"]  = r4m.get(r["ticker"])
            r["rank_12w"] = r12m.get(r["ticker"])
        results.sort(key=lambda x: (x["ret_4w"] or -999), reverse=True)
        return results
    except Exception:
        return []


def _render_sector_rotation_tab(quadrant: str, regime: str) -> None:
    """Live sector rotation panel: 4W + 12W momentum ranks + regime alignment."""
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">SECTOR ROTATION MONITOR</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"11 SPDR sector ETFs ranked by 4-week and 12-week momentum. "
        f"Current regime: **{regime}** · **{quadrant}** quadrant. "
        f"Bright bars = regime-aligned sectors."
    )

    with st.spinner("Loading sector momentum data..."):
        sectors = _fetch_sector_momentum()

    if not sectors:
        st.warning("Could not fetch sector ETF data. Click Refresh Data and try again.")
        return

    aligned = set(_QUADRANT_SECTOR_ALIGNMENT.get(quadrant, []))

    # ── Regime-confirmed leaders callout ──────────────────────────────────────
    confirmed = [s for s in sectors[:5] if s["ticker"] in aligned]
    if confirmed:
        _names = " &nbsp;·&nbsp; ".join(
            f'<span style="font-weight:700;color:#f1f5f9;">{s["ticker"]}</span> '
            f'<span style="color:#94a3b8;">{s["name"]} {s["ret_4w"]:+.1f}%</span>'
            for s in confirmed
        )
        st.markdown(
            f'<div style="background:#0a2218;border-left:3px solid #22c55e;'
            f'padding:8px 14px;border-radius:0 4px 4px 0;margin:8px 0 14px 0;">'
            f'<span style="font-size:10px;color:#22c55e;font-weight:700;letter-spacing:0.06em;">'
            f'REGIME-CONFIRMED LEADERS (top 5 momentum ∩ {quadrant} alignment)</span><br>'
            f'<span style="font-size:13px;">{_names}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Momentum bar charts ────────────────────────────────────────────────────
    col_4w, col_12w = st.columns(2)

    def _bar_colors(data, ret_key):
        return [
            "#22c55e" if ((s[ret_key] or 0) > 0 and s["ticker"] in aligned)
            else "#86efac" if (s[ret_key] or 0) > 0
            else "#ef4444" if s["ticker"] in aligned
            else "#fca5a5"
            for s in data
        ]

    with col_4w:
        st.markdown(
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
            f'font-weight:700;margin-bottom:2px;">4-WEEK MOMENTUM</div>',
            unsafe_allow_html=True,
        )
        _d4 = sorted([s for s in sectors if s["ret_4w"] is not None], key=lambda x: x["ret_4w"])
        fig4 = go.Figure(go.Bar(
            x=[s["ret_4w"] for s in _d4],
            y=[s["ticker"] for s in _d4],
            orientation="h",
            marker_color=_bar_colors(_d4, "ret_4w"),
            text=[f"{s['ret_4w']:+.1f}%" for s in _d4],
            textposition="outside",
        ))
        apply_dark_layout(fig4, height=340)
        fig4.update_layout(
            margin=dict(l=10, r=55, t=10, b=10),
            xaxis=dict(ticksuffix="%", gridcolor=COLORS["grid"]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_12w:
        st.markdown(
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
            f'font-weight:700;margin-bottom:2px;">12-WEEK MOMENTUM</div>',
            unsafe_allow_html=True,
        )
        _d12 = sorted([s for s in sectors if s["ret_12w"] is not None], key=lambda x: x["ret_12w"])
        fig12 = go.Figure(go.Bar(
            x=[s["ret_12w"] for s in _d12],
            y=[s["ticker"] for s in _d12],
            orientation="h",
            marker_color=_bar_colors(_d12, "ret_12w"),
            text=[f"{s['ret_12w']:+.1f}%" for s in _d12],
            textposition="outside",
        ))
        apply_dark_layout(fig12, height=340)
        fig12.update_layout(
            margin=dict(l=10, r=55, t=10, b=10),
            xaxis=dict(ticksuffix="%", gridcolor=COLORS["grid"]),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig12, use_container_width=True)

    st.caption("Bright green/red = regime-aligned sector.  Light green/red = momentum only (no quadrant alignment).")

    # ── Ranked table ──────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin:14px 0 6px 0;">FULL RANKING TABLE</div>',
        unsafe_allow_html=True,
    )
    _tbl = (
        f'<table style="width:100%;border-collapse:collapse;'
        f'font-family:JetBrains Mono,monospace;font-size:12px;">'
        f'<tr style="border-bottom:2px solid {COLORS["bloomberg_orange"]};">'
    )
    for _h in ["4W #", "Sector", "ETF", "Price", "4W %", "12W #", "12W %", "26W %", "Regime"]:
        _tbl += (f'<th style="padding:5px 10px;text-align:left;'
                 f'color:{COLORS["bloomberg_orange"]};">{_h}</th>')
    _tbl += "</tr>"
    for i, s in enumerate(sectors):
        _bg   = COLORS["surface"] if i % 2 == 0 else COLORS["bg"]
        _aln  = s["ticker"] in aligned
        _badge = (
            '<span style="color:#22c55e;font-size:10px;font-weight:700;">✓ ALIGNED</span>'
            if _aln else
            '<span style="color:#475569;font-size:10px;">—</span>'
        )
        _4w  = s.get("ret_4w");  _4s  = f"{_4w:+.1f}%"  if _4w  is not None else "—"
        _12w = s.get("ret_12w"); _12s = f"{_12w:+.1f}%" if _12w is not None else "—"
        _26w = s.get("ret_26w"); _26s = f"{_26w:+.1f}%" if _26w is not None else "—"
        _4c  = "#22c55e" if (_4w  or 0) > 0 else "#ef4444"
        _12c = "#22c55e" if (_12w or 0) > 0 else "#ef4444"
        _26c = "#22c55e" if (_26w or 0) > 0 else "#ef4444"
        _rc  = "#22c55e" if (s.get("rank_4w") or 99) <= 3 else COLORS["text_dim"]
        _r12c = "#22c55e" if (s.get("rank_12w") or 99) <= 3 else COLORS["text_dim"]
        _tbl += (
            f'<tr style="background:{_bg};">'
            f'<td style="padding:5px 10px;color:{_rc};font-weight:700;">#{s.get("rank_4w","?")}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text"]};">{s["name"]}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["bloomberg_orange"]};font-weight:700;">{s["ticker"]}</td>'
            f'<td style="padding:5px 10px;color:{COLORS["text_dim"]};">${s["price"]:,.2f}</td>'
            f'<td style="padding:5px 10px;color:{_4c};font-weight:600;">{_4s}</td>'
            f'<td style="padding:5px 10px;color:{_r12c};font-weight:700;">#{s.get("rank_12w","?")}</td>'
            f'<td style="padding:5px 10px;color:{_12c};">{_12s}</td>'
            f'<td style="padding:5px 10px;color:{_26c};">{_26s}</td>'
            f'<td style="padding:5px 10px;">{_badge}</td>'
            f'</tr>'
        )
    _tbl += "</table>"
    st.markdown(_tbl, unsafe_allow_html=True)

    # ── Quadrant alignment legend ──────────────────────────────────────────────
    with st.expander("Quadrant Sector Alignment Guide", expanded=False):
        st.markdown("""
| Quadrant | Environment | Favored Sectors |
|---|---|---|
| **Goldilocks** | Growth ↑, Inflation ↓ | Tech (XLK), Comm (XLC), Discr (XLY), Industrials (XLI), Financials (XLF) |
| **Reflation** | Growth ↑, Inflation ↑ | Energy (XLE), Materials (XLB), Financials (XLF), Industrials (XLI), Discr (XLY) |
| **Stagflation** | Growth ↓, Inflation ↑ | Energy (XLE), Materials (XLB), Utilities (XLU), Staples (XLP), Health Care (XLV) |
| **Deflation** | Growth ↓, Inflation ↓ | Utilities (XLU), Health Care (XLV), Staples (XLP), Real Estate (XLRE) |
""")


def render():
    st.title("Macro Dashboard")
    st.caption("Global macro monitor — Risk-On / Risk-Off workflow")

    if st.button("Refresh Data"):
        st.cache_data.clear()

    _FRED_SERIES_IDS = [
        "T10Y2Y", "BAMLH0A0HYM2", "M2SL", "SAHMREALTIME", "UNRATE",
        "PCEPILFE", "PNFI", "THREEFYTP10",
        "INDPRO", "NFCI", "DGS10", "ICSA", "USSLIND",
        "UMCSENT", "PERMIT", "FEDFUNDS",
    ]

    load_start = datetime.now()
    with st.status("MACRO DASHBOARD · INITIALIZING...", expanded=True) as status:
        t0 = datetime.now()
        with ThreadPoolExecutor(max_workers=3) as executor:
            fred_future = executor.submit(warm_fred_cache, _FRED_SERIES_IDS)
            core_future = executor.submit(fetch_core_data)
            gamma_future = executor.submit(_compute_spy_gamma_mode_with_retry, 1)

            future_labels = {
                fred_future: "Federal Reserve (FRED) — 15 series",
                core_future: "Market prices — 13 tickers",
                gamma_future: "SPY options chain — gamma exposure",
            }

            st.write("⏳ Connecting to data sources...")
            for future in as_completed(future_labels):
                label = future_labels[future]
                future.result()  # raise if failed
                st.write(f"✓ {label}")

            core_snaps = core_future.result()
            gamma = gamma_future.result()
        t_fetch = (datetime.now() - t0).total_seconds()

        t1 = datetime.now()
        st.write("⏳ Computing risk regime signals...")
        # Collect warmed FRED data (hits st.cache — no new network requests)
        fred_ids = {
            "yield_curve": "T10Y2Y",
            "credit_spread": "BAMLH0A0HYM2",
            "m2": "M2SL",
            "sahm": "SAHMREALTIME",
            "unrate": "UNRATE",
            "core_pce": "PCEPILFE",
            "capex": "PNFI",
            "icsa": "ICSA",
            "lei": "USSLIND",
            "term_premium": "THREEFYTP10",
            "ism": "INDPRO",
            "fci": "NFCI",
            "dgs10": "DGS10",
            "umcsent": "UMCSENT",
            "permit": "PERMIT",
            "fedfunds": "FEDFUNDS",
        }
        fred_data = {k: fetch_fred_series_safe(v) for k, v in fred_ids.items()}
        macro = _build_macro_dashboard(core_snaps, gamma_data=gamma, fred_data=fred_data)
        snaps = core_snaps
        macro["sector_rotation"] = _sector_rotation_recs(macro["quadrant"], macro["macro_regime"], snaps)
        macro["tactical_opps"] = _tactical_opportunities(macro, snaps)
        macro["snaps"] = snaps
        t_macro = (datetime.now() - t1).total_seconds()
        st.write("✓ Risk regime signals — 21 signals")

        status.update(label="MACRO DASHBOARD · READY", state="complete", expanded=False)

    total_load = (datetime.now() - load_start).total_seconds()
    cache_expiry = datetime.now() + timedelta(seconds=14400)
    st.markdown(
        f'<div style="font-size:11px;color:{COLORS["text_dim"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'padding:2px 0;letter-spacing:0.03em;">'
        f'FETCH {t_fetch:.1f}s | SIGNALS {t_macro:.1f}s | TOTAL {total_load:.1f}s — '
        f'cached until ~{cache_expiry.strftime("%H:%M")} (4h TTL)</div>',
        unsafe_allow_html=True,
    )

    tab1, tab_sector = st.tabs(["📊 Macro Dashboard", "🔄 Sector Rotation"])

    with tab1:
        # ── Ticker Bar ──
        _TICKER_BAR = [
            ("SPY", "SPY"), ("NDX", "QQQ"), ("DJ30", "DIA"), ("IWM", "IWM"),
            ("GOLD", "GLD"), ("SILVER", "SLV"), ("OIL", "USO"), ("TLT", "TLT"),
        ]
        cells = []
        for label, ticker in _TICKER_BAR:
            snap = snaps.get(ticker)
            if snap and snap.latest_price:
                pct_1d = snap.pct_change_1d or 0.0
                pct_ytd = snap.pct_change_ytd
                color_1d = COLORS["green"] if pct_1d >= 0 else COLORS["red"]
                arrow = "▲" if pct_1d >= 0 else "▼"
                if pct_ytd is not None:
                    color_ytd = COLORS["green"] if pct_ytd >= 0 else COLORS["red"]
                    ytd_html = f'<div style="font-size:10px;color:{color_ytd};">YTD {pct_ytd:+.2f}%</div>'
                else:
                    ytd_html = '<div style="font-size:10px;color:#555;">YTD —</div>'
                cells.append(
                    f'<div style="display:inline-block;padding:4px 12px;text-align:center;">'
                    f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};letter-spacing:0.06em;">{label}</div>'
                    f'<div style="font-size:15px;font-weight:700;color:{COLORS["text"]};">${snap.latest_price:,.2f}</div>'
                    f'<div style="font-size:12px;color:{color_1d};">{arrow} {pct_1d:+.2f}%</div>'
                    f'{ytd_html}'
                    f'</div>'
                )
        if cells:
            bar_html = (
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:4px;padding:6px 4px;margin-bottom:12px;display:flex;'
                f'justify-content:space-around;font-family:\'JetBrains Mono\',Consolas,monospace;">'
                + "".join(cells) + '</div>'
            )
            st.markdown(bar_html, unsafe_allow_html=True)

        regime = macro["macro_regime"]
        regime_color = COLORS["green"] if regime == "Risk-On" else COLORS["red"] if regime == "Risk-Off" else COLORS["yellow"]

        # ── Gauge + Top-level metrics ──
        col_gauge, col_metrics = st.columns([1, 2])
        with col_gauge:
            gauge_fig = _make_gauge(macro["macro_score"], regime, regime_color)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_metrics:
            m1, m2, m3 = st.columns(3)
            m1.markdown(bloomberg_metric("Macro Score", str(macro["macro_score"])), unsafe_allow_html=True)
            m2.markdown(bloomberg_metric("Quadrant", macro["quadrant"]), unsafe_allow_html=True)
            m3.markdown(bloomberg_metric("Regime", regime, regime_color), unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:11px;color:{COLORS["text_dim"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                f'margin-top:8px;letter-spacing:0.03em;">'
                f'CONFIDENCE {_confidence_label(macro["avg_confidence"])} ({macro["avg_confidence"]}%) '
                f'| GROWTH {macro["growth_dir"]} | INFLATION {macro["inflation_dir"]}</div>',
                unsafe_allow_html=True,
            )

        # ── Signal Radar ──
        radar_fig = _make_category_radar(macro["signals"])
        if radar_fig:
            _section_header("Signal Radar")
            st.plotly_chart(radar_fig, use_container_width=True)

        # ── Regime History ──
        _section_header("Regime History")
        history_fig = _make_regime_history()
        if history_fig:
            st.plotly_chart(history_fig, use_container_width=True)
        st.caption(f"Daily macro verdict: {regime}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # ── Current Events ──
        _section_header("Current Events")
        from services.news_feed import (
            fetch_financial_headlines, load_news_inbox, save_to_inbox, clear_inbox,
            headlines_to_text, inbox_to_text,
        )
        from services.claude_client import GROQ_API_URL
        import requests as _req_ce

        _headlines = fetch_financial_headlines()
        _inbox = load_news_inbox()

        _ce_col1, _ce_col2 = st.columns([3, 1])
        with _ce_col1:
            if _headlines:
                st.markdown(
                    f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                    f'letter-spacing:0.08em;margin-bottom:4px;">RSS HEADLINES</div>',
                    unsafe_allow_html=True,
                )
                for _h in _headlines[:12]:
                    st.markdown(
                        f'<div style="padding:3px 0;border-bottom:1px solid #1e293b;">'
                        f'<span style="font-size:10px;color:#f97316;">[{_h["source"]}]</span> '
                        f'<a href="{_h.get("url","#")}" target="_blank" style="font-size:12px;color:#C8D8E8;text-decoration:none;">'
                        f'{_h["title"][:90]}{"…" if len(_h["title"])>90 else ""}</a>'
                        f'<span style="font-size:10px;color:#475569;margin-left:6px;">{_h["date"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("RSS feeds unavailable — check connection.")

        with _ce_col2:
            if st.button("🔄 Refresh Headlines", key="refresh_headlines"):
                fetch_financial_headlines.clear()
                st.rerun()

        if _inbox:
            st.markdown(
                f'<div style="font-size:10px;font-weight:700;color:#64748b;'
                f'letter-spacing:0.08em;margin:10px 0 4px 0;">📥 INBOX ({len(_inbox)} items)</div>',
                unsafe_allow_html=True,
            )
            for _item in reversed(_inbox[-5:]):
                _ts = _item.get("ts", "")[:16].replace("T", " ")
                _src = _item.get("source", "manual")
                st.markdown(
                    f'<div style="background:#0f172a;border-left:3px solid #f97316;'
                    f'padding:6px 10px;margin:3px 0;border-radius:0 4px 4px 0;">'
                    f'<span style="font-size:10px;color:#64748b;">{_ts} · {_src}</span><br>'
                    f'<span style="font-size:12px;color:#C8D8E8;">{_item.get("text","")[:200]}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑 Clear Inbox", key="clear_inbox_btn"):
                clear_inbox()
                st.rerun()

        with st.expander("📝 Add Note / Paste X Post", expanded=False):
            _note_text = st.text_area("Paste an X post, news snippet, or your own note:", key="ce_note_input", height=100)
            if st.button("💾 Save to Inbox", key="ce_save_btn"):
                if _note_text.strip():
                    save_to_inbox(_note_text.strip(), source="manual")
                    st.success("Saved to inbox.")
                    st.rerun()

        # AI News Digest
        _ce_has_claude = bool(os.getenv("XAI_API_KEY"))
        _ce_tier_opts = ["⚡ Freeloader Mode"] + (["🧠 Regard Mode"] if _ce_has_claude else [])
        _ced1, _ced2 = st.columns([2, 2])
        with _ced1:
            _ce_tier = st.radio("Engine", _ce_tier_opts, horizontal=True, key="ce_engine_radio")
            st.markdown(
                '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
                'margin-top:-10px;margin-bottom:2px;">'
                '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
                '</div>',
                unsafe_allow_html=True,
            )
        with _ced2:
            _ce_run = st.button("🗞 Generate News Digest", key="ce_digest_btn", type="primary")

        if _ce_run:
            _ce_ctx = headlines_to_text(_headlines, 15) + "\n\n" + inbox_to_text(_inbox, 10)
            _ce_prompt = (
                f"You are a macro strategist. Summarize these financial headlines and notes "
                f"into a 3-4 sentence digest for an investor. Focus on: rate expectations, "
                f"geopolitical risks, sector rotation signals, and anything surprising.\n\n"
                f"Headlines & Notes:\n{_ce_ctx}"
            )
            _groq_key = os.getenv("GROQ_API_KEY", "")
            _digest = None
            if _ce_tier != "⚡ Freeloader Mode" and _ce_has_claude:
                try:
                    import anthropic as _ac
                    _ac_client = _ac.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                    _digest = _ac_client.messages.create(
                        model="grok-4-1-fast-reasoning", max_tokens=400, temperature=0.3,
                        messages=[{"role": "user", "content": _ce_prompt}],
                    ).content[0].text.strip()
                except Exception as _ace:
                    st.error(f"Claude error: {_ace}")
            elif _groq_key:
                try:
                    _gr = _req_ce.post(
                        GROQ_API_URL,
                        headers={"Authorization": f"Bearer {_groq_key}", "Content-Type": "application/json"},
                        json={"model": "llama-3.3-70b-versatile",
                              "messages": [{"role": "user", "content": _ce_prompt}],
                              "max_tokens": 400, "temperature": 0.3},
                        timeout=20,
                    )
                    if _gr.ok:
                        _digest = _gr.json()["choices"][0]["message"]["content"].strip()
                    elif _gr.status_code == 400 and _ce_has_claude:
                        import anthropic as _ac2
                        _digest = _ac2.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
                            model="grok-4-1-fast-reasoning", max_tokens=400, temperature=0.3,
                            messages=[{"role": "user", "content": _ce_prompt}],
                        ).content[0].text.strip()
                except Exception as _ge:
                    st.error(f"News digest error: {_ge}")

            if _digest:
                st.session_state["_current_events_digest"] = _digest
                st.session_state["_current_events_digest_ts"] = datetime.now()
                st.session_state["_current_events_engine"] = _ce_tier

        _digest_val = st.session_state.get("_current_events_digest")
        if _digest_val:
            _digest_ts = st.session_state.get("_current_events_digest_ts")
            _digest_age = f"{int((datetime.now()-_digest_ts).total_seconds()/60)}m ago" if _digest_ts else ""
            st.markdown(
                f'<div style="background:#0f172a;border-left:3px solid #22c55e;'
                f'padding:10px 14px;border-radius:0 4px 4px 0;margin-top:8px;">'
                f'<div style="font-size:10px;color:#475569;margin-bottom:4px;">'
                f'NEWS DIGEST · {st.session_state.get("_current_events_engine","")}'
                f'{" · " + _digest_age if _digest_age else ""}</div>'
                f'<div style="font-size:13px;color:#C8D8E8;">{_digest_val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Dalio Quadrant Scatter ──
        with st.expander("📈 Quadrant History — Growth vs Inflation", expanded=False):
            _qs_fig = _make_quadrant_scatter()
            if _qs_fig:
                st.plotly_chart(_qs_fig, use_container_width=True)
                st.caption("Each dot = one trading day. X-axis = growth signal (yield curve, equities, ISM, unemployment). Y-axis = inflation signal (PCE trend, commodities). Current dot is highlighted with date label.")
            else:
                st.info("Accumulating history — quadrant scatter will appear after the first regime snapshot is saved.")

        # ── Core Signals ──
        _section_header(f"Core Signals ({len(macro['signals'])})")
        _render_signals_table(macro["signals"])

        na_count = sum(1 for s in macro["signals"] if "N/A" in s.get("Value", ""))
        if na_count:
            cols = st.columns([6, 1])
            cols[0].warning(
                f"{na_count} signal(s) showing N/A — FRED data may be temporarily unavailable. "
                "Click 'Retry' to refresh. Signals auto-recover on next cache cycle."
            )
            if cols[1].button("Retry", key="retry_signals"):
                st.cache_data.clear()
                st.rerun()

        # ── Signal Health ──
        with st.expander("Signal Health", expanded=False):
            _md_age_days = _age_days
            health_rows = []
            _expected_freq = {
                "Yield Curve (10Y-2Y)": 1, "Credit Spreads (HY vs Treasuries)": 1,
                "Financial Conditions Index": 7, "VIX (Equity Volatility)": 1,
                "Equity Trend (S&P, Nasdaq, Dow)": 1, "Commodity Trend (Oil + Copper)": 1,
                "US Dollar Index (DXY proxy)": 1, "Initial Jobless Claims": 7,
                "HYG/LQD Ratio (Credit Risk Appetite)": 1, "Copper/Gold Ratio (Growth vs Safety)": 1,
                "Global Liquidity (M2 proxy)": 30, "Unemployment Trend (Sahm context)": 30,
                "Core Inflation (PCE)": 30, "Industrial Production": 30,
                "Term Premium": 7, "S&P 500 P/E (CAPE proxy)": 1,
                "Corporate CAPEX vs Liquidity": 90, "Leading Economic Index": 30,
                "Gamma Exposure (Dealer Positioning)": 1,
                "Consumer Sentiment (Michigan)": 30, "Building Permits": 30,
            }
            for sig in macro["signals"]:
                name = sig["Indicator"]
                conf = sig["Confidence"]
                expected = _expected_freq.get(name, 7)
                conf_color = COLORS["green"] if conf >= 75 else (COLORS["yellow"] if conf >= 50 else COLORS["red"])
                stale_flag = "STALE" if conf < 50 else ""
                health_rows.append({
                    "Signal": name,
                    "Confidence": f"{conf}%",
                    "Expected Freq": f"{expected}d",
                    "Status": stale_flag,
                })
            health_df = pd.DataFrame(health_rows)

            def _color_status(val):
                if val == "STALE":
                    return f"color: {COLORS['red']}; font-weight: bold"
                return ""

            styled_health = health_df.style.map(_color_status, subset=["Status"])
            st.dataframe(styled_health, use_container_width=True, hide_index=True)

        # ── Yield Curve Regime ──
        yc_regime = macro.get("yield_curve_regime", {})
        if yc_regime.get("regime") != "Unknown":
            _section_header("Yield Curve Regime")
            regime_name = yc_regime["regime"]
            inv_tag = " **(Inverted)**" if yc_regime.get("inverted") else ""

            yc_descriptions = {
                "Bull Steepening": "Curve widening with rates falling — Fed easing expectations, positive for risk assets",
                "Bear Steepening": "Curve widening with rates rising — inflation fears, long-end selling off",
                "Bull Flattening": "Curve narrowing with rates falling — flight to safety, slowing growth expectations",
                "Bear Flattening": "Curve narrowing with rates rising — Fed tightening, short-end rates rising faster",
            }

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Regime:** {regime_name}{inv_tag}")
            c2.markdown(f"**10Y-2Y Spread:** {yc_regime['spread_now']} bps ({yc_regime['spread_change']:+.3f} 30d chg)")
            c3.markdown(f"**10Y Yield:** {yc_regime['ten_year_now']}% ({yc_regime['rate_direction']})")
            st.caption(yc_descriptions.get(regime_name, ""))

        # ── Signal Changes ──
        history = _load_history()
        if len(history) >= 2:
            prev = history[-2]
            curr = history[-1]
            prev_sigs = prev.get("signals_summary", {})
            curr_sigs = curr.get("signals_summary", {})
            changes = []
            for name, curr_score in curr_sigs.items():
                prev_score = prev_sigs.get(name)
                if prev_score is not None:
                    prev_label = _label_from_score(prev_score)
                    curr_label = _label_from_score(curr_score)
                    if prev_label != curr_label:
                        changes.append(f"**{name}**: {prev_label} → {curr_label}")
            if changes:
                with st.expander(f"Signal Changes vs Previous Session ({prev.get('date', '?')})", expanded=True):
                    for c in changes:
                        st.markdown(f"- {c}")

        # ── Valuation ──
        _section_header("Valuation")
        cape_txt = "N/A" if macro["cape"] is None else f"{macro['cape']:.2f}x"
        st.markdown(bloomberg_metric("S&P 500 P/E", cape_txt), unsafe_allow_html=True)
        st.caption(macro["valuation"])

        # ── Cycle Stage ──
        _section_header("Cycle Stage")

        c1, c2, c3, c4 = st.columns(4)
        capex_lvl = macro.get("capex_level")
        m2_lvl = macro.get("m2_level")
        capex_yoy_val = macro.get("capex_yoy")
        m2_yoy_val = macro.get("m2_yoy")

        c1.markdown(bloomberg_metric("CAPEX (PNFI)",
            f"${capex_lvl:,.0f}B" if capex_lvl is not None else "N/A"), unsafe_allow_html=True)
        c2.markdown(bloomberg_metric("M2 Money Supply",
            f"${m2_lvl:,.0f}B" if m2_lvl is not None else "N/A"), unsafe_allow_html=True)
        c3.markdown(bloomberg_metric("CAPEX YoY",
            f"{capex_yoy_val:+.1f}%" if capex_yoy_val is not None else "N/A"), unsafe_allow_html=True)
        c4.markdown(bloomberg_metric("M2 YoY",
            f"{m2_yoy_val:+.1f}%" if m2_yoy_val is not None else "N/A"), unsafe_allow_html=True)

        capliq_txt = "N/A" if macro["capex_vs_liquidity"] is None else f"{macro['capex_vs_liquidity']:.2f}pp"
        st.markdown(bloomberg_metric("CAPEX vs Liquidity Spread", capliq_txt), unsafe_allow_html=True)
        st.caption(macro["cycle_stage"])

        # ── Summary ──
        _section_header("Summary")
        for line in macro["summary"]:
            st.markdown(f"- {line}")

        # ── Portfolio Bias ──
        _section_header("Portfolio Bias")
        bias_items = list(macro["portfolio_bias"].items())
        cols = st.columns(len(bias_items))
        for col, (sleeve, bias) in zip(cols, bias_items):
            col.markdown(bloomberg_metric(sleeve, bias), unsafe_allow_html=True)

        # ── Sector Rotation ──
        _section_header("Sector Rotation")
        sector_recs = macro.get("sector_rotation", [])
        if sector_recs:
            col_favor, col_avoid = st.columns(2)
            with col_favor:
                st.markdown(
                    f'<div style="color:{COLORS["green"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Favor</div>',
                    unsafe_allow_html=True,
                )
                for rec in sector_recs:
                    if rec["action"] == "Favor":
                        mom_str = f" ({rec['momentum_30d']:+.1f}% 30d)" if rec["momentum_30d"] is not None else ""
                        st.markdown(f"- **{rec['ticker']}** {rec['label']}{mom_str} — {rec['reason']}")
            with col_avoid:
                st.markdown(
                    f'<div style="color:{COLORS["red"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:13px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Avoid</div>',
                    unsafe_allow_html=True,
                )
                for rec in sector_recs:
                    if rec["action"] == "Avoid":
                        mom_str = f" ({rec['momentum_30d']:+.1f}% 30d)" if rec["momentum_30d"] is not None else ""
                        st.markdown(f"- **{rec['ticker']}** {rec['label']}{mom_str} — {rec['reason']}")
        else:
            st.markdown("Sector rotation data unavailable.")

        # ── AI Regime Plays ──────────────────────────────────────────────────
        _rp_has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
        _rp_tier_opts = ["⚡ Freeloader Mode", "🧠 Regard Mode", "👑 Highly Regarded Mode"] if _rp_has_anthropic else ["⚡ Freeloader Mode"]
        _rp_tier_map  = {
            "⚡ Freeloader Mode":                (False, None),
            "🧠 Regard Mode":         (True,  "grok-4-1-fast-reasoning"),
            "👑 Highly Regarded Mode": (True,  "claude-sonnet-4-6"),
        }
        _prev_rp_tier = st.session_state.get("_rp_tier_prev")
        _sel_rp_tier = st.radio(
            "Engine", _rp_tier_opts, horizontal=True, key="regime_plays_engine_radio",
            help="Sonnet gives the most nuanced regime play synthesis"
        )
        st.markdown(
            '<div style="font-size:10px;color:#64748b;font-family:\'JetBrains Mono\',Consolas,monospace;'
            'margin-top:-10px;margin-bottom:2px;">'
            '⚡ llama-3.3-70b &nbsp;·&nbsp; 🧠 grok-4-1-fast-reasoning &nbsp;·&nbsp; 👑 claude-sonnet-4-6'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption("💡 🧠 Grok 4.1 sufficient here — structured JSON output, Groq also works well")
        _use_cl_rp, _cl_rp_model = _rp_tier_map[_sel_rp_tier]
        st.session_state["_rp_tier_prev"] = _sel_rp_tier

        _gen_rp = st.button("Generate Regime Plays", type="primary", key="gen_regime_plays_btn")
        if _gen_rp or st.session_state.get("_rp_plays_result"):
            if _gen_rp:
                from services.claude_client import suggest_regime_plays
                _top_sigs = macro.get("top_signals", [])[:5]
                _sig_lines = [f"- {s['name']}: z={s['score']:+.2f}" for s in _top_sigs]
                _signal_summary = "\n".join(_sig_lines) if _sig_lines else "No signal data available."
                _score_norm = (macro.get("macro_score", 50) - 50) / 50
                with st.spinner("Generating AI regime plays..."):
                    _plays = suggest_regime_plays(
                        macro.get("macro_regime", "Unknown"),
                        _score_norm,
                        _signal_summary,
                        use_claude=_use_cl_rp,
                        model=_cl_rp_model,
                    )
                st.session_state["_rp_plays_result"] = _plays
                st.session_state["_rp_plays_last_tier"] = _sel_rp_tier
                from services.play_log import append_play as _append_play
                _append_play("AI Regime Plays", _sel_rp_tier, _plays,
                             meta={"regime": macro.get("macro_regime"), "score": round(_score_norm, 3)})
            else:
                _plays = st.session_state["_rp_plays_result"]

            _section_header("AI Regime Plays", badge=st.session_state.get("_rp_plays_last_tier", _sel_rp_tier))
            if _plays.get("rationale"):
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border-left:3px solid {COLORS["bloomberg_orange"]};'
                    f'padding:10px 14px;margin-bottom:12px;font-size:13px;color:{COLORS["text"]};">'
                    f'{_plays["rationale"]}</div>',
                    unsafe_allow_html=True,
                )
            _col_s, _col_st, _col_b = st.columns(3)
            with _col_s:
                st.markdown(
                    f'<div style="color:{COLORS["bloomberg_orange"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Sectors</div>',
                    unsafe_allow_html=True,
                )
                for item in (_plays.get("sectors") or []):
                    _conv = item.get("conviction", 1)
                    _stars = "★" * _conv + "☆" * (3 - _conv)
                    st.markdown(f"`{_stars}` {item['name']}")
            with _col_st:
                st.markdown(
                    f'<div style="color:{COLORS["bloomberg_orange"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Stocks</div>',
                    unsafe_allow_html=True,
                )
                for item in (_plays.get("stocks") or []):
                    _conv = item.get("conviction", 1)
                    _stars = "★" * _conv + "☆" * (3 - _conv)
                    st.markdown(f"`{_stars}` **{item['ticker']}** — {item.get('reason', '')}")
            with _col_b:
                st.markdown(
                    f'<div style="color:{COLORS["bloomberg_orange"]};font-family:\'JetBrains Mono\',Consolas,monospace;'
                    f'font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:6px;">Bonds</div>',
                    unsafe_allow_html=True,
                )
                for item in (_plays.get("bonds") or []):
                    _conv = item.get("conviction", 1)
                    _stars = "★" * _conv + "☆" * (3 - _conv)
                    st.markdown(f"`{_stars}` **{item['ticker']}** — {item.get('reason', '')}")

        # ── Risk Management Alerts ──
        _section_header("Risk Management Alerts")
        for alert in macro.get("risk_alerts", ["No elevated risk signals detected."]):
            st.markdown(
                f'<div style="border-left:2px solid {COLORS["red"]};padding:4px 10px;margin:4px 0;'
                f'font-family:\'JetBrains Mono\',Consolas,monospace;font-size:13px;color:{COLORS["text"]};">{alert}</div>',
                unsafe_allow_html=True,
            )

        # ── Tactical Opportunities ──
        _section_header("Tactical Opportunities")
        tactical = macro.get("tactical_opps", [])
        if tactical:
            for opp in tactical:
                ticker_strs = []
                for t in opp["tickers"]:
                    snap = snaps.get(t)
                    mom = snap.pct_change_30d if snap and snap.pct_change_30d is not None else None
                    mom_str = f" ({mom:+.1f}% 30d)" if mom is not None else ""
                    ticker_strs.append(f"**{t}**{mom_str}")
                st.markdown(f"- **{opp['signal']}**: {opp['opportunity']} — {', '.join(ticker_strs)}")
        else:
            st.markdown("No strong cross-signal opportunities detected currently.")

        # ── SPY Options Sentiment (pre-fetched in parallel) ──
        _section_header("SPY Options Sentiment")
        if gamma:
            asof = gamma.get("asof")
            if asof:
                try:
                    asof_ts = pd.to_datetime(asof, errors="coerce")
                    if pd.notna(asof_ts):
                        if asof_ts.tzinfo is not None:
                            asof_ts = asof_ts.tz_convert(None)
                        st.caption(f"Last options fetch: {asof_ts.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                except Exception:
                    pass

            st.markdown(f"- Current market price: {gamma['price']:.2f}")
            st.markdown(f"- Current market sentiment: {gamma['zone']}")
            st.markdown(f"- Call Wall price: {gamma['call_wall']:.2f}" if gamma["call_wall"] is not None else "- Call Wall price: N/A")
            st.markdown(f"- Put Wall price: {gamma['put_wall']:.2f}" if gamma["put_wall"] is not None else "- Put Wall price: N/A")

            m1, m2 = st.columns(2)
            m1.markdown(bloomberg_metric("Call Wall", f"{gamma['call_wall']:.2f}" if gamma["call_wall"] is not None else "N/A", COLORS["green"]), unsafe_allow_html=True)
            m2.markdown(bloomberg_metric("Put Wall", f"{gamma['put_wall']:.2f}" if gamma["put_wall"] is not None else "N/A", COLORS["red"]), unsafe_allow_html=True)

            fig = go.Figure()
            bar_colors = [COLORS["green"] if val >= 0 else COLORS["red"] for val in gamma["net_gamma"]]
            fig.add_trace(go.Bar(
                x=gamma["strikes"],
                y=gamma["net_gamma"],
                marker_color=bar_colors,
                name="Net Gamma Proxy",
                opacity=0.65,
                hovertemplate="Strike %{x:.0f}<br>Net Gamma %{y:.0f}<extra></extra>",
            ))

            fig.add_vline(x=gamma["price"], line_color=COLORS["blue"], line_dash="dash", line_width=2)
            if gamma["call_wall"] is not None:
                fig.add_vline(x=gamma["call_wall"], line_color=COLORS["green"], line_dash="dash", line_width=1)
            if gamma["put_wall"] is not None:
                fig.add_vline(x=gamma["put_wall"], line_color=COLORS["red"], line_dash="dash", line_width=1)

            fig.update_layout(
                title="SPY Strike vs Dealer Gamma Proxy",
                xaxis_title="Strike",
                yaxis_title="Net Gamma Proxy",
                height=360,
                margin=dict(l=20, r=20, t=50, b=20),
                showlegend=False,
            )
            apply_dark_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("SPY options sentiment currently unavailable.")
            if st.button("Retry SPY Gamma", key="retry_gamma"):
                st.cache_data.clear()
                st.rerun()

    with tab_sector:
        _render_sector_rotation_tab(macro["quadrant"], macro["macro_regime"])
