"""
Module 0: Macro Dashboard

Daily macro regime indicator using 16 cross-asset signals:
- FRED macro series (yield curve, credit spreads, ISM, FCI, etc.)
- ETF proxies (equities, commodities, FX, volatility)
- SPY options chain (dealer gamma positioning)

Output:
- Risk-On / Neutral / Risk-Off verdict with Macro Score (0-100)
- Ray Dalio quadrant (Goldilocks / Reflation / Stagflation / Deflation)
- Valuation (CAPE + Buffett Indicator)
- Cycle stage (CAPEX vs Liquidity)
- Portfolio bias by asset class
- SPY gamma sentiment (zone, flip, call wall, put wall)

Architecture:
- 16-indicator scoring engine (_build_macro_dashboard)
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
from concurrent.futures import ThreadPoolExecutor

from services.market_data import (
    fetch_batch_safe, AssetSnapshot,
    fetch_fred_series_safe, fetch_options_chain_snapshot_safe,
)
from utils.theme import COLORS, apply_dark_layout

# ─────────────────────────────────────────────
# HISTORY PERSISTENCE
# ─────────────────────────────────────────────

_HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "regime_history.json")


def _load_history() -> list[dict]:
    """Load regime history from JSON file."""
    if os.path.exists(_HISTORY_FILE):
        try:
            with open(_HISTORY_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_snapshot(macro: dict):
    """Persist today's regime snapshot (one entry per calendar day)."""
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    history = _load_history()
    today = datetime.now().strftime("%Y-%m-%d")

    # Map macro_score (0-100) to -1..+1 for history chart continuity
    score_normalized = (macro["macro_score"] - 50) / 50.0

    history = [h for h in history if h.get("date") != today]
    history.append({
        "date": today,
        "score": round(score_normalized, 3),
        "regime": macro["macro_regime"],
        "signal_count": len(macro["signals"]),
        "macro_score": macro["macro_score"],
        "quadrant": macro["quadrant"],
        "signals_summary": {s["Indicator"]: s["Score"] for s in macro["signals"]},
    })

    # Keep last 365 days max
    history = sorted(history, key=lambda x: x["date"])[-365:]

    with open(_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────
# TICKER UNIVERSE
# ─────────────────────────────────────────────

TICKERS = {
    # Yield curve proxies
    "SHY": "2Y Treasury ETF",
    "IEF": "10Y Treasury ETF",
    "TLT": "20Y+ Treasury ETF",
    # Credit spreads
    "JNK": "HY Corporate Bonds",
    "LQD": "IG Corporate Bonds",
    "HYG": "HY (Alt)",
    # Inflation
    "TIP": "TIPS ETF",
    "RINF": "Inflation Expectations",
    # Equity - US
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "^DJI": "Dow Jones Industrial Average",
    "IWM": "Russell 2000",
    "XLF": "Financials",
    "XLE": "Energy",
    # Equity - International
    "EEM": "Emerging Markets",
    "EFA": "Developed ex-US",
    "FXI": "China",
    # Commodities
    "GLD": "Gold",
    "SLV": "Silver",
    "CPER": "Copper",
    "USO": "Oil (WTI)",
    # Volatility
    "^VIX": "VIX",
    # USD
    "UUP": "USD Bull ETF",
    # Crypto proxy
    "IBIT": "Bitcoin ETF",
    # Ticker bar extras
    "DIA": "Dow Jones ETF",
    "CL=F": "WTI Crude Oil",
    "GC=F": "Gold Futures",
}


# ─────────────────────────────────────────────
# SCORING UTILITIES
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _label_from_score(score: float) -> str:
    if score >= 0.2:
        return "Risk-On"
    elif score <= -0.2:
        return "Risk-Off"
    return "Neutral"


def _score_to_bucket(score: float) -> tuple[str, str]:
    if score >= 0.2:
        return "🟢", "Risk-On"
    if score <= -0.2:
        return "🔴", "Risk-Off"
    return "🟡", "Neutral"


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


def _interpret_valuation(cape: float | None, buffett: float | None) -> str:
    if cape is None and buffett is None:
        return "Valuation data unavailable."
    if cape is not None and buffett is not None:
        if cape > 25 and buffett > 150:
            return "Both P/E and Buffett Indicator point to an expensive equity market."
        if cape < 18 and buffett < 110:
            return "Both P/E and Buffett Indicator suggest relatively attractive long-term valuation."
    if cape is not None:
        if cape > 23:
            return "S&P 500 P/E is elevated versus long-run norms, implying lower forward return expectations."
        if cape < 18:
            return "S&P 500 P/E is below long-run highs, valuation risk appears more moderate."
    if buffett is not None:
        if buffett > 145:
            return "Buffett Indicator is elevated, signaling stretched market-cap-to-GDP conditions."
        if buffett < 110:
            return "Buffett Indicator is in a more balanced zone relative to GDP."
    return "Valuation is mixed; neither clearly cheap nor deeply stretched."


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
    ism_score = score_map.get("ISM Manufacturing", 0)
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
        opps.append({"signal": "Credit tight + ISM rising", "opportunity": "Favor high-yield over investment-grade bonds", "tickers": ["JNK", "HYG"]})

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

        # 200d MA
        if len(s) >= 200:
            ma200 = float(s.rolling(200).mean().iloc[-1])
            pct = (current / ma200 - 1) * 100
            note = "above" if pct > 0 else "below"
            levels.append({"asset": ticker, "level_type": "200d MA", "price": round(ma200, 2), "current": round(current, 2), "pct_away": round(pct, 2), "note": f"{note} 200d trend"})

        # 50d MA
        if len(s) >= 50:
            ma50 = float(s.rolling(50).mean().iloc[-1])
            pct = (current / ma50 - 1) * 100
            note = "above" if pct > 0 else "below"
            levels.append({"asset": ticker, "level_type": "50d MA", "price": round(ma50, 2), "current": round(current, 2), "pct_away": round(pct, 2), "note": f"{note} 50d trend"})

        # 52-week high/low
        if len(s) >= 252:
            window = s.iloc[-252:]
        else:
            window = s
        hi52 = float(window.max())
        lo52 = float(window.min())
        pct_hi = (current / hi52 - 1) * 100
        pct_lo = (current / lo52 - 1) * 100
        levels.append({"asset": ticker, "level_type": "52w High", "price": round(hi52, 2), "current": round(current, 2), "pct_away": round(pct_hi, 2), "note": "from 52-week high"})
        levels.append({"asset": ticker, "level_type": "52w Low", "price": round(lo52, 2), "current": round(current, 2), "pct_away": round(pct_lo, 2), "note": "from 52-week low"})

    # SPY gamma levels
    gamma = macro.get("gamma")
    if gamma:
        spy_current = gamma.get("price", 0)
        for level_name, key in [("Gamma Flip", "gamma_flip"), ("Call Wall", "call_wall"), ("Put Wall", "put_wall")]:
            val = gamma.get(key)
            if val is not None and spy_current:
                pct = (spy_current / val - 1) * 100
                levels.append({"asset": "SPY", "level_type": level_name, "price": round(val, 2), "current": round(spy_current, 2), "pct_away": round(pct, 2), "note": "options-derived level"})

    return levels


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_all_data() -> dict[str, AssetSnapshot]:
    """Fetch all ticker data via shared market_data service."""
    return fetch_batch_safe(TICKERS, period="1y", interval="1d")


# ─────────────────────────────────────────────
# SPY GAMMA MODE
# ─────────────────────────────────────────────

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


@st.cache_data(ttl=3600)
def _fetch_spy_pe() -> float:
    """Fetch SPY trailing P/E with raise-on-failure to avoid caching None."""
    try:
        val = yf.Ticker("SPY").info.get("trailingPE")
        if val is not None:
            return float(val)
    except Exception:
        pass
    raise _SpyPeFetchError("Failed to fetch SPY P/E")


def _build_macro_dashboard(snaps: dict[str, AssetSnapshot], low_compute_mode: bool = False) -> dict:
    fred_ids = {
        "yield_curve": "T10Y2Y",
        "credit_spread": "BAMLH0A0HYM2",
        "m2": "M2SL",
        "sahm": "SAHMREALTIME",
        "unrate": "UNRATE",
        "core_pce": "PCEPILFE",
        "wilshire": "WILL5000INDFC",  # Wilshire 5000 Full Cap Price Index (daily)
        "gdp": "GDP",
        "capex": "PNFI",  # Private Nonresidential Fixed Investment (quarterly BEA)
        "term_premium": "THREEFYTP10",
        "ism": "INDPRO",  # Industrial Production Index (monthly, replaces discontinued NAPM)
        "fci": "NFCI",
        "dgs10": "DGS10",  # 10-Year Treasury yield (for yield curve regime classification)
    }
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {k: executor.submit(fetch_fred_series_safe, v) for k, v in fred_ids.items()}
        fred = {k: f.result() for k, f in futures.items()}

    # SPY trailing P/E as CAPE proxy (FRED has no Shiller CAPE series)
    try:
        spy_pe = _fetch_spy_pe()
    except _SpyPeFetchError:
        spy_pe = None

    indicators = []

    yc = _safe_latest(fred["yield_curve"])
    yc_score = _clamp_score((yc or 0.0), 1.0)
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
    cs_score = _clamp_score((4.0 - (cs or 4.0)), 2.0)
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
    liquidity_score = _clamp_score(((m2_yoy or 0.0) - 2.0), 4.0)
    indicators.append(("Global Liquidity (M2 proxy)", m2_yoy, "% YoY", liquidity_score, _confidence_from_age(fred["m2"], expected_days=45)))

    sahm = _safe_latest(fred["sahm"])
    if sahm is None:
        unrate = fred["unrate"].dropna() if fred["unrate"] is not None else None
        if unrate is not None and len(unrate) >= 12:
            sahm = float(unrate.iloc[-1] - unrate.iloc[-12:].min())
    unemp_score = _clamp_score((0.4 - (sahm or 0.0)), 0.4)
    indicators.append(("Unemployment Trend (Sahm context)", sahm, "delta", unemp_score, _confidence_from_age(fred["sahm"] if fred["sahm"] is not None else fred["unrate"], expected_days=45)))

    core_yoy = _yoy_latest(fred["core_pce"], periods=12)
    core_infl_score = _clamp_score((2.4 - (core_yoy or 2.4)), 1.5)
    indicators.append(("Core Inflation (PCE)", core_yoy, "% YoY", core_infl_score, _confidence_from_age(fred["core_pce"], expected_days=45)))

    eq_components = []
    for ticker in ("SPY", "QQQ", "^DJI"):
        s = snaps.get(ticker).series if snaps.get(ticker) else None
        if s is not None and len(s) >= 200:
            ma200 = s.rolling(200).mean().iloc[-1]
            if ma200 and ma200 != 0:
                eq_components.append(float((s.iloc[-1] / ma200 - 1) * 100))
    eq_trend = float(np.mean(eq_components)) if eq_components else None
    equity_score = _clamp_score((eq_trend or 0.0), 5.0)
    indicators.append(("Equity Trend (S&P, Nasdaq, Dow)", eq_trend, "% vs 200d MA", equity_score, _confidence_from_snap("SPY", "QQQ", "^DJI", snaps=snaps)))

    cape = float(spy_pe) if spy_pe is not None else None
    cape_score = _clamp_score((25.0 - (cape or 25.0)), 10.0)
    indicators.append(("S&P 500 P/E (CAPE proxy)", cape, "x", cape_score, 85 if cape is not None else 0))

    wilshire = _safe_latest(fred["wilshire"])
    gdp = _safe_latest(fred["gdp"])
    # WILL5000INDFC ≈ market cap in billions (1 point ≈ $1B), GDP also in billions
    buffett = (wilshire / gdp * 100) if (wilshire is not None and gdp and gdp != 0) else None
    buffett_score = _clamp_score((120.0 - (buffett or 120.0)), 40.0)
    indicators.append(("Buffett Indicator (Mkt Cap / GDP)", buffett, "%", buffett_score, int(round(np.mean([
        _confidence_from_age(fred["wilshire"], expected_days=7),
        _confidence_from_age(fred["gdp"], expected_days=120),
    ])))))

    capex_yoy = _yoy_latest(fred["capex"], periods=4)
    capex_vs_liquidity = (capex_yoy - m2_yoy) if (capex_yoy is not None and m2_yoy is not None) else None
    capliq_score = _clamp_score((capex_vs_liquidity or 0.0), 5.0)
    indicators.append(("Corporate CAPEX vs Liquidity", capex_vs_liquidity, "pp", capliq_score, int(round(np.mean([
        _confidence_from_age(fred["capex"], expected_days=120),
        _confidence_from_age(fred["m2"], expected_days=45),
    ])))))

    gamma_data = _compute_spy_gamma_mode(max_expiries=1 if low_compute_mode else 2)
    gamma_score = 0.0
    if gamma_data and len(gamma_data["net_gamma"]) > 0:
        nearest = int(np.argmin(np.abs(gamma_data["strikes"] - gamma_data["price"])))
        gamma_score = _clamp_score(float(gamma_data["net_gamma"][nearest]), 10000.0)
    gamma_conf = 85 if gamma_data else 35
    indicators.append(("Gamma Exposure (Dealer Positioning)", gamma_score, "score", gamma_score, gamma_conf))

    term = _safe_latest(fred["term_premium"])
    term_score = _clamp_score((term or 0.0), 0.75)
    indicators.append(("Term Premium", term, "%", term_score, _confidence_from_age(fred["term_premium"], expected_days=14)))

    indpro_yoy = _yoy_latest(fred["ism"], periods=12)
    ism_score = _clamp_score((indpro_yoy or 0.0), 5.0)  # Positive YoY = expansion = risk-on
    indicators.append(("Industrial Production", indpro_yoy, "% YoY", ism_score, _confidence_from_age(fred["ism"], expected_days=45)))

    fci = _safe_latest(fred["fci"])
    fci_score = _clamp_score((-(fci or 0.0)), 0.5)
    indicators.append(("Financial Conditions Index", fci, "index", fci_score, _confidence_from_age(fred["fci"], expected_days=14)))

    vix_snap = snaps.get("^VIX")
    vix = vix_snap.latest_price if vix_snap else None
    vix_score = _clamp_score((20.0 - (vix or 20.0)), 8.0)  # VIX 20 = neutral, lower = risk-on
    indicators.append(("VIX (Equity Volatility)", vix, "level", vix_score, _confidence_from_snap("^VIX", snaps=snaps)))

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
        "Buffett Indicator (Mkt Cap / GDP)": "Valuation",
        "Corporate CAPEX vs Liquidity": "Growth",
        "Gamma Exposure (Dealer Positioning)": "Positioning",
        "Term Premium": "Rates",
        "Industrial Production": "Growth",
        "Financial Conditions Index": "Credit",
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
            "Value": display_value,
            "Score": round(float(score), 3),
            "Confidence": f"{_confidence_label(confidence)} ({confidence}%)",
        })

    aggregate = float(np.mean(scores)) if scores else 0.0
    macro_score = int(round((aggregate + 1.0) * 50))

    if macro_score >= 60:
        macro_regime = "Risk-On"
    elif macro_score <= 40:
        macro_regime = "Risk-Off"
    else:
        macro_regime = "Neutral"

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

    valuation_text = _interpret_valuation(cape, buffett)

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

    result = {
        "signals": signal_rows,
        "macro_score": macro_score,
        "avg_confidence": int(round(float(np.mean(confidence_scores)))) if confidence_scores else 0,
        "macro_regime": macro_regime,
        "quadrant": quadrant,
        "growth_dir": growth_dir,
        "inflation_dir": inflation_dir,
        "valuation": valuation_text,
        "cape": cape,
        "buffett": buffett,
        "cycle_stage": cycle_stage,
        "capex_vs_liquidity": capex_vs_liquidity,
        "summary": summary[:5],
        "portfolio_bias": _portfolio_bias(macro_regime),
        "gamma": gamma_data,
        "low_compute_mode": low_compute_mode,
    }

    result["sector_rotation"] = _sector_rotation_recs(quadrant, macro_regime, snaps)
    result["risk_alerts"] = _risk_management_alerts(result, snaps)
    result["tactical_opps"] = _tactical_opportunities(result, snaps)
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
    snaps = fetch_all_data()
    macro = _build_macro_dashboard(snaps, low_compute_mode=True)
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


def _make_regime_history(timeframe: str = "All") -> go.Figure | None:
    """Time-series chart of historical regime scores."""
    history = _load_history()
    if len(history) < 2:
        return None

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    days_map = {"1D": 1, "1W": 7, "1M": 30, "6M": 182, "1Y": 365, "All": None}
    days = days_map.get(timeframe)
    if days is not None:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df = df[df["date"] >= cutoff]
        if len(df) < 2:
            return None

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


def _regime_timeframe_summary(timeframe: str) -> dict | None:
    """Compute summary stats for the selected timeframe window."""
    history = _load_history()
    if len(history) < 2:
        return None

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    days_map = {"1W": 7, "1M": 30, "6M": 182, "1Y": 365}
    days = days_map.get(timeframe)
    if days is None:
        return None
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    df = df[df["date"] >= cutoff]
    if len(df) < 2:
        return None

    # Use macro_score if available, else legacy score
    if "macro_score" in df.columns:
        scores = df["macro_score"].dropna()
        thresh_hi, thresh_lo = 60, 40
    else:
        scores = df["score"].dropna()
        thresh_hi, thresh_lo = 0.35, -0.35

    if len(scores) < 2:
        return None

    avg_score = round(scores.mean(), 1)

    # Classify each day
    risk_on = int((scores >= thresh_hi).sum())
    risk_off = int((scores <= thresh_lo).sum())
    neutral = int(len(scores) - risk_on - risk_off)

    # Dominant regime
    counts = {"Risk-On": risk_on, "Neutral": neutral, "Risk-Off": risk_off}
    dominant = max(counts, key=counts.get)

    # Trend: compare first half avg vs second half avg
    mid = len(scores) // 2
    first_half = scores.iloc[:mid].mean()
    second_half = scores.iloc[mid:].mean()
    diff = second_half - first_half
    # Use a small threshold to avoid noise
    threshold = 3 if "macro_score" in df.columns else 0.05
    if diff > threshold:
        trend = "Improving"
    elif diff < -threshold:
        trend = "Deteriorating"
    else:
        trend = "Stable"

    # Period verdict from avg score
    if "macro_score" in df.columns:
        verdict = "Risk-On" if avg_score >= 60 else ("Risk-Off" if avg_score <= 40 else "Neutral")
    else:
        verdict = "Risk-On" if avg_score >= 0.35 else ("Risk-Off" if avg_score <= -0.35 else "Neutral")

    return {
        "avg_score": avg_score,
        "verdict": verdict,
        "dominant_regime": dominant,
        "risk_on_days": risk_on,
        "risk_off_days": risk_off,
        "neutral_days": neutral,
        "total_days": len(scores),
        "trend": trend,
    }


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

def _section_header(title: str):
    """Render a Bloomberg-style uppercase orange monospace section header."""
    c_orange = COLORS["orange"]
    c_border = COLORS["border"]
    st.markdown(
        f"<div style='font-family:Consolas,monospace;font-size:13px;font-weight:700;"
        f"color:{c_orange};text-transform:uppercase;letter-spacing:1.5px;"
        f"padding:8px 0 4px 0;border-bottom:1px solid {c_border};margin:18px 0 10px 0;'>"
        f"{title}</div>",
        unsafe_allow_html=True,
    )


def _render_signals_table(signals: list[dict]):
    """Render core signals as a Bloomberg-style HTML table."""
    c_border = COLORS["border"]
    c_text = COLORS["text"]
    c_text_dim = COLORS["text_dim"]
    c_green = COLORS["green"]
    c_red = COLORS["red"]
    c_yellow = COLORS["yellow"]
    c_orange = COLORS["orange"]
    c_surface = COLORS["surface_dark"]

    rows = ""
    for s in signals:
        name = s.get("Signal", "")
        value = s.get("Value", "N/A")
        score = s.get("Score", 0)
        direction = s.get("Direction", "Neutral")
        confidence = s.get("Confidence", 0)

        # Score cell color
        try:
            score_val = float(score)
        except (ValueError, TypeError):
            score_val = 0.0
        if score_val > 0.05:
            score_color = c_green
        elif score_val < -0.05:
            score_color = c_red
        else:
            score_color = c_text_dim

        # Direction cell bg
        if "Risk-On" in str(direction) or "Bullish" in str(direction):
            dir_bg = "rgba(0,212,170,0.12)"
            dir_color = c_green
        elif "Risk-Off" in str(direction) or "Bearish" in str(direction):
            dir_bg = "rgba(255,75,75,0.12)"
            dir_color = c_red
        else:
            dir_bg = "transparent"
            dir_color = c_text_dim

        # Confidence bar
        try:
            conf_val = int(confidence)
        except (ValueError, TypeError):
            conf_val = 0
        if conf_val >= 70:
            conf_color = c_green
        elif conf_val >= 40:
            conf_color = c_yellow
        else:
            conf_color = c_red
        conf_bar = (
            f"<div style='display:flex;align-items:center;gap:6px;'>"
            f"<div style='flex:1;height:4px;background:{c_border};border-radius:2px;'>"
            f"<div style='width:{conf_val}%;height:100%;background:{conf_color};border-radius:2px;'></div>"
            f"</div>"
            f"<span style='font-size:10px;color:{conf_color};'>{conf_val}%</span></div>"
        )

        rows += (
            f"<tr style='border-bottom:1px solid {c_border};'>"
            f"<td style='padding:4px 8px;font-size:12px;color:{c_text};'>{name}</td>"
            f"<td style='padding:4px 8px;font-size:12px;color:{c_text};text-align:right;'>{value}</td>"
            f"<td style='padding:4px 8px;font-size:12px;color:{score_color};text-align:right;font-weight:600;'>{score}</td>"
            f"<td style='padding:4px 8px;font-size:12px;color:{dir_color};background:{dir_bg};text-align:center;'>{direction}</td>"
            f"<td style='padding:4px 12px;min-width:100px;'>{conf_bar}</td>"
            f"</tr>"
        )

    header_style = (
        f"font-family:Consolas,monospace;font-size:10px;font-weight:700;"
        f"color:{c_orange};text-transform:uppercase;letter-spacing:1px;"
        f"padding:6px 8px;border-bottom:2px solid {c_border};"
    )
    html = (
        f"<div style='background:{c_surface};border:1px solid {c_border};border-radius:4px;overflow:hidden;'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:Consolas,monospace;'>"
        f"<thead><tr>"
        f"<th style='{header_style}text-align:left;'>Signal</th>"
        f"<th style='{header_style}text-align:right;'>Value</th>"
        f"<th style='{header_style}text-align:right;'>Score</th>"
        f"<th style='{header_style}text-align:center;'>Direction</th>"
        f"<th style='{header_style}text-align:left;'>Confidence</th>"
        f"</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────

def render():
    # Local color aliases to avoid escaped quotes in f-strings
    c_orange = COLORS["orange"]
    c_surface_dark = COLORS["surface_dark"]
    c_border = COLORS["border"]
    c_text = COLORS["text"]
    c_text_dim = COLORS["text_dim"]

    st.markdown(
        f"<div style='font-family:Consolas,monospace;font-size:22px;font-weight:700;"
        f"color:{c_orange};text-transform:uppercase;letter-spacing:2px;'>"
        f"MACRO DASHBOARD</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{c_text_dim};margin-top:-4px;font-family:Consolas,monospace;font-size:11px;'>"
        "Global macro monitor — Risk-On / Risk-Off workflow</p>",
        unsafe_allow_html=True,
    )

    if st.button("Refresh Data"):
        st.cache_data.clear()

    low_compute_mode = st.toggle(
        "Low Compute Mode",
        value=True,
        help="Reduces options processing load and disables gamma chart rendering to conserve free Streamlit usage.",
    )

    with st.spinner("Building macro dashboard..."):
        load_start = datetime.now()
        snaps = fetch_all_data()
        macro = _build_macro_dashboard(snaps, low_compute_mode=low_compute_mode)
        load_secs = (datetime.now() - load_start).total_seconds()

    # Cache freshness indicator — batch TTL is 2h, FRED is 12h
    cache_expiry = datetime.now() + timedelta(seconds=7200)
    st.caption(
        f"Data loaded in {load_secs:.1f}s — cached until ~{cache_expiry.strftime('%H:%M')} "
        f"(batch 2h / FRED 12h). No need to refresh unless markets have moved significantly."
    )

    regime = macro["macro_regime"]
    regime_color = COLORS["green"] if regime == "Risk-On" else COLORS["red"] if regime == "Risk-Off" else COLORS["yellow"]

    # ── Market Ticker Bar ──
    TICKER_BAR = [
        ("QQQ",  "Nasdaq 100 (QQQ)"),
        ("DIA",  "Dow 30 (DIA)"),
        ("SPY",  "S&P 500 (SPY)"),
        ("IWM",  "Russell 2000 (IWM)"),
        ("GC=F", "Gold"),
        ("SLV",  "Silver (SLV)"),
        ("CL=F", "WTI Crude"),
        ("TLT",  "TLT (20Y+)"),
    ]
    ticker_tf = st.radio(
        "Timeframe", ["Daily", "Weekly", "Monthly", "YTD"],
        horizontal=True, key="ticker_bar_tf",
    )
    tf_field = {"Daily": "pct_change_1d", "Weekly": "pct_change_5d", "Monthly": "pct_change_30d", "YTD": "pct_change_ytd"}[ticker_tf]

    bar_snaps = snaps  # reuse already-fetched data
    ticker_cells = []
    for ticker, label in TICKER_BAR:
        snap = bar_snaps.get(ticker)
        price = snap.latest_price if snap else None
        pct = getattr(snap, tf_field, None) if snap else None
        price_str = f"${price:,.0f}" if price is not None and price >= 1000 else f"${price:,.2f}" if price is not None else "N/A"
        if pct is not None:
            arrow = "▲" if pct >= 0 else "▼"
            pct_color = COLORS["green"] if pct >= 0 else COLORS["red"]
            pct_str = f"<span style='color:{pct_color};font-size:11px;'>{arrow} {pct:+.2f}%</span>"
        else:
            pct_str = f"<span style='color:{c_text_dim};font-size:11px;'>N/A</span>"
        ticker_cells.append(
            f"<div style='text-align:center;padding:4px 10px;border-right:1px solid {c_border};'>"
            f"<div style='color:{c_text_dim};font-size:11px;'>{label}</div>"
            f"<div style='font-size:14px;font-weight:700;color:{c_text};'>{price_str}</div>"
            f"<div>{pct_str}</div></div>"
        )
    # Remove border from last cell
    ticker_cells[-1] = ticker_cells[-1].replace(f"border-right:1px solid {c_border};", "")
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;background:{c_surface_dark};"
        f"border:1px solid {c_border};border-radius:4px;padding:6px 4px;"
        f"font-family:Consolas,monospace;overflow-x:auto;'>{''.join(ticker_cells)}</div>",
        unsafe_allow_html=True,
    )

    # ── Gauge + Top-level metrics ──
    col_gauge, col_metrics = st.columns([1, 2])
    with col_gauge:
        gauge_fig = _make_gauge(macro["macro_score"], regime, regime_color)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_metrics:
        def _metric_card(label, value, color=COLORS["text"]):
            return (
                f"<div style='background:{c_surface_dark};border:1px solid {c_border};"
                f"border-radius:4px;padding:10px 14px;text-align:center;'>"
                f"<div style='font-family:Consolas,monospace;font-size:10px;color:{c_orange};"
                f"text-transform:uppercase;letter-spacing:1px;'>{label}</div>"
                f"<div style='font-family:Consolas,monospace;font-size:20px;font-weight:700;"
                f"color:{color};margin-top:4px;'>{value}</div></div>"
            )
        cards_html = (
            f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;'>"
            f"{_metric_card('Macro Score', macro['macro_score'])}"
            f"{_metric_card('Quadrant', macro['quadrant'])}"
            f"{_metric_card('Regime', regime, regime_color)}"
            f"</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-family:Consolas,monospace;font-size:11px;color:{c_text_dim};margin-top:6px;'>"
            f"Confidence: {_confidence_label(macro['avg_confidence'])} ({macro['avg_confidence']}%) "
            f"&nbsp;|&nbsp; Growth: {macro['growth_dir']} &nbsp;|&nbsp; Inflation: {macro['inflation_dir']}</div>",
            unsafe_allow_html=True,
        )

    # ── Signal Radar ──
    radar_fig = _make_category_radar(macro["signals"])
    if radar_fig:
        _section_header("Signal Radar")
        st.plotly_chart(radar_fig, use_container_width=True)

    # ── Regime History ──
    _section_header("Regime History")
    timeframe = st.radio("Timeframe", ["1D", "1W", "1M", "6M", "1Y", "All"], index=0, horizontal=True)

    if timeframe != "All":
        summary = _regime_timeframe_summary(timeframe)
        if summary:
            v_color = COLORS["green"] if summary["verdict"] == "Risk-On" else COLORS["red"] if summary["verdict"] == "Risk-Off" else COLORS["yellow"]
            t_color = COLORS["green"] if summary["trend"] == "Improving" else COLORS["red"] if summary["trend"] == "Deteriorating" else COLORS["yellow"]
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{timeframe} Verdict:** <span style='color:{v_color}'>{summary['verdict']}</span>", unsafe_allow_html=True)
            c2.markdown(f"**Avg Score:** {summary['avg_score']}")
            c3.markdown(f"**Trend:** <span style='color:{t_color}'>{summary['trend']}</span>", unsafe_allow_html=True)
            st.caption(f"{summary['risk_on_days']} Risk-On / {summary['neutral_days']} Neutral / {summary['risk_off_days']} Risk-Off days (of {summary['total_days']})")

        if timeframe == "1D":
            st.caption("Note: Some signals (ISM, Unemployment, Inflation, GDP, CAPEX) update monthly or quarterly and may not show daily changes.")

    history_fig = _make_regime_history(timeframe=timeframe)
    if history_fig:
        st.plotly_chart(history_fig, use_container_width=True)

    st.markdown(
        f"<p style='color:{regime_color};font-size:11px;margin-top:12px;'>"
        f"Daily macro verdict: {regime}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )

    # ── Core Signals ──
    _section_header(f"Core Signals ({len(macro['signals'])})")
    _render_signals_table(macro["signals"])

    na_count = sum(1 for s in macro["signals"] if "N/A" in s.get("Value", ""))
    if na_count:
        cols = st.columns([6, 1])
        cols[0].caption(f"⚠ {na_count} signal(s) show N/A — FRED API or yfinance fetch failures.")
        if cols[1].button("Retry", key="retry_signals"):
            st.cache_data.clear()
            st.rerun()

    # ── Yield Curve Regime ──
    yc_regime = macro.get("yield_curve_regime", {})
    if yc_regime.get("regime") != "Unknown":
        _section_header("Yield Curve Regime")
        regime_name = yc_regime["regime"]
        is_bullish = regime_name.startswith("Bull")
        yc_color = COLORS["green"] if is_bullish else COLORS["red"]
        inv_tag = " **(Inverted)**" if yc_regime.get("inverted") else ""

        yc_descriptions = {
            "Bull Steepening": "Curve widening with rates falling — Fed easing expectations, positive for risk assets",
            "Bear Steepening": "Curve widening with rates rising — inflation fears, long-end selling off",
            "Bull Flattening": "Curve narrowing with rates falling — flight to safety, slowing growth expectations",
            "Bear Flattening": "Curve narrowing with rates rising — Fed tightening, short-end rates rising faster",
        }

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Regime:** <span style='color:{yc_color}'>{regime_name}</span>{inv_tag}", unsafe_allow_html=True)
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
    buffett_txt = "N/A" if macro["buffett"] is None else f"{macro['buffett']:.2f}%"
    st.markdown(
        f"<div style='display:flex;gap:12px;'>"
        f"<div style='flex:1;background:{c_surface_dark};border:1px solid {c_border};"
        f"border-radius:4px;padding:14px;text-align:center;'>"
        f"<div style='font-family:Consolas,monospace;font-size:10px;color:{c_orange};"
        f"text-transform:uppercase;letter-spacing:1px;'>S&P 500 P/E</div>"
        f"<div style='font-family:Consolas,monospace;font-size:26px;font-weight:700;"
        f"color:{c_text};margin-top:6px;'>{cape_txt}</div></div>"
        f"<div style='flex:1;background:{c_surface_dark};border:1px solid {c_border};"
        f"border-radius:4px;padding:14px;text-align:center;'>"
        f"<div style='font-family:Consolas,monospace;font-size:10px;color:{c_orange};"
        f"text-transform:uppercase;letter-spacing:1px;'>Buffett Indicator</div>"
        f"<div style='font-family:Consolas,monospace;font-size:26px;font-weight:700;"
        f"color:{c_text};margin-top:6px;'>{buffett_txt}</div></div></div>"
        f"<div style='font-family:Consolas,monospace;font-size:11px;color:{c_text_dim};"
        f"margin-top:8px;'>{macro['valuation']}</div>",
        unsafe_allow_html=True,
    )

    # ── Cycle Stage ──
    _section_header("Cycle Stage")
    capliq_txt = "N/A" if macro["capex_vs_liquidity"] is None else f"{macro['capex_vs_liquidity']:.2f}pp"
    st.markdown(
        f"<div style='background:{c_surface_dark};border:1px solid {c_border};"
        f"border-radius:4px;padding:14px;text-align:center;max-width:320px;'>"
        f"<div style='font-family:Consolas,monospace;font-size:10px;color:{c_orange};"
        f"text-transform:uppercase;letter-spacing:1px;'>CAPEX vs Liquidity</div>"
        f"<div style='font-family:Consolas,monospace;font-size:26px;font-weight:700;"
        f"color:{c_text};margin-top:6px;'>{capliq_txt}</div>"
        f"<div style='font-family:Consolas,monospace;font-size:12px;color:{c_text_dim};"
        f"margin-top:4px;'>{macro['cycle_stage']}</div></div>",
        unsafe_allow_html=True,
    )

    # ── Summary ──
    _section_header("Summary")
    for line in macro["summary"]:
        st.markdown(f"- {line}")

    # ── Portfolio Bias ──
    _section_header("Portfolio Bias")
    bias_cards = []
    for sleeve, bias in macro["portfolio_bias"].items():
        if "Overweight" in bias:
            tag_color = COLORS["green"]
            tag_bg = "rgba(0,212,170,0.12)"
        elif "Underweight" in bias:
            tag_color = COLORS["red"]
            tag_bg = "rgba(255,75,75,0.12)"
        else:
            tag_color = COLORS["text_dim"]
            tag_bg = "rgba(136,136,136,0.10)"
        bias_cards.append(
            f"<div style='background:{c_surface_dark};border:1px solid {c_border};"
            f"border-radius:4px;padding:8px 14px;text-align:center;'>"
            f"<div style='font-family:Consolas,monospace;font-size:11px;color:{c_text};'>{sleeve}</div>"
            f"<div style='display:inline-block;margin-top:4px;padding:2px 8px;border-radius:3px;"
            f"background:{tag_bg};font-family:Consolas,monospace;font-size:11px;font-weight:600;"
            f"color:{tag_color};'>{bias}</div></div>"
        )
    st.markdown(
        f"<div style='display:flex;flex-wrap:wrap;gap:8px;'>{''.join(bias_cards)}</div>",
        unsafe_allow_html=True,
    )

    # ── Sector Rotation ──
    _section_header("Sector Rotation")
    sector_recs = macro.get("sector_rotation", [])
    if sector_recs:
        col_favor, col_avoid = st.columns(2)
        with col_favor:
            st.markdown("**Favor**")
            for rec in sector_recs:
                if rec["action"] == "Favor":
                    mom_str = f" ({rec['momentum_30d']:+.1f}% 30d)" if rec["momentum_30d"] is not None else ""
                    st.markdown(f"- **{rec['ticker']}** {rec['label']}{mom_str} — {rec['reason']}")
        with col_avoid:
            st.markdown("**Avoid**")
            for rec in sector_recs:
                if rec["action"] == "Avoid":
                    mom_str = f" ({rec['momentum_30d']:+.1f}% 30d)" if rec["momentum_30d"] is not None else ""
                    st.markdown(f"- **{rec['ticker']}** {rec['label']}{mom_str} — {rec['reason']}")
    else:
        st.markdown("Sector rotation data unavailable.")

    # ── Risk Management Alerts ──
    _section_header("Risk Management Alerts")
    for alert in macro.get("risk_alerts", ["No elevated risk signals detected."]):
        st.markdown(f"- {alert}")

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

    # ── SPY Options Sentiment ──
    _section_header("SPY Options Sentiment")
    gamma = macro["gamma"]
    if gamma:
        if macro.get("low_compute_mode"):
            st.caption("Cached mode active (Low Compute Mode).")

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
        if gamma["gamma_flip"] is None:
            st.markdown("- Gamma Flip price: N/A")
        else:
            location = "above stock price" if gamma["gamma_flip"] > gamma["price"] else "below stock price"
            st.markdown(f"- Gamma Flip price: {gamma['gamma_flip']:.2f} ({location})")
        st.markdown(f"- Call Wall price: {gamma['call_wall']:.2f}" if gamma["call_wall"] is not None else "- Call Wall price: N/A")
        st.markdown(f"- Put Wall price: {gamma['put_wall']:.2f}" if gamma["put_wall"] is not None else "- Put Wall price: N/A")

        m1, m2, m3 = st.columns(3)
        m1.metric("Gamma Flip", f"{gamma['gamma_flip']:.2f}" if gamma["gamma_flip"] is not None else "N/A")
        m2.metric("Call Wall", f"{gamma['call_wall']:.2f}" if gamma["call_wall"] is not None else "N/A")
        m3.metric("Put Wall", f"{gamma['put_wall']:.2f}" if gamma["put_wall"] is not None else "N/A")

        if macro.get("low_compute_mode"):
            st.caption("Gamma chart disabled in Low Compute Mode.")
        else:
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
            if gamma["gamma_flip"] is not None:
                fig.add_vline(x=gamma["gamma_flip"], line_color=COLORS["yellow"], line_dash="dot", line_width=2)
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

