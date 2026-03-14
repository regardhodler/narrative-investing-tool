"""
Module 0: Macro Dashboard

Daily macro regime indicator using 15 cross-asset signals:
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
- 15-indicator scoring engine (_build_macro_dashboard)
- Daily regime history persistence (JSON snapshots)
- Shared data layer via services/market_data.py
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from services.market_data import (
    fetch_batch, AssetSnapshot,
    fetch_fred_series, fetch_options_chain_snapshot,
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
        if cape > 30 and buffett > 150:
            return "Both CAPE and Buffett Indicator point to an expensive equity market."
        if cape < 20 and buffett < 110:
            return "Both CAPE and Buffett Indicator suggest relatively attractive long-term valuation."
    if cape is not None:
        if cape > 28:
            return "CAPE is elevated versus long-run norms, implying lower forward return expectations."
        if cape < 20:
            return "CAPE is below long-run highs, valuation risk appears more moderate."
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


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_all_data() -> dict[str, AssetSnapshot]:
    """Fetch all ticker data via shared market_data service."""
    return fetch_batch(TICKERS, period="1y", interval="1d")


# ─────────────────────────────────────────────
# SPY GAMMA MODE
# ─────────────────────────────────────────────

def _compute_spy_gamma_mode(max_expiries: int = 2) -> dict | None:
    snap = fetch_options_chain_snapshot("SPY", max_expiries=max_expiries)
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

def _build_macro_dashboard(snaps: dict[str, AssetSnapshot], low_compute_mode: bool = False) -> dict:
    fred_ids = {
        "yield_curve": "T10Y2Y",
        "credit_spread": "BAMLH0A0HYM2",
        "m2": "M2SL",
        "sahm": "SAHMREALTIME",
        "unrate": "UNRATE",
        "core_pce": "PCEPILFE",
        "cape": "CAPE",
        "wilshire": "WILL5000INDFC",
        "gdp": "GDP",
        "capex": "NCBDBIQ027S",
        "term_premium": "THREEFYTP10",
        "ism": "NAPM",
        "fci": "NFCI",
    }
    fred = {k: fetch_fred_series(v) for k, v in fred_ids.items()}

    indicators = []

    yc = _safe_latest(fred["yield_curve"])
    yc_score = _clamp_score((yc or 0.0), 1.0)
    indicators.append(("Yield Curve (10Y-2Y)", yc, "bps", yc_score, _confidence_from_age(fred["yield_curve"], expected_days=14)))

    cs = _safe_latest(fred["credit_spread"])
    cs_score = _clamp_score((4.0 - (cs or 4.0)), 2.0)
    indicators.append(("Credit Spreads (HY vs Treasuries)", cs, "%", cs_score, _confidence_from_age(fred["credit_spread"], expected_days=7)))

    oil = snaps.get("USO").pct_change_30d if snaps.get("USO") else None
    copper = snaps.get("CPER").pct_change_30d if snaps.get("CPER") else None
    commodity_trend = np.nanmean([oil if oil is not None else np.nan, copper if copper is not None else np.nan])
    commodity_trend = None if np.isnan(commodity_trend) else float(commodity_trend)
    commodity_score = _clamp_score((commodity_trend or 0.0), 5.0)
    indicators.append(("Commodity Trend (Oil + Copper)", commodity_trend, "% 30d", commodity_score, _confidence_from_snap("USO", "CPER", snaps=snaps)))

    dxy = snaps.get("UUP").pct_change_30d if snaps.get("UUP") else None
    if dxy is None:
        dxy_score = 0.0
    elif dxy > 0:
        dxy_score = -_clamp_score(abs(dxy), 3.0)
    elif dxy < 0:
        dxy_score = _clamp_score(abs(dxy), 3.0)
    else:
        dxy_score = 0.0
    indicators.append(("US Dollar Index (DXY proxy, up=Risk-Off)", dxy, "% 30d", dxy_score, _confidence_from_snap("UUP", snaps=snaps)))

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

    cape = _safe_latest(fred["cape"])
    cape_score = _clamp_score((25.0 - (cape or 25.0)), 10.0)
    indicators.append(("Shiller CAPE", cape, "x", cape_score, _confidence_from_age(fred["cape"], expected_days=60)))

    wilshire = _safe_latest(fred["wilshire"])
    gdp = _safe_latest(fred["gdp"])
    buffett = (wilshire / gdp * 100) if (wilshire is not None and gdp and gdp != 0) else None
    buffett_score = _clamp_score((120.0 - (buffett or 120.0)), 40.0)
    indicators.append(("Buffett Indicator (Mkt Cap / GDP)", buffett, "%", buffett_score, int(round(np.mean([
        _confidence_from_age(fred["wilshire"], expected_days=45),
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

    ism = _safe_latest(fred["ism"])
    ism_score = _clamp_score(((ism or 50.0) - 50.0), 5.0)
    indicators.append(("ISM Manufacturing", ism, "index", ism_score, _confidence_from_age(fred["ism"], expected_days=45)))

    fci = _safe_latest(fred["fci"])
    fci_score = _clamp_score((-(fci or 0.0)), 0.5)
    indicators.append(("Financial Conditions Index", fci, "index", fci_score, _confidence_from_age(fred["fci"], expected_days=14)))

    signal_rows = []
    scores = []
    confidence_scores = []
    for name, value, unit, score, confidence in indicators:
        emoji, verdict = _score_to_bucket(score)
        scores.append(score)
        confidence_scores.append(confidence)
        display_value = "N/A" if value is None else f"{value:.2f} {unit}".strip()
        signal_rows.append({
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


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────

def render():
    st.markdown("## Macro Dashboard")
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};margin-top:-10px;'>"
        "Concise global macro monitor for daily Risk-On / Risk-Off workflow</p>",
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
        snaps = fetch_all_data()
        macro = _build_macro_dashboard(snaps, low_compute_mode=low_compute_mode)

    regime = macro["macro_regime"]
    regime_color = COLORS["green"] if regime == "Risk-On" else COLORS["red"] if regime == "Risk-Off" else COLORS["yellow"]

    # ── Gauge + Top-level metrics ──
    col_gauge, col_metrics = st.columns([1, 2])
    with col_gauge:
        gauge_fig = _make_gauge(macro["macro_score"], regime, regime_color)
        st.plotly_chart(gauge_fig, use_container_width=True)

    with col_metrics:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Macro Score (0-100)", macro["macro_score"])
        col_b.metric("Macro Quadrant", macro["quadrant"])
        col_c.metric("Current Regime", f"{regime}")
        st.caption(f"Signal confidence: {_confidence_label(macro['avg_confidence'])} ({macro['avg_confidence']}%).")
        st.caption(f"Growth: {macro['growth_dir']} | Inflation: {macro['inflation_dir']}")

    # ── Core Signals ──
    st.markdown(f"### Core Signals ({len(macro['signals'])})")
    st.dataframe(pd.DataFrame(macro["signals"]), use_container_width=True, hide_index=True)

    # ── Valuation ──
    st.markdown("### Valuation")
    cape_txt = "N/A" if macro["cape"] is None else f"{macro['cape']:.2f}x"
    buffett_txt = "N/A" if macro["buffett"] is None else f"{macro['buffett']:.2f}%"
    st.markdown(f"- CAPE: {cape_txt}")
    st.markdown(f"- Buffett Indicator: {buffett_txt}")
    st.markdown(f"- Interpretation: {macro['valuation']}")

    # ── Cycle Stage ──
    st.markdown("### Cycle Stage")
    capliq_txt = "N/A" if macro["capex_vs_liquidity"] is None else f"{macro['capex_vs_liquidity']:.2f}pp"
    st.markdown(f"- CAPEX vs Liquidity: {capliq_txt}")
    st.markdown(f"- Stage: {macro['cycle_stage']}")

    # ── Summary ──
    st.markdown("### Summary")
    for line in macro["summary"]:
        st.markdown(f"- {line}")

    # ── Portfolio Bias ──
    st.markdown("### Portfolio Bias")
    for sleeve, bias in macro["portfolio_bias"].items():
        st.markdown(f"- {sleeve}: {bias}")

    # ── SPY Options Sentiment ──
    st.markdown("### SPY Options Sentiment Mode")
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

    # ── Regime History ──
    history_fig = _make_regime_history()
    if history_fig:
        st.plotly_chart(history_fig, use_container_width=True)

    st.markdown(
        f"<p style='color:{regime_color};font-size:11px;margin-top:12px;'>"
        f"Daily macro verdict: {regime}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )
