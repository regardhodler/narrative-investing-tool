"""
Module 0: Market Risk Regime Indicator

Determines if markets are in "Risk-On" or "Risk-Off" mode using:
- Yield curve (2Y/10Y spread via SHY/TLT)
- Credit spreads (JNK/LQD/TLT ratios)
- Inflation proxies (TIP, breakevens)
- Commodities (Oil, Gold, Silver, Copper, Gold/Copper ratio)
- US & Global equity indices
- Volatility (VIX, MOVE proxy via TLT vol)
- USD strength (DXY proxy via UUP)
- Truflation (via public API)

Architecture:
- Declarative signal definitions (SIGNAL_DEFS)
- Z-score based adaptive thresholds (1Y lookback)
- Daily regime history persistence (JSON snapshots)
- Shared data layer via services/market_data.py
"""

import json
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

from services.market_data import (
    fetch_batch, fetch_truflation, zscore, ratio_latest, AssetSnapshot,
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


def _save_snapshot(regime_data: dict):
    """Persist today's regime snapshot (one entry per calendar day)."""
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    history = _load_history()
    today = datetime.now().strftime("%Y-%m-%d")

    # Replace today's entry if it exists, otherwise append
    history = [h for h in history if h.get("date") != today]
    history.append({
        "date": today,
        "score": regime_data["aggregate_score"],
        "regime": regime_data["regime"],
        "signal_count": len(regime_data["signals"]),
        "signals_summary": {
            s["name"]: round(s["score"], 3) for s in regime_data["signals"]
        },
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
# DECLARATIVE SIGNAL DEFINITIONS (#7)
# ─────────────────────────────────────────────

SIGNAL_DEFS = [
    # ── Yield Curve ──
    {
        "name": "Curve Steepening (SHY vs TLT momentum)",
        "category": "Yield Curve",
        "method": "spread_momentum",
        "tickers": ("SHY", "TLT"),
        "higher_is_bullish": True,
        "interpretation": "Steepening = growth expectations rising",
    },
    # ── Credit ──
    {
        "name": "JNK/TLT Ratio",
        "category": "Credit",
        "method": "ratio_zscore",
        "tickers": ("JNK", "TLT"),
        "higher_is_bullish": True,
        "interpretation": "High = HY outperforming Treasuries = risk-on",
    },
    {
        "name": "HY vs IG Momentum (JNK - LQD)",
        "category": "Credit",
        "method": "spread_momentum",
        "tickers": ("JNK", "LQD"),
        "higher_is_bullish": True,
        "interpretation": "HY outperforming = credit appetite = risk-on",
    },
    # ── Volatility ──
    {
        "name": "VIX Fear Index",
        "category": "Volatility",
        "method": "level_zscore",
        "tickers": ("^VIX",),
        "higher_is_bullish": False,
        "interpretation": "VIX low = complacency (risk-on), high = fear (risk-off)",
    },
    {
        "name": "Bond Volatility (TLT 20d realized vol)",
        "category": "Volatility",
        "method": "realized_vol",
        "tickers": ("TLT",),
        "higher_is_bullish": False,
        "interpretation": "High bond vol = rate uncertainty = risk-off (MOVE proxy)",
    },
    # ── Equities ──
    {
        "name": "S&P 500 (30d momentum)",
        "category": "Equities",
        "method": "momentum_zscore",
        "tickers": ("SPY",),
        "higher_is_bullish": True,
        "interpretation": "Positive trend = equities bid = risk-on",
    },
    {
        "name": "Small Cap vs Large Cap (IWM - SPY)",
        "category": "Equities",
        "method": "spread_momentum",
        "tickers": ("IWM", "SPY"),
        "higher_is_bullish": True,
        "interpretation": "IWM leading = growth optimism = risk-on",
    },
    {
        "name": "SPY Trend (above 50d MA)",
        "category": "Equities",
        "method": "ma_position",
        "tickers": ("SPY",),
        "ma_period": 50,
        "higher_is_bullish": True,
        "interpretation": "Price > 50d MA = uptrend intact = risk-on",
    },
    {
        "name": "SPY Trend (above 200d MA)",
        "category": "Equities",
        "method": "ma_position",
        "tickers": ("SPY",),
        "ma_period": 200,
        "higher_is_bullish": True,
        "interpretation": "Price > 200d MA = secular bull = risk-on",
    },
    # ── Global ──
    {
        "name": "EM vs Developed Markets",
        "category": "Global",
        "method": "spread_momentum",
        "tickers": ("EEM", "EFA"),
        "higher_is_bullish": True,
        "interpretation": "EM outperforming = global growth bid = risk-on",
    },
    # ── Commodities ──
    {
        "name": "Gold Momentum (30d)",
        "category": "Commodities",
        "method": "gold_context",
        "tickers": ("GLD", "SPY"),
        "higher_is_bullish": True,
        "interpretation": "Gold up + equities down = risk-off flight to safety",
    },
    {
        "name": "Gold/Copper Ratio",
        "category": "Commodities",
        "method": "ratio_zscore",
        "tickers": ("GLD", "CPER"),
        "higher_is_bullish": False,
        "interpretation": "High Gold/Copper = defensive positioning = risk-off",
    },
    {
        "name": "Copper (30d momentum)",
        "category": "Commodities",
        "method": "momentum_zscore",
        "tickers": ("CPER",),
        "higher_is_bullish": True,
        "interpretation": "Dr. Copper rising = global industrial demand = risk-on",
    },
    {
        "name": "Oil (WTI, 30d momentum)",
        "category": "Commodities",
        "method": "momentum_zscore",
        "tickers": ("USO",),
        "higher_is_bullish": True,
        "interpretation": "Oil rising = economic activity",
    },
    # ── FX ──
    {
        "name": "USD Strength (UUP, 30d)",
        "category": "FX",
        "method": "momentum_zscore",
        "tickers": ("UUP",),
        "higher_is_bullish": False,
        "interpretation": "USD falling = global liquidity expanding = risk-on",
    },
    # ── Inflation ──
    {
        "name": "TIPS Inflation Expectations",
        "category": "Inflation",
        "method": "momentum_zscore",
        "tickers": ("TIP",),
        "higher_is_bullish": True,
        "interpretation": "TIPS rising = reflation = mild risk-on, sharp = stagflation risk",
    },
    # ── Sector Rotation ──
    {
        "name": "Financials (XLF, 30d)",
        "category": "Sector Rotation",
        "method": "momentum_zscore",
        "tickers": ("XLF",),
        "higher_is_bullish": True,
        "interpretation": "Financials leading = rate expectations rising = risk-on",
    },
]

# Category weights for aggregate score
CATEGORY_WEIGHTS = {
    "Yield Curve": 1.5,
    "Credit": 1.5,
    "Volatility": 1.5,
    "Equities": 1.2,
    "Global": 1.0,
    "Commodities": 0.8,
    "FX": 0.9,
    "Inflation": 0.8,
    "Sector Rotation": 0.7,
}


# ─────────────────────────────────────────────
# SIGNAL COMPUTATION ENGINE (#1 fix + #6 z-scores)
# ─────────────────────────────────────────────

def _clamp(val: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _zscore_to_score(z: float, higher_is_bullish: bool) -> float:
    """Convert z-score to -1..+1 score. Z beyond ±2 saturates."""
    score = _clamp(z / 2.0)
    if not higher_is_bullish:
        score = -score
    return score


def _label_from_score(score: float) -> str:
    if score >= 0.2:
        return "Risk-On"
    elif score <= -0.2:
        return "Risk-Off"
    return "Neutral"


def _compute_signal(defn: dict, snaps: dict[str, AssetSnapshot]) -> dict | None:
    """Compute a single signal from its declarative definition."""
    method = defn["method"]
    tickers = defn["tickers"]
    bullish = defn.get("higher_is_bullish", True)

    def snap(t):
        return snaps.get(t)

    def series(t):
        s = snap(t)
        return s.series if s else None

    def pct30(t):
        s = snap(t)
        return s.pct_change_30d if s else None

    # Discount weight if any ticker data is stale
    stale = any(snap(t) and snap(t).stale for t in tickers if snap(t))

    try:
        if method == "momentum_zscore":
            # Z-score of 30d return vs 1Y distribution of rolling 30d returns
            s = series(tickers[0])
            if s is None or len(s) < 60:
                return None
            rolling_ret = s.pct_change(22) * 100
            rolling_ret = rolling_ret.dropna()
            if len(rolling_ret) < 20:
                return None
            z = float((rolling_ret.iloc[-1] - rolling_ret.mean()) / rolling_ret.std()) if rolling_ret.std() > 0 else 0
            score = _zscore_to_score(z, bullish)
            val = round(float(rolling_ret.iloc[-1]), 2)
            return _build_signal(defn, val, "%", score, stale)

        elif method == "level_zscore":
            # Z-score of current level vs 1Y distribution
            s = series(tickers[0])
            if s is None or len(s) < 60:
                return None
            z = zscore(s, lookback=252)
            if z is None:
                return None
            score = _zscore_to_score(z, bullish)
            val = round(float(s.iloc[-1]), 2)
            return _build_signal(defn, val, "pts", score, stale)

        elif method == "spread_momentum":
            # Difference in 30d returns between two tickers
            p1, p2 = pct30(tickers[0]), pct30(tickers[1])
            if p1 is None or p2 is None:
                return None
            spread = p1 - p2
            # Z-score the spread vs historical rolling spread
            s1, s2 = series(tickers[0]), series(tickers[1])
            if s1 is not None and s2 is not None and len(s1) >= 60 and len(s2) >= 60:
                # Align series
                combined = pd.DataFrame({"a": s1, "b": s2}).dropna()
                if len(combined) >= 60:
                    r1 = combined["a"].pct_change(22) * 100
                    r2 = combined["b"].pct_change(22) * 100
                    hist_spread = (r1 - r2).dropna()
                    if len(hist_spread) >= 20 and hist_spread.std() > 0:
                        z = float((hist_spread.iloc[-1] - hist_spread.mean()) / hist_spread.std())
                        score = _zscore_to_score(z, bullish)
                        return _build_signal(defn, round(spread, 2), "% spread", score, stale)
            # Fallback: simple threshold
            score = _clamp(spread / 3.0)
            if not bullish:
                score = -score
            return _build_signal(defn, round(spread, 2), "% spread", score, stale)

        elif method == "ratio_zscore":
            # Z-score of price ratio
            s1, s2 = series(tickers[0]), series(tickers[1])
            if s1 is None or s2 is None:
                return None
            combined = pd.DataFrame({"a": s1, "b": s2}).dropna()
            if len(combined) < 60:
                return None
            ratio_series = combined["a"] / combined["b"]
            z = zscore(ratio_series, lookback=252)
            if z is None:
                return None
            score = _zscore_to_score(z, bullish)
            val = round(float(ratio_series.iloc[-1]), 4)
            return _build_signal(defn, val, "ratio", score, stale)

        elif method == "realized_vol":
            # 20-day realized volatility, z-scored vs 1Y
            s = series(tickers[0])
            if s is None or len(s) < 60:
                return None
            log_ret = np.log(s / s.shift(1)).dropna()
            rvol = log_ret.rolling(20).std() * np.sqrt(252) * 100
            rvol = rvol.dropna()
            if len(rvol) < 60:
                return None
            z = float((rvol.iloc[-1] - rvol.mean()) / rvol.std()) if rvol.std() > 0 else 0
            score = _zscore_to_score(z, bullish)
            val = round(float(rvol.iloc[-1]), 1)
            return _build_signal(defn, val, "% ann.", score, stale)

        elif method == "ma_position":
            # How far above/below moving average (as %)
            s = series(tickers[0])
            ma_period = defn.get("ma_period", 50)
            if s is None or len(s) < ma_period + 20:
                return None
            ma = s.rolling(ma_period).mean()
            pct_above = ((s / ma) - 1) * 100
            pct_above = pct_above.dropna()
            if len(pct_above) < 20:
                return None
            z = float((pct_above.iloc[-1] - pct_above.mean()) / pct_above.std()) if pct_above.std() > 0 else 0
            score = _zscore_to_score(z, bullish)
            val = round(float(pct_above.iloc[-1]), 2)
            return _build_signal(defn, val, f"% vs {ma_period}d MA", score, stale)

        elif method == "gold_context":
            # Gold momentum contextualized: gold up + equities down = risk-off
            gld_pct = pct30(tickers[0])
            spy_pct = pct30(tickers[1])
            if gld_pct is None:
                return None
            spy_pct = spy_pct or 0
            if gld_pct > 0 and spy_pct < -1:
                score = -0.7  # safe haven demand
            elif gld_pct > 0 and spy_pct > 1:
                score = 0.3   # reflation
            elif gld_pct < -2:
                score = 0.4   # gold selling = risk appetite
            else:
                score = 0.0   # neutral
            return _build_signal(defn, round(gld_pct, 2), "%", score, stale)

    except Exception:
        return None

    return None


def _build_signal(defn: dict, value, unit: str, score: float, stale: bool) -> dict:
    score = round(_clamp(score), 3)
    return {
        "category": defn["category"],
        "name": defn["name"],
        "value": value,
        "unit": unit,
        "score": score,
        "label": _label_from_score(score),
        "interpretation": defn["interpretation"],
        "stale": stale,
    }


# ─────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_all_data() -> dict[str, AssetSnapshot]:
    """Fetch all ticker data via shared market_data service."""
    return fetch_batch(TICKERS, period="1y", interval="1d")


def compute_regime(snaps: dict[str, AssetSnapshot]) -> dict:
    """Compute individual signals and aggregate risk regime score."""
    signals = []

    # Process declarative signals
    for defn in SIGNAL_DEFS:
        result = _compute_signal(defn, snaps)
        if result:
            signals.append(result)

    # Truflation (special case — external API, not yfinance)
    tf = fetch_truflation()
    if tf and isinstance(tf, dict):
        tf_val = tf.get("inflation") or tf.get("current") or tf.get("value")
        if tf_val is not None:
            try:
                tf_float = float(str(tf_val).replace("%", ""))
                # Lower inflation = Fed can ease = risk-on
                score = _clamp((3.0 - tf_float) / 2.0)  # 1% → +1, 3% → 0, 5% → -1
                signals.append({
                    "category": "Inflation",
                    "name": "Truflation Real-Time CPI",
                    "value": round(tf_float, 2),
                    "unit": "%",
                    "score": round(score, 3),
                    "label": _label_from_score(score),
                    "interpretation": "Low truflation = Fed can stay accommodative = risk-on",
                    "stale": False,
                })
            except Exception:
                pass

    # Aggregate score (weighted by category, discounted if stale)
    if signals:
        total_weight = 0.0
        weighted_score = 0.0
        for s in signals:
            w = CATEGORY_WEIGHTS.get(s["category"], 1.0)
            if s.get("stale"):
                w *= 0.6  # discount stale data
            weighted_score += s["score"] * w
            total_weight += w
        aggregate = weighted_score / total_weight if total_weight > 0 else 0.0
    else:
        aggregate = 0.0

    # Map aggregate to regime
    if aggregate >= 0.35:
        regime = "RISK-ON"
        regime_color = COLORS["green"]
        regime_emoji = "🟢"
        regime_desc = "Markets are in an appetite-for-risk environment. Growth assets, high-yield credit, EM equities, and cyclicals are favored."
    elif aggregate <= -0.35:
        regime = "RISK-OFF"
        regime_color = COLORS["red"]
        regime_emoji = "🔴"
        regime_desc = "Defensive posture indicated. Treasuries, gold, USD, and low-volatility assets are being favored by the market."
    else:
        regime = "NEUTRAL"
        regime_color = COLORS["yellow"]
        regime_emoji = "🟡"
        regime_desc = "Mixed signals across asset classes. Markets are in a transition or consolidation phase with no clear directional bias."

    result = {
        "signals": signals,
        "aggregate_score": round(aggregate, 3),
        "regime": regime,
        "regime_color": regime_color,
        "regime_emoji": regime_emoji,
        "regime_desc": regime_desc,
    }

    # Persist daily snapshot
    try:
        _save_snapshot(result)
    except Exception:
        pass  # non-critical

    return result


def get_current_regime() -> dict:
    """Public accessor for other modules to consume regime data."""
    snaps = fetch_all_data()
    data = compute_regime(snaps)
    return {
        "regime": data["regime"],
        "score": data["aggregate_score"],
        "signals": data["signals"],
    }


# ─────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────

def _make_gauge(score: float, regime: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        title={"text": f"Risk Regime Score ({score:+.2f})", "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": " / 100", "font": {"size": 28, "color": color}},
        gauge={
            "axis": {"range": [-100, 100], "tickwidth": 1, "tickcolor": COLORS["text_dim"]},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": COLORS["surface"],
            "borderwidth": 0,
            "steps": [
                {"range": [-100, -35], "color": "#2d1b1b"},
                {"range": [-35, 35], "color": "#1e1e10"},
                {"range": [35, 100], "color": "#1b2d1b"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": round(score * 100, 1),
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    apply_dark_layout(fig)
    return fig


def _make_signal_bar(signals: list) -> go.Figure:
    df = pd.DataFrame(signals).sort_values("score")
    colors = [
        COLORS["green"] if s > 0.2 else (COLORS["red"] if s < -0.2 else COLORS["yellow"])
        for s in df["score"]
    ]

    fig = go.Figure(go.Bar(
        x=df["score"],
        y=df["name"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in df["score"]],
        textposition="outside",
        textfont=dict(color=COLORS["text"], size=11),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(380, len(signals) * 30),
        margin=dict(l=20, r=60, t=30, b=20),
        xaxis=dict(
            range=[-1.2, 1.2],
            zeroline=True,
            zerolinecolor=COLORS["grid"],
            zerolinewidth=2,
        ),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        showlegend=False,
    )
    apply_dark_layout(fig)
    return fig


def _make_category_radar(signals: list) -> go.Figure:
    df = pd.DataFrame(signals)
    cat_scores = df.groupby("category")["score"].mean().reset_index()

    categories = cat_scores["category"].tolist()
    scores = cat_scores["score"].tolist()
    categories += [categories[0]]
    scores += [scores[0]]

    fig = go.Figure(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill="toself",
        fillcolor="rgba(75, 159, 255, 0.15)",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(color=COLORS["blue"], size=6),
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["surface"],
            radialaxis=dict(
                visible=True, range=[-1, 1],
                gridcolor=COLORS["grid"],
                tickfont=dict(size=9, color=COLORS["text_dim"]),
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=COLORS["text"]),
                gridcolor=COLORS["grid"],
            ),
        ),
        height=320,
        margin=dict(l=40, r=40, t=30, b=30),
        showlegend=False,
    )
    apply_dark_layout(fig)
    return fig


def _make_regime_history() -> go.Figure | None:
    """Time-series chart of historical regime scores (#5)."""
    history = _load_history()
    if len(history) < 2:
        return None

    df = pd.DataFrame(history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    colors = [
        COLORS["green"] if s >= 0.35 else (COLORS["red"] if s <= -0.35 else COLORS["yellow"])
        for s in df["score"]
    ]

    fig = go.Figure()

    # Background bands for regime zones
    fig.add_hrect(y0=0.35, y1=1.0, fillcolor="#1b2d1b", opacity=0.5, line_width=0)
    fig.add_hrect(y0=-0.35, y1=0.35, fillcolor="#1e1e10", opacity=0.5, line_width=0)
    fig.add_hrect(y0=-1.0, y1=-0.35, fillcolor="#2d1b1b", opacity=0.5, line_width=0)

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["score"],
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2),
        marker=dict(color=colors, size=6),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Score: %{y:+.3f}<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(y=0.35, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
    fig.add_hline(y=-0.35, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
    fig.add_hline(y=0, line_color=COLORS["grid"], opacity=0.3)

    fig.update_layout(
        height=300,
        margin=dict(l=40, r=20, t=30, b=30),
        yaxis=dict(range=[-1.1, 1.1], title="Regime Score"),
        xaxis=dict(title=""),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Regime Score History")
    return fig


def _make_price_sparklines(snaps: dict[str, AssetSnapshot]) -> go.Figure | None:
    """Mini sparklines for key assets."""
    assets = [
        ("SPY", "S&P 500"), ("TLT", "TLT (Bonds)"), ("GLD", "Gold"),
        ("^VIX", "VIX"), ("UUP", "USD"), ("JNK", "JNK HY"),
    ]
    valid = [(t, l) for t, l in assets if snaps.get(t) and snaps[t].series is not None
             and len(snaps[t].series) > 1]
    if not valid:
        return None

    n = len(valid)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[l for _, l in valid])

    for i, (ticker, label) in enumerate(valid):
        row = i // cols + 1
        col = i % cols + 1
        s = snaps[ticker].series
        # Use last 3 months (~63 trading days)
        s = s.iloc[-63:] if len(s) > 63 else s
        norm = (s / s.iloc[0] - 1) * 100
        color = COLORS["green"] if norm.iloc[-1] >= 0 else COLORS["red"]
        fig.add_trace(
            go.Scatter(
                x=norm.index, y=norm.values,
                mode="lines",
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.08)",
                name=label,
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>%{{y:.1f}}%<extra></extra>",
            ),
            row=row, col=col,
        )

    fig.update_layout(height=220 * rows, margin=dict(l=20, r=20, t=40, b=20))
    apply_dark_layout(fig)
    fig.update_xaxes(showticklabels=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(ticksuffix="%", gridcolor=COLORS["grid"], tickfont=dict(size=9))
    return fig


def _make_heatmap(signals: list) -> go.Figure:
    """Category heatmap — easier to read than radar for many signals."""
    df = pd.DataFrame(signals)
    cat_order = list(CATEGORY_WEIGHTS.keys())
    # Pivot: each category gets its signals as rows
    cats = []
    names = []
    scores = []
    for cat in cat_order:
        cat_sigs = df[df["category"] == cat]
        for _, row in cat_sigs.iterrows():
            cats.append(cat)
            names.append(row["name"])
            scores.append(row["score"])

    if not names:
        return go.Figure()

    fig = go.Figure(go.Heatmap(
        z=[scores],
        x=names,
        y=["Score"],
        colorscale=[
            [0.0, COLORS["red"]],
            [0.5, COLORS["yellow"]],
            [1.0, COLORS["green"]],
        ],
        zmin=-1, zmax=1,
        text=[[f"{s:+.2f}" for s in scores]],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="<b>%{x}</b><br>Score: %{z:+.3f}<extra></extra>",
        showscale=False,
    ))
    fig.update_layout(
        height=120,
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(visible=False),
    )
    apply_dark_layout(fig)
    return fig


# ─────────────────────────────────────────────
# RENDER
# ─────────────────────────────────────────────

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
    }


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

    return {
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
    regime_emoji = "🟢" if regime == "Risk-On" else "🔴" if regime == "Risk-Off" else "🟡"

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Macro Score (0-100)", macro["macro_score"])
    col_b.metric("Macro Quadrant", macro["quadrant"])
    col_c.metric("Current Regime", f"{regime_emoji} {regime}")
    st.caption(f"Signal confidence: {_confidence_label(macro['avg_confidence'])} ({macro['avg_confidence']}%).")

    st.markdown(f"### Core Signals ({len(macro['signals'])})")
    st.dataframe(pd.DataFrame(macro["signals"]), use_container_width=True, hide_index=True)

    st.markdown("### Valuation")
    cape_txt = "N/A" if macro["cape"] is None else f"{macro['cape']:.2f}x"
    buffett_txt = "N/A" if macro["buffett"] is None else f"{macro['buffett']:.2f}%"
    st.markdown(f"- CAPE: {cape_txt}")
    st.markdown(f"- Buffett Indicator: {buffett_txt}")
    st.markdown(f"- Interpretation: {macro['valuation']}")

    st.markdown("### Cycle Stage")
    capliq_txt = "N/A" if macro["capex_vs_liquidity"] is None else f"{macro['capex_vs_liquidity']:.2f}pp"
    st.markdown(f"- CAPEX vs Liquidity: {capliq_txt}")
    st.markdown(f"- Stage: {macro['cycle_stage']}")

    st.markdown("### Summary")
    for line in macro["summary"]:
        st.markdown(f"- {line}")

    st.markdown("### Portfolio Bias")
    for sleeve, bias in macro["portfolio_bias"].items():
        st.markdown(f"- {sleeve}: {bias}")

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

    st.markdown(
        f"<p style='color:{regime_color};font-size:11px;margin-top:12px;'>"
        f"Daily macro verdict: {regime}. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        unsafe_allow_html=True,
    )
