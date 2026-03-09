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
        title={"text": "Risk Regime Score", "font": {"size": 16, "color": COLORS["text"]}},
        number={"suffix": "", "font": {"size": 28, "color": color}},
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
                fillcolor=color + "15",
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

def render():
    st.markdown("## Market Risk Regime")
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};margin-top:-10px;'>"
        "Real-time cross-asset risk-on / risk-off dashboard</p>",
        unsafe_allow_html=True,
    )

    col_refresh, _ = st.columns([1, 4])
    with col_refresh:
        if st.button("Refresh Data"):
            st.cache_data.clear()

    with st.spinner("Fetching cross-asset data..."):
        snaps = fetch_all_data()

    regime_data = compute_regime(snaps)
    signals = regime_data["signals"]
    aggregate = regime_data["aggregate_score"]
    regime = regime_data["regime"]
    regime_color = regime_data["regime_color"]
    regime_emoji = regime_data["regime_emoji"]
    regime_desc = regime_data["regime_desc"]

    # ── Hero Banner ──
    banner_bg = (
        "rgba(31,48,31,0.9)" if regime == "RISK-ON" else
        "rgba(48,24,24,0.9)" if regime == "RISK-OFF" else
        "rgba(40,38,20,0.9)"
    )
    st.markdown(
        f"""
        <div style="background:{banner_bg};border:1px solid {regime_color};border-radius:12px;
                    padding:24px 32px;margin-bottom:24px;">
          <div style="display:flex;align-items:center;gap:16px;">
            <span style="font-size:48px;">{regime_emoji}</span>
            <div>
              <div style="font-size:32px;font-weight:700;color:{regime_color};letter-spacing:2px;">
                {regime}
              </div>
              <div style="font-size:14px;color:#c9d1d9;margin-top:4px;max-width:600px;">
                {regime_desc}
              </div>
            </div>
            <div style="margin-left:auto;text-align:right;">
              <div style="font-size:12px;color:{COLORS['text_dim']};">Aggregate Score</div>
              <div style="font-size:36px;font-weight:700;color:{regime_color};">
                {aggregate:+.2f}
              </div>
              <div style="font-size:11px;color:{COLORS['text_dim']};">-1 (off) to +1 (on)</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Signal Heatmap (compact overview) ──
    if signals:
        st.plotly_chart(_make_heatmap(signals), use_container_width=True)

    # ── Gauge + Radar ──
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_make_gauge(aggregate, regime, regime_color), use_container_width=True)
    with col2:
        if len(signals) >= 3:
            st.plotly_chart(_make_category_radar(signals), use_container_width=True)
        else:
            st.info("Not enough data for radar chart.")

    # ── Regime History (#5) ──
    history_fig = _make_regime_history()
    if history_fig:
        st.markdown("### Regime History")
        st.plotly_chart(history_fig, use_container_width=True)

    # ── Signal Bar Chart ──
    st.markdown("### Individual Signal Scores")
    if signals:
        st.plotly_chart(_make_signal_bar(signals), use_container_width=True)
    else:
        st.warning("No signals could be computed. Check network connectivity.")

    # ── Signal Detail Table ──
    with st.expander("Signal Detail Table", expanded=False):
        if signals:
            df_signals = pd.DataFrame(signals)[
                ["category", "name", "value", "unit", "label", "score", "stale", "interpretation"]
            ]
            df_signals = df_signals.rename(columns={
                "category": "Category", "name": "Signal", "value": "Value",
                "unit": "Unit", "label": "Verdict", "score": "Score",
                "stale": "Stale", "interpretation": "Notes",
            })
            df_signals["Score"] = df_signals["Score"].map(lambda x: f"{x:+.3f}")
            df_signals["Stale"] = df_signals["Stale"].map(lambda x: "⚠️" if x else "")

            def color_verdict(val):
                if val == "Risk-On":
                    return f"color: {COLORS['green']}"
                elif val == "Risk-Off":
                    return f"color: {COLORS['red']}"
                return f"color: {COLORS['yellow']}"

            st.dataframe(
                df_signals.style.applymap(color_verdict, subset=["Verdict"]),
                use_container_width=True,
                hide_index=True,
            )

    # ── Sparklines ──
    st.markdown("### 3-Month Price Performance")
    spark_fig = _make_price_sparklines(snaps)
    if spark_fig:
        st.plotly_chart(spark_fig, use_container_width=True)
    else:
        st.info("Sparkline data unavailable.")

    # ── Truflation ──
    tf_data = fetch_truflation()
    if tf_data and isinstance(tf_data, dict):
        st.markdown("### Truflation Real-Time Inflation")
        tf_cols = st.columns(3)
        for i, (k, v) in enumerate(list(tf_data.items())[:6]):
            with tf_cols[i % 3]:
                st.metric(k.replace("_", " ").title(), v)

    # ── Signal Changes (from history) ──
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

    # ── How it Works ──
    with st.expander("How the Regime Score Works", expanded=False):
        st.markdown(f"""
**Aggregate Score** is a weighted average of {len(signals)} cross-asset signals, ranging from **-1 (full Risk-Off)** to **+1 (full Risk-On)**.

**Adaptive Thresholds:** Signals use z-scores computed against their own 1-year distribution, so thresholds automatically adjust to changing market conditions.

| Regime | Score | Characteristics |
|---|---|---|
| Risk-On | > +0.35 | Equities bid, HY credit tight, VIX low, USD weak, copper rising |
| Neutral | -0.35 to +0.35 | Mixed signals, transition phase |
| Risk-Off | < -0.35 | Flight to safety, TLT bid, VIX elevated, gold/USD rising |

**Signal Categories & Weights:**
- Yield Curve, Credit Spreads, Volatility: **1.5x** (highest weight)
- Equities (US): **1.2x**
- FX (USD), Global Equities: **0.9-1.0x**
- Commodities, Inflation: **0.8x**
- Sector Rotation: **0.7x**

**New in this version:** Gold/Copper ratio, bond volatility (MOVE proxy), MA trend signals, z-score adaptive thresholds, regime history tracking, stale data discounting.

Data refreshes every **60 minutes** (cached). Click Refresh to force update.
        """)

    st.markdown(
        f"<p style='color:{COLORS['text_dim']};font-size:11px;margin-top:24px;'>"
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        "Data via Yahoo Finance &amp; Truflation API | For informational purposes only.</p>",
        unsafe_allow_html=True,
    )
