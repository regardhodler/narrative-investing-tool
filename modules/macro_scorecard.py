"""Macro Regime Scorecard — mechanical 4-factor regime scoring, trigger system,
position sizing model, and action summary.

Framework:
  1. Regime Score: Growth / Inflation / Liquidity / Credit each -1 to +1
  2. Trigger System: short-term directional signals (oil, yields, spreads, equities, vol)
  3. Position Sizing: composite alignment × conviction × risk → allocation %
  4. Action Summary: ADD / REDUCE / HOLD
"""

from __future__ import annotations

import json
import os

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

from utils.theme import COLORS, apply_dark_layout

_REGIME_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json"
)


# ── Constants ────────────────────────────────────────────────────────────────

_REGIME_BANDS = [
    (+2, +4, "Overheat",    "#ef4444", "Inflation risk dominant. Growth strong, spreads tight."),
    (+1, +1, "Expansion",   "#22c55e", "Growth accelerating, inflation moderate, liquidity ample."),
    ( 0,  0, "Neutral",     "#f59e0b", "Mixed signals. No dominant regime. Stay selective."),
    (-1, -1, "Contraction", "#f97316", "Growth slowing. Defensive rotation warranted."),
    (-2, -3, "Stagflation", "#a855f7", "Slow growth + sticky inflation. Hardest regime for equities."),
    (-4, -4, "Crisis",      "#ef4444", "All factors negative. Maximum defensiveness."),
]

_ASSETS = [
    ("Gold",             "GLD",  "Inflation hedge / safe haven"),
    ("Silver",           "SLV",  "Industrial + monetary hedge"),
    ("Utilities",        "XLU",  "Defensive yield, rate-sensitive"),
    ("Long Bonds",       "TLT",  "Duration play, deflation hedge"),
    ("Energy",           "XLE",  "Oil/gas — inflation + growth proxy"),
    ("Copper (FCX)",     "FCX",  "Global growth barometer"),
    ("Hedge",            "VXX",  "Volatility / tail-risk insurance"),
]

# Regime alignment scores per asset per regime (1=weak, 5=strong)
# [Overheat, Expansion, Neutral, Contraction, Stagflation, Crisis]
_ASSET_REGIME_ALIGNMENT = {
    "GLD": [4, 2, 3, 4, 5, 5],
    "SLV": [3, 3, 2, 3, 4, 3],
    "XLU": [1, 2, 3, 4, 3, 5],
    "TLT": [1, 2, 3, 5, 2, 5],
    "XLE": [5, 4, 3, 2, 4, 1],
    "FCX": [4, 5, 3, 2, 2, 1],
    "VXX": [2, 1, 2, 4, 4, 5],
}


@st.cache_data(ttl=900)
def _fetch_price_data() -> dict:
    """Fetch recent price history for scorecard assets + macro tickers."""
    tickers = ["GLD", "SLV", "XLU", "TLT", "XLE", "FCX", "VXX",
               "USO", "SPY", "^VIX", "^TNX", "HYG"]
    try:
        raw = yf.download(tickers, period="3mo", interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return {}
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw[["Close"]]
        close = close.dropna(how="all")
        out = {}
        for t in tickers:
            if t in close.columns:
                s = close[t].dropna()
                if len(s) >= 2:
                    out[t] = s
        return out
    except Exception:
        return {}


def _regime_stability() -> int:
    """Count consecutive sessions where regime classification matches the most recent entry."""
    try:
        with open(_REGIME_HISTORY_PATH) as f:
            records = json.load(f)
    except Exception:
        return 0
    if not records:
        return 0
    records = sorted(records, key=lambda r: r["date"])
    current_regime = records[-1]["regime"]
    count = 0
    for r in reversed(records):
        if r["regime"] == current_regime:
            count += 1
        else:
            break
    return count


def _pct_change(series: pd.Series, n: int) -> float | None:
    """N-day percentage change."""
    if series is None or len(series) < n + 1:
        return None
    return float((series.iloc[-1] / series.iloc[-(n+1)] - 1) * 100)


def _score_direction(val: float | None, pos_thresh: float, neg_thresh: float) -> int:
    """Return +1, 0, -1 based on value vs thresholds."""
    if val is None:
        return 0
    if val > pos_thresh:
        return 1
    if val < neg_thresh:
        return -1
    return 0


def _classify_regime(total: int) -> tuple[str, str, str]:
    """Return (label, color, description) for a total score."""
    for lo, hi, label, color, desc in _REGIME_BANDS:
        if lo <= total <= hi:
            return label, color, desc
    if total > 4:
        return "Overheat", "#ef4444", "Extreme growth + inflation."
    return "Crisis", "#ef4444", "All factors deeply negative."


# ── Factor scoring ────────────────────────────────────────────────────────────

def _score_growth(prices: dict, regime_ctx: dict) -> tuple[int, str]:
    """Growth: accelerating=+1, neutral=0, slowing=-1."""
    signals = regime_ctx.get("signals_summary", {})

    # Use existing regime signal scores as primary inputs
    eq_score    = signals.get("Equity Trend (S&P, Nasdaq, Dow)", 0.0)
    lei_score   = signals.get("Leading Economic Index", 0.0)
    indpro      = signals.get("Industrial Production", 0.0)
    unemp       = signals.get("Unemployment Trend (Sahm context)", 0.0)

    composite = (eq_score + lei_score + indpro - unemp) / 4

    if composite > 0.15:
        return +1, f"Accelerating (eq={eq_score:+.2f}, LEI={lei_score:+.2f})"
    if composite < -0.15:
        return -1, f"Slowing (eq={eq_score:+.2f}, LEI={lei_score:+.2f}, unemp={unemp:+.2f})"
    return 0, f"Neutral (composite={composite:+.2f})"


def _score_inflation(regime_ctx: dict, fred_data: dict | None) -> tuple[int, str]:
    """Inflation: rising=+1 (reflationary), stable=0, falling=-1 (disinflationary)."""
    signals = regime_ctx.get("signals_summary", {})
    infl_score = signals.get("Core Inflation (PCE)", 0.0)
    comm_score = signals.get("Commodity Trend (Oil + Copper)", 0.0)

    composite = (infl_score + comm_score) / 2

    # Context: rising inflation in expansion = mild +1; in slowdown = stagflationary (still +1 by factor def)
    if composite > 0.15:
        return +1, f"Rising (PCE={infl_score:+.2f}, commodities={comm_score:+.2f})"
    if composite < -0.15:
        return -1, f"Falling (PCE={infl_score:+.2f}, commodities={comm_score:+.2f})"
    return 0, f"Stable (composite={composite:+.2f})"


def _score_liquidity(regime_ctx: dict) -> tuple[int, str]:
    """Liquidity: easing=+1, neutral=0, tightening=-1."""
    signals = regime_ctx.get("signals_summary", {})
    m2_score  = signals.get("Net Liquidity (M2 − Drains)", signals.get("Global Liquidity (M2 proxy)", 0.0))
    fci_score = signals.get("Financial Conditions Index", 0.0)

    composite = (m2_score + fci_score) / 2

    if composite > 0.15:
        return +1, f"Easing (M2={m2_score:+.2f}, FCI={fci_score:+.2f})"
    if composite < -0.15:
        return -1, f"Tightening (M2={m2_score:+.2f}, FCI={fci_score:+.2f})"
    return 0, f"Neutral (composite={composite:+.2f})"


def _score_credit(regime_ctx: dict) -> tuple[int, str]:
    """Credit: tightening=+1, neutral=0, widening=-1."""
    signals = regime_ctx.get("signals_summary", {})
    cs_score  = signals.get("Credit Spreads (HY vs Treasuries)", 0.0)
    hyg_score = signals.get("HYG/LQD Ratio (Credit Risk Appetite)", 0.0)

    # For credit factor: positive = spread TIGHTENING (risk-on) = +1
    composite = (cs_score + hyg_score) / 2

    if composite > 0.15:
        return +1, f"Tightening (spreads={cs_score:+.2f}, HYG/LQD={hyg_score:+.2f})"
    if composite < -0.15:
        return -1, f"Widening (spreads={cs_score:+.2f}, HYG/LQD={hyg_score:+.2f})"
    return 0, f"Neutral (composite={composite:+.2f})"


# ── Trigger system ────────────────────────────────────────────────────────────

def _evaluate_triggers(prices: dict) -> dict:
    """Evaluate medium-term directional triggers (20-day, suited for 3-month hold style)."""
    oil_20d   = _pct_change(prices.get("USO"), 20)
    spy_20d   = _pct_change(prices.get("SPY"), 20)
    tlt_20d   = _pct_change(prices.get("TLT"), 20)
    hyg_20d   = _pct_change(prices.get("HYG"), 20)

    vix_series = prices.get("^VIX")
    vix_now   = float(vix_series.iloc[-1]) if vix_series is not None and len(vix_series) else None
    vix_21ago = float(vix_series.iloc[-22]) if vix_series is not None and len(vix_series) >= 22 else None
    vix_delta = (vix_now - vix_21ago) if (vix_now and vix_21ago) else None

    tnx_series = prices.get("^TNX")
    tnx_20d = _pct_change(tnx_series, 20)

    def _dir(val, pos=1.0, neg=-1.0, label_pos="↑", label_neg="↓", label_flat="→"):
        if val is None:
            return "—", "#64748b", 0
        if val > pos:
            return f"{label_pos} {val:+.1f}%", "#22c55e", 1
        if val < neg:
            return f"{label_neg} {val:+.1f}%", "#ef4444", -1
        return f"{label_flat} {val:+.1f}%", "#94a3b8", 0

    oil_lbl,  oil_clr,  oil_dir  = _dir(oil_20d,   3.0,  -3.0)
    spy_lbl,  spy_clr,  spy_dir  = _dir(spy_20d,   4.0,  -4.0)
    tlt_lbl,  tlt_clr,  tlt_dir  = _dir(tlt_20d,   2.5,  -2.5)
    hyg_lbl,  hyg_clr,  hyg_dir  = _dir(hyg_20d,   1.5,  -1.5)

    if vix_now is not None:
        vix_lbl = f"{'↑' if (vix_delta or 0) > 5 else ('↓' if (vix_delta or 0) < -5 else '→')} {vix_now:.1f}"
        vix_clr = "#ef4444" if (vix_delta or 0) > 5 else ("#22c55e" if (vix_delta or 0) < -5 else "#94a3b8")
        vix_dir = 1 if (vix_delta or 0) > 5 else (-1 if (vix_delta or 0) < -5 else 0)
    else:
        vix_lbl, vix_clr, vix_dir = "—", "#64748b", 0

    tnx_lbl,  tnx_clr,  tnx_dir  = _dir(tnx_20d,  5.0, -5.0, "↑", "↓", "→")

    # Determine active trigger
    spread_widening = hyg_dir < 0
    eq_weak         = spy_dir < 0
    eq_strong       = spy_dir > 0
    oil_rising      = oil_dir > 0
    vix_spiking     = vix_dir > 0
    yields_falling  = tnx_dir < 0

    if spread_widening and vix_spiking:
        trigger = "Crisis"
        trigger_color = "#ef4444"
        confidence = "High" if (hyg_20d or 0) < -3.0 and (vix_delta or 0) > 8 else "Medium"
    elif spread_widening and eq_weak:
        trigger = "Risk-Off"
        trigger_color = "#f97316"
        confidence = "High" if (hyg_20d or 0) < -2.0 and (spy_20d or 0) < -5.0 else "Medium"
    elif yields_falling and eq_strong:
        trigger = "Risk-On"
        trigger_color = "#22c55e"
        confidence = "High" if (tnx_20d or 0) < -6.0 and (spy_20d or 0) > 5.0 else "Medium"
    elif oil_rising and (vix_dir >= 0):
        trigger = "Inflation Shock"
        trigger_color = "#f59e0b"
        confidence = "Medium" if (oil_20d or 0) > 8.0 else "Low"
    else:
        trigger = "None"
        trigger_color = "#64748b"
        confidence = "—"

    return {
        "rows": [
            ("Oil (USO 20d)",         oil_lbl,  oil_clr),
            ("Bond Yields (TNX 20d)", tnx_lbl,  tnx_clr),
            ("Credit Spreads (HYG 20d)", hyg_lbl, hyg_clr),
            ("Equities (SPY 20d)",    spy_lbl,  spy_clr),
            ("Volatility (VIX Δ20d)", vix_lbl,  vix_clr),
        ],
        "trigger":        trigger,
        "trigger_color":  trigger_color,
        "confidence":     confidence,
    }


# ── Position sizing ───────────────────────────────────────────────────────────

def _position_sizing(regime_label: str, prices: dict) -> list[dict]:
    """Score each asset and compute recommended allocation."""
    regime_idx = {
        "Overheat": 0, "Expansion": 1, "Neutral": 2,
        "Contraction": 3, "Stagflation": 4, "Crisis": 5,
    }.get(regime_label, 2)

    rows = []
    for name, ticker, desc in _ASSETS:
        alignment = _ASSET_REGIME_ALIGNMENT.get(ticker, [3, 3, 3, 3, 3, 3])[regime_idx]

        # Conviction from momentum (1-5)
        series = prices.get(ticker)
        mom_20 = _pct_change(series, 20) if series is not None else None
        if mom_20 is None:
            conviction = 3
        elif mom_20 > 5:
            conviction = 5
        elif mom_20 > 2:
            conviction = 4
        elif mom_20 > -2:
            conviction = 3
        elif mom_20 > -5:
            conviction = 2
        else:
            conviction = 1

        # Risk (5=stable, 1=volatile) — based on 20d std of returns
        if series is not None and len(series) >= 22:
            ret_std = float(series.pct_change().dropna().iloc[-20:].std() * 100)
            if ret_std < 0.8:
                risk = 5
            elif ret_std < 1.5:
                risk = 4
            elif ret_std < 2.5:
                risk = 3
            elif ret_std < 4.0:
                risk = 2
            else:
                risk = 1
        else:
            risk = 3

        score = alignment * conviction * risk

        if score >= 80:
            alloc = "10–20%"
            alloc_mid = 15.0
            row_color = "#14532d"
        elif score >= 50:
            alloc = "5–10%"
            alloc_mid = 7.5
            row_color = "#1e3a2a"
        elif score >= 25:
            alloc = "2–5%"
            alloc_mid = 3.5
            row_color = "#1a1f2e"
        else:
            alloc = "0–2%"
            alloc_mid = 1.0
            row_color = "#1a1a1a"

        mom_str = f"{mom_20:+.1f}%" if mom_20 is not None else "—"

        rows.append({
            "Asset":      name,
            "Ticker":     ticker,
            "Desc":       desc,
            "Alignment":  alignment,
            "Conviction": conviction,
            "Risk":       risk,
            "Score":      score,
            "Allocation": alloc,
            "AllocMid":   alloc_mid,
            "Mom20":      mom_str,
            "RowColor":   row_color,
        })

    return sorted(rows, key=lambda r: r["Score"], reverse=True)


def _action_summary(rows: list[dict], regime_label: str, trigger: str) -> dict[str, list[str]]:
    """Generate ADD / REDUCE / HOLD action bullets."""
    add, reduce, hold = [], [], []

    for r in rows:
        if r["Score"] >= 80:
            add.append(f"{r['Asset']} ({r['Ticker']}) — score {r['Score']}, {r['Allocation']}")
        elif r["Score"] >= 50:
            hold.append(f"{r['Asset']} ({r['Ticker']}) — score {r['Score']}, {r['Allocation']}")
        else:
            reduce.append(f"{r['Asset']} ({r['Ticker']}) — score {r['Score']}, trim to {r['Allocation']}")

    # Trigger overrides
    if trigger == "Risk-Off":
        add.insert(0, "⚠ Risk-Off trigger active — prioritize Gold, TLT, Utilities over cyclicals")
    elif trigger == "Crisis":
        add.insert(0, "🚨 Crisis trigger — maximize VXX hedge, exit FCX/XLE")
    elif trigger == "Inflation Shock":
        add.insert(0, "🛢 Inflation Shock — overweight Energy and Gold vs long bonds")
    elif trigger == "Risk-On":
        add.insert(0, "✦ Risk-On trigger — lean into FCX, XLE; reduce VXX hedge")

    return {"ADD": add, "REDUCE": reduce, "HOLD": hold}


# ── Render helpers ────────────────────────────────────────────────────────────

def _badge(text: str, bg: str, fg: str = "#fff") -> str:
    return (
        f'<span style="background:{bg};color:{fg};font-size:11px;font-weight:700;'
        f'letter-spacing:0.06em;padding:2px 10px;border-radius:3px;">{text}</span>'
    )


def _section_header(title: str):
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin:20px 0 8px 0;border-bottom:1px solid #1e293b;'
        f'padding-bottom:4px;">{title}</div>',
        unsafe_allow_html=True,
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">MACRO REGIME SCORECARD</div>'
        f'<div style="font-size:11px;color:#64748b;margin-bottom:14px;">'
        f'Mechanical 4-factor framework · Long-hold edition · Growth / Inflation / Liquidity / Credit</div>',
        unsafe_allow_html=True,
    )

    # Pull regime context from session state (populated by Risk Regime module)
    regime_ctx = st.session_state.get("_regime_context") or {}

    if not regime_ctx:
        st.info(
            "Run the **Risk Regime** module first to populate macro signals. "
            "This scorecard uses live signal data already computed there."
        )
        return

    with st.spinner("Fetching price data..."):
        prices = _fetch_price_data()

    # ── Compute scores ────────────────────────────────────────────────────────
    g_score, g_detail = _score_growth(prices, regime_ctx)
    i_score, i_detail = _score_inflation(regime_ctx, None)
    l_score, l_detail = _score_liquidity(regime_ctx)
    c_score, c_detail = _score_credit(regime_ctx)
    total = g_score + i_score + l_score + c_score

    # Use Risk Regime's quadrant classification as the authoritative label —
    # it uses 21 weighted z-scored signals and detects borderline stagflation/
    # goldilocks that the ±0.15 binary threshold here would round to Neutral.
    _rr_quadrant = regime_ctx.get("quadrant", "")
    _quadrant_map = {
        "Stagflation":  ("Stagflation",  "#a855f7", "Slow growth + sticky inflation. Hardest regime for equities."),
        "Goldilocks":   ("Goldilocks",   "#22c55e", "Growth expanding, inflation contained. Best regime for equities."),
        "Reflation":    ("Reflation",    "#f59e0b", "Growth recovering + inflation rising. Cyclicals and commodities lead."),
        "Deflation":    ("Deflation",    "#60a5fa", "Growth slowing + inflation falling. Duration and defensives lead."),
        "Overheat":     ("Overheat",     "#ef4444", "Growth strong but inflation risk rising. Late-cycle caution."),
    }
    if _rr_quadrant and _rr_quadrant in _quadrant_map:
        regime_label, regime_color, regime_desc = _quadrant_map[_rr_quadrant]
    else:
        regime_label, regime_color, regime_desc = _classify_regime(total)

    trigger_data = _evaluate_triggers(prices)
    sizing_rows  = _position_sizing(regime_label, prices)
    actions      = _action_summary(sizing_rows, regime_label, trigger_data["trigger"])

    _stability = _regime_stability()

    tab1, tab2, tab3, tab4 = st.tabs([
        "① Regime Score", "② Entry Window", "③ Position Sizing", "④ Action Summary"
    ])

    # ── TAB 1: Regime Score ───────────────────────────────────────────────────
    with tab1:
        # Total score display
        col_score, col_label = st.columns([1, 3])
        with col_score:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid {regime_color}40;'
                f'border-radius:8px;padding:18px 20px;text-align:center;">'
                f'<div style="font-size:11px;color:#64748b;letter-spacing:0.08em;margin-bottom:4px;">TOTAL SCORE</div>'
                f'<div style="font-size:48px;font-weight:700;color:{regime_color};'
                f'font-family:\'JetBrains Mono\',monospace;">{total:+d}</div>'
                f'<div style="font-size:11px;color:#64748b;">range: −4 to +4</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_label:
            _stab_color = "#22c55e" if _stability >= 10 else ("#f59e0b" if _stability >= 5 else "#94a3b8")
            _stab_note  = "High confidence" if _stability >= 10 else ("Building" if _stability >= 5 else "Early — watch for flip")
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;'
                f'padding:18px 20px;height:100%;">'
                f'<div style="font-size:11px;color:#64748b;letter-spacing:0.08em;margin-bottom:6px;">REGIME CLASSIFICATION</div>'
                f'<div style="font-size:28px;font-weight:700;color:{regime_color};">{regime_label}</div>'
                f'<div style="font-size:12px;color:#94a3b8;margin-top:6px;">{regime_desc}</div>'
                f'<div style="font-size:11px;color:#475569;margin-top:4px;">'
                + (f'Source: Risk Regime quadrant (21-signal z-score)' if _rr_quadrant and _rr_quadrant in _quadrant_map else f'Source: 4-factor composite (score {total:+d})')
                + f'</div>'
                f'<div style="font-size:12px;color:{_stab_color};margin-top:8px;font-weight:600;">'
                f'Stable {_stability} session{"s" if _stability != 1 else ""} · {_stab_note}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Factor table
        _section_header("FACTOR BREAKDOWN")

        factors = [
            ("Growth",    g_score, g_detail, "Accelerating", "Slowing"),
            ("Inflation", i_score, i_detail, "Rising",       "Falling"),
            ("Liquidity", l_score, l_detail, "Easing",       "Tightening"),
            ("Credit",    c_score, c_detail, "Tightening",   "Widening"),
        ]

        rows_html = ""
        for name, score, detail, pos_label, neg_label in factors:
            score_color = "#22c55e" if score > 0 else ("#ef4444" if score < 0 else "#f59e0b")
            verdict = pos_label if score > 0 else (neg_label if score < 0 else "Neutral")
            bar_pct = int((score + 1) / 2 * 100)
            bar_color = score_color

            rows_html += (
                f'<tr style="border-bottom:1px solid #1e293b;">'
                f'<td style="padding:10px 12px;font-size:13px;font-weight:700;color:#e2e8f0;width:120px;">{name}</td>'
                f'<td style="padding:10px 12px;text-align:center;width:80px;">'
                f'<span style="font-size:22px;font-weight:700;color:{score_color};">{score:+d}</span></td>'
                f'<td style="padding:10px 12px;width:100px;">'
                f'<span style="font-size:12px;font-weight:600;color:{score_color};">{verdict}</span></td>'
                f'<td style="padding:10px 12px;">'
                f'<div style="background:#1e293b;border-radius:3px;height:6px;width:100%;">'
                f'<div style="background:{bar_color};height:6px;border-radius:3px;width:{bar_pct}%;"></div>'
                f'</div></td>'
                f'<td style="padding:10px 12px;font-size:11px;color:#64748b;">{detail}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;background:#0f172a;'
            f'border-radius:8px;overflow:hidden;">'
            f'<thead><tr style="background:#1e293b;">'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">FACTOR</th>'
            f'<th style="padding:8px 12px;text-align:center;font-size:10px;color:#64748b;letter-spacing:0.08em;">SCORE</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">DIRECTION</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">GAUGE</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.08em;">SIGNAL INPUTS</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>',
            unsafe_allow_html=True,
        )

        # Regime scale legend
        _section_header("REGIME SCALE")
        legend_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;">'
        for lo, hi, label, color, _ in _REGIME_BANDS:
            score_range = f"{lo:+d}" if lo == hi else f"{lo:+d} to {hi:+d}"
            legend_html += (
                f'<div style="background:#0f172a;border:1px solid {color}40;border-radius:6px;'
                f'padding:6px 12px;text-align:center;">'
                f'<div style="font-size:11px;font-weight:700;color:{color};">{label}</div>'
                f'<div style="font-size:10px;color:#64748b;">{score_range}</div>'
                f'</div>'
            )
        legend_html += '</div>'
        st.markdown(legend_html, unsafe_allow_html=True)

    # ── TAB 2: Triggers ───────────────────────────────────────────────────────
    with tab2:
        # Active trigger banner
        t_color = trigger_data["trigger_color"]
        t_conf  = trigger_data["confidence"]
        st.markdown(
            f'<div style="background:#0f172a;border:1px solid {t_color}60;border-radius:8px;'
            f'padding:14px 20px;display:flex;align-items:center;gap:16px;margin-bottom:16px;">'
            f'<div>'
            f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;margin-bottom:2px;">ACTIVE TRIGGER</div>'
            f'<div style="font-size:22px;font-weight:700;color:{t_color};">{trigger_data["trigger"]}</div>'
            f'</div>'
            f'<div style="border-left:1px solid #1e293b;padding-left:16px;">'
            f'<div style="font-size:10px;color:#64748b;letter-spacing:0.08em;margin-bottom:2px;">CONFIDENCE</div>'
            f'<div style="font-size:16px;font-weight:700;color:{t_color};">{t_conf}</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        _section_header("ENTRY WINDOW SIGNALS (20D)")
        for label, value, color in trigger_data["rows"]:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;border-bottom:1px solid #1e293b;">'
                f'<span style="font-size:12px;color:#94a3b8;">{label}</span>'
                f'<span style="font-size:13px;font-weight:700;color:{color};'
                f'font-family:\'JetBrains Mono\',monospace;">{value}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Trigger map reference
        with st.expander("Trigger Reference"):
            st.markdown("""
All signals use **20-day lookback** — designed for quarterly entry decisions, not daily noise.

| Trigger | Condition (20d) | Implication |
|---------|-----------------|-------------|
| **Risk-Off** | Spreads widening + equities weak | Reduce cyclicals, add Gold/TLT |
| **Risk-On** | Yields falling + equities strong | Add FCX/XLE, reduce hedge |
| **Inflation Shock** | Oil rising sharply | Gold + Energy > Bonds |
| **Crisis** | Spreads widening rapidly + VIX spiking | Max hedge, exit growth |
| **None** | Mixed/flat signals | Stay with regime positioning |
""")

    # ── TAB 3: Position Sizing ────────────────────────────────────────────────
    with tab3:
        _section_header(f"POSITION SIZING — {regime_label.upper()} REGIME")

        # Allocation bar chart
        fig = go.Figure()
        alloc_colors = ["#22c55e" if r["AllocMid"] >= 10 else
                        "#60a5fa" if r["AllocMid"] >= 5 else
                        "#f59e0b" if r["AllocMid"] >= 2 else "#ef4444"
                        for r in sizing_rows]
        fig.add_trace(go.Bar(
            x=[r["Asset"] for r in sizing_rows],
            y=[r["AllocMid"] for r in sizing_rows],
            marker_color=alloc_colors,
            text=[r["Allocation"] for r in sizing_rows],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Score: %{customdata[0]}<br>Alloc: %{customdata[1]}<extra></extra>",
            customdata=[[r["Score"], r["Allocation"]] for r in sizing_rows],
        ))
        apply_dark_layout(fig, title="Recommended Allocation by Regime Score", height=280)
        fig.update_layout(
            yaxis=dict(title="Allocation Midpoint (%)", range=[0, 25]),
            margin=dict(t=40, b=30, l=50, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed table
        table_rows = ""
        for r in sizing_rows:
            alloc_color = ("#22c55e" if r["AllocMid"] >= 10 else
                           "#60a5fa" if r["AllocMid"] >= 5 else
                           "#f59e0b" if r["AllocMid"] >= 2 else "#94a3b8")
            table_rows += (
                f'<tr style="border-bottom:1px solid #1e293b;">'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:#e2e8f0;">{r["Asset"]}</td>'
                f'<td style="padding:8px 12px;font-size:11px;color:#64748b;">{r["Ticker"]}</td>'
                f'<td style="padding:8px 12px;text-align:center;font-size:12px;color:#94a3b8;">{r["Alignment"]}</td>'
                f'<td style="padding:8px 12px;text-align:center;font-size:12px;color:#94a3b8;">{r["Conviction"]}</td>'
                f'<td style="padding:8px 12px;text-align:center;font-size:12px;color:#94a3b8;">{r["Risk"]}</td>'
                f'<td style="padding:8px 12px;text-align:center;font-size:13px;font-weight:700;color:#e2e8f0;">{r["Score"]}</td>'
                f'<td style="padding:8px 12px;font-size:12px;font-weight:700;color:{alloc_color};">{r["Allocation"]}</td>'
                f'<td style="padding:8px 12px;font-size:11px;color:#64748b;">{r["Mom20"]}</td>'
                f'</tr>'
            )

        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;background:#0f172a;border-radius:8px;overflow:hidden;">'
            f'<thead><tr style="background:#1e293b;">'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;letter-spacing:0.06em;">ASSET</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;">TICKER</th>'
            f'<th style="padding:8px 12px;text-align:center;font-size:10px;color:#64748b;">ALIGNMENT</th>'
            f'<th style="padding:8px 12px;text-align:center;font-size:10px;color:#64748b;">CONVICTION</th>'
            f'<th style="padding:8px 12px;text-align:center;font-size:10px;color:#64748b;">RISK</th>'
            f'<th style="padding:8px 12px;text-align:center;font-size:10px;color:#64748b;">SCORE</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;">ALLOCATION</th>'
            f'<th style="padding:8px 12px;text-align:left;font-size:10px;color:#64748b;">20D MOM</th>'
            f'</tr></thead><tbody>{table_rows}</tbody></table>',
            unsafe_allow_html=True,
        )

        with st.expander("Scoring Methodology"):
            st.markdown("""
**Score = Alignment × Conviction × Risk** (max = 5×5×5 = 125)

| Range | Allocation |
|-------|------------|
| 80–125 | 10–20% |
| 50–79 | 5–10% |
| 25–49 | 2–5% |
| 0–24 | 0–2% |

- **Alignment (1–5)**: How well this asset fits the current regime classification
- **Conviction (1–5)**: Derived from 20-day momentum — strong trend = higher conviction
- **Risk (1–5)**: Inverse of 20-day return volatility — lower vol = higher risk score
""")

    # ── TAB 4: Action Summary ─────────────────────────────────────────────────
    with tab4:
        _section_header("ACTION SUMMARY")

        for category, bullet_color in [("ADD", "#22c55e"), ("HOLD", "#f59e0b"), ("REDUCE", "#ef4444")]:
            items = actions.get(category, [])
            st.markdown(
                f'<div style="font-size:13px;font-weight:700;color:{bullet_color};'
                f'letter-spacing:0.08em;margin:14px 0 6px 0;">{category}</div>',
                unsafe_allow_html=True,
            )
            if items:
                for item in items:
                    st.markdown(
                        f'<div style="display:flex;align-items:flex-start;gap:8px;'
                        f'padding:5px 0;border-bottom:1px solid #1e293b;">'
                        f'<span style="color:{bullet_color};font-weight:700;flex-shrink:0;">▸</span>'
                        f'<span style="font-size:12px;color:#e2e8f0;">{item}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div style="font-size:12px;color:#475569;padding:4px 0;">— None at current regime</div>',
                    unsafe_allow_html=True,
                )

        # Regime + trigger context footer
        st.markdown(
            f'<div style="margin-top:20px;padding:12px 16px;background:#0f172a;'
            f'border:1px solid #1e293b;border-radius:6px;">'
            f'<span style="font-size:11px;color:#64748b;">Regime: </span>'
            f'<span style="font-size:11px;font-weight:700;color:{regime_color};">{regime_label} ({total:+d})</span>'
            f' &nbsp;·&nbsp; '
            f'<span style="font-size:11px;color:#64748b;">Trigger: </span>'
            f'<span style="font-size:11px;font-weight:700;color:{trigger_data["trigger_color"]};">'
            f'{trigger_data["trigger"]} ({trigger_data["confidence"]})</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Store in session state for export
        st.session_state["_macro_scorecard"] = {
            "regime": regime_label,
            "total_score": total,
            "factors": {"growth": g_score, "inflation": i_score, "liquidity": l_score, "credit": c_score},
            "trigger": trigger_data["trigger"],
            "trigger_confidence": trigger_data["confidence"],
            "actions": actions,
        }
