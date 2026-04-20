"""Cross-Asset Correlation Monitor — contagion detection and regime-shift early warning."""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from services.market_data import fetch_correlation_matrix
from utils.theme import COLORS, apply_dark_layout

_HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "correlation_history.json")
_MAX_ENTRIES = 365

# Fixed cross-asset universe — chosen for regime-signal diversity
_UNIVERSE = ("SPY", "QQQ", "TLT", "GLD", "UUP", "HYG", "^VIX", "USO", "EEM")

_LABELS = {
    "SPY": "S&P 500", "QQQ": "Nasdaq", "TLT": "20Y Bonds", "GLD": "Gold",
    "UUP": "US Dollar", "HYG": "HY Credit", "^VIX": "VIX", "USO": "Oil", "EEM": "EM Equity",
}

_PERIOD_MAP = {"1M": "1mo", "3M": "3mo", "6M": "6mo"}


def _contagion_score(corr: pd.DataFrame) -> float:
    """Average absolute correlation across all off-diagonal pairs, scaled 0-100."""
    mask = np.ones(corr.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    abs_mean = np.abs(corr.values[mask]).mean()
    return min(100.0, abs_mean / 0.8 * 100.0)


def _contagion_zone(score: float) -> tuple[str, str]:
    """Return (label, color) for a contagion score."""
    if score >= 80:
        return "CRISIS CORRELATION", "#ff1744"
    if score >= 60:
        return "CONTAGION RISK", "#ef4444"
    if score >= 30:
        return "ELEVATED", "#f59e0b"
    return "HEALTHY DIVERSIFICATION", "#00c853"


def _render_heatmap(corr: pd.DataFrame):
    """Annotated Plotly heatmap of pairwise correlations."""
    labels = [_LABELS.get(t, t) for t in corr.columns]
    z = corr.values

    text = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, family="JetBrains Mono, Consolas, monospace"),
        colorscale=[
            [0.0, "#1e88e5"],
            [0.5, COLORS["bg"]],
            [1.0, "#ef4444"],
        ],
        zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text="Corr", font=dict(size=10)),
            tickfont=dict(size=10), len=0.6,
        ),
        hoverongaps=False,
    ))

    apply_dark_layout(fig,
        title=None,
        xaxis=dict(side="bottom", tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        margin=dict(l=100, r=40, t=20, b=80),
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_contagion_card(score: float):
    """Large metric card for the contagion score."""
    label, color = _contagion_zone(score)
    pill_html = (
        f'<span style="background:{color}22;color:{color};font-size:10px;font-weight:700;'
        f'padding:3px 10px;border-radius:12px;letter-spacing:0.08em;border:1px solid {color}44;">'
        f'{label}</span>'
    )
    card_html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'padding:16px 20px;border-radius:6px;font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'text-align:center;">'
        f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
        f'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;">CONTAGION INDEX</div>'
        f'<div style="font-size:36px;font-weight:900;color:{color};margin-bottom:8px;">'
        f'{score:.0f}</div>'
        f'{pill_html}'
        f'<div style="font-size:9px;color:{COLORS["text_dim"]};margin-top:10px;">'
        f'Avg |corr| across {len(_UNIVERSE)} cross-asset pairs</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


def _render_change_alerts(current: pd.DataFrame, baseline: pd.DataFrame):
    """Flag pairs where correlation shifted significantly between current and baseline."""
    threshold = 0.3
    rows = []
    tickers = list(current.columns)
    for i, t1 in enumerate(tickers):
        for t2 in tickers[i + 1:]:
            cur = current.loc[t1, t2]
            base = baseline.loc[t1, t2]
            delta = cur - base
            if abs(delta) >= threshold:
                rows.append({
                    "Pair": f"{_LABELS.get(t1, t1)} / {_LABELS.get(t2, t2)}",
                    "30d": round(cur, 2),
                    "90d": round(base, 2),
                    "Delta": round(delta, 2),
                    "Direction": "CONVERGING" if delta > 0 else "DIVERGING",
                })

    if not rows:
        st.markdown(
            f'<div style="color:{COLORS["text_dim"]};font-size:12px;padding:12px;">'
            f'No significant correlation shifts detected (threshold: &plusmn;{threshold:.1f})</div>',
            unsafe_allow_html=True,
        )
        return

    df = pd.DataFrame(rows).sort_values("Delta", key=abs, ascending=False)

    header = (
        f'<tr style="border-bottom:1px solid {COLORS["border"]};">'
        + "".join(
            f'<th style="text-align:left;padding:6px 10px;font-size:10px;color:{COLORS["bloomberg_orange"]};'
            f'letter-spacing:0.08em;text-transform:uppercase;">{col}</th>'
            for col in df.columns
        )
        + "</tr>"
    )
    body = ""
    for _, row in df.iterrows():
        color = "#ef4444" if row["Direction"] == "CONVERGING" else "#4b9fff"
        cells = ""
        for col in df.columns:
            val = row[col]
            if col == "Direction":
                cells += (
                    f'<td style="padding:6px 10px;font-size:11px;">'
                    f'<span style="color:{color};font-weight:700;">{val}</span></td>'
                )
            elif col == "Delta":
                sign = "+" if val > 0 else ""
                cells += (
                    f'<td style="padding:6px 10px;font-size:11px;color:{color};font-weight:700;">'
                    f'{sign}{val}</td>'
                )
            else:
                cells += f'<td style="padding:6px 10px;font-size:11px;color:{COLORS["text"]};">{val}</td>'
        body += f'<tr style="border-bottom:1px solid {COLORS["border"]}11;">{cells}</tr>'

    table_html = (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;overflow:hidden;font-family:\'JetBrains Mono\',Consolas,monospace;">'
        f'<table style="width:100%;border-collapse:collapse;">{header}{body}</table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def _load_history() -> list[dict]:
    if not os.path.exists(_HISTORY_PATH):
        return []
    try:
        with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _log_snapshot(score: float, zone_label: str, shifts: list[dict]):
    """Save today's contagion snapshot. One entry per day, keeps last 365."""
    today = datetime.now().strftime("%Y-%m-%d")
    history = _load_history()
    history = [h for h in history if h.get("date") != today]
    history.append({
        "date": today,
        "contagion_score": round(score, 1),
        "zone": zone_label,
        "top_shifts": shifts[:5],
    })
    if len(history) > _MAX_ENTRIES:
        history = history[-_MAX_ENTRIES:]
    os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
    with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def _render_history_chart():
    """Line chart of contagion score over time with zone bands."""
    history = _load_history()
    if len(history) < 2:
        st.caption("Log at least 2 days to see the trend chart.")
        return

    dates = [h["date"] for h in history]
    scores = [h["contagion_score"] for h in history]

    fig = go.Figure()

    # Zone bands
    fig.add_hrect(y0=0, y1=30, fillcolor="#00c853", opacity=0.07, line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="#f59e0b", opacity=0.07, line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="#ef4444", opacity=0.07, line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="#ff1744", opacity=0.10, line_width=0)

    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode="lines+markers",
        line=dict(color=COLORS["bloomberg_orange"], width=2),
        marker=dict(size=5, color=COLORS["bloomberg_orange"]),
        hovertemplate="Date: %{x}<br>Contagion: %{y:.0f}<extra></extra>",
    ))

    # Zone threshold lines
    for level, label in [(30, "Elevated"), (60, "Contagion"), (80, "Crisis")]:
        fig.add_hline(y=level, line_dash="dot", line_color=COLORS["grid"],
                      annotation_text=label, annotation_position="top left",
                      annotation_font=dict(size=9, color=COLORS["text_dim"]))

    apply_dark_layout(fig,
        title=None,
        yaxis=dict(title="Contagion Index", range=[0, 100]),
        xaxis=dict(title=None),
        height=280,
        margin=dict(l=50, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render():
    st.markdown(
        f'<div style="font-size:10px;color:{COLORS["text_dim"]};letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:4px;">Cross-Asset</div>'
        f'<div style="font-size:22px;font-weight:900;color:{COLORS["text"]};margin-bottom:2px;">'
        f'CORRELATION MONITOR</div>'
        f'<div style="height:2px;margin-bottom:16px;'
        f'background:linear-gradient(90deg,{COLORS["bloomberg_orange"]},'
        f'{COLORS["bloomberg_orange"]}44,transparent);border-radius:1px;"></div>',
        unsafe_allow_html=True,
    )

    with st.expander("How to read the Contagion Index"):
        st.markdown(
            f"""
- **0–30**: Assets behaving independently — normal, healthy
- **30–60**: Starting to move together — pay attention
- **60–80**: Broad risk-off selling — cross-check your QIR signals
- **80+**: Everything correlated — crisis-level, similar to COVID/GFC
""")

    period = st.radio(
        "Lookback", list(_PERIOD_MAP.keys()),
        horizontal=True, key="corr_lookback", label_visibility="collapsed",
    )
    yf_period = _PERIOD_MAP[period]

    corr = fetch_correlation_matrix(_UNIVERSE, period=yf_period)
    if corr is None or corr.empty:
        st.warning("Failed to fetch correlation data. Markets may be closed.")
        return

    # Contagion score card + heatmap side by side
    col_score, col_map = st.columns([1, 3])
    with col_score:
        score = _contagion_score(corr)

        # Auto-log once per calendar day (session-key guard)
        _today = datetime.now().strftime("%Y-%m-%d")
        if st.session_state.get("_corr_last_logged") != _today:
            _c30 = fetch_correlation_matrix(_UNIVERSE, period="1mo")
            _c90 = fetch_correlation_matrix(_UNIVERSE, period="3mo")
            _auto_shifts = []
            if _c30 is not None and _c90 is not None:
                _tks = list(_c30.columns)
                for _i, _t1 in enumerate(_tks):
                    for _t2 in _tks[_i + 1:]:
                        _d = _c30.loc[_t1, _t2] - _c90.loc[_t1, _t2]
                        if abs(_d) >= 0.3:
                            _auto_shifts.append({"pair": f"{_t1}/{_t2}", "delta": round(_d, 2),
                                                  "cur": round(_c30.loc[_t1, _t2], 2),
                                                  "base": round(_c90.loc[_t1, _t2], 2)})
            _zone_label, _ = _contagion_zone(score)
            _log_snapshot(score, _zone_label, _auto_shifts)
            st.session_state["_corr_last_logged"] = _today

        _render_contagion_card(score)
    with col_map:
        _render_heatmap(corr)

    # Log button + history chart
    col_log, col_hist_label = st.columns([1, 5])
    with col_log:
        if st.button("Log Today", key="corr_log_today", type="primary"):
            current_30d_log = fetch_correlation_matrix(_UNIVERSE, period="1mo")
            baseline_90d_log = fetch_correlation_matrix(_UNIVERSE, period="3mo")
            shifts = []
            if current_30d_log is not None and baseline_90d_log is not None:
                tickers_log = list(current_30d_log.columns)
                for i, t1 in enumerate(tickers_log):
                    for t2 in tickers_log[i + 1:]:
                        cur = current_30d_log.loc[t1, t2]
                        base = baseline_90d_log.loc[t1, t2]
                        delta = cur - base
                        if abs(delta) >= 0.3:
                            shifts.append({"pair": f"{t1}/{t2}", "delta": round(delta, 2),
                                           "cur": round(cur, 2), "base": round(base, 2)})
            zone_label, _ = _contagion_zone(score)
            _log_snapshot(score, zone_label, shifts)
            st.toast(f"Logged contagion {score:.0f} ({zone_label})", icon="✅")
    with col_hist_label:
        st.markdown(
            f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;text-transform:uppercase;margin:8px 0;">CONTAGION HISTORY</div>',
            unsafe_allow_html=True,
        )
    _render_history_chart()

    # Correlation change alerts
    st.markdown(
        f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;text-transform:uppercase;margin:20px 0 8px 0;">'
        f'CORRELATION SHIFT ALERTS (30d vs 90d)</div>',
        unsafe_allow_html=True,
    )
    current_30d = fetch_correlation_matrix(_UNIVERSE, period="1mo")
    baseline_90d = fetch_correlation_matrix(_UNIVERSE, period="3mo")
    if current_30d is not None and baseline_90d is not None:
        _render_change_alerts(current_30d, baseline_90d)
    else:
        st.caption("Insufficient data for change detection.")
