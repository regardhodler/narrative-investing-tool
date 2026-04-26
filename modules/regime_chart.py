"""
Regime Chart — visual backtest for both HMM brains vs SPX history.

Layout per brain: a 3-row Plotly subplot sharing X axis.
  row 1: SPX log price (clearly visible)
  row 2: Regime ribbon (heatmap strip — state color over time)
  row 3: LL z-score with trigger threshold + trigger-fire markers

A check-engine status banner sits at the top; lead-time tables under each chart;
a Shadow vs Main LL comparison closes the page.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from services.hmm_regime import (
    compute_full_state_path,
    get_ci_anchor,
    get_state_color,
    load_hmm_brain,
)
from services.hmm_shadow import (
    compute_full_shadow_state_path,
    load_shadow_brain,
)
from utils.theme import COLORS, apply_dark_layout


_CRISIS_MARKERS = [
    ("1973-10-19", "Oil Shock"),
    ("1987-10-19", "Black Monday"),
    ("2000-03-10", "Dot-com"),
    ("2008-10-09", "GFC"),
    ("2018-02-05", "Volmageddon"),
    ("2020-03-23", "COVID"),
    ("2022-10-12", "Rate Shock"),
]

def _trigger_anchor() -> float:
    """Negative-signed CI anchor used as the LL z-score trigger threshold."""
    return -get_ci_anchor()

_SPX_LINE_COLOR = "#B0B8C5"
_LL_LINE_COLOR = "#4B9FFF"
_TRIGGER_FIRE_COLOR = "#FF1744"

_RETRAIN_DUE_DAYS = 90

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_MAIN_HISTORY = os.path.join(_DATA_DIR, "hmm_state_history.json")
_SHADOW_HISTORY = os.path.join(_DATA_DIR, "hmm_shadow_history.json")


# ────────────────────────────────────────────────────────────────────────────
# Data loaders
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_spx_close() -> pd.Series:
    import yfinance as yf
    df = yf.download("^GSPC", start="1960-01-01", progress=False, auto_adjust=True)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


@st.cache_data(ttl=3600, show_spinner=False)
def _load_shadow_path(_cache_key: str = "") -> pd.DataFrame | None:
    return compute_full_shadow_state_path()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_main_path(_cache_key: str = "") -> pd.DataFrame | None:
    return compute_full_state_path()


# ────────────────────────────────────────────────────────────────────────────
# Check-engine helpers
# ────────────────────────────────────────────────────────────────────────────

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _last_history_date(history_path: str) -> str | None:
    if not os.path.exists(history_path):
        return None
    try:
        with open(history_path) as f:
            hist = json.load(f)
        if not hist:
            return None
        return hist[-1].get("date")
    except Exception:
        return None


def _brain_health(brain, history_path: str, label: str) -> dict:
    """Return status dict with severity (ok/warn/error), short message, details list."""
    if brain is None:
        return {
            "label": label,
            "severity": "error",
            "icon": "❌",
            "color": "#ef4444",
            "message": "Brain not trained",
            "details": [f"{label}: no trained brain found. Train it first."],
        }

    details = []
    severity = "ok"

    try:
        trained_dt = datetime.fromisoformat(brain.trained_at.replace("Z", "+00:00"))
        days_since = (datetime.now(timezone.utc) - trained_dt).days
    except Exception:
        days_since = None

    if days_since is None:
        details.append(f"{label}: training timestamp unreadable.")
        severity = "warn"
    elif days_since >= _RETRAIN_DUE_DAYS:
        details.append(f"{label}: trained {days_since} days ago — retrain due (>{_RETRAIN_DUE_DAYS}d).")
        severity = "warn"
    else:
        details.append(f"{label}: trained {days_since} days ago.")

    today = _today_utc()
    last_date = _last_history_date(history_path)
    if last_date is None:
        details.append(f"{label}: history file missing or empty — score today to start logging.")
        if severity == "ok":
            severity = "warn"
    elif last_date != today:
        details.append(f"{label}: today's reading not logged yet (last entry: {last_date}). Score Today in QIR.")
        if severity == "ok":
            severity = "warn"
    else:
        details.append(f"{label}: today's reading logged ({last_date}).")

    icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}[severity]
    color = {"ok": "#22c55e", "warn": "#f59e0b", "error": "#ef4444"}[severity]
    msg_short = {
        "ok": "Healthy",
        "warn": "Attention",
        "error": "Not trained",
    }[severity]
    return {
        "label": label,
        "severity": severity,
        "icon": icon,
        "color": color,
        "message": msg_short,
        "days_since": days_since,
        "last_logged": last_date,
        "details": details,
    }


def _render_status_banner(main_status: dict, shadow_status: dict) -> None:
    """One-line check-engine banner with hover tooltips per brain."""
    def _pill(s: dict) -> str:
        tooltip = " | ".join(s["details"])
        return (
            f'<span title="{tooltip}" '
            f'style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#0f172a;border:1px solid {s["color"]};'
            f'border-radius:14px;padding:4px 12px;margin-right:8px;'
            f'font-size:11px;color:{COLORS["text"]};">'
            f'<span style="font-size:14px;">{s["icon"]}</span>'
            f'<span style="color:{s["color"]};font-weight:700;">{s["label"]}</span>'
            f'<span style="color:#94a3b8;">{s["message"]}</span>'
            f"</span>"
        )

    overall = "ok"
    for s in (main_status, shadow_status):
        if s["severity"] == "error":
            overall = "error"
            break
        elif s["severity"] == "warn" and overall == "ok":
            overall = "warn"
    overall_color = {"ok": "#22c55e", "warn": "#f59e0b", "error": "#ef4444"}[overall]
    overall_label = {
        "ok": "ALL SYSTEMS NOMINAL",
        "warn": "CHECK ENGINE",
        "error": "ACTION REQUIRED",
    }[overall]

    st.markdown(
        f'<div style="background:#0a0d12;border:1px solid {overall_color};'
        f'border-left:4px solid {overall_color};border-radius:5px;'
        f'padding:8px 14px;margin-bottom:14px;'
        f'display:flex;align-items:center;gap:14px;">'
        f'<span style="font-size:9px;color:{overall_color};font-weight:800;'
        f'letter-spacing:0.12em;">{overall_label}</span>'
        f'<div>{_pill(main_status)}{_pill(shadow_status)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Trigger / lead-time helpers
# ────────────────────────────────────────────────────────────────────────────

def _trigger_clusters(brain_df: pd.DataFrame, threshold: float | None = None,
                      min_gap_days: int = 30) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """Group consecutive trigger-fire days (z < threshold) into clusters separated by
    at least `min_gap_days`. Returns list of (start, end, min_z) tuples."""
    if brain_df is None or brain_df.empty:
        return []
    if threshold is None:
        threshold = _trigger_anchor()
    z = brain_df["ll_zscore"]
    fire = z < threshold
    if not fire.any():
        return []
    fire_dates = z.index[fire.values]
    if len(fire_dates) == 0:
        return []
    clusters: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    cluster_start = fire_dates[0]
    prev = fire_dates[0]
    for d in fire_dates[1:]:
        if (d - prev).days > min_gap_days:
            window = z.loc[cluster_start:prev]
            clusters.append((cluster_start, prev, float(window.min())))
            cluster_start = d
        prev = d
    window = z.loc[cluster_start:prev]
    clusters.append((cluster_start, prev, float(window.min())))
    return clusters


def _build_lead_time_table(brain_df: pd.DataFrame, spx: pd.Series,
                           threshold: float | None = None,
                           window_days: int = 180) -> pd.DataFrame:
    if brain_df is None or brain_df.empty:
        return pd.DataFrame()
    if threshold is None:
        threshold = _trigger_anchor()

    rows = []
    for date_str, label in _CRISIS_MARKERS:
        crisis_date = pd.Timestamp(date_str)
        if crisis_date < brain_df.index[0] or crisis_date > brain_df.index[-1]:
            continue

        win_start = crisis_date - pd.Timedelta(days=window_days)
        win_end = crisis_date + pd.Timedelta(days=window_days)
        window_df = brain_df.loc[win_start:win_end]
        if window_df.empty:
            continue

        fires = window_df[window_df["ll_zscore"] < threshold]
        if fires.empty:
            rows.append({
                "Crisis": label,
                "Date": crisis_date.strftime("%Y-%m-%d"),
                "First trigger": "—",
                "State at trigger": "MISSED",
                "Min z": f"{window_df['ll_zscore'].min():.3f}",
                "Lead time (days)": "—",
            })
            continue

        first_trigger = fires.index[0]
        state_at_trigger = str(fires.iloc[0]["state_label"])
        min_z = float(window_df["ll_zscore"].min())

        spx_window = spx.loc[max(win_start, spx.index[0]):min(win_end, spx.index[-1])]
        spx_bottom_date = spx_window.idxmin() if not spx_window.empty else crisis_date

        lead_days = (spx_bottom_date - first_trigger).days
        sign = "before" if lead_days > 0 else ("after" if lead_days < 0 else "same day")
        lead_str = f"{abs(lead_days)} {sign}" if lead_days != 0 else "0"

        rows.append({
            "Crisis": label,
            "Date": crisis_date.strftime("%Y-%m-%d"),
            "First trigger": first_trigger.strftime("%Y-%m-%d"),
            "State at trigger": state_at_trigger,
            "Min z": f"{min_z:.3f}",
            "Lead time (days)": lead_str,
        })

    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────────
# Regime ribbon (heatmap strip)
# ────────────────────────────────────────────────────────────────────────────

def _ribbon_trace(brain_df: pd.DataFrame, state_labels: list[str]) -> go.Heatmap:
    """Build a 1-row heatmap where each cell's color matches its state label."""
    n_states = len(state_labels)
    color_by_idx = {i: get_state_color(state_labels[i]) for i in range(n_states)}

    z_row = brain_df["state_idx"].astype(int).values.reshape(1, -1)

    if n_states == 1:
        colorscale = [[0.0, color_by_idx[0]], [1.0, color_by_idx[0]]]
    else:
        colorscale = [[i / (n_states - 1), color_by_idx[i]] for i in range(n_states)]

    text_row = np.array([state_labels[int(i)] for i in z_row[0]]).reshape(1, -1)
    return go.Heatmap(
        x=brain_df.index, y=["Regime"],
        z=z_row, zmin=0, zmax=max(n_states - 1, 1),
        colorscale=colorscale,
        showscale=False,
        hovertemplate="%{x|%Y-%m-%d}<br>Regime: %{text}<extra></extra>",
        text=text_row,
    )


def _legend_html(state_labels: list[str], title: str) -> str:
    cells = []
    for lbl in state_labels:
        c = get_state_color(lbl)
        cells.append(
            f'<span style="display:inline-flex;align-items:center;gap:6px;'
            f'margin-right:14px;font-size:11px;color:{COLORS["text"]};">'
            f'<span style="width:12px;height:12px;background:{c};'
            f'border:1px solid {COLORS["border"]};border-radius:2px;"></span>'
            f"{lbl}</span>"
        )
    return (
        f'<div style="margin:6px 0 4px 0;">'
        f'<span style="font-size:10px;color:{COLORS["bloomberg_orange"]};'
        f'text-transform:uppercase;letter-spacing:0.08em;margin-right:10px;">'
        f"{title}</span>{''.join(cells)}</div>"
    )


# ────────────────────────────────────────────────────────────────────────────
# Brain chart builder
# ────────────────────────────────────────────────────────────────────────────

def _build_brain_chart(
    title: str,
    subtitle: str,
    spx: pd.Series,
    brain_df: pd.DataFrame,
    state_labels: list[str],
) -> go.Figure:
    """3-row stacked figure: SPX log | Regime ribbon | LL z-score."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.55, 0.08, 0.37],
        subplot_titles=("S&P 500 (Log)", "Regime", "LL z-score"),
    )

    spx_clip = spx.loc[spx.index >= brain_df.index[0]]

    # Crisis vertical dashed lines spanning all rows
    chart_start = brain_df.index[0]
    chart_end = brain_df.index[-1]
    for date_str, label in _CRISIS_MARKERS:
        d = pd.Timestamp(date_str)
        if d < chart_start or d > chart_end:
            continue
        fig.add_vline(
            x=d, line=dict(color=COLORS["text_dim"], dash="dot", width=1),
            opacity=0.45,
        )
        fig.add_annotation(
            x=d, y=1.0, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=9, color=COLORS["text"]),
            xanchor="center", yanchor="bottom", yshift=2,
        )

    # ── Row 1: SPX log ────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=spx_clip.index, y=spx_clip.values,
            mode="lines", name="SPX Price (Log)",
            line=dict(color=_SPX_LINE_COLOR, width=1.3),
            hovertemplate="%{x|%Y-%m-%d}<br>SPX $%{y:,.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(type="log", title_text="SPX (Log)", row=1, col=1)

    # ── Row 2: Regime ribbon (heatmap) ────────────────────────────────────
    fig.add_trace(_ribbon_trace(brain_df, state_labels), row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=1, fixedrange=True)

    # ── Row 3: LL z-score ─────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=brain_df.index, y=brain_df["ll_zscore"],
            mode="lines", name="LL z-score",
            line=dict(color=_LL_LINE_COLOR, width=1.1),
            hovertemplate="%{x|%Y-%m-%d}<br>z=%{y:.3f}<extra></extra>",
        ),
        row=3, col=1,
    )

    fig.add_hline(
        y=_trigger_anchor(),
        line=dict(color=COLORS["red"], dash="dash", width=1),
        row=3, col=1,
    )

    clusters = _trigger_clusters(brain_df)
    if clusters:
        marker_x, marker_y = [], []
        for start, end, _min_z in clusters:
            window = brain_df.loc[start:end]
            d_min = window["ll_zscore"].idxmin()
            marker_x.append(d_min)
            marker_y.append(float(window["ll_zscore"].min()))
        fig.add_trace(
            go.Scatter(
                x=marker_x, y=marker_y, mode="markers",
                marker=dict(size=8, color=_TRIGGER_FIRE_COLOR,
                            line=dict(color="#FFFFFF", width=0.5)),
                name="Trigger Fire",
                hovertemplate="%{x|%Y-%m-%d}<br>min z=%{y:.3f}<extra></extra>",
            ),
            row=3, col=1,
        )

    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines",
        line=dict(color=COLORS["red"], dash="dash", width=1),
        name=f"Trigger Anchor ({_trigger_anchor():.4f})",
    ), row=3, col=1)

    fig.update_yaxes(title_text="LL z-score", row=3, col=1)
    fig.update_xaxes(title_text="Year", row=3, col=1)

    apply_dark_layout(
        fig,
        height=640,
        showlegend=True,
        margin=dict(l=70, r=40, t=90, b=50),
        hovermode="x unified",
        title=dict(
            text=(f"<b>{title}</b><br>"
                  f"<span style='font-size:11px;color:#888888;'>{subtitle}</span>"),
            x=0.5, xanchor="center",
            font=dict(color=COLORS["bloomberg_orange"], size=14,
                      family="JetBrains Mono, Consolas, monospace"),
        ),
        legend=dict(
            x=0.01, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(14,17,23,0.75)",
            bordercolor=COLORS["border"], borderwidth=1,
            font=dict(size=10, color=COLORS["text"]),
        ),
    )
    for ann in fig.layout.annotations:
        if ann.text in ("S&P 500 (Log)", "Regime", "LL z-score"):
            ann.update(font=dict(color=COLORS["bloomberg_orange"], size=11,
                                 family="JetBrains Mono, Consolas, monospace"))
    return fig


def _build_comparison_chart(shadow_df: pd.DataFrame, main_df: pd.DataFrame) -> go.Figure:
    overlap_start = max(shadow_df.index[0], main_df.index[0])
    overlap_end = min(shadow_df.index[-1], main_df.index[-1])
    s = shadow_df.loc[overlap_start:overlap_end]
    m = main_df.loc[overlap_start:overlap_end]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s["ll_zscore"], mode="lines", name="Shadow LL",
        line=dict(color="#4B9FFF", width=1.0),
        hovertemplate="%{x|%Y-%m-%d}<br>Shadow z=%{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=m.index, y=m["ll_zscore"], mode="lines", name="Main LL",
        line=dict(color="#FF8C00", width=1.0),
        hovertemplate="%{x|%Y-%m-%d}<br>Main z=%{y:.3f}<extra></extra>",
    ))
    fig.add_hline(
        y=_trigger_anchor(),
        line=dict(color=COLORS["red"], dash="dash", width=1),
        annotation_text=f"Trigger ({_trigger_anchor():.4f})",
        annotation_position="bottom right",
    )

    for date_str, label in _CRISIS_MARKERS:
        d = pd.Timestamp(date_str)
        if d < overlap_start or d > overlap_end:
            continue
        fig.add_vline(
            x=d, line=dict(color=COLORS["text_dim"], dash="dot", width=1),
            opacity=0.4,
        )
        fig.add_annotation(
            x=d, y=1.0, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(size=9, color=COLORS["text_dim"]),
            xanchor="center", yanchor="bottom", yshift=2,
        )

    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="LL z-score")

    apply_dark_layout(
        fig,
        height=380,
        showlegend=True,
        margin=dict(l=70, r=40, t=70, b=50),
        hovermode="x unified",
        title=dict(
            text=("<b>Shadow vs Main — LL z-score head-to-head</b><br>"
                  f"<span style='font-size:11px;color:#888888;'>"
                  f"Overlap window: {overlap_start.date()} → {overlap_end.date()}</span>"),
            x=0.5, xanchor="center",
            font=dict(color=COLORS["bloomberg_orange"], size=14,
                      family="JetBrains Mono, Consolas, monospace"),
        ),
        legend=dict(
            x=0.01, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(14,17,23,0.75)",
            bordercolor=COLORS["border"], borderwidth=1,
            font=dict(size=10, color=COLORS["text"]),
        ),
    )
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Main render
# ────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        f'<h2 style="color:{COLORS["bloomberg_orange"]};margin:0 0 4px 0;'
        f'font-family:JetBrains Mono,Consolas,monospace;">REGIME CHART — VISUAL BACKTEST</h2>'
        f'<div style="color:{COLORS["text_dim"]};font-size:12px;margin-bottom:10px;">'
        f"Each brain plotted against SPX (log) with a regime-color ribbon and trigger fires. "
        f"Lead-time tables below each chart show how early the brain caught each known bottom."
        f"</div>",
        unsafe_allow_html=True,
    )

    main_brain = load_hmm_brain()
    shadow_brain = load_shadow_brain()

    # ── Check-engine status banner ─────────────────────────────────────────
    main_status = _brain_health(main_brain, _MAIN_HISTORY, "Main")
    shadow_status = _brain_health(shadow_brain, _SHADOW_HISTORY, "Shadow")
    _render_status_banner(main_status, shadow_status)

    if main_brain is None and shadow_brain is None:
        st.warning("No trained brains found. Train the HMM and Shadow HMM first.")
        return

    with st.spinner("Loading SPX history and decoding regime paths..."):
        spx = _load_spx_close()
        main_key = getattr(main_brain, "trained_at", "") if main_brain else ""
        shadow_key = getattr(shadow_brain, "trained_at", "") if shadow_brain else ""
        shadow_df = _load_shadow_path(shadow_key) if shadow_brain else None
        main_df = _load_main_path(main_key) if main_brain else None

    if spx is None or spx.empty:
        st.error("Failed to load SPX price history from yfinance.")
        return

    # ── Shadow Brain ────────────────────────────────────────────────────────
    if shadow_brain is not None and shadow_df is not None and not shadow_df.empty:
        shadow_fig = _build_brain_chart(
            title="Shadow Brain — 60-Year Synthesis",
            subtitle="SPX (log) · regime ribbon · LL z-score · trigger fires",
            spx=spx,
            brain_df=shadow_df,
            state_labels=shadow_brain.state_labels,
        )
        st.plotly_chart(shadow_fig, use_container_width=True)
        st.markdown(_legend_html(shadow_brain.state_labels, "Shadow regimes"),
                    unsafe_allow_html=True)
        lead_table = _build_lead_time_table(shadow_df, spx)
        if not lead_table.empty:
            st.markdown(
                f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin:8px 0 4px 0;">'
                f"SHADOW BRAIN — LEAD-TIME BACKTEST</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(lead_table, use_container_width=True, hide_index=True)
    else:
        st.info("Shadow brain not available — train it first.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Brain ──────────────────────────────────────────────────────────
    if main_brain is not None and main_df is not None and not main_df.empty:
        main_subtitle = (
            f"SPX (log) · regime ribbon · LL z-score · trigger fires "
            f"&middot; data window: {main_df.index[0].date()} → {main_df.index[-1].date()}"
        )
        main_fig = _build_brain_chart(
            title=f"Main HMM Brain — {main_brain.lookback_years}yr Lookback",
            subtitle=main_subtitle,
            spx=spx,
            brain_df=main_df,
            state_labels=main_brain.state_labels,
        )
        st.plotly_chart(main_fig, use_container_width=True)
        st.markdown(_legend_html(main_brain.state_labels, "Main regimes"),
                    unsafe_allow_html=True)
        lead_table = _build_lead_time_table(main_df, spx)
        if not lead_table.empty:
            st.markdown(
                f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;'
                f'text-transform:uppercase;letter-spacing:0.08em;margin:8px 0 4px 0;">'
                f"MAIN BRAIN — LEAD-TIME BACKTEST</div>",
                unsafe_allow_html=True,
            )
            st.dataframe(lead_table, use_container_width=True, hide_index=True)
    else:
        st.info("Main HMM brain not available — train it first.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Comparison ──────────────────────────────────────────────────────────
    if (shadow_df is not None and not shadow_df.empty
            and main_df is not None and not main_df.empty):
        comp_fig = _build_comparison_chart(shadow_df, main_df)
        st.plotly_chart(comp_fig, use_container_width=True)

    # ── Footer metadata ─────────────────────────────────────────────────────
    notes = []
    if main_brain is not None and main_df is not None:
        notes.append(
            f"Main HMM trained {main_brain.training_start} → {main_brain.training_end} "
            f"(lookback {main_brain.lookback_years}y, BIC {main_brain.bic:.0f})"
        )
    if shadow_brain is not None and shadow_df is not None:
        notes.append(
            f"Shadow HMM trained {shadow_brain.training_start} → {shadow_brain.training_end} "
            f"(BIC {shadow_brain.bic:.0f}, n_obs {shadow_brain.n_obs:,})"
        )
    if notes:
        st.markdown(
            f'<div style="color:{COLORS["text_dim"]};font-size:11px;margin-top:8px;">'
            + "<br>".join(notes) + "</div>",
            unsafe_allow_html=True,
        )
