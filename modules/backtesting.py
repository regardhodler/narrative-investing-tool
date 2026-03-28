"""Backtesting — test historical signal strategies and view results."""

import json
import os

import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from utils.theme import COLORS, apply_dark_layout
from services.backtest_engine import (
    backtest_sma_crossover,
    backtest_vix_spike,
    backtest_regime_flip,
    backtest_insider_cluster,
    walk_forward_sma,
    walk_forward_vix,
    BacktestResult,
)

_REGIME_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json"
)


def _load_regime_history() -> pd.DataFrame:
    """Load regime_history.json and return as a tidy DataFrame."""
    try:
        with open(_REGIME_HISTORY_PATH) as f:
            records = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()

    rows = []
    for r in records:
        row = {
            "date": pd.to_datetime(r["date"]),
            "score": r.get("score", 0.0),
            "macro_score": r.get("macro_score", 50),
            "regime": r.get("regime", "Neutral"),
            "quadrant": r.get("quadrant", ""),
        }
        for sig, val in (r.get("signals_summary") or {}).items():
            row[sig] = val
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


@st.cache_data(ttl=3600)
def _fetch_spy_returns(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY daily closes for a date range."""
    try:
        raw = yf.download("SPY", start=start, end=end, interval="1d",
                          progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]["SPY"]
        else:
            close = raw["Close"]
        return close.pct_change().dropna().rename("spy_return")
    except Exception:
        return pd.DataFrame()


def _render_regime_history():
    """Regime Signal History — score timeline, signal heatmap, SPY overlay."""
    df = _load_regime_history()

    if df.empty:
        st.info("No regime history yet. Run the Risk Regime module to start building history.")
        return

    n_days = len(df)
    date_range = f"{df['date'].min().strftime('%b %d')} → {df['date'].max().strftime('%b %d, %Y')}"

    st.markdown(
        f'<div style="font-size:11px;color:#64748b;margin-bottom:12px;">'
        f'{n_days} sessions recorded · {date_range}'
        + (" · History building — more sessions = higher accuracy" if n_days < 30 else "")
        + f'</div>',
        unsafe_allow_html=True,
    )

    # ── Regime score timeline ────────────────────────────────────────────────
    fig_score = go.Figure()

    # Background bands
    fig_score.add_hrect(y0=0.2, y1=1.0, fillcolor="rgba(34,197,94,0.07)", line_width=0, annotation_text="Risk-On", annotation_position="right")
    fig_score.add_hrect(y0=-1.0, y1=-0.2, fillcolor="rgba(239,68,68,0.07)", line_width=0, annotation_text="Risk-Off", annotation_position="right")
    fig_score.add_hline(y=0, line_dash="dot", line_color="#334155", line_width=1)

    # Regime score line
    colors_score = [
        "#22c55e" if s >= 0.2 else ("#ef4444" if s <= -0.2 else "#f59e0b")
        for s in df["score"]
    ]
    fig_score.add_trace(go.Scatter(
        x=df["date"], y=df["score"],
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2),
        marker=dict(size=6, color=colors_score, line=dict(color="#1e293b", width=1)),
        name="Regime Score",
        hovertemplate="<b>%{x|%b %d}</b><br>Score: %{y:.3f}<extra></extra>",
    ))

    apply_dark_layout(fig_score, title="Regime Score History", height=280)
    fig_score.update_layout(
        yaxis=dict(title="Score (−1 to +1)", range=[-1.1, 1.1], tickformat=".2f"),
        margin=dict(t=40, b=30, l=60, r=80),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # ── Regime vs SPY overlay ────────────────────────────────────────────────
    _start = (df["date"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    _end   = (df["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    spy_ret = _fetch_spy_returns(_start, _end)

    if not spy_ret.empty:
        # Align on date
        df_spy = df.set_index("date")[["score", "regime"]].copy()
        spy_aligned = spy_ret.copy()
        spy_aligned.index = pd.to_datetime(spy_aligned.index).normalize()

        merged = df_spy.join(spy_aligned, how="inner")
        if not merged.empty:
            # Cumulative SPY return
            cum_spy = (1 + merged["spy_return"]).cumprod() - 1

            fig_spy = go.Figure()
            fig_spy.add_trace(go.Bar(
                x=merged.index,
                y=merged["spy_return"] * 100,
                marker_color=[
                    "#22c55e" if v >= 0 else "#ef4444"
                    for v in merged["spy_return"]
                ],
                name="SPY Daily %",
                yaxis="y2",
                opacity=0.5,
                hovertemplate="<b>%{x|%b %d}</b><br>SPY: %{y:+.2f}%<extra></extra>",
            ))
            fig_spy.add_trace(go.Scatter(
                x=merged.index,
                y=merged["score"],
                mode="lines+markers",
                line=dict(color="#60a5fa", width=2),
                marker=dict(size=5),
                name="Regime Score",
                hovertemplate="<b>%{x|%b %d}</b><br>Score: %{y:.3f}<extra></extra>",
            ))

            apply_dark_layout(fig_spy, title="Regime Score vs SPY Daily Returns", height=280)
            fig_spy.update_layout(
                yaxis=dict(title="Regime Score", range=[-1.1, 1.1], tickformat=".2f"),
                yaxis2=dict(
                    title="SPY Return (%)",
                    overlaying="y", side="right",
                    tickformat=".1f", showgrid=False,
                ),
                margin=dict(t=40, b=30, l=60, r=80),
                legend=dict(orientation="h", y=1.08, x=0),
            )
            st.plotly_chart(fig_spy, use_container_width=True)

            # Direction accuracy
            if len(merged) >= 3:
                correct = ((merged["score"] > 0) & (merged["spy_return"] > 0)) | \
                          ((merged["score"] < 0) & (merged["spy_return"] < 0))
                accuracy = correct.mean() * 100
                acc_color = "#22c55e" if accuracy >= 55 else ("#f59e0b" if accuracy >= 45 else "#ef4444")
                st.markdown(
                    f'<div style="font-size:12px;color:#94a3b8;margin-bottom:12px;">'
                    f'Regime → SPY direction accuracy: '
                    f'<b style="color:{acc_color};">{accuracy:.0f}%</b>'
                    f' ({correct.sum()}/{len(merged)} sessions) '
                    f'<span style="color:#475569;font-size:11px;">— needs 30+ sessions for statistical significance</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Signal heatmap ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin:16px 0 6px 0;">SIGNAL SCORE HEATMAP</div>',
        unsafe_allow_html=True,
    )

    signal_cols = [c for c in df.columns if c not in ("date", "score", "macro_score", "regime", "quadrant")]
    if signal_cols:
        heat_df = df[["date"] + signal_cols].set_index("date")[signal_cols].T
        # Fill NaN with 0
        heat_df = heat_df.fillna(0)

        date_labels = [d.strftime("%b %d") for d in heat_df.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=heat_df.values.tolist(),
            x=date_labels,
            y=heat_df.index.tolist(),
            colorscale=[
                [0.0, "#b91c1c"],   # -1 = deep red (risk-off)
                [0.5, "#1e293b"],   # 0  = dark neutral
                [1.0, "#15803d"],   # +1 = deep green (risk-on)
            ],
            zmid=0, zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text="Score", font=dict(color="#aaa", size=10)),
                thickness=12, len=0.8,
                tickfont=dict(color="#aaa", size=10),
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.3f}<extra></extra>",
        ))
        apply_dark_layout(fig_heat, title="", height=max(360, 22 * len(signal_cols)))
        fig_heat.update_layout(
            margin=dict(t=10, b=40, l=220, r=60),
            xaxis=dict(tickfont=dict(color="#94a3b8", size=10)),
            yaxis=dict(tickfont=dict(color="#94a3b8", size=10), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Quadrant transition log ───────────────────────────────────────────────
    if "quadrant" in df.columns and len(df) >= 2:
        transitions = []
        for i in range(1, len(df)):
            prev_q = df.iloc[i - 1]["quadrant"]
            curr_q = df.iloc[i]["quadrant"]
            if prev_q != curr_q:
                transitions.append({
                    "Date": df.iloc[i]["date"].strftime("%b %d, %Y"),
                    "From": prev_q,
                    "To": curr_q,
                    "Score": f"{df.iloc[i]['score']:+.3f}",
                })
        if transitions:
            st.markdown(
                f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin:16px 0 6px 0;">QUADRANT TRANSITIONS</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(pd.DataFrame(transitions), use_container_width=True, hide_index=True)


def _render_results(result: BacktestResult):
    """Render backtest results: metrics, equity curve, trade table, return distribution."""
    if result.num_trades == 0:
        st.warning("No trades generated for this signal. Try different parameters or a longer lookback.")
        return

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trades", result.num_trades)
    m2.metric("Win Rate", f"{result.win_rate:.1f}%")
    m3.metric("Avg Return", f"{result.avg_return:+.2f}%")
    m4.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")

    col_chart, col_hist = st.columns(2)

    # Equity curve
    with col_chart:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=result.equity_curve,
            mode="lines",
            line=dict(color=COLORS["accent"], width=2),
            name="Equity",
        ))
        fig.add_hline(y=10000, line_dash="dash", line_color=COLORS["text_dim"])
        apply_dark_layout(fig, title=f"Equity Curve — {result.signal_name} on {result.ticker}",
                          yaxis_title="Portfolio Value ($)", height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Return distribution histogram
    with col_hist:
        returns = [t["return_pct"] for t in result.trades]
        colors = [COLORS["positive"] if r > 0 else COLORS["negative"] for r in returns]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=returns,
            marker_color=COLORS["accent"],
            nbinsx=20,
            name="Returns",
        ))
        fig2.add_vline(x=0, line_dash="dash", line_color=COLORS["text_dim"])
        apply_dark_layout(fig2, title="Return Distribution", xaxis_title="Return (%)",
                          yaxis_title="Count", height=350)
        st.plotly_chart(fig2, use_container_width=True)

    # Trade table
    st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'margin:12px 0 6px 0;">TRADE LOG</div>', unsafe_allow_html=True)
    df = pd.DataFrame(result.trades)
    df.columns = ["Entry Date", "Exit Date", "Entry Price", "Exit Price", "Return %"]
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_walk_forward():
    """Walk-forward validation panel."""
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">WALK-FORWARD VALIDATION</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Splits history into N sliding windows (train → test). "
        "Measures out-of-sample (OOS) performance to test whether the signal generalizes "
        "or is overfit to the sample. Industry standard: walk-forward is the minimum "
        "bar before trusting any backtest."
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2 = st.columns([2, 1])
    with c1:
        _wf_signal = st.radio("Strategy", ["SMA Crossover", "VIX Spike"],
                              horizontal=True, key="wf_signal")
    with c2:
        _wf_lb = st.slider("Total Lookback (years)", 3, 10, 5, key="wf_lb")

    if _wf_signal == "SMA Crossover":
        p1, p2, p3, p4 = st.columns(4)
        with p1: _wf_ticker  = st.text_input("Ticker", value="SPY", key="wf_ticker").upper().strip()
        with p2: _wf_short   = st.number_input("Short SMA", value=50, min_value=5, max_value=100, key="wf_short")
        with p3: _wf_long    = st.number_input("Long SMA", value=200, min_value=50, max_value=400, key="wf_long")
        with p4: _wf_hold    = st.number_input("Hold Days", value=20, min_value=1, max_value=252, key="wf_hold_sma")
    else:
        p1, p2 = st.columns(2)
        with p1: _wf_thresh  = st.number_input("VIX Threshold", value=25.0, min_value=15.0, max_value=50.0, step=1.0, key="wf_vix_thresh")
        with p2: _wf_hold    = st.number_input("Hold Days", value=20, min_value=1, max_value=252, key="wf_hold_vix")

    w1, w2 = st.columns(2)
    with w1: _train_m = st.slider("Train Window (months)", 6, 24, 12, key="wf_train")
    with w2: _test_m  = st.slider("Test Window (months)",  1, 6,  3,  key="wf_test")

    if st.button("RUN WALK-FORWARD", type="primary", key="wf_run"):
        with st.spinner("Running walk-forward validation..."):
            if _wf_signal == "SMA Crossover":
                _wf_res = walk_forward_sma(_wf_ticker, _wf_short, _wf_long, _wf_hold,
                                           _train_m, _test_m, _wf_lb)
            else:
                _wf_res = walk_forward_vix(_wf_thresh, _wf_hold, _train_m, _test_m, _wf_lb)
        st.session_state["wf_result"] = _wf_res

    _wf = st.session_state.get("wf_result")
    if not _wf or not _wf.get("windows"):
        if _wf and not _wf.get("windows"):
            st.warning("Not enough data or no signals fired in the selected period. "
                       "Try a longer lookback or lower VIX threshold.")
        else:
            st.info("Configure parameters above and click **RUN WALK-FORWARD**.")
        return

    _windows  = _wf["windows"]
    _conf     = _wf.get("confidence", "INSUFFICIENT DATA")
    _oos_wr   = _wf.get("oos_win_rate", 0)
    _oos_ret  = _wf.get("oos_avg_return", 0)
    _oos_n    = _wf.get("oos_total_trades", 0)
    _is_wr    = _wf.get("in_sample_win_rate", 0)
    _is_ret   = _wf.get("in_sample_avg_return", 0)

    # ── Confidence verdict ─────────────────────────────────────────────────────
    _conf_color = {
        "HIGH": "#22c55e", "MODERATE": "#f59e0b",
        "LOW": "#ef4444", "INSUFFICIENT DATA": "#64748b",
    }.get(_conf, "#64748b")
    _conf_bg = {
        "HIGH": "#0a2218", "MODERATE": "#1a1200",
        "LOW": "#1f0a0a", "INSUFFICIENT DATA": "#0f172a",
    }.get(_conf, "#0f172a")
    _conf_msg = {
        "HIGH":   "Signal generalizes well out-of-sample. OOS win rate ≥55% and positive avg return.",
        "MODERATE": "Signal shows some OOS edge but not consistently. Use with caution.",
        "LOW":    "Signal appears overfit to in-sample data. OOS performance is poor.",
        "INSUFFICIENT DATA": f"Only {_oos_n} OOS trades — need at least 5 to assess. Try longer lookback.",
    }.get(_conf, "")

    st.markdown(
        f'<div style="border:1px solid {_conf_color}44;border-radius:8px;'
        f'padding:12px 18px;background:{_conf_bg};margin:10px 0;">'
        f'<span style="font-size:11px;color:{_conf_color};font-weight:700;letter-spacing:0.08em;">'
        f'OOS CONFIDENCE: {_conf}</span><br>'
        f'<span style="font-size:13px;color:#f1f5f9;">{_conf_msg}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── OOS vs IS comparison metrics ──────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Windows", len(_windows))
    m2.metric("OOS Trades", _oos_n)
    m3.metric("OOS Win Rate", f"{_oos_wr:.1f}%",
              delta=f"{_oos_wr - _is_wr:+.1f}% vs IS")
    m4.metric("OOS Avg Return", f"{_oos_ret:+.2f}%",
              delta=f"{_oos_ret - _is_ret:+.2f}% vs IS")
    _final_eq = _wf["oos_equity"][-1] if _wf.get("oos_equity") else 10000
    m5.metric("OOS $10k→", f"${_final_eq:,.0f}")

    st.caption(
        "OOS = out-of-sample (test period). IS = in-sample (train period). "
        "Large IS→OOS gap = overfit. Small gap = signal generalizes."
    )

    # ── OOS equity curve ──────────────────────────────────────────────────────
    _eq = _wf.get("oos_equity", [])
    if len(_eq) > 1:
        _eq_color = COLORS["positive"] if _eq[-1] >= _eq[0] else COLORS["negative"]
        fig_eq = go.Figure()
        fig_eq.add_hline(y=10000, line_dash="dash", line_color=COLORS["text_dim"])
        fig_eq.add_trace(go.Scatter(
            y=_eq,
            mode="lines",
            line=dict(color=_eq_color, width=2),
            fill="tozeroy",
            fillcolor=f"{'rgba(34,197,94,0.08)' if _eq[-1] >= _eq[0] else 'rgba(239,68,68,0.08)'}",
            name="OOS Equity",
        ))
        apply_dark_layout(fig_eq, title="Concatenated Out-of-Sample Equity Curve",
                          yaxis_title="Portfolio Value ($)", height=280)
        fig_eq.update_layout(margin=dict(t=40, b=20, l=60, r=20))
        st.plotly_chart(fig_eq, use_container_width=True)

    # ── Per-window bar chart ──────────────────────────────────────────────────
    _wf_with_trades = [w for w in _windows if w["test_trades"] > 0]
    if _wf_with_trades:
        _wlabels = [f"W{w['n']}\n{w['test_start'][:7]}" for w in _wf_with_trades]
        _oos_wrs  = [w["test_win_rate"]  for w in _wf_with_trades]
        _oos_rets = [w["test_avg_return"] for w in _wf_with_trades]

        fig_bars = go.Figure()
        fig_bars.add_trace(go.Bar(
            name="OOS Win Rate %",
            x=_wlabels,
            y=_oos_wrs,
            marker_color=[COLORS["positive"] if v >= 50 else COLORS["negative"] for v in _oos_wrs],
            text=[f"{v:.0f}%" for v in _oos_wrs],
            textposition="outside",
        ))
        fig_bars.add_hline(y=50, line_dash="dash", line_color=COLORS["text_dim"],
                           annotation_text="50% baseline")
        apply_dark_layout(fig_bars, title="OOS Win Rate by Window", height=280)
        fig_bars.update_layout(
            yaxis=dict(range=[0, 110], ticksuffix="%"),
            margin=dict(t=40, b=30, l=50, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # ── Per-window table ──────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'margin:12px 0 6px 0;">WINDOW-BY-WINDOW RESULTS</div>',
        unsafe_allow_html=True,
    )
    _tbl_rows = []
    for w in _windows:
        _tbl_rows.append({
            "Window":         f"W{w['n']}",
            "Test Period":    f"{w['test_start']} → {w['test_end']}",
            "IS Trades":      w["train_trades"],
            "IS Win %":       f"{w['train_win_rate']:.0f}%",
            "IS Avg Ret":     f"{w['train_avg_return']:+.2f}%",
            "OOS Trades":     w["test_trades"],
            "OOS Win %":      f"{w['test_win_rate']:.0f}%",
            "OOS Avg Ret":    f"{w['test_avg_return']:+.2f}%",
        })
    if _tbl_rows:
        st.dataframe(pd.DataFrame(_tbl_rows), use_container_width=True, hide_index=True)

    with st.expander("What is Walk-Forward Validation?", expanded=False):
        st.markdown("""
**Why single backtests lie:**
A single backtest uses ALL the data to measure performance — but the parameters were often chosen
*because* they looked good on that same data. This is overfitting.

**How walk-forward works:**
1. Split history into N windows of (train + test) months each
2. On each window: the strategy runs only on the **train period** to validate signal firing
3. Performance is measured on the **test period** — data the strategy never "saw"
4. Repeat across all windows, concatenate out-of-sample results

**Reading the output:**
- **HIGH confidence**: OOS win rate ≥ 55% and positive OOS avg return → signal likely generalizes
- **MODERATE**: Borderline — reduce position size, watch live
- **LOW**: Signal is overfit — don't trade it live
- **Small IS→OOS gap**: The signal is robust. Large gap = overfit.
""")


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">BACKTESTING ENGINE</div>',
        unsafe_allow_html=True,
    )

    tab_bt, tab_wf, tab_history = st.tabs(["⚙ Strategy Backtest", "📊 Walk-Forward Validation", "📈 Regime Signal History"])

    with tab_history:
        _render_regime_history()

    with tab_wf:
        _render_walk_forward()

    with tab_bt:
        signal = st.selectbox(
            "Signal Strategy",
            ["SMA Crossover", "VIX Spike", "Regime Flip", "Insider Cluster"],
            key="bt_signal",
        )

        # Dynamic parameters
        st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin:8px 0 4px 0;">'
                    f'PARAMETERS</div>', unsafe_allow_html=True)

        if signal == "SMA Crossover":
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                ticker = st.text_input("Ticker", value="SPY", key="bt_sma_ticker").upper().strip()
            with col2:
                short_w = st.number_input("Short SMA", value=50, min_value=5, max_value=100, key="bt_short")
            with col3:
                long_w = st.number_input("Long SMA", value=200, min_value=50, max_value=400, key="bt_long")
            with col4:
                hold = st.number_input("Hold Days", value=20, min_value=1, max_value=252, key="bt_hold_sma")
            lookback = st.slider("Lookback (years)", 1, 10, 5, key="bt_lb_sma")

        elif signal == "VIX Spike":
            col1, col2 = st.columns(2)
            with col1:
                threshold = st.number_input("VIX Threshold", value=25.0, min_value=15.0, max_value=50.0,
                                            step=1.0, key="bt_vix_thresh")
            with col2:
                hold = st.number_input("Hold Days", value=20, min_value=1, max_value=252, key="bt_hold_vix")
            lookback = st.slider("Lookback (years)", 1, 10, 5, key="bt_lb_vix")

        elif signal == "Regime Flip":
            hold = st.number_input("Hold Days after flip", value=20, min_value=1, max_value=252, key="bt_hold_reg")

        elif signal == "Insider Cluster":
            col1, col2, col3 = st.columns(3)
            with col1:
                ticker = st.text_input("Ticker", value="AAPL", key="bt_ins_ticker").upper().strip()
            with col2:
                min_buys = st.number_input("Min Buys", value=3, min_value=2, max_value=10, key="bt_min_buys")
            with col3:
                hold = st.number_input("Hold Days", value=20, min_value=1, max_value=252, key="bt_hold_ins")
            cluster_days = st.slider("Cluster Window (days)", 7, 90, 30, key="bt_cluster_days")

        # ── Interpretive Tips ──
        with st.expander("ℹ️ How These Strategies Work"):
            st.markdown("""
**SMA Crossover (Golden Cross)**
- **Short SMA** (default 50): The faster-moving average. When this crosses ABOVE the Long SMA, it's a **golden cross** (buy signal).
- **Long SMA** (default 200): The slower-moving average. Acts as the trend baseline.
- **How it works**: A golden cross means short-term momentum is overtaking the long-term trend — historically a bullish signal. The backtest buys on each golden cross and holds for the specified number of days.

**VIX Spike**
- **VIX Threshold** (default 25): When VIX crosses above this level, fear is elevated — the strategy buys SPY as a contrarian bet that fear is overdone.
- **Hold Days**: How long to hold after the spike entry.

**Regime Flip**
- Buys SPY when the Risk Regime indicator flips from **Risk-Off → Risk-On** (sourced from saved regime history).
- **Hold Days**: How long to hold after the flip.

**Insider Cluster**
- Detects clusters of insider **purchases** (not sales) on the same stock within a rolling window.
- **Min Buys**: Minimum number of insider buys required to trigger (default 3).
- **Cluster Window**: The time window in days to look for clustered buys (default 30).
- A cluster of insider buying suggests insiders believe the stock is undervalued.

---

**Reading Your Results**
- **Win Rate** = percentage of trades that were profitable.
- **Avg Return** = mean return per trade — positive means the strategy was profitable on average.
- **Max Drawdown** = worst peak-to-trough decline in the equity curve — lower is better.
- Compare strategies and parameter settings to find which signals historically worked best for a given ticker.
""")

        # Run backtest
        if st.button("RUN BACKTEST", type="primary", key="bt_run"):
            with st.spinner("Running backtest..."):
                if signal == "SMA Crossover":
                    result = backtest_sma_crossover(ticker, short_w, long_w, hold, lookback)
                elif signal == "VIX Spike":
                    result = backtest_vix_spike(threshold, hold, lookback)
                elif signal == "Regime Flip":
                    result = backtest_regime_flip(hold)
                elif signal == "Insider Cluster":
                    result = backtest_insider_cluster(ticker, min_buys, cluster_days, hold)
                else:
                    return

                st.session_state["bt_result"] = result

        result = st.session_state.get("bt_result")
        if result:
            _render_results(result)
