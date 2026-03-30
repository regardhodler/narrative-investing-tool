"""Backtesting — test historical signal strategies and view results."""

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


def _load_regime_history() -> list[dict]:
    """Load regime history from local JSON — no circular imports, mirrors backtest_engine pattern."""
    import json as _json
    _hf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json")
    if not os.path.exists(_hf):
        return []
    try:
        with open(_hf) as f:
            return _json.load(f)
    except Exception:
        return []


@st.cache_data(ttl=3600, show_spinner=False)
def _run_regime_conditional_backtest(ticker: str, target_quadrant: str, _history_key: str) -> dict:
    """Compute per-period returns for ticker during target_quadrant regime periods.

    _history_key is a cache-busting string (e.g. last date in history) to avoid stale results.
    """
    from services.backtest_engine import _compute_metrics
    history = _load_regime_history()
    if not history:
        return {"error": "No regime history found. Run QIR at least once to build history."}

    history = sorted(history, key=lambda x: x.get("date", ""))

    # Find contiguous windows matching target_quadrant
    windows, regime_periods, non_regime_periods = [], [], []
    in_regime = False
    start = None
    for entry in history:
        q = entry.get("quadrant", "")
        d = entry.get("date", "")
        if q == target_quadrant:
            if not in_regime:
                in_regime = True
                start = d
        else:
            if in_regime:
                windows.append((start, d))
                in_regime = False
    if in_regime and start:
        windows.append((start, history[-1]["date"]))

    if not windows:
        return {"error": f"No '{target_quadrant}' regime periods found in history ({len(history)} days)."}

    # Download price data for full span
    overall_start = windows[0][0]
    try:
        raw = yf.download(ticker, start=overall_start, progress=False, auto_adjust=True)
        if raw.empty:
            return {"error": f"No price data returned for {ticker}."}
        prices = raw["Close"].squeeze().dropna()
        # Normalize index to timezone-naive for consistent date comparisons
        if hasattr(prices.index, "tz") and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
    except Exception as e:
        return {"error": f"Price download failed: {e}"}

    def _get_price(date_str: str, after: bool = True) -> float | None:
        idx = pd.to_datetime(date_str)
        candidates = prices.loc[prices.index >= idx] if after else prices.loc[prices.index <= idx]
        return float(candidates.iloc[0]) if len(candidates) else None

    periods = []
    for ws, we in windows:
        ep = _get_price(ws, after=True)
        xp = _get_price(we, after=True)
        if ep and xp and ep > 0:
            ret = (xp / ep - 1) * 100
            days = (pd.to_datetime(we) - pd.to_datetime(ws)).days
            periods.append({"start": ws, "end": we, "return_pct": round(ret, 2), "num_days": days})
            regime_periods.append(ret)

    # Non-regime periods: everything else in the history span
    all_dates = [e["date"] for e in history]
    regime_dates = set()
    for ws, we in windows:
        regime_dates.update(d for d in all_dates if ws <= d <= we)
    non_regime_dates = [d for d in all_dates if d not in regime_dates]
    if non_regime_dates:
        for i in range(0, len(non_regime_dates) - 30, 30):
            ep = _get_price(non_regime_dates[i], after=True)
            xp = _get_price(non_regime_dates[min(i + 30, len(non_regime_dates) - 1)], after=True)
            if ep and xp and ep > 0:
                non_regime_periods.append((xp / ep - 1) * 100)

    # Compute metrics directly from return list (avoid _compute_metrics which expects list[dict])
    if regime_periods:
        wins = sum(1 for r in regime_periods if r > 0)
        win_rate = wins / len(regime_periods) * 100
        avg_return = float(np.mean(regime_periods))
        # Max drawdown on cumulative equity curve
        equity = np.cumprod([1 + r / 100 for r in regime_periods])
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak * 100
        max_dd = float(np.min(drawdowns))
    else:
        win_rate = avg_return = max_dd = 0.0
    metrics = {"win_rate": win_rate, "avg_return": avg_return, "max_drawdown": max_dd}

    return {
        "periods": periods,
        "regime_win_rate":       round(metrics["win_rate"], 1),
        "regime_avg_return":     round(metrics["avg_return"], 2),
        "regime_max_drawdown":   round(metrics["max_drawdown"], 2),
        "non_regime_avg_return": round(float(np.mean(non_regime_periods)), 2) if non_regime_periods else 0.0,
        "total_regime_periods":  len(periods),
        "ticker": ticker,
        "quadrant": target_quadrant,
    }


def _render_regime_conditional() -> None:
    """UI for Regime-Conditional Backtest tab."""
    st.caption("How does a ticker perform historically during specific macro quadrant regimes?")

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col1:
        rc_ticker = st.text_input("Ticker", value="SPY", key="rc_ticker").upper().strip()
    with col2:
        _current_q = (st.session_state.get("_regime_context") or {}).get("quadrant", "")
        _q_options = ["Current (auto)", "Goldilocks", "Reflation", "Stagflation", "Deflation"]
        rc_quadrant_sel = st.selectbox("Quadrant", _q_options, key="rc_quadrant")
        target_q = _current_q if rc_quadrant_sel == "Current (auto)" else rc_quadrant_sel
        if rc_quadrant_sel == "Current (auto)" and _current_q:
            st.caption(f"Auto: {_current_q}")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_rc = st.button("RUN", type="primary", key="rc_run", use_container_width=True)

    if not target_q and rc_quadrant_sel == "Current (auto)":
        st.info("Run QIR first to auto-populate the current quadrant, or select one manually.")
        return

    if run_rc or st.session_state.get("_rc_result"):
        if run_rc:
            history = _load_regime_history()
            _hkey = history[-1]["date"] if history else "none"
            result = _run_regime_conditional_backtest(rc_ticker, target_q, _hkey)
            st.session_state["_rc_result"] = result
            st.session_state["_rc_ticker"] = rc_ticker
            st.session_state["_rc_quadrant"] = target_q

        result = st.session_state.get("_rc_result", {})
        if "error" in result:
            st.warning(result["error"])
            return

        # Metric row
        m1, m2, m3, m4 = st.columns(4)
        _oc = COLORS["bloomberg_orange"]
        m1.metric("Regime Periods", result["total_regime_periods"])
        m2.metric("Win Rate", f"{result['regime_win_rate']}%")
        m3.metric("Avg Return", f"{result['regime_avg_return']:+.2f}%")
        _delta = result["regime_avg_return"] - result["non_regime_avg_return"]
        m4.metric("vs Non-Regime", f"{result['non_regime_avg_return']:+.2f}%",
                  delta=f"{_delta:+.2f}pp")

        # Bar chart
        periods = result.get("periods", [])
        if periods:
            _labels = [f"{p['start'][:7]}→{p['end'][:7]}" for p in periods]
            _rets   = [p["return_pct"] for p in periods]
            _colors = [COLORS["positive"] if r >= 0 else COLORS["negative"] for r in _rets]
            fig = go.Figure(go.Bar(
                x=_labels, y=_rets,
                marker_color=_colors,
                text=[f"{r:+.1f}%" for r in _rets],
                textposition="outside",
                textfont=dict(size=10),
            ))
            apply_dark_layout(fig)
            fig.update_layout(
                height=280, margin=dict(l=0, r=0, t=30, b=0),
                title=dict(text=f"{result['ticker']} returns in {result['quadrant']} periods",
                           font=dict(size=11, color=_oc), x=0, xanchor="left"),
                yaxis=dict(title="Return %", ticksuffix="%"),
                showlegend=False,
            )
            fig.add_hline(y=0, line_color="#334155", line_width=1)
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📋 Period Detail"):
                st.dataframe(
                    pd.DataFrame(periods).rename(columns={
                        "start": "Start", "end": "End",
                        "return_pct": "Return %", "num_days": "Days"
                    }),
                    use_container_width=True, hide_index=True,
                )

        # Auto-query open positions
        with st.expander(f"🗂 How do my open positions perform in {target_q}?"):
            try:
                import json as _json2
                _jf = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trade_journal.json")
                _trades = _json2.load(open(_jf)) if os.path.exists(_jf) else []
                _open_tickers = list({t["ticker"].upper() for t in _trades if t.get("status") == "open"})
                if not _open_tickers:
                    st.info("No open positions in trade journal.")
                else:
                    history2 = _load_regime_history()
                    _hkey2 = history2[-1]["date"] if history2 else "none"
                    _rows = []
                    with st.spinner(f"Querying {len(_open_tickers)} positions..."):
                        for _tk in _open_tickers:
                            try:
                                _r = _run_regime_conditional_backtest(_tk, target_q, _hkey2)
                                if "error" not in _r:
                                    _rows.append({
                                        "Ticker": _tk,
                                        "Regime Win Rate": f"{_r['regime_win_rate']}%",
                                        "Regime Avg Return": f"{_r['regime_avg_return']:+.2f}%",
                                        "vs Non-Regime": f"{_r['regime_avg_return'] - _r['non_regime_avg_return']:+.2f}pp",
                                        "Periods": _r["total_regime_periods"],
                                    })
                            except Exception:
                                pass
                    if _rows:
                        _df = pd.DataFrame(_rows).sort_values("Regime Avg Return", ascending=False)
                        st.dataframe(_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No results — try different quadrant or check ticker symbols.")
            except Exception as _ex:
                st.warning(f"Could not load trade journal: {_ex}")


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">BACKTESTING ENGINE</div>',
        unsafe_allow_html=True,
    )

    tab_bt, tab_wf, tab_rc = st.tabs(["⚙ Strategy Backtest", "📊 Walk-Forward Validation", "🗂 Regime-Conditional"])

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

    with tab_rc:
        _render_regime_conditional()
