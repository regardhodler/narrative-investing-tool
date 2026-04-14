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
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Trades", result.num_trades)
    m2.metric("Win Rate", f"{result.win_rate:.1f}%")
    m3.metric("Avg Return", f"{result.avg_return:+.2f}%")
    m4.metric("Max Drawdown", f"{result.max_drawdown:.1f}%")
    final_eq = result.equity_curve[-1] if result.equity_curve else 10000
    m5.metric("$10k→", f"${final_eq:,.0f}")

    # Exit reason breakdown
    if result.exit_reason_counts:
        _exit_icons = {"profit_target": "🎯", "trailing_stop": "🛑", "data_end": "📅"}
        _parts = " &nbsp;|&nbsp; ".join(
            f'{_exit_icons.get(k,"")}<span style="color:{COLORS["bloomberg_orange"]}">{k.replace("_"," ").title()}</span>: <b>{v}</b>'
            for k, v in sorted(result.exit_reason_counts.items(), key=lambda x: -x[1])
        )
        st.markdown(
            f'<div style="font-size:11px;color:{COLORS["text_dim"]};margin:4px 0 12px 0;">'
            f'Exit breakdown: {_parts}</div>',
            unsafe_allow_html=True,
        )

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
        p1, p2, p3 = st.columns(3)
        with p1: _wf_ticker  = st.text_input("Ticker", value="SPY", key="wf_ticker").upper().strip()
        with p2: _wf_short   = st.number_input("Short SMA", value=50, min_value=5, max_value=100, key="wf_short")
        with p3: _wf_long    = st.number_input("Long SMA", value=200, min_value=50, max_value=400, key="wf_long")
    else:
        p1, = st.columns(1)
        with p1: _wf_thresh  = st.number_input("VIX Threshold", value=25.0, min_value=15.0, max_value=50.0, step=1.0, key="wf_vix_thresh")

    w1, w2, w3, w4 = st.columns(4)
    with w1: _train_m    = st.slider("Train Window (months)", 6, 24, 12, key="wf_train")
    with w2: _test_m     = st.slider("Test Window (months)",  1, 6,  3,  key="wf_test")
    with w3: _atr_stop   = st.number_input("ATR Stop ×", value=2.0, min_value=0.5, max_value=5.0, step=0.5, key="wf_atr_stop", help="Trailing stop = N × ATR(14)")
    with w4: _atr_tgt    = st.number_input("ATR Target ×", value=3.0, min_value=0.5, max_value=8.0, step=0.5, key="wf_atr_tgt", help="Profit target = N × ATR(14)")

    if st.button("RUN WALK-FORWARD", type="primary", key="wf_run"):
        with st.spinner("Running walk-forward validation..."):
            if _wf_signal == "SMA Crossover":
                _wf_res = walk_forward_sma(_wf_ticker, _wf_short, _wf_long,
                                           _train_m, _test_m, _wf_lb,
                                           atr_stop_mult=_atr_stop, atr_target_mult=_atr_tgt)
            else:
                _wf_res = walk_forward_vix(_wf_thresh, _train_m, _test_m, _wf_lb,
                                           atr_stop_mult=_atr_stop, atr_target_mult=_atr_tgt)
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


def _render_atr_replay():
    """ATR Replay — simulate a single trade on any ticker from any past date."""
    from datetime import datetime, timedelta
    from services.forecast_tracker import backtest_atr
    from utils.theme import apply_dark_layout

    st.markdown(
        f'<div style="color:{COLORS["bloomberg_orange"]};font-size:11px;font-weight:700;'
        f'letter-spacing:0.08em;margin-bottom:4px;">🔁 ATR REPLAY — SINGLE TRADE SIMULATOR</div>'
        f'<div style="color:#94a3b8;font-size:11px;margin-bottom:14px;">'
        f'Simulate the exact ATR trailing stop/target engine on any ticker from any past date. '
        f'Same 2×ATR stop / 3×ATR target used in live Forecast Tracker.</div>',
        unsafe_allow_html=True,
    )

    with st.form("atr_replay_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            rp_ticker = st.text_input("Ticker", placeholder="SPY", value="SPY")
        with c2:
            rp_date = st.date_input("Entry Date", value=datetime.now() - timedelta(days=30))
        with c3:
            rp_dir = st.selectbox("Direction", ["Buy", "Sell"])
        rp_submit = st.form_submit_button("▶ Run Replay", use_container_width=True)

    if rp_submit:
        if not rp_ticker.strip():
            st.error("Enter a ticker.")
            return
        with st.spinner(f"Fetching {rp_ticker.upper()} data from {rp_date}…"):
            result = backtest_atr(rp_ticker.strip().upper(), str(rp_date), rp_dir)

        if result is None:
            st.error("Could not fetch price data. Check ticker and date.")
            return

        outcome    = result.get("outcome", "open")
        ret        = result.get("return_pct")
        exit_r     = result.get("exit_reason", "")
        exit_label = {"profit_target": "🎯 Profit Target", "trailing_stop": "🛑 Trailing Stop",
                      "still_open": "📡 Still Open", "data_end": "📅 End of Data"}.get(exit_r, exit_r)
        out_color  = COLORS["positive"] if outcome == "correct" else (COLORS["negative"] if outcome == "incorrect" else COLORS["blue"])
        out_label  = {"correct": "✅ CORRECT", "incorrect": "❌ WRONG", "open": "📡 STILL OPEN"}.get(outcome, outcome.upper())

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val, color) in zip([c1, c2, c3, c4], [
            ("OUTCOME",   out_label,                                    out_color),
            ("RETURN",    f"{ret:+.2f}%" if ret is not None else "—",   out_color),
            ("EXIT",      exit_label,                                   COLORS["bloomberg_orange"]),
            ("EXIT DATE", result.get("exit_date") or "—",              COLORS["text_dim"]),
        ]):
            with col:
                st.markdown(
                    f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                    f'border-left:3px solid {color};padding:8px 12px;border-radius:4px;margin-bottom:8px;">'
                    f'<div style="font-size:10px;color:{COLORS["bloomberg_orange"]};text-transform:uppercase;'
                    f'letter-spacing:0.08em;">{label}</div>'
                    f'<div style="font-size:16px;color:{color};font-weight:700;margin-top:3px;">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        atr    = result.get("atr_at_log")
        stop_l = result.get("stop_at_log")
        tgt_l  = result.get("target_at_log")
        entry  = result.get("price_at_entry")
        if atr:
            c1, c2, c3, c4 = st.columns(4)
            for col, (label, val) in zip([c1, c2, c3, c4], [
                ("ENTRY PRICE",    f"${entry:.2f}" if entry else "—"),
                ("ATR(14)",        f"${atr:.2f}"),
                ("STOP (2×ATR)",   f"${stop_l:.2f}" if stop_l else "—"),
                ("TARGET (3×ATR)", f"${tgt_l:.2f}" if tgt_l else "—"),
            ]):
                with col:
                    st.metric(label, val)

        dates  = result.get("dates", [])
        closes = result.get("closes", [])
        highs  = result.get("highs", [])
        lows   = result.get("lows", [])

        if dates and closes:
            is_short  = rp_dir == "Sell"
            stop_dist = (atr or 0) * 2.0
            watermark = entry or closes[0]
            stop_path = []
            for h, l in zip(highs, lows):
                if is_short:
                    if l < watermark:
                        watermark = l
                    stop_path.append(watermark + stop_dist)
                else:
                    if h > watermark:
                        watermark = h
                    stop_path.append(watermark - stop_dist)

            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=dates, open=closes, high=highs, low=lows, close=closes,
                increasing_line_color=COLORS["positive"], decreasing_line_color=COLORS["negative"],
                name="Price", showlegend=False,
            ))
            if entry:
                fig.add_hline(y=entry, line_color=COLORS["text_dim"], line_dash="dot", line_width=1,
                              annotation_text=f"Entry ${entry:.2f}", annotation_position="left")
            if tgt_l:
                fig.add_hline(y=tgt_l, line_color=COLORS["positive"], line_dash="dash", line_width=1,
                              annotation_text=f"Target ${tgt_l:.2f}", annotation_position="left")
            if stop_path:
                fig.add_trace(go.Scatter(x=dates, y=stop_path, mode="lines",
                                         line=dict(color=COLORS["negative"], width=1, dash="dash"),
                                         name="Trailing Stop"))
            if result.get("exit_date") and result.get("exit_price"):
                fig.add_trace(go.Scatter(x=[result["exit_date"]], y=[result["exit_price"]],
                                         mode="markers",
                                         marker=dict(size=12, color=out_color, symbol="x"),
                                         name=exit_label))
            apply_dark_layout(fig)
            fig.update_layout(height=340, margin=dict(l=60, r=20, t=20, b=40),
                              xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)


def _render_crash_stress_test():
    """Crash Stress Test — replay REGARD signals through historical crashes."""
    from services.backtest_engine import CRASH_SCENARIOS, run_crash_simulation, run_all_crash_simulations

    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:4px;">CRASH STRESS TEST SIMULATOR</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Reconstructs REGARD's regime score through major historical crashes using the same "
        "FRED + yfinance signals and z-score math as the live engine. Answers: would REGARD "
        "have warned you early enough to avoid the drawdown and buy the dip?"
    )

    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        crash_options = {k: v["name"] for k, v in CRASH_SCENARIOS.items()}
        selected = st.selectbox("Select crash scenario", list(crash_options.keys()),
                                format_func=lambda k: crash_options[k], key="crash_select")
    with c2:
        run_one = st.button("Run Simulation", type="primary", key="crash_run_one", use_container_width=True)
    with c3:
        run_all = st.button("Run All Crashes", key="crash_run_all", use_container_width=True)

    if run_all:
        results = run_all_crash_simulations()
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.1em;margin:16px 0 8px 0;">ALL CRASHES — COMPARISON TABLE</div>',
            unsafe_allow_html=True,
        )
        rows = []
        for r in results:
            if r.get("error"):
                continue
            rows.append({
                "Crash": r["crash_name"],
                "Peak": r["peak_date"],
                "Trough": r["trough_date"],
                "Drawdown": f"{r['max_drawdown']:+.1f}%" if r.get("max_drawdown") else "—",
                "Warning": r.get("warning_date", "—"),
                "Lead (days)": r.get("warning_lead_days", "—"),
                "Avoided": f"{r['avoided_drawdown']:+.1f}%" if r.get("avoided_drawdown") else "—",
                "Dip Buy": r.get("dip_buy_date", "—"),
                "60d Return": f"{r['dip_buy_return_60d']:+.1f}%" if r.get("dip_buy_return_60d") else "—",
            })
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            warnings_fired = sum(1 for r in results if r.get("warning_date"))
            valid_leads = [r["warning_lead_days"] for r in results if r.get("warning_lead_days") and r["warning_lead_days"] > 0]
            valid_avoided = [r["avoided_drawdown"] for r in results if r.get("avoided_drawdown") and r["avoided_drawdown"] < 0]
            dip_returns = [r["dip_buy_return_60d"] for r in results if r.get("dip_buy_return_60d")]
            avg_lead = np.mean(valid_leads) if valid_leads else 0
            avg_avoided = np.mean(valid_avoided) if valid_avoided else 0
            avg_dip = np.mean(dip_returns) if dip_returns else 0

            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:6px;padding:12px;margin:8px 0;">'
                f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">VERDICT</div>'
                f'<div style="font-size:12px;color:{COLORS["text"]};">'
                f'REGARD fired early warnings in <b>{warnings_fired}/{len(results)}</b> crashes '
                f'with avg <b>{avg_lead:.0f} days</b> lead time. '
                f'Avg avoided drawdown: <b>{avg_avoided:.1f}%</b>. '
                f'Avg dip buy 60d return: <b>{avg_dip:+.1f}%</b>.</div>'
                f'<div style="font-size:9px;color:{COLORS["text_dim"]};margin-top:6px;">'
                f'Signals not available historically: GEX, options flow, whale tracking, AI debate, news sentiment. '
                f'Results use FRED macro + VIX + SPY trend only.</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        return

    if not run_one:
        sc = CRASH_SCENARIOS[selected]
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;padding:12px;margin:8px 0;">'
            f'<div style="font-size:12px;color:{COLORS["text"]};">'
            f'<b>{sc["name"]}</b><br>'
            f'<span style="color:{COLORS["text_dim"]};">{sc["context"]}</span><br>'
            f'Peak: {sc["peak"]} → Trough: {sc["trough"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    result = run_crash_simulation(selected)
    if result.get("error"):
        st.error(result["error"])
        return

    _warn_html = ""
    if result.get("warning_date"):
        _wc = COLORS["yellow"]
        _warn_html = (
            f'<div style="margin-top:8px;padding:8px;background:#1a1a0a;border:1px solid {_wc};border-radius:4px;">'
            f'<span style="color:{_wc};font-weight:700;">EARLY WARNING: {result["warning_date"]}</span>'
            f' — <b>{result["warning_lead_days"]} days</b> before the bottom<br>'
            f'<span style="color:{COLORS["text_dim"]};">SPY at warning: ${result["warning_spy"]:,.2f} → '
            f'SPY at trough: ${result["spy_at_trough"]:,.2f}</span><br>'
            f'<span style="color:{COLORS["negative"]};">Avoided drawdown: {result["avoided_drawdown"]:+.1f}%</span>'
            f'</div>'
        )

    _dip_html = ""
    if result.get("dip_buy_date"):
        _dc = COLORS["positive"]
        _r20 = f'{result["dip_buy_return_20d"]:+.1f}%' if result.get("dip_buy_return_20d") else "—"
        _r60 = f'{result["dip_buy_return_60d"]:+.1f}%' if result.get("dip_buy_return_60d") else "—"
        _dip_html = (
            f'<div style="margin-top:6px;padding:8px;background:#0a1a0a;border:1px solid {_dc};border-radius:4px;">'
            f'<span style="color:{_dc};font-weight:700;">DIP BUY SIGNAL: {result["dip_buy_date"]}</span>'
            f' — SPY ${result["dip_buy_spy"]:,.2f}<br>'
            f'<span style="color:{COLORS["text_dim"]};">20d return: </span><span style="color:{_dc};">{_r20}</span>'
            f' &nbsp;|&nbsp; '
            f'<span style="color:{COLORS["text_dim"]};">60d return: </span><span style="color:{_dc};">{_r60}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;padding:14px;margin:8px 0;">'
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;letter-spacing:0.1em;margin-bottom:6px;">'
        f'{result["crash_name"]}</div>'
        f'<div style="font-size:10px;color:{COLORS["text_dim"]};margin-bottom:8px;">{result["context"]}</div>'
        f'<div style="font-size:12px;color:{COLORS["text"]};">'
        f'Peak: <b>{result["peak_date"]}</b> (${result["spy_at_peak"]:,.2f}) → '
        f'Trough: <b>{result["trough_date"]}</b> (${result["spy_at_trough"]:,.2f}) — '
        f'<span style="color:{COLORS["negative"]};font-weight:700;">{result["max_drawdown"]:+.1f}%</span>'
        f'</div>'
        f'{_warn_html}{_dip_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Signal Timeline Chart
    snaps = result.get("snapshots", [])
    if snaps:
        dates_c = [s["date"] for s in snaps]
        scores = [s["regime_score"] for s in snaps]
        spy_prices = [s.get("spy_price") for s in snaps]

        fig = go.Figure()

        for i in range(len(dates_c) - 1):
            color = "rgba(34,197,94,0.08)" if scores[i] > 0 else "rgba(239,68,68,0.08)" if scores[i] < -0.15 else "rgba(100,100,100,0.05)"
            fig.add_vrect(x0=dates_c[i], x1=dates_c[i + 1], fillcolor=color, layer="below", line_width=0)

        fig.add_trace(go.Scatter(
            x=dates_c, y=spy_prices, mode="lines",
            line=dict(color=COLORS["text"], width=1.5),
            name="SPY", yaxis="y2",
        ))

        fig.add_trace(go.Scatter(
            x=dates_c, y=scores, mode="lines",
            line=dict(color=COLORS["bloomberg_orange"], width=2),
            name="Regime Score",
            fill="tozeroy",
            fillcolor="rgba(255,143,0,0.1)",
        ))

        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["text_dim"], line_width=0.5)
        fig.add_hline(y=-0.15, line_dash="dot", line_color=COLORS["negative"], line_width=0.5,
                       annotation_text="Risk-Off", annotation_position="bottom right")

        if result.get("warning_date"):
            _wi = next((i for i, d in enumerate(dates_c) if d == result["warning_date"]), None)
            if _wi is not None:
                fig.add_trace(go.Scatter(
                    x=[result["warning_date"]], y=[scores[_wi]],
                    mode="markers+text", text=["WARNING"],
                    textposition="top center", textfont=dict(size=9, color=COLORS["yellow"]),
                    marker=dict(size=12, color=COLORS["yellow"], symbol="triangle-down"),
                    showlegend=False,
                ))

        if result.get("dip_buy_date"):
            _di = next((i for i, d in enumerate(dates_c) if d == result["dip_buy_date"]), None)
            if _di is not None:
                fig.add_trace(go.Scatter(
                    x=[result["dip_buy_date"]], y=[scores[_di]],
                    mode="markers+text", text=["BUY DIP"],
                    textposition="top center", textfont=dict(size=9, color=COLORS["positive"]),
                    marker=dict(size=12, color=COLORS["positive"], symbol="triangle-up"),
                    showlegend=False,
                ))

        _event_markers = [("PEAK", result["peak_date"], COLORS["bloomberg_orange"])]
        if result.get("warning_date"):
            _event_markers.append(("WARNING", result["warning_date"], COLORS["yellow"]))
        _event_markers.append(("TROUGH", result["trough_date"], COLORS["negative"]))
        if result.get("dip_buy_date"):
            _event_markers.append(("DIP BUY", result["dip_buy_date"], COLORS["positive"]))

        for _em_label, _em_date, _em_color in _event_markers:
            fig.add_shape(type="line", x0=_em_date, x1=_em_date, y0=0, y1=1,
                          yref="paper", line=dict(color=_em_color, width=1, dash="dash"))
            fig.add_annotation(x=_em_date, y=1.02, yref="paper", text=_em_label,
                               showarrow=False, font=dict(size=8, color=_em_color))

        apply_dark_layout(fig, title=f"REGARD Signal Timeline — {result['crash_name']}", height=420)
        fig.update_layout(
            yaxis=dict(title="Regime Score", range=[-1.1, 1.1]),
            yaxis2=dict(title="SPY Price ($)", overlaying="y", side="right", type="log"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Signal Breakdown
    sfw = result.get("signal_first_warnings", {})
    if sfw:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.1em;margin:12px 0 6px 0;">SIGNAL BREAKDOWN — WHICH FIRED FIRST?</div>',
            unsafe_allow_html=True,
        )

        sorted_signals = sorted(sfw.items(), key=lambda x: x[1].get("lead_days", 0), reverse=True)

        _signal_labels = {
            "yield_curve": "Yield Curve (10Y-2Y)", "yield_curve_3m": "Yield Curve (3M-10Y)",
            "credit_hy": "Credit Spreads (HY)", "credit_ig": "Credit Spreads (IG)",
            "fci": "Financial Conditions (NFCI)", "icsa": "Initial Claims",
            "vix": "VIX", "spy_trend": "SPY Trend (SMA/RSI)", "real_yield": "Real Yields (TIPS)",
            "indpro": "Industrial Production", "umcsent": "Consumer Sentiment",
            "permit": "Building Permits", "credit_impulse": "Credit Impulse",
            "fedfunds": "Fed Funds Rate", "totbkcr": "Bank Credit",
        }

        rows_html = []
        for name, info in sorted_signals:
            label = _signal_labels.get(name, name)
            lead = info.get("lead_days", 0)
            if lead > 0:
                lead_str = f'<span style="color:{COLORS["positive"]};">{lead}d before peak</span>'
            elif lead == 0:
                lead_str = f'<span style="color:{COLORS["yellow"]};">at peak</span>'
            else:
                lead_str = f'<span style="color:{COLORS["negative"]};">{abs(lead)}d after peak</span>'
            rows_html.append(
                f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
                f'border-bottom:1px solid {COLORS["border"]};">'
                f'<span style="color:{COLORS["text"]};">{label}</span>'
                f'<span style="font-size:11px;">{info["date"]} — z={info["z_score"]:+.2f} — {lead_str}</span>'
                f'</div>'
            )

        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;padding:10px;font-size:11px;">'
            + "".join(rows_html)
            + '</div>',
            unsafe_allow_html=True,
        )

    # Simulated QIR Snapshots
    from services.backtest_engine import build_qir_snapshot, _build_hmm_historical_inference, _load_all_historical_data as _load_hist

    _hist_data = _load_hist()
    _hmm_data = _build_hmm_historical_inference()

    key_dates = []
    _peak_dt = pd.Timestamp(result["peak_date"])
    key_dates.append(("6M BEFORE", str((_peak_dt - pd.Timedelta(days=180)).date())))
    key_dates.append(("3M BEFORE", str((_peak_dt - pd.Timedelta(days=90)).date())))
    if result.get("warning_date"):
        key_dates.append(("WARNING", result["warning_date"]))
    key_dates.append(("PEAK", result["peak_date"]))
    key_dates.append(("TROUGH", result["trough_date"]))
    if result.get("dip_buy_date"):
        key_dates.append(("DIP BUY", result["dip_buy_date"]))
    key_dates.sort(key=lambda x: x[1])

    if key_dates:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.1em;margin:16px 0 8px 0;">SIMULATED QIR SNAPSHOTS</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "What REGARD's QIR dashboard would have shown at each key moment. "
            "Options, sentiment, and events scores are unavailable in historical mode. "
            "Regime velocity = regime score change over prior 7 calendar days."
        )

        from services.backtest_engine import reconstruct_regime_at_date
        _prev_scores = {}
        for label, date in key_dates:
            _dt = pd.Timestamp(date)
            _prior_dt = _dt - pd.Timedelta(days=7)
            _spy = _hist_data.get("spy")
            if _spy is not None and not _spy.empty:
                _candidates = _spy[_spy.index <= _prior_dt]
                if len(_candidates) > 0:
                    _prior_date_str = str(_candidates.index[-1])[:10]
                    _prior_regime = reconstruct_regime_at_date(_prior_date_str, _hist_data)
                    _prev_scores[date] = _prior_regime["regime_score"]
                else:
                    _prev_scores[date] = None
            else:
                _prev_scores[date] = None

        for label, date in key_dates:
            snap = build_qir_snapshot(date, _hist_data, _hmm_data, prev_regime_score=_prev_scores.get(date))
            _r = snap["regime"]
            _h = snap["hmm"]

            _lc = {
                "6M BEFORE": COLORS["text_dim"], "3M BEFORE": COLORS["text"],
                "WARNING": COLORS["yellow"], "PEAK": COLORS["bloomberg_orange"],
                "TROUGH": COLORS["negative"], "DIP BUY": COLORS["positive"],
            }.get(label, COLORS["text"])

            _ec = COLORS["positive"] if "BUY" in snap["entry_signal"] else (
                COLORS["negative"] if "SELL" in snap["entry_signal"] else COLORS["yellow"]
            )
            _lean_c = COLORS["positive"] if snap["lean"] == "BULLISH" else COLORS["negative"]

            _vel = snap.get("regime_velocity")
            _vel_html = ""
            if _vel is not None:
                _vel_color = COLORS["positive"] if _vel > 0.02 else (COLORS["negative"] if _vel < -0.02 else COLORS["yellow"])
                _vel_arrow = "↑" if _vel > 0.02 else ("↓" if _vel < -0.02 else "→")
                _vel_label = "ACCELERATING" if abs(_vel) > 0.10 else ("FLIPPING" if abs(_vel) > 0.05 else "DRIFTING" if abs(_vel) > 0.02 else "STABLE")
                _vel_html = (
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;padding:4px 8px;'
                    f'background:#0d1117;border:1px solid {_vel_color}40;border-radius:4px;">'
                    f'<span style="color:{_vel_color};font-size:14px;font-weight:700;">{_vel_arrow}</span>'
                    f'<span style="color:{_vel_color};font-size:10px;font-weight:700;">REGIME VELOCITY: {_vel:+.3f}/week</span>'
                    f'<span style="color:{COLORS["text_dim"]};font-size:9px;">({_vel_label})</span>'
                    f'</div>'
                )

            _hmm_html = ""
            if not _h:
                _hmm_html = (
                    f'<div style="margin-top:6px;padding:6px 8px;background:#0d1117;border:1px solid {COLORS["border"]};'
                    f'border-radius:4px;font-size:9px;color:{COLORS["text_dim"]};">'
                    f'HMM BRAIN STATE: N/A — TIPS real yield data (DFII10) starts 2003; '
                    f'5yr z-score warmup means HMM coverage begins ~2008</div>'
                )
            if _h:
                _hc = {"Bull": COLORS["positive"], "Neutral": COLORS["yellow"],
                       "Early Stress": COLORS["orange"], "Stress": COLORS["negative"],
                       "Late Cycle": COLORS["red"], "Crisis": "#FF0040"}.get(_h["state_label"], COLORS["text"])

                _fc_rows = ""
                if _h.get("labels") and _h.get("forecast_1m") and _h.get("forecast_2m"):
                    for i, lbl in enumerate(_h["labels"]):
                        _now_p = _h["probs"][i] * 100 if i < len(_h["probs"]) else 0
                        _1m_p = _h["forecast_1m"][i] * 100 if i < len(_h["forecast_1m"]) else 0
                        _2m_p = _h["forecast_2m"][i] * 100 if i < len(_h["forecast_2m"]) else 0
                        _bold = "font-weight:700;" if i == _h["state_idx"] else ""
                        _fc_rows += (
                            f'<div style="display:flex;justify-content:space-between;{_bold}font-size:9px;padding:1px 0;">'
                            f'<span style="width:80px;">{lbl}</span>'
                            f'<span style="width:50px;text-align:right;">{_now_p:.0f}%</span>'
                            f'<span style="width:50px;text-align:right;">{_1m_p:.0f}%</span>'
                            f'<span style="width:50px;text-align:right;">{_2m_p:.0f}%</span>'
                            f'</div>'
                        )

                _hmm_html = (
                    f'<div style="margin-top:6px;padding:8px;background:#0d1117;border:1px solid {COLORS["border"]};border-radius:4px;">'
                    f'<div style="font-size:9px;color:{COLORS["text_dim"]};font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">HMM BRAIN STATE</div>'
                    f'<div style="font-size:14px;color:{_hc};font-weight:700;">{_h["state_label"]}</div>'
                    f'<div style="font-size:10px;color:{COLORS["text_dim"]};margin-top:2px;">'
                    f'Confidence: {_h["confidence"]:.1%} &nbsp;|&nbsp; '
                    f'Persistence: {_h["persistence"]}d &nbsp;|&nbsp; '
                    f'Entropy: {_h["entropy"]:.2f} &nbsp;|&nbsp; '
                    f'LL z: {_h["ll_zscore"]:+.1f}</div>'
                    f'<div style="margin-top:6px;font-size:9px;color:{COLORS["text_dim"]};">'
                    f'<div style="display:flex;justify-content:space-between;font-weight:700;padding:1px 0;border-bottom:1px solid {COLORS["border"]};">'
                    f'<span style="width:80px;">State</span>'
                    f'<span style="width:50px;text-align:right;">Now</span>'
                    f'<span style="width:50px;text-align:right;">1M</span>'
                    f'<span style="width:50px;text-align:right;">2M</span></div>'
                    f'{_fc_rows}</div>'
                    f'<div style="font-size:9px;color:{COLORS["text_dim"]};margin-top:4px;">'
                    f'Transition risk: 1M={_h["transition_risk_1m"]:.0%} | 2M={_h["transition_risk_2m"]:.0%} &nbsp;|&nbsp; '
                    f'Kelly mult: {snap["hmm_kelly_mult"]:.2f}x</div>'
                    f'</div>'
                )

            _setup_html = (
                f'<span style="background:#0a2010;color:{COLORS["positive"]};padding:2px 8px;border-radius:3px;'
                f'font-size:10px;font-weight:700;">BUY SETUP</span>'
                if snap["lean"] == "BULLISH" else
                f'<span style="background:#200a0a;color:{COLORS["negative"]};padding:2px 8px;border-radius:3px;'
                f'font-size:10px;font-weight:700;">SHORT SETUP</span>'
            )

            # Top/Bottom proximity block
            _tb = snap.get("top_bottom", {})
            _top_pct = _tb.get("top_pct", 0)
            _bot_pct = _tb.get("bottom_pct", 0)
            _top_sigs = _tb.get("top_signals", [])
            _bot_sigs = _tb.get("bottom_signals", [])
            _tb_html = ""
            if _top_sigs or _bot_sigs:
                _tb_rows = ""
                if _top_sigs:
                    _top_c = COLORS["negative"] if _top_pct >= 50 else (COLORS["yellow"] if _top_pct >= 25 else COLORS["positive"])
                    _tb_rows += (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;">'
                        f'<span style="color:{COLORS["negative"]};font-size:9px;font-weight:700;">▲ MARKET TOP</span>'
                        f'<span style="color:{_top_c};font-size:13px;font-weight:900;">{_top_pct}%</span>'
                        f'</div>'
                        + "".join(f'<div style="font-size:8px;color:#475569;padding:1px 0 1px 8px;">{"●" if v >= 50 else "○"} {n}</div>'
                                  for n, v in zip(_top_sigs, [_tb.get("top_pct",0)]*len(_top_sigs)))
                    )
                if _bot_sigs:
                    _bot_c = COLORS["positive"] if _bot_pct >= 50 else (COLORS["yellow"] if _bot_pct >= 25 else COLORS["text_dim"])
                    _tb_rows += (
                        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;margin-top:3px;">'
                        f'<span style="color:{COLORS["positive"]};font-size:9px;font-weight:700;">▼ MARKET BOTTOM</span>'
                        f'<span style="color:{_bot_c};font-size:13px;font-weight:900;">{_bot_pct}%</span>'
                        f'</div>'
                        + "".join(f'<div style="font-size:8px;color:#475569;padding:1px 0 1px 8px;">{"●" if v >= 50 else "○"} {n}</div>'
                                  for n, v in zip(_bot_sigs, [_tb.get("bottom_pct",0)]*len(_bot_sigs)))
                    )
                _tb_html = (
                    f'<div style="margin-top:6px;padding:6px 10px;background:#0a0f1a;'
                    f'border:1px solid #1e293b;border-radius:4px;">'
                    f'<div style="font-size:8px;color:#475569;font-weight:700;letter-spacing:0.08em;margin-bottom:3px;">'
                    f'TOP / BOTTOM PROXIMITY</div>'
                    f'{_tb_rows}'
                    f'</div>'
                )

            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:6px;padding:12px;margin:6px 0;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<div>'
                f'<span style="color:{_lc};font-weight:700;font-size:12px;">{label}</span>'
                f' <span style="color:{COLORS["text_dim"]};font-size:11px;">{date}</span>'
                f' &nbsp; {_setup_html}'
                f'</div>'
                f'<div style="font-size:11px;">'
                f'<span style="color:{COLORS["text_dim"]};">SPY</span> '
                f'<span style="color:{COLORS["text"]};font-weight:700;">${snap["spy_price"]:,.2f}</span>'
                f' &nbsp;|&nbsp; '
                f'<span style="color:{COLORS["text_dim"]};">VIX</span> '
                f'<span style="color:{COLORS["text"]};font-weight:700;">{snap["vix"]:.1f}</span>'
                f'</div></div>'
                f'{_vel_html}'
                f'<div style="display:flex;gap:12px;font-size:10px;margin-bottom:4px;">'
                f'<div><span style="color:{COLORS["text_dim"]};">Regime:</span> '
                f'<span style="color:{COLORS["text"]};font-weight:700;">{_r["regime_label"]}</span> '
                f'<span style="color:{COLORS["text_dim"]};">({_r["regime_score"]:+.3f})</span></div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Macro:</span> '
                f'<b>{snap["macro_score"]:.0f}</b>/100</div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Tech:</span> '
                f'<b>{snap["tech_score"]:.0f}</b>/100</div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Opts:</span> '
                f'<span style="color:{COLORS["text_dim"]};">N/A</span></div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Sent:</span> '
                f'<span style="color:{COLORS["text_dim"]};">N/A</span></div>'
                f'</div>'
                f'<div style="display:flex;gap:12px;font-size:10px;margin-bottom:2px;">'
                f'<div><span style="color:{COLORS["text_dim"]};">Entry Signal:</span> '
                f'<span style="color:{_ec};font-weight:700;">{snap["entry_signal"]}</span></div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Lean:</span> '
                f'<span style="color:{_lean_c};font-weight:700;">{snap["lean"]} ({snap["lean_pct"]:.0f}%)</span></div>'
                f'<div><span style="color:{COLORS["text_dim"]};">Conviction:</span> '
                f'<b>{snap["conviction"]:.0f}</b>/100</div>'
                f'<div><span style="color:{COLORS["text_dim"]};">GEX:</span> '
                f'<span style="color:{COLORS["text_dim"]};">N/A (no hist options)</span></div>'
                f'</div>'
                f'{_hmm_html}'
                f'{_tb_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="font-size:9px;color:{COLORS["text_dim"]};margin-top:12px;">'
        f'Historical reconstruction uses FRED macro data + VIX + SPY trend only. '
        f'Signals unavailable in historical mode: GEX dealer positioning, options flow, '
        f'whale tracking, StockTwits sentiment, news digest, AI debate. '
        f'HMM state is reconstructed from the trained brain applied to historical features. '
        f'This is a stress test of the quantitative macro signal stack, not the full REGARD system.</div>',
        unsafe_allow_html=True,
    )


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">BACKTESTING ENGINE</div>',
        unsafe_allow_html=True,
    )

    tab_bt, tab_wf, tab_rc, tab_replay, tab_crash = st.tabs([
        "⚙ Strategy Backtest", "📊 Walk-Forward Validation",
        "🗂 Regime-Conditional", "🔁 ATR Replay", "💥 Crash Stress Test",
    ])

    with tab_wf:
        _render_walk_forward()

    with tab_bt:
        signal = st.selectbox(
            "Signal Strategy",
            ["SMA Crossover", "VIX Spike", "Regime Flip", "Insider Cluster"],
            key="bt_signal",
        )

        st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin:8px 0 4px 0;">'
                    f'PARAMETERS</div>', unsafe_allow_html=True)

        if signal == "SMA Crossover":
            col1, col2, col3 = st.columns(3)
            with col1:
                ticker = st.text_input("Ticker", value="SPY", key="bt_sma_ticker").upper().strip()
            with col2:
                short_w = st.number_input("Short SMA", value=50, min_value=5, max_value=100, key="bt_short")
            with col3:
                long_w = st.number_input("Long SMA", value=200, min_value=50, max_value=400, key="bt_long")
            lookback = st.slider("Lookback (years)", 1, 10, 5, key="bt_lb_sma")

        elif signal == "VIX Spike":
            col1, = st.columns(1)
            with col1:
                threshold = st.number_input("VIX Threshold", value=25.0, min_value=15.0, max_value=50.0,
                                            step=1.0, key="bt_vix_thresh")
            lookback = st.slider("Lookback (years)", 1, 10, 5, key="bt_lb_vix")

        elif signal == "Regime Flip":
            st.caption("No parameters needed — fires on Risk-Off → Risk-On regime transitions from saved history.")

        elif signal == "Insider Cluster":
            col1, col2 = st.columns(2)
            with col1:
                ticker = st.text_input("Ticker", value="AAPL", key="bt_ins_ticker").upper().strip()
            with col2:
                min_buys = st.number_input("Min Buys", value=3, min_value=2, max_value=10, key="bt_min_buys")
            cluster_days = st.slider("Cluster Window (days)", 7, 90, 30, key="bt_cluster_days")

        # ATR exit parameters
        st.markdown(f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};margin:10px 0 4px 0;">'
                    f'ATR EXIT PARAMETERS</div>', unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            atr_stop = st.number_input("Trailing Stop ×ATR", value=2.0, min_value=0.5, max_value=5.0,
                                       step=0.5, key="bt_atr_stop",
                                       help="Stop = 2×ATR below entry (long). Trails the high watermark.")
        with ac2:
            atr_tgt = st.number_input("Profit Target ×ATR", value=3.0, min_value=0.5, max_value=8.0,
                                      step=0.5, key="bt_atr_tgt",
                                      help="Target = 3×ATR above entry. R:R = target÷stop.")
        _rr = round(atr_tgt / atr_stop, 2) if atr_stop > 0 else "—"
        st.caption(f"R:R ratio: **{_rr}:1** — target fires {atr_tgt}×ATR from entry, stop at {atr_stop}×ATR.")

        with st.expander("ℹ️ How These Strategies Work"):
            st.markdown("""
**SMA Crossover (Golden Cross)**
- When the Short SMA crosses **above** the Long SMA, it's a golden cross (buy signal).
- Exit: ATR trailing stop (moves up with high watermark) or profit target.

**VIX Spike**
- Buys SPY when VIX crosses above threshold — contrarian bet that fear is overdone.
- Exit: ATR trailing stop or profit target on SPY.

**Regime Flip**
- Buys SPY when Risk Regime flips from Risk-Off → Risk-On (sourced from regime history).
- Exit: ATR trailing stop or profit target.

**Insider Cluster**
- Fires on 3+ insider **purchases** within a rolling window on the same stock.
- Exit: ATR trailing stop or profit target.

---
**ATR Exit vs Fixed Hold Days**
Fixed hold days close trades at an arbitrary calendar date — a great call early looks wrong, a bad call gets lucky. ATR trailing stop lets winners run and cuts losers at a volatility-calibrated level, giving cleaner signal measurement.
""")

        if st.button("RUN BACKTEST", type="primary", key="bt_run"):
            with st.spinner("Running backtest..."):
                if signal == "SMA Crossover":
                    result = backtest_sma_crossover(ticker, short_w, long_w, lookback,
                                                    atr_stop_mult=atr_stop, atr_target_mult=atr_tgt)
                elif signal == "VIX Spike":
                    result = backtest_vix_spike(threshold, lookback,
                                                atr_stop_mult=atr_stop, atr_target_mult=atr_tgt)
                elif signal == "Regime Flip":
                    result = backtest_regime_flip(atr_stop_mult=atr_stop, atr_target_mult=atr_tgt)
                elif signal == "Insider Cluster":
                    result = backtest_insider_cluster(ticker, min_buys, cluster_days,
                                                      atr_stop_mult=atr_stop, atr_target_mult=atr_tgt)
                else:
                    return
                st.session_state["bt_result"] = result

        result = st.session_state.get("bt_result")
        if result:
            _render_results(result)

    with tab_rc:
        _render_regime_conditional()

    with tab_replay:
        _render_atr_replay()

    with tab_crash:
        _render_crash_stress_test()
