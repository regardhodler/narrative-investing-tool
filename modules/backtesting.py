"""Backtesting — test historical signal strategies and view results."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.theme import COLORS, apply_dark_layout
from services.backtest_engine import (
    backtest_sma_crossover,
    backtest_vix_spike,
    backtest_regime_flip,
    backtest_insider_cluster,
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


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">BACKTESTING ENGINE</div>',
        unsafe_allow_html=True,
    )

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
