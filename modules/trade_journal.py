"""Trade Journal — log trades, track live P&L, and analyze performance by signal source."""

import json
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from datetime import date
from utils.journal import load_journal, add_trade, close_trade, delete_trade, update_trade
from utils.theme import COLORS, apply_dark_layout
from services.sec_client import search_ticker_by_name

_REGIME_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "regime_history.json")
_SIGNAL_SOURCES = ["Manual", "Insider Buying", "Options Flow", "AI Valuation",
                   "Risk Regime", "Whale Movement", "SMA Crossover", "Scorecard"]


def _get_latest_regime() -> str:
    """Read latest regime from regime_history.json."""
    if not os.path.exists(_REGIME_FILE):
        return ""
    try:
        with open(_REGIME_FILE) as f:
            history = json.load(f)
        if history:
            latest = history[-1]
            return f"{latest['regime']} ({latest['macro_score']})"
    except Exception:
        pass
    return ""


def _get_live_prices(tickers: list[str]) -> dict[str, float]:
    """Fetch current prices for open positions (5-min cache)."""
    if not tickers:
        return {}
    from services.market_data import fetch_batch_safe
    snaps = fetch_batch_safe({t: t for t in tickers}, period="5d", interval="1d")
    return {t: s.latest_price for t, s in snaps.items() if s.latest_price}


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">TRADE JOURNAL</div>',
        unsafe_allow_html=True,
    )

    journal = load_journal()
    open_trades = [t for t in journal if t["status"] == "open"]
    closed_trades = [t for t in journal if t["status"] == "closed"]

    # --- Add Trade Form ---
    with st.expander("ADD NEW TRADE", expanded=not journal):
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Ticker", key="jrnl_ticker", placeholder="AAPL").upper().strip()
            if ticker:
                matches = search_ticker_by_name(ticker)
                exact = [m for m in matches if m["ticker"] == ticker]
                if exact:
                    st.caption(f":green[✓ {exact[0]['name']}]")
                else:
                    # Fallback to yfinance for non-US tickers (TSX, etc.)
                    try:
                        info = yf.Ticker(ticker).info
                        name = info.get("shortName") or info.get("longName")
                        if name:
                            st.caption(f":green[✓ {name}]")
                        else:
                            st.caption("⚠ Ticker not found")
                    except Exception:
                        st.caption("⚠ Ticker not found")
            direction = st.selectbox("Direction", ["Long", "Short"], key="jrnl_dir")
            signal_source = st.selectbox(
                "Signal Source", _SIGNAL_SOURCES, key="jrnl_signal",
            )
        with col2:
            # Auto-fill price
            default_price = 0.0
            if ticker:
                try:
                    tk = yf.Ticker(ticker)
                    hist = tk.history(period="1d", auto_adjust=True)
                    if hist is not None and not hist.empty:
                        default_price = round(float(hist["Close"].iloc[-1]), 2)
                except Exception:
                    pass
            entry_price = st.number_input("Entry Price", value=default_price, min_value=0.0,
                                          step=0.01, key="jrnl_price")
            position_size = st.number_input("Shares", value=100, min_value=1, step=1, key="jrnl_size")
            entry_date = st.date_input("Entry Date", value=date.today(), key="jrnl_date")
            notes = st.text_input("Notes", key="jrnl_notes", placeholder="Reason for trade...")

        # Thesis and regime context
        thesis = st.text_area("Trade Thesis", key="jrnl_thesis",
                              placeholder="Why are you taking this trade?...", height=80)
        regime_label = _get_latest_regime()
        if regime_label:
            st.markdown(
                f'<div style="font-size:12px;color:{COLORS["text_dim"]};">'
                f'Macro Regime at Entry: <b style="color:{COLORS["bloomberg_orange"]};">{regime_label}</b></div>',
                unsafe_allow_html=True,
            )

        if st.button("LOG TRADE", type="primary", key="jrnl_add"):
            if ticker and entry_price > 0:
                add_trade(ticker, direction, entry_price, position_size, signal_source, notes,
                         entry_date=str(entry_date), thesis=thesis, regime_at_entry=regime_label)
                st.success(f"Logged {direction} {ticker} @ ${entry_price:.2f}")
                st.rerun()
            else:
                st.error("Enter a valid ticker and price.")

    # Smaller action buttons (EDIT / CLOSE / DEL)
    st.markdown(
        '<style>[data-testid="stButton"] button[kind="secondary"]'
        '{font-size:11px;padding:2px 8px;min-height:28px;}</style>',
        unsafe_allow_html=True,
    )

    # --- Tabs ---
    tab_open, tab_closed, tab_analytics = st.tabs(["OPEN POSITIONS", "CLOSED TRADES", "ANALYTICS"])

    # --- Open Positions ---
    with tab_open:
        if not open_trades:
            st.info("No open positions.")
        else:
            live_tickers = list({t["ticker"] for t in open_trades})
            prices = _get_live_prices(live_tickers)

            # Compute total portfolio value for portfolio % calc
            total_portfolio_val = 0.0
            for trade in open_trades:
                cur = prices.get(trade["ticker"]) or trade["entry_price"]
                total_portfolio_val += cur * trade["position_size"]

            for trade in open_trades:
                tid = trade["id"]
                current = prices.get(trade["ticker"])
                entry = trade["entry_price"]
                size = trade["position_size"]
                direction = trade["direction"]

                if current and entry > 0:
                    if direction == "Long":
                        pnl = (current - entry) * size
                        pnl_pct = (current / entry - 1) * 100
                    else:
                        pnl = (entry - current) * size
                        pnl_pct = (entry / current - 1) * 100 if current > 0 else 0
                    color = COLORS["positive"] if pnl >= 0 else COLORS["negative"]
                else:
                    pnl = 0
                    pnl_pct = 0
                    color = COLORS["text_dim"]
                    current = current or 0

                pos_val = (current or entry) * size
                port_pct = (pos_val / total_portfolio_val * 100) if total_portfolio_val > 0 else 0

                col1, col2, col3, col4, col5, col6 = st.columns([2, 1.5, 1.2, 1, 1.2, 1.2])
                with col1:
                    st.markdown(
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{trade["ticker"]}</span>'
                        f' <span style="color:{COLORS["text_dim"]};font-size:11px;">{direction} · {trade["signal_source"]}'
                        f' · {trade["entry_date"]}</span>',
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(f'Entry: **${entry:.2f}** × {size}')
                with col3:
                    st.markdown(f'Now: **${current:.2f}**')
                with col4:
                    st.markdown(
                        f'<span style="color:{COLORS["text_dim"]};font-size:12px;">{port_pct:.1f}% port</span>',
                        unsafe_allow_html=True,
                    )
                with col5:
                    st.markdown(
                        f'<span style="color:{color};font-weight:700;">${pnl:+,.2f} ({pnl_pct:+.1f}%)</span>',
                        unsafe_allow_html=True,
                    )
                with col6:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.button("EDIT", key=f"edit_{tid}"):
                            st.session_state[f"editing_{tid}"] = True
                            st.rerun()
                    with c2:
                        if st.button("CLOSE", key=f"close_{tid}"):
                            close_trade(tid, current)
                            st.rerun()
                    with c3:
                        if st.button("DEL", key=f"del_{tid}"):
                            delete_trade(tid)
                            st.rerun()

                # Inline edit form
                if st.session_state.get(f"editing_{tid}"):
                    with st.expander(f"EDITING {trade['ticker']}", expanded=True):
                        with st.form(key=f"editform_{tid}"):
                            ec1, ec2 = st.columns(2)
                            with ec1:
                                ed_ticker = st.text_input("Ticker", value=trade["ticker"], key=f"ed_tk_{tid}")
                                ed_dir = st.selectbox("Direction", ["Long", "Short"],
                                                      index=0 if trade["direction"] == "Long" else 1,
                                                      key=f"ed_dir_{tid}")
                                sig_idx = _SIGNAL_SOURCES.index(trade["signal_source"]) if trade["signal_source"] in _SIGNAL_SOURCES else 0
                                ed_signal = st.selectbox("Signal Source", _SIGNAL_SOURCES,
                                                         index=sig_idx, key=f"ed_sig_{tid}")
                            with ec2:
                                ed_price = st.number_input("Entry Price", value=float(entry),
                                                           min_value=0.0, step=0.01, key=f"ed_px_{tid}")
                                ed_size = st.number_input("Shares", value=int(size),
                                                          min_value=1, step=1, key=f"ed_sz_{tid}")
                                ed_date = st.date_input("Entry Date",
                                                        value=date.fromisoformat(trade["entry_date"]),
                                                        key=f"ed_dt_{tid}")
                            ed_notes = st.text_input("Notes", value=trade.get("notes", ""), key=f"ed_nt_{tid}")
                            ed_thesis = st.text_area("Thesis", value=trade.get("thesis", ""),
                                                     height=80, key=f"ed_th_{tid}")
                            sb1, sb2 = st.columns(2)
                            with sb1:
                                save_clicked = st.form_submit_button("SAVE", type="primary")
                            with sb2:
                                cancel_clicked = st.form_submit_button("CANCEL")
                        if save_clicked:
                            update_trade(tid,
                                         ticker=ed_ticker.upper().strip(),
                                         direction=ed_dir,
                                         entry_price=ed_price,
                                         position_size=ed_size,
                                         entry_date=str(ed_date),
                                         signal_source=ed_signal,
                                         notes=ed_notes,
                                         thesis=ed_thesis)
                            st.session_state.pop(f"editing_{tid}", None)
                            st.rerun()
                        if cancel_clicked:
                            st.session_state.pop(f"editing_{tid}", None)
                            st.rerun()

                st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:4px 0;"></div>',
                            unsafe_allow_html=True)

    # --- Closed Trades ---
    with tab_closed:
        if not closed_trades:
            st.info("No closed trades yet.")
        else:
            rows = []
            for t in closed_trades:
                entry = t["entry_price"]
                exit_p = t["exit_price"] or 0
                size = t["position_size"]
                if t["direction"] == "Long":
                    pnl = (exit_p - entry) * size
                    pnl_pct = (exit_p / entry - 1) * 100 if entry > 0 else 0
                else:
                    pnl = (entry - exit_p) * size
                    pnl_pct = (entry / exit_p - 1) * 100 if exit_p > 0 else 0
                rows.append({
                    "Ticker": t["ticker"],
                    "Dir": t["direction"],
                    "Entry": f"${entry:.2f}",
                    "Exit": f"${exit_p:.2f}",
                    "Shares": size,
                    "P&L": f"${pnl:+,.2f}",
                    "Return": f"{pnl_pct:+.1f}%",
                    "Signal": t["signal_source"],
                    "Regime": t.get("regime_at_entry", ""),
                    "Entry Date": t["entry_date"],
                    "Exit Date": t.get("exit_date", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Edit buttons for closed trades
            for t in closed_trades:
                tid = t["id"]
                col_label, col_btn = st.columns([5, 1])
                with col_label:
                    st.markdown(
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{t["ticker"]}</span>'
                        f' <span style="color:{COLORS["text_dim"]};font-size:12px;">'
                        f'{t["direction"]} · {t["entry_date"]} → {t.get("exit_date", "")}</span>',
                        unsafe_allow_html=True,
                    )
                with col_btn:
                    if st.button("EDIT", key=f"cedit_{tid}"):
                        st.session_state[f"cediting_{tid}"] = True
                        st.rerun()

                if st.session_state.get(f"cediting_{tid}"):
                    with st.expander(f"EDITING {t['ticker']}", expanded=True):
                        with st.form(key=f"ceditform_{tid}"):
                            ec1, ec2 = st.columns(2)
                            with ec1:
                                ed_ticker = st.text_input("Ticker", value=t["ticker"], key=f"ced_tk_{tid}")
                                ed_dir = st.selectbox("Direction", ["Long", "Short"],
                                                      index=0 if t["direction"] == "Long" else 1,
                                                      key=f"ced_dir_{tid}")
                                sig_idx = _SIGNAL_SOURCES.index(t["signal_source"]) if t["signal_source"] in _SIGNAL_SOURCES else 0
                                ed_signal = st.selectbox("Signal Source", _SIGNAL_SOURCES,
                                                         index=sig_idx, key=f"ced_sig_{tid}")
                                ed_notes = st.text_input("Notes", value=t.get("notes", ""), key=f"ced_nt_{tid}")
                            with ec2:
                                ed_entry_px = st.number_input("Entry Price", value=float(t["entry_price"]),
                                                              min_value=0.0, step=0.01, key=f"ced_epx_{tid}")
                                ed_exit_px = st.number_input("Exit Price", value=float(t["exit_price"] or 0),
                                                             min_value=0.0, step=0.01, key=f"ced_xpx_{tid}")
                                ed_size = st.number_input("Shares", value=int(t["position_size"]),
                                                          min_value=1, step=1, key=f"ced_sz_{tid}")
                                ed_entry_dt = st.date_input("Entry Date",
                                                            value=date.fromisoformat(t["entry_date"]),
                                                            key=f"ced_edt_{tid}")
                                exit_dt_val = date.fromisoformat(t["exit_date"]) if t.get("exit_date") else date.today()
                                ed_exit_dt = st.date_input("Exit Date", value=exit_dt_val,
                                                           key=f"ced_xdt_{tid}")
                            ed_thesis = st.text_area("Thesis", value=t.get("thesis", ""),
                                                     height=80, key=f"ced_th_{tid}")
                            ed_reopen = st.checkbox("Re-open this trade (clear exit data)",
                                                    key=f"ced_reopen_{tid}")
                            sb1, sb2 = st.columns(2)
                            with sb1:
                                save_clicked = st.form_submit_button("SAVE", type="primary")
                            with sb2:
                                cancel_clicked = st.form_submit_button("CANCEL")
                        if save_clicked:
                            updates = dict(
                                ticker=ed_ticker.upper().strip(),
                                direction=ed_dir,
                                entry_price=ed_entry_px,
                                exit_price=ed_exit_px,
                                position_size=ed_size,
                                entry_date=str(ed_entry_dt),
                                exit_date=str(ed_exit_dt),
                                signal_source=ed_signal,
                                notes=ed_notes,
                                thesis=ed_thesis,
                            )
                            if ed_reopen:
                                updates.update(status="open", exit_price=None, exit_date=None)
                            update_trade(tid, **updates)
                            st.session_state.pop(f"cediting_{tid}", None)
                            st.rerun()
                        if cancel_clicked:
                            st.session_state.pop(f"cediting_{tid}", None)
                            st.rerun()

            # Show thesis details for trades that have one
            trades_with_thesis = [t for t in closed_trades if t.get("thesis")]
            if trades_with_thesis:
                with st.expander("Trade Theses"):
                    for t in trades_with_thesis:
                        st.markdown(
                            f'**{t["ticker"]}** ({t["entry_date"]}) — {t["thesis"]}',
                        )

    # --- Analytics ---
    with tab_analytics:
        if not closed_trades and not open_trades:
            st.info("Close some trades to see analytics.")
            return

        # Compute P&L for each closed trade
        pnls = []
        ytd_pnls = []
        signals = []
        current_year = str(date.today().year)
        for t in closed_trades:
            entry = t["entry_price"]
            exit_p = t["exit_price"] or 0
            size = t["position_size"]
            if t["direction"] == "Long":
                pnl = (exit_p - entry) * size
            else:
                pnl = (entry - exit_p) * size
            pnls.append(pnl)
            signals.append(t["signal_source"])
            if t.get("exit_date", "").startswith(current_year):
                ytd_pnls.append(pnl)

        # Unrealized P&L from ALL open positions
        unrealized_total = 0.0
        if open_trades:
            live_tickers = list({t["ticker"] for t in open_trades})
            live_prices = _get_live_prices(live_tickers)
            for t in open_trades:
                price = live_prices.get(t["ticker"])
                if price is None:
                    continue
                size = t["position_size"]
                if t["direction"] == "Long":
                    unrealized_total += (price - t["entry_price"]) * size
                else:
                    unrealized_total += (t["entry_price"] - price) * size

        total_pnl = sum(pnls)
        ytd_pnl = sum(ytd_pnls) + unrealized_total
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Metrics row — shrink font so values aren't clipped
        st.markdown(
            """<style>
            [data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 14px !important;
            }
            [data-testid="stMetric"] [data-testid="stMetricLabel"] {
                font-size: 10px !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total P&L (realized)", f"${total_pnl:+,.2f}")
        m2.metric("Unrealized P&L", f"${unrealized_total:+,.2f}")
        m3.metric("YTD P&L (incl. open)", f"${ytd_pnl:+,.2f}")
        m4.metric("Win Rate", f"{win_rate:.0f}%")
        m5.metric("Avg Win / Loss", f"${avg_win:+,.2f} / ${avg_loss:+,.2f}")

        if not closed_trades:
            st.info("Close some trades to see full analytics (equity curve, signal breakdown).")
            return

        # Equity curve
        equity = [10000]  # assume $10k starting
        for p in pnls:
            equity.append(equity[-1] + p)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=equity, mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
            marker=dict(size=4),
            name="Equity",
        ))
        apply_dark_layout(fig, title="Equity Curve", yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Performance by signal source
        df_signals = pd.DataFrame({"signal": signals, "pnl": pnls})
        grouped = df_signals.groupby("signal").agg(
            total_pnl=("pnl", "sum"),
            count=("pnl", "count"),
            win_rate=("pnl", lambda x: (x > 0).sum() / len(x) * 100),
        ).reset_index()

        fig2 = go.Figure()
        colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in grouped["total_pnl"]]
        fig2.add_trace(go.Bar(
            x=grouped["signal"], y=grouped["total_pnl"],
            marker_color=colors,
            text=[f"${v:+,.0f}<br>{n} trades<br>{wr:.0f}% win"
                  for v, n, wr in zip(grouped["total_pnl"], grouped["count"], grouped["win_rate"])],
            textposition="auto",
        ))
        apply_dark_layout(fig2, title="P&L by Signal Source", yaxis_title="Total P&L ($)")
        st.plotly_chart(fig2, use_container_width=True)
