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

# Canadian exchange suffixes recognised by yfinance
_CAD_SUFFIXES = (".TO", ".V", ".TSX", ".CN", ".NE", ".VN")


def _ticker_currency(ticker: str) -> str:
    """Return 'CAD' for TSX-listed tickers, 'USD' for everything else."""
    return "CAD" if ticker.upper().endswith(_CAD_SUFFIXES) else "USD"


@st.cache_data(ttl=3600)
def _get_usdcad() -> float:
    """Fetch live USD/CAD spot rate (cached 1 hr). Falls back to 1.36 if unavailable."""
    try:
        raw = yf.download("USDCAD=X", period="5d", interval="1d", progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            rate = float(close.dropna().iloc[-1])
            if 1.0 < rate < 2.0:  # sanity check
                return rate
    except Exception:
        pass
    return 1.36  # reasonable fallback


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

        # Compute P&L for each closed trade — all values converted to CAD
        _usdcad_metrics = _get_usdcad()
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
            if _ticker_currency(t["ticker"]) == "USD":
                pnl *= _usdcad_metrics
            pnls.append(pnl)
            signals.append(t["signal_source"])
            if t.get("exit_date", "").startswith(current_year):
                ytd_pnls.append(pnl)

        # Unrealized P&L from ALL open positions — converted to CAD
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
                    pnl_native = (price - t["entry_price"]) * size
                else:
                    pnl_native = (t["entry_price"] - price) * size
                if _ticker_currency(t["ticker"]) == "USD":
                    pnl_native *= _usdcad_metrics
                unrealized_total += pnl_native

        total_pnl = sum(pnls)
        ytd_pnl = sum(ytd_pnls) + unrealized_total
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0

        # Total portfolio value (open positions at live price, in CAD)
        _port_value_cad = 0.0
        if open_trades:
            _live_for_val = _get_live_prices(list({t["ticker"] for t in open_trades}))
            for t in open_trades:
                _px = (_live_for_val.get(t["ticker"]) or t["entry_price"])
                _val_native = _px * t["position_size"]
                _port_value_cad += _val_native * _usdcad_metrics if _ticker_currency(t["ticker"]) == "USD" else _val_native

        # Metrics row — shrink font so values aren't clipped
        st.markdown(
            """<style>
            [data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 12px !important;
            }
            [data-testid="stMetric"] [data-testid="stMetricLabel"] {
                font-size: 10px !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )
        m0, m1, m2, m3, m4, m5 = st.columns(6)
        m0.metric("Portfolio Value (CAD)", f"C${_port_value_cad:,.2f}")
        m1.metric("Total P&L (realized, CAD)", f"C${total_pnl:+,.2f}")
        m2.metric("Unrealized P&L (CAD)", f"C${unrealized_total:+,.2f}")
        m3.metric("YTD P&L (incl. open, CAD)", f"C${ytd_pnl:+,.2f}")
        m4.metric("Win Rate", f"{win_rate:.0f}%")
        m5.metric("Avg Win / Loss", f"C${avg_win:+,.2f} / C${avg_loss:+,.2f}")

        # ── Unrealized P&L Since Entry (open positions history) ───────────────
        if open_trades:
            st.markdown(
                f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin:18px 0 6px 0;">UNREALIZED P&L SINCE ENTRY</div>',
                unsafe_allow_html=True,
            )

            @st.cache_data(ttl=3600)
            def _fetch_open_history(trades_key: tuple, usdcad: float) -> pd.DataFrame:
                """
                Returns a daily series of cumulative unrealized P&L in CAD.
                trades_key is a tuple of (ticker, entry_date, entry_price, size, direction).
                """
                import yfinance as _yf
                if not trades_key:
                    return pd.DataFrame()

                # Unpack tuples back into dicts
                trades_list = [
                    {"ticker": tk, "entry_date": ed, "entry_price": ep, "position_size": sz, "direction": dr}
                    for tk, ed, ep, sz, dr in trades_key
                ]

                tickers = [t["ticker"] for t in trades_list]
                earliest = min(t["entry_date"] for t in trades_list)

                raw = _yf.download(
                    tickers if len(tickers) > 1 else tickers[0],
                    start=earliest,
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if raw is None or raw.empty:
                    return pd.DataFrame()

                if isinstance(raw.columns, pd.MultiIndex):
                    prices = raw["Close"]
                else:
                    prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

                # Ensure all tickers are present
                for tk in tickers:
                    if tk not in prices.columns:
                        prices[tk] = float("nan")

                # Daily portfolio unrealized P&L in CAD
                daily_pnl = pd.Series(0.0, index=prices.index)
                for t in trades_list:
                    tk = t["ticker"]
                    entry_date = pd.Timestamp(t["entry_date"])
                    entry_price = t["entry_price"]
                    size = t["position_size"]
                    direction = t["direction"]
                    ccy = _ticker_currency(tk)

                    col = prices[tk].copy()
                    col = col[col.index >= entry_date]
                    if col.empty:
                        continue

                    if direction == "Long":
                        pnl_native = (col - entry_price) * size
                    else:
                        pnl_native = (entry_price - col) * size

                    if ccy == "USD":
                        pnl_native = pnl_native * usdcad

                    daily_pnl = daily_pnl.add(pnl_native, fill_value=0.0)

                return daily_pnl.dropna()

            # Build a hashable key from open trades
            _trades_key = tuple(
                (t["ticker"], t["entry_date"], t["entry_price"], t["position_size"], t["direction"])
                for t in sorted(open_trades, key=lambda x: x["entry_date"])
            )
            _usdcad_now = _get_usdcad()
            _daily_upnl = _fetch_open_history(_trades_key, _usdcad_now)

            if not _daily_upnl.empty:
                _color_line = COLORS.get("positive", "#00e676") if float(_daily_upnl.iloc[-1]) >= 0 else COLORS.get("negative", "#f44336")
                fig_upnl = go.Figure()
                fig_upnl.add_trace(go.Scatter(
                    x=_daily_upnl.index,
                    y=_daily_upnl.values,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(0,200,100,0.08)" if float(_daily_upnl.iloc[-1]) >= 0 else "rgba(244,67,54,0.08)",
                    line=dict(color=_color_line, width=2),
                    hovertemplate="<b>%{x|%b %d, %Y}</b><br>C$%{y:+,.2f}<extra></extra>",
                ))
                apply_dark_layout(fig_upnl)
                fig_upnl.update_layout(
                    height=280,
                    margin=dict(t=20, b=24, l=24, r=24),
                    yaxis_title="Unrealized P&L (C$)",
                    showlegend=False,
                )
                fig_upnl.add_hline(y=0, line_dash="dot", line_color="#555")
                st.plotly_chart(fig_upnl, use_container_width=True, key="jrnl_upnl_chart")
            else:
                st.info("Not enough price history to build the chart.")

        # ── Portfolio Allocation Pie ──────────────────────────────────────────
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;margin:18px 0 6px 0;">PORTFOLIO ALLOCATION</div>',
            unsafe_allow_html=True,
        )

        _all_trades = open_trades + closed_trades
        if _all_trades:
            # Use live price for open positions; entry price for closed
            _live = _get_live_prices(list({t["ticker"] for t in open_trades})) if open_trades else {}
            _usdcad = _get_usdcad()

            # Build allocation in CAD — USD positions are converted at live spot rate
            _alloc: dict[str, float] = {}
            _currencies: dict[str, str] = {}
            for t in _all_trades:
                price = (_live.get(t["ticker"]) if t["status"] == "open" else None) or t["entry_price"]
                val_native = abs(price * t["position_size"])
                ccy = _ticker_currency(t["ticker"])
                val_cad = val_native * _usdcad if ccy == "USD" else val_native
                _alloc[t["ticker"]] = _alloc.get(t["ticker"], 0.0) + val_cad
                _currencies[t["ticker"]] = ccy

            _total_val = sum(_alloc.values())
            _labels = list(_alloc.keys())
            _values = [_alloc[k] for k in _labels]
            _pcts   = [v / _total_val * 100 for v in _values]

            # Show the rate used
            st.markdown(
                f'<div style="font-size:11px;color:{COLORS["text_dim"]};margin-bottom:6px;">'
                f'All values in <b style="color:#fff;">CAD</b> · '
                f'USD/CAD rate: <b style="color:{COLORS["bloomberg_orange"]};">{_usdcad:.4f}</b> '
                f'(live, 1hr cache)</div>',
                unsafe_allow_html=True,
            )

            _pie_col, _tbl_col = st.columns([3, 2])
            with _pie_col:
                fig_pie = go.Figure(go.Pie(
                    labels=_labels,
                    values=_values,
                    texttemplate="%{label}<br>%{percent}",
                    textposition="outside",
                    hole=0.45,
                    marker=dict(
                        colors=[
                            "#ff9800", "#2196f3", "#4caf50", "#e91e63",
                            "#9c27b0", "#00bcd4", "#ff5722", "#8bc34a",
                            "#ffc107", "#607d8b", "#3f51b5", "#009688",
                        ][:len(_labels)],
                        line=dict(color="#1a1a1a", width=2),
                    ),
                    hovertemplate="<b>%{label}</b><br>C$%{value:,.0f}<br>%{percent}<extra></extra>",
                ))
                apply_dark_layout(fig_pie)
                fig_pie.update_layout(
                    height=340,
                    margin=dict(t=20, b=20, l=20, r=20),
                    showlegend=False,
                    annotations=[dict(
                        text=f"C${_total_val:,.0f}",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=14, color="#ccc"),
                    )],
                )
                st.plotly_chart(fig_pie, use_container_width=True, key="jrnl_alloc_pie")

            with _tbl_col:
                _tbl_rows = sorted(
                    [
                        {
                            "Ticker": k,
                            "Ccy": _currencies.get(k, "USD"),
                            "Value (CAD)": f"C${v:,.0f}",
                            "Weight": f"{p:.1f}%",
                        }
                        for k, v, p in zip(_labels, _values, _pcts)
                    ],
                    key=lambda r: float(r["Weight"].rstrip("%")),
                    reverse=True,
                )
                st.dataframe(pd.DataFrame(_tbl_rows), use_container_width=True, hide_index=True)

        # ── Portfolio Correlation ─────────────────────────────────────────────
        # Shown for any journal entries (open OR closed) — does NOT need closed trades
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;margin:18px 0 6px 0;">PORTFOLIO CORRELATION</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:11px;color:#888;line-height:1.6;margin-bottom:10px;">'
            'Pearson correlation (ρ) measures how similarly two assets move day-to-day. '
            '<b style="color:#ccc;">ρ = +1.0</b> means perfect lockstep (both rise/fall together). '
            '<b style="color:#ccc;">ρ = 0</b> means no relationship. '
            '<b style="color:#ccc;">ρ = −1.0</b> means perfect offset (one rises when the other falls). '
            'For portfolio health, lower average ρ is better — positions with low or negative correlation '
            'reduce overall volatility without sacrificing expected return. '
            'The <b style="color:#ccc;">R² = ρ²</b> figure tells you what % of daily variance is shared '
            '(e.g. ρ = 0.5 → R² = 25% shared, 75% independent).</div>',
            unsafe_allow_html=True,
        )

        all_tickers = list({t["ticker"] for t in open_trades + closed_trades})

        if len(all_tickers) < 2:
            st.info("Need at least 2 tickers in your journal to compute correlations.")
        else:
            cr_col1, cr_col2 = st.columns([1, 3])
            with cr_col1:
                corr_period = st.selectbox(
                    "Lookback Period",
                    options=["3mo", "6mo", "1y", "2y", "5y"],
                    index=2,
                    key="jrnl_corr_period",
                )

            @st.cache_data(ttl=3600)
            def _fetch_corr_data(tickers: tuple, period: str) -> pd.DataFrame:
                raw = yf.download(
                    list(tickers), period=period, interval="1d",
                    progress=False, auto_adjust=True,
                )
                if raw.empty:
                    return pd.DataFrame()
                if isinstance(raw.columns, pd.MultiIndex):
                    close = raw["Close"]
                else:
                    close = raw[["Close"]] if "Close" in raw.columns else raw
                close = close.dropna(how="all")
                # Keep only tickers that actually downloaded
                close = close.loc[:, close.notna().sum() > 10]
                return close.pct_change().dropna(how="all")

            returns = _fetch_corr_data(tuple(sorted(all_tickers)), corr_period)

            if returns.empty or returns.shape[1] < 2:
                st.warning("Could not fetch enough price history for these tickers.")
            else:
                corr = returns.corr()
                labels = corr.columns.tolist()

                # ── Heatmap ──────────────────────────────────────────────
                z = corr.values.tolist()
                text = [[f"{v:.2f}" for v in row] for row in z]

                fig_corr = go.Figure(go.Heatmap(
                    z=z,
                    x=labels,
                    y=labels,
                    text=text,
                    texttemplate="%{text}",
                    colorscale=[
                        [0.0,  "#1565C0"],   # strong negative = blue
                        [0.5,  "#212121"],   # zero = near-black
                        [1.0,  "#C62828"],   # strong positive = red
                    ],
                    zmid=0,
                    zmin=-1,
                    zmax=1,
                    showscale=True,
                    colorbar=dict(
                        title=dict(text="ρ", font=dict(color="#aaa")),
                        thickness=14,
                        len=0.8,
                        tickfont=dict(color="#aaa", size=11),
                    ),
                ))
                apply_dark_layout(fig_corr)
                fig_corr.update_layout(
                    title=f"Return Correlation ({corr_period})",
                    height=max(320, 80 * len(labels)),
                    margin=dict(t=40, b=40, l=60, r=20),
                    xaxis=dict(side="bottom", tickfont=dict(color="#ccc", size=11)),
                    yaxis=dict(tickfont=dict(color="#ccc", size=11), autorange="reversed"),
                )
                st.plotly_chart(fig_corr, use_container_width=True, key="jrnl_corr_heatmap")

                # ── Highly correlated pairs warning ───────────────────────
                high_pairs = []
                for i, t1 in enumerate(labels):
                    for j, t2 in enumerate(labels):
                        if j <= i:
                            continue
                        v = corr.loc[t1, t2]
                        if abs(v) >= 0.70:
                            high_pairs.append({
                                "Pair": f"{t1} / {t2}",
                                "Correlation": f"{v:+.2f}",
                                "Type": "⚠ High positive" if v >= 0.70 else "✦ Inverse hedge",
                            })

                if high_pairs:
                    st.markdown(
                        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};'
                        f'margin:6px 0 4px 0;">Significant Pairs  (|ρ| ≥ 0.70)</div>',
                        unsafe_allow_html=True,
                    )
                    st.dataframe(
                        pd.DataFrame(high_pairs),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.markdown(
                        '<div style="font-size:12px;color:#4caf50;margin-top:6px;">'
                        '✓ No highly correlated pairs — portfolio appears well diversified.</div>',
                        unsafe_allow_html=True,
                    )

                # ── Average pairwise correlation (concentration gauge) ────
                upper = corr.where(
                    pd.DataFrame(
                        [[i < j for j in range(len(labels))] for i in range(len(labels))],
                        index=labels, columns=labels,
                    )
                ).stack()
                avg_corr = float(upper.mean()) if not upper.empty else 0.0
                gauge_color = (
                    COLORS.get("negative", "#f44336") if avg_corr > 0.5
                    else COLORS.get("bloomberg_orange", "#ff9800") if avg_corr > 0.25
                    else COLORS.get("positive", "#00e676")
                )
                label_text = "Concentrated" if avg_corr > 0.5 else ("Moderate" if avg_corr > 0.25 else "Diversified")

                if avg_corr > 0.5:
                    _corr_context = (
                        "Your holdings move together strongly. A market downturn is likely to hit "
                        "all positions simultaneously — consider adding uncorrelated assets (e.g. bonds, "
                        "gold, inverse ETFs) or reducing position count."
                    )
                elif avg_corr > 0.25:
                    """Moderate correlation — some diversification benefit but meaningful overlap. """
                    _corr_context = (
                        "Some diversification benefit, but your holdings share moderate common exposure. "
                        "Watch for sector concentration (e.g. all tech) or macro factor overlap "
                        "(e.g. all rate-sensitive). Consider whether any single macro shock "
                        "could hit multiple positions at once."
                    )
                else:
                    _r2 = avg_corr ** 2 * 100  # % variance shared
                    _near_boundary = avg_corr >= 0.20
                    _boundary_note = (
                        f" Note: at {avg_corr:+.2f} you are close to the Moderate threshold (0.25) — "
                        f"adding one more correlated position could push you into that band."
                        if _near_boundary else ""
                    )
                    _corr_context = (
                        f"Your holdings share only {_r2:.1f}% of their daily return variance on average "
                        f"(R\u00b2 = \u03c1\u00b2 = {avg_corr:.2f}\u00b2). "
                        f"In practice this means roughly {100 - _r2:.0f}% of each position's daily move "
                        f"is driven by its own story, not the rest of your portfolio. "
                        f"This reduces overall portfolio volatility compared to a concentrated book — "
                        f"a single bad position is unlikely to drag everything down simultaneously.{_boundary_note}"
                    )

                st.markdown(
                    f'<div style="font-size:12px;color:{COLORS["text_dim"]};margin-top:8px;">'
                    f'Avg pairwise ρ: <b style="color:{gauge_color};">{avg_corr:+.2f}</b>'
                    f' — <span style="color:{gauge_color};font-weight:700;">{label_text}</span>'
                    f'<br><span style="font-size:11px;color:#888;">{_corr_context}</span></div>',
                    unsafe_allow_html=True,
                )

        # ── Equity Curve + Signal Breakdown (closed trades only) ─────────────
        if not closed_trades:
            st.info("Close some trades to see the equity curve and signal breakdown.")
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
