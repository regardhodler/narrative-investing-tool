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


@st.cache_data(ttl=3600)
def _fetch_risk_prices(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    """Fetch daily close prices for risk calculations."""
    import yfinance as yf
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(list(tickers), period=period, interval="1d",
                          progress=False, auto_adjust=True, threads=True)
        if raw is None or raw.empty:
            return pd.DataFrame()
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        closes.index = pd.to_datetime(closes.index).normalize()
        return closes.ffill().dropna(how="all")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400)
def _fetch_sector(ticker: str) -> str:
    """Fetch sector from yfinance .info. Returns 'Unknown' on failure."""
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or info.get("industryDisp") or "Unknown"
    except Exception:
        return "Unknown"


def _render_risk_matrix(open_trades: list):
    """Render the portfolio risk matrix tab."""
    import numpy as np
    import plotly.graph_objects as go
    from utils.theme import apply_dark_layout

    if not open_trades:
        st.info("No open positions. Add trades in the Open Positions tab.")
        return

    tickers = tuple(sorted({t["ticker"].upper() for t in open_trades}))
    bench = "SPY"

    with st.spinner("Loading risk data…"):
        prices = _fetch_risk_prices(tickers + (bench,), period="1y")

    if prices.empty:
        st.warning("Could not fetch price history for risk calculations.")
        return

    # ── Daily returns ────────────────────────────────────────────────────────
    returns = prices.pct_change().dropna(how="all")
    port_tickers = [t for t in tickers if t in returns.columns]
    if not port_tickers:
        st.warning("No return data available for open positions.")
        return

    # ── Position weights by market value ────────────────────────────────────
    prices_latest = {
        t: float(prices[t].dropna().iloc[-1]) if t in prices.columns else 0
        for t in port_tickers
    }
    position_values = {}
    for trade in open_trades:
        tk = trade["ticker"].upper()
        if tk in port_tickers:
            px = prices_latest.get(tk, trade.get("entry_price", 0))
            position_values[tk] = position_values.get(tk, 0) + px * trade.get("position_size", 0)

    total_val = sum(position_values.values())
    weights = {tk: v / total_val for tk, v in position_values.items()} if total_val > 0 else {tk: 1/len(port_tickers) for tk in port_tickers}

    # ── Portfolio daily return ────────────────────────────────────────────────
    port_ret = sum(
        returns[tk] * weights.get(tk, 0)
        for tk in port_tickers if tk in returns.columns
    )

    # ── VaR (95% and 99% — historical simulation) ───────────────────────────
    port_ret_clean = port_ret.dropna()
    var_95 = float(np.percentile(port_ret_clean, 5)) * 100 if len(port_ret_clean) > 20 else None
    var_99 = float(np.percentile(port_ret_clean, 1)) * 100 if len(port_ret_clean) > 20 else None
    var_95_dollar = var_95 / 100 * total_val if var_95 and total_val else None
    cvar_95 = float(port_ret_clean[port_ret_clean <= np.percentile(port_ret_clean, 5)].mean()) * 100 if len(port_ret_clean) > 20 else None

    # ── Beta per position ─────────────────────────────────────────────────────
    spy_ret = returns[bench].dropna() if bench in returns.columns else None
    betas = {}
    if spy_ret is not None:
        for tk in port_tickers:
            if tk in returns.columns:
                aligned = pd.concat([returns[tk], spy_ret], axis=1).dropna()
                if len(aligned) > 20:
                    cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                    betas[tk] = round(cov[0, 1] / cov[1, 1], 2) if cov[1, 1] != 0 else 1.0
    port_beta = sum(betas.get(tk, 1.0) * weights.get(tk, 0) for tk in port_tickers)

    # ── Sector concentration ──────────────────────────────────────────────────
    sectors = {tk: _fetch_sector(tk) for tk in port_tickers}
    sector_weights: dict[str, float] = {}
    for tk, w in weights.items():
        s = sectors.get(tk, "Unknown")
        sector_weights[s] = sector_weights.get(s, 0) + w

    # ── Correlation matrix ────────────────────────────────────────────────────
    ret_df = returns[[t for t in port_tickers if t in returns.columns]].dropna()
    corr = ret_df.corr() if len(ret_df.columns) > 1 else None

    # ─────────────────────────────────────────────────────────────────────────
    # DISPLAY
    # ─────────────────────────────────────────────────────────────────────────

    # ── Risk summary metrics ──────────────────────────────────────────────────
    m_cols = st.columns(5)
    def _risk_metric(col, label, val, color="#ccc", sub=""):
        col.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;padding:10px 14px;text-align:center;">'
            f'<div style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;margin-bottom:4px;">{label}</div>'
            f'<div style="font-size:18px;font-weight:700;color:{color};">{val}</div>'
            f'{"<div style=font-size:11px;color:#64748b;margin-top:2px;>" + sub + "</div>" if sub else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

    beta_color = "#ef4444" if port_beta > 1.5 else ("#f59e0b" if port_beta > 1.1 else "#22c55e")
    var_color  = "#ef4444" if var_95 and var_95 < -3 else ("#f59e0b" if var_95 and var_95 < -1.5 else "#22c55e")
    max_wt     = max(weights.values()) * 100 if weights else 0
    conc_color = "#ef4444" if max_wt > 40 else ("#f59e0b" if max_wt > 25 else "#22c55e")

    _risk_metric(m_cols[0], "Portfolio Beta", f"{port_beta:.2f}", beta_color, f"vs {bench}")
    _risk_metric(m_cols[1], "VaR 95% (1-day)", f"{var_95:.1f}%" if var_95 else "—", var_color,
                 f"${var_95_dollar:,.0f}" if var_95_dollar else "")
    _risk_metric(m_cols[2], "CVaR 95%", f"{cvar_95:.1f}%" if cvar_95 else "—", "#ef4444", "expected tail loss")
    _risk_metric(m_cols[3], "Max Position Wt", f"{max_wt:.0f}%", conc_color,
                 max(weights, key=weights.get) if weights else "")
    _risk_metric(m_cols[4], "Positions", str(len(port_tickers)), "#94a3b8",
                 f"${total_val:,.0f} total" if total_val else "")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Correlation heatmap + Sector pie ─────────────────────────────────────
    col_corr, col_sector = st.columns([2, 1])

    with col_corr:
        if corr is not None and len(corr) > 1:
            fig_corr = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=[
                    [0.0,  "#1a4a8a"],
                    [0.5,  "#1e293b"],
                    [1.0,  "#8a1a1a"],
                ],
                zmin=-1, zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr.values],
                texttemplate="%{text}",
                textfont={"size": 11},
                showscale=True,
                colorbar=dict(thickness=12, len=0.8),
            ))
            apply_dark_layout(fig_corr, title="Position Correlation Matrix", height=320)
            fig_corr.update_layout(margin=dict(l=40, r=20, t=40, b=40))
            st.plotly_chart(fig_corr, use_container_width=True)
        elif len(port_tickers) == 1:
            st.info("Add more positions to see a correlation matrix.")
        else:
            st.warning("Not enough price history for correlation matrix.")

    with col_sector:
        if sector_weights:
            sorted_sectors = dict(sorted(sector_weights.items(), key=lambda x: x[1], reverse=True))
            sector_colors = [
                "#FF8811", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7",
                "#ef4444", "#06b6d4", "#84cc16", "#f97316", "#64748b",
            ]
            fig_pie = go.Figure(go.Pie(
                labels=list(sorted_sectors.keys()),
                values=[v * 100 for v in sorted_sectors.values()],
                marker_colors=sector_colors[:len(sorted_sectors)],
                textinfo="label+percent",
                textfont=dict(size=11),
                hole=0.4,
            ))
            apply_dark_layout(fig_pie, title="Sector Concentration", height=320)
            fig_pie.update_layout(showlegend=False, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Per-position risk table ───────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin:8px 0 6px 0;">PER-POSITION RISK BREAKDOWN</div>',
        unsafe_allow_html=True,
    )

    rows_html = ""
    for tk in sorted(port_tickers, key=lambda t: weights.get(t, 0), reverse=True):
        w     = weights.get(tk, 0) * 100
        beta  = betas.get(tk, None)
        sec   = sectors.get(tk, "—")
        val   = position_values.get(tk, 0)
        tk_ret = returns[tk].dropna() if tk in returns.columns else pd.Series()
        tk_var = float(np.percentile(tk_ret, 5)) * 100 if len(tk_ret) > 20 else None
        w_color = "#ef4444" if w > 40 else ("#f59e0b" if w > 25 else "#22c55e")
        b_color = "#ef4444" if beta and beta > 1.5 else ("#f59e0b" if beta and beta > 1.1 else "#22c55e")

        # Direction of each trade
        dirs = [t["direction"].upper() for t in open_trades if t["ticker"].upper() == tk]
        dir_str = "/".join(sorted(set(dirs)))

        rows_html += (
            f'<tr style="border-bottom:1px solid {COLORS["border"]}22;">'
            f'<td style="padding:6px 10px;color:{COLORS["bloomberg_orange"]};font-weight:700;">{tk}</td>'
            f'<td style="padding:6px 10px;color:#888;font-size:11px;">{dir_str}</td>'
            f'<td style="padding:6px 10px;color:{w_color};font-weight:600;">{w:.1f}%</td>'
            f'<td style="padding:6px 10px;color:#ccc;">${val:,.0f}</td>'
            f'<td style="padding:6px 10px;color:{b_color};">{f"{beta:.2f}" if beta is not None else "—"}</td>'
            f'<td style="padding:6px 10px;color:#ef4444;">{f"{tk_var:.1f}%" if tk_var else "—"}</td>'
            f'<td style="padding:6px 10px;color:#64748b;font-size:11px;">{sec}</td>'
            f'</tr>'
        )

    st.markdown(
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;overflow:auto;">'
        f'<table style="width:100%;border-collapse:collapse;">'
        f'<thead><tr style="border-bottom:1px solid {COLORS["border"]};">'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">TICKER</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">DIR</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">PORT WT</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">MKT VALUE</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">BETA</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">VaR 95%</th>'
        f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">SECTOR</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>',
        unsafe_allow_html=True,
    )

    # ── Diversification ratio ─────────────────────────────────────────────────
    if len(port_tickers) > 1 and var_95 and len(port_ret_clean) > 20:
        indiv_vars = []
        for tk in port_tickers:
            if tk in returns.columns:
                tk_ret = returns[tk].dropna()
                if len(tk_ret) > 20:
                    indiv_vars.append(abs(float(np.percentile(tk_ret, 5))) * weights.get(tk, 0))
        wsum_var = sum(indiv_vars)
        port_var_abs = abs(var_95 / 100)
        div_ratio = round(wsum_var / port_var_abs, 2) if port_var_abs > 0 else None
        if div_ratio:
            div_color = "#22c55e" if div_ratio >= 1.2 else ("#f59e0b" if div_ratio >= 1.05 else "#ef4444")
            div_label = "Good" if div_ratio >= 1.2 else ("Moderate" if div_ratio >= 1.05 else "Low")
            st.markdown(
                f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
                f'border-radius:6px;padding:10px 14px;margin:10px 0 4px 0;display:flex;align-items:center;gap:16px;">'
                f'<span style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;">DIVERSIFICATION RATIO</span>'
                f'<span style="font-size:18px;font-weight:700;color:{div_color};">{div_ratio}×</span>'
                f'<span style="font-size:11px;color:{div_color};">{div_label}</span>'
                f'<span style="font-size:11px;color:#475569;">— sum of individual VaRs ÷ portfolio VaR '
                f'(higher = more diversification benefit from low correlation)</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Stress test scenarios ─────────────────────────────────────────────────
    if total_val and betas:
        st.markdown(
            f'<div style="font-size:12px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;margin:14px 0 6px 0;">STRESS TEST SCENARIOS</div>',
            unsafe_allow_html=True,
        )
        _stress_scenarios = [
            ("2008 GFC",         -0.57, "SPY peak-to-trough Oct 07 – Mar 09"),
            ("COVID Crash",      -0.34, "SPY Feb 19 – Mar 23, 2020 (33 days)"),
            ("2022 Rate Shock",  -0.25, "SPY Jan – Oct 2022, Fed 0→4.5%"),
            ("Flash Crash",      -0.10, "Intraday -10% shock (1-day stress)"),
            ("VIX Spike +200%",  -0.15, "Hypothetical vol regime shift"),
        ]
        _s_rows = ""
        for _s_label, _spy_shock, _s_desc in _stress_scenarios:
            # Beta-adjusted loss per position, weighted by portfolio weight
            # Longs lose beta × shock; Shorts gain beta × shock
            _port_impact = 0.0
            for trade in open_trades:
                tk = trade["ticker"].upper()
                if tk not in port_tickers:
                    continue
                _w = weights.get(tk, 0)
                _b = betas.get(tk, 1.0)
                _dir = trade.get("direction", "long").lower()
                _sign = 1 if _dir == "long" else -1
                _port_impact += _w * _b * _spy_shock * _sign
            _loss_pct = _port_impact * 100
            _loss_dollar = _port_impact * total_val
            _s_color = "#ef4444" if _loss_pct < -15 else ("#f59e0b" if _loss_pct < -5 else "#22c55e")
            _s_rows += (
                f'<tr>'
                f'<td style="padding:6px 12px;font-weight:700;color:#e2e8f0;">{_s_label}</td>'
                f'<td style="padding:6px 12px;color:#64748b;font-size:10px;">{_s_desc}</td>'
                f'<td style="padding:6px 12px;text-align:right;color:#94a3b8;">{_spy_shock*100:+.0f}%</td>'
                f'<td style="padding:6px 12px;text-align:right;font-weight:700;color:{_s_color};">{_loss_pct:+.1f}%</td>'
                f'<td style="padding:6px 12px;text-align:right;font-weight:700;color:{_s_color};">'
                f'{"−" if _loss_dollar < 0 else "+"}${abs(_loss_dollar):,.0f}</td>'
                f'</tr>'
            )
        _hdr = "padding:6px 12px;font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.06em;"
        st.markdown(
            f'<div style="border:1px solid {COLORS["border"]};border-radius:6px;overflow:hidden;margin-bottom:10px;">'
            f'<table style="width:100%;border-collapse:collapse;font-size:12px;font-family:monospace;">'
            f'<thead><tr style="background:{COLORS["surface"]};">'
            f'<th style="{_hdr};text-align:left;">SCENARIO</th>'
            f'<th style="{_hdr};text-align:left;">CONTEXT</th>'
            f'<th style="{_hdr};text-align:right;">SPY SHOCK</th>'
            f'<th style="{_hdr};text-align:right;">PORT IMPACT</th>'
            f'<th style="{_hdr};text-align:right;">$ P&L</th>'
            f'</tr></thead>'
            f'<tbody>{_s_rows}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )
        st.caption("Beta-adjusted estimates. Longs lose β × shock; shorts gain β × shock. Assumes beta stability — actual losses may differ significantly in tail events.")

    # ── Risk flags ────────────────────────────────────────────────────────────
    flags = []
    if port_beta > 1.4:
        flags.append(f"⚠ High portfolio beta ({port_beta:.2f}) — amplified SPY moves in both directions")
    if max_wt > 40:
        top_tk = max(weights, key=weights.get)
        flags.append(f"⚠ Concentration risk: {top_tk} is {max_wt:.0f}% of portfolio")
    if corr is not None and len(corr) > 1:
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = [(c, r) for c in corr.columns for r in corr.index
                           if r != c and upper.loc[r, c] > 0.75 if pd.notna(upper.loc[r, c] if hasattr(upper, 'loc') else 0)]
        if high_corr_pairs[:1]:
            pair = high_corr_pairs[0]
            v = corr.loc[pair[1], pair[0]]
            flags.append(f"⚠ High correlation: {pair[0]} / {pair[1]} ({v:.2f}) — limited diversification benefit")
    if len(sector_weights) == 1:
        flags.append(f"⚠ All positions in one sector ({list(sector_weights.keys())[0]}) — zero sector diversification")

    if flags:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        for flag in flags:
            st.markdown(
                f'<div style="background:#2d1a0a;border-left:3px solid #f59e0b;'
                f'border-radius:0 4px 4px 0;padding:8px 14px;margin-bottom:6px;'
                f'font-size:12px;color:#fbbf24;">{flag}</div>',
                unsafe_allow_html=True,
            )


def _regime_badge(direction: str, regime: str) -> tuple:
    """Return (badge_text, color) for a position direction × current regime."""
    if not regime:
        return "⚪ No Regime", "#64748b"
    is_risk_off = "Risk-Off" in regime
    if direction.lower() == "long":
        return ("❌ Misaligned", "#ef4444") if is_risk_off else ("✅ Aligned", "#22c55e")
    else:
        return ("✅ Aligned", "#22c55e") if is_risk_off else ("❌ Misaligned", "#ef4444")


def render():
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.1em;margin-bottom:12px;">MY REGARDED PORTFOLIO</div>',
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
    tab_open, tab_intel, tab_risk, tab_closed, tab_analytics, tab_perf, tab_playlog = st.tabs(["OPEN POSITIONS", "🧠 PORTFOLIO INTELLIGENCE", "⚠ RISK MATRIX", "CLOSED TRADES", "ANALYTICS", "📈 PERFORMANCE", "📋 AI PLAY LOG"])

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

            _cur_regime = st.session_state.get("_regime_context", {}).get("regime", "")

            # ── Portfolio Allocation Chart ─────────────────────────────────
            if total_portfolio_val > 0:
                import plotly.graph_objects as go
                from utils.theme import apply_dark_layout

                # Aggregate by ticker (multiple tranches → combined weight)
                _alloc: dict[str, dict] = {}
                for _t in open_trades:
                    _tk = _t["ticker"]
                    _cur = prices.get(_tk) or _t["entry_price"]
                    _val = _cur * _t["position_size"]
                    _pnl = ((_cur - _t["entry_price"]) * _t["position_size"]
                            if _t["direction"] == "Long"
                            else (_t["entry_price"] - _cur) * _t["position_size"])
                    if _tk not in _alloc:
                        _alloc[_tk] = {"val": 0.0, "pnl": 0.0, "direction": _t["direction"]}
                    _alloc[_tk]["val"] += _val
                    _alloc[_tk]["pnl"] += _pnl

                _alloc_sorted = sorted(_alloc.items(), key=lambda x: x[1]["val"], reverse=True)
                _tks   = [a[0] for a in _alloc_sorted]
                _wts   = [a[1]["val"] / total_portfolio_val * 100 for a in _alloc_sorted]
                _pnls  = [a[1]["pnl"] for a in _alloc_sorted]
                _dirs  = [a[1]["direction"] for a in _alloc_sorted]
                _bar_colors = [
                    COLORS["positive"] if p >= 0 else COLORS["negative"]
                    for p in _pnls
                ]
                _text_labels = [
                    f"{w:.1f}%  {'+' if p >= 0 else ''}${p:,.0f}"
                    for w, p in zip(_wts, _pnls)
                ]

                _fig_alloc = go.Figure(go.Bar(
                    x=_wts, y=_tks,
                    orientation="h",
                    marker_color=_bar_colors,
                    text=_text_labels,
                    textposition="outside",
                    textfont=dict(size=10, color=COLORS["text_dim"]),
                    hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
                ))
                apply_dark_layout(_fig_alloc)
                _fig_alloc.update_layout(
                    height=max(160, len(_tks) * 28 + 40),
                    margin=dict(l=0, r=80, t=24, b=0),
                    xaxis=dict(title="Portfolio Weight %", ticksuffix="%",
                               showgrid=True, gridcolor=COLORS["border"]),
                    yaxis=dict(autorange="reversed"),
                    title=dict(text="PORTFOLIO ALLOCATION",
                               font=dict(size=11, color=COLORS["bloomberg_orange"]),
                               x=0, xanchor="left"),
                    showlegend=False,
                )
                # Concentration warning line at 15%
                _fig_alloc.add_vline(x=15, line_dash="dot",
                                     line_color=COLORS["bloomberg_orange"] + "66",
                                     annotation_text="15% cap",
                                     annotation_font_size=9,
                                     annotation_font_color=COLORS["bloomberg_orange"])

                with st.expander("📊 Allocation Chart", expanded=True):
                    st.plotly_chart(_fig_alloc, use_container_width=True)

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
                _badge_text, _badge_color = _regime_badge(direction, _cur_regime)
                with col1:
                    _earn_badge = ""
                    try:
                        from services.market_data import fetch_earnings_date
                        _earn_info = fetch_earnings_date(trade["ticker"])
                        if _earn_info:
                            _ed = _earn_info["days_away"]
                            _ec = "#ef4444" if _ed <= 3 else ("#f59e0b" if _ed <= 14 else "#64748b")
                            _earn_badge = (
                                f' <span style="background:#1e293b;border:1px solid {_ec};border-radius:3px;'
                                f'padding:1px 6px;font-size:10px;color:{_ec};">'
                                f'{"⚠" if _ed <= 7 else "📅"} {_earn_info["date"]} ({_ed}d)</span>'
                            )
                    except Exception:
                        pass
                    st.markdown(
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;">{trade["ticker"]}</span>'
                        f' <span style="color:{COLORS["text_dim"]};font-size:11px;">{direction} · {trade["signal_source"]}'
                        f' · {trade["entry_date"]}</span>'
                        f' <span style="color:{_badge_color};font-size:11px;font-weight:600;margin-left:6px;">{_badge_text}</span>'
                        f'{_earn_badge}',
                        unsafe_allow_html=True,
                    )
                    if trade.get("regime_at_entry") and _cur_regime and trade["regime_at_entry"] != _cur_regime:
                        st.caption(f"⚠ Entered in {trade['regime_at_entry']} · now {_cur_regime}")
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

    # --- Portfolio Intelligence ---
    with tab_intel:
        from datetime import datetime as _dt
        from services.claude_client import analyze_portfolio
        from services.play_log import append_play

        # ── Regime Strip (always visible) ─────────────────────────────────────
        import json as _json_ri, os as _os_ri
        _rc_pi = st.session_state.get("_regime_context") or {}
        _quadrant_pi = _rc_pi.get("quadrant", "")
        _raw_score_pi = _rc_pi.get("score", 0)
        if isinstance(_raw_score_pi, float) and -1.0 <= _raw_score_pi <= 1.0:
            _score100_pi = int((_raw_score_pi + 1.0) * 50.0)
        else:
            _score100_pi = int(_raw_score_pi)
        _REGIME_HIST_PI = _os_ri.path.join(
            _os_ri.path.dirname(_os_ri.path.dirname(__file__)), "data", "regime_history.json"
        )
        _stability_pi = 0
        try:
            with open(_REGIME_HIST_PI) as _f_ri:
                _hist_pi = _json_ri.load(_f_ri)
            if _hist_pi:
                _hist_pi = sorted(_hist_pi, key=lambda r: r.get("date", ""))
                _cur_q_pi = _hist_pi[-1].get("quadrant", _hist_pi[-1].get("regime", ""))
                for _hr_pi in reversed(_hist_pi):
                    _hq_pi = _hr_pi.get("quadrant", _hr_pi.get("regime", ""))
                    if _hq_pi == _cur_q_pi:
                        _stability_pi += 1
                    else:
                        break
        except Exception:
            pass
        _qcolors_pi = {
            "Stagflation": "#a855f7", "Goldilocks": "#22c55e",
            "Reflation": "#f59e0b",   "Deflation":  "#3b82f6",
        }
        _qc_pi = _qcolors_pi.get(_quadrant_pi, "#888")
        _stab_c_pi = "#22c55e" if _stability_pi >= 10 else ("#f59e0b" if _stability_pi >= 5 else "#94a3b8")
        if _quadrant_pi or _rc_pi:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap;'
                f'background:#0A1520;border:1px solid #1E3A4A;border-radius:6px;'
                f'padding:8px 16px;margin-bottom:12px;">'
                f'<span style="font-size:10px;color:{COLORS["bloomberg_orange"]};'
                f'font-weight:700;letter-spacing:0.08em;">MACRO REGIME</span>'
                f'<span style="color:{_qc_pi};font-weight:700;font-size:13px;">'
                f'{_quadrant_pi or "—"}</span>'
                f'<span style="color:#888;font-size:11px;">Score {_score100_pi}/100</span>'
                f'<span style="color:{_stab_c_pi};font-size:11px;">'
                f'· Stable {_stability_pi} session{"s" if _stability_pi != 1 else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("Run Risk Regime → Regime Overview to load macro context, then return here for sizing guidance.")

        # Section A — Context freshness panel
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">CONTEXT FRESHNESS</div>',
            unsafe_allow_html=True,
        )
        _ctx_items = [
            ("Regime", "_regime_context_ts"),
            ("Rate Path", "_rate_path_probs_ts"),
            ("Doom Briefing", "_doom_briefing_ts"),
            ("Black Swans", "_custom_swans_ts"),
            ("Whale Summary", "_whale_summary_ts"),
            ("Fed Plays", "_fed_plays_result_ts"),
            ("Current Events", "_current_events_digest_ts"),
        ]
        _ctx_cols = st.columns(len(_ctx_items))
        _now = _dt.now()
        _missing_critical = []
        for _ci, (_label, _ts_key) in enumerate(_ctx_items):
            _ts = st.session_state.get(_ts_key)
            if _ts is None:
                _icon = "✗"
                _color = "#ef4444"
                _age_str = "missing"
                if _label in ("Regime", "Rate Path"):
                    _missing_critical.append(_label)
            else:
                _age_min = (_now - _ts).total_seconds() / 60
                if _age_min < 120:
                    _icon = "✅"
                    _color = "#22c55e"
                    _age_str = f"{int(_age_min)}m ago"
                elif _age_min < 360:
                    _icon = "⚠"
                    _color = "#f59e0b"
                    _age_str = f"{int(_age_min // 60)}h ago"
                else:
                    _icon = "✗"
                    _color = "#ef4444"
                    _age_str = f"{int(_age_min // 60)}h ago"
                    if _label in ("Regime", "Rate Path"):
                        _missing_critical.append(_label)
            _ctx_cols[_ci].markdown(
                f'<div style="text-align:center;font-size:11px;">'
                f'<div style="color:{_color};font-size:16px;">{_icon}</div>'
                f'<div style="color:#ccc;font-weight:600;">{_label}</div>'
                f'<div style="color:#888;">{_age_str}</div></div>',
                unsafe_allow_html=True,
            )
        # Also flag missing presence (not just stale timestamps)
        for _k, _lbl in [("_regime_context", "Regime"), ("_rate_path_probs", "Rate Path")]:
            if not st.session_state.get(_k) and _lbl not in _missing_critical:
                _missing_critical.append(_lbl)
        if _missing_critical:
            st.markdown(
                f'<div style="background:#1a0d00;border:1px solid #f59e0b55;border-radius:6px;'
                f'padding:8px 14px;margin-bottom:8px;font-size:11px;">'
                f'<span style="color:#f59e0b;font-weight:700;">⚠ Data Quality</span>'
                f'<span style="color:#94a3b8;margin-left:8px;"><b>Missing / stale:</b> {", ".join(_missing_critical)}</span>'
                f'<span style="color:#64748b;margin-left:8px;">— run ⚡ Quick Intel Run to refresh</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)

        # Section B — Engine selector + Run button
        st.markdown(
            f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
            f'letter-spacing:0.08em;margin-bottom:8px;">PORTFOLIO ANALYSIS ENGINE</div>',
            unsafe_allow_html=True,
        )
        import os as _os
        _pi_has_claude = bool(_os.getenv("ANTHROPIC_API_KEY"))
        _pi_tier_opts = ["⚡ Groq"]
        if _pi_has_claude:
            _pi_tier_opts += ["🧠 Regard Mode", "👑 Highly Regarded Mode"]
        _sel_pi_tier = st.radio("Analysis Engine", _pi_tier_opts, horizontal=True, key="portfolio_intel_engine")
        st.caption("💡 👑 Highly Regarded Mode strongly recommended — this is the highest-stakes synthesis task (actual position decisions)")
        if not _pi_has_claude:
            st.markdown(
                '<div style="background:#1a1200;border:1px solid #f59e0b44;border-radius:4px;'
                'padding:6px 12px;font-size:10px;color:#f59e0b;margin-bottom:4px;">'
                '🔒 <b>Regard / Highly Regarded Mode unavailable</b> — set ANTHROPIC_API_KEY in Streamlit secrets to unlock Claude analysis</div>',
                unsafe_allow_html=True,
            )

        if not open_trades:
            st.info("No open positions to analyze.")
        else:
            if st.button("🧠 Run Portfolio Analysis", type="primary", key="run_portfolio_intel"):
                # Assemble upstream context from session_state
                _rc = st.session_state.get("_regime_context") or {}
                _rp_probs = st.session_state.get("_rate_path_probs") or []
                # Find dominant rate path (list of {scenario, prob} dicts)
                _dominant_rp = {}
                if _rp_probs:
                    _dom = max(_rp_probs, key=lambda r: r.get("prob", 0), default=None)
                    if _dom:
                        _dominant_rp = {
                            "scenario": _dom.get("scenario", ""),
                            "prob_pct": round(_dom.get("prob", 0) * 100),
                        }
                _rp_plays = st.session_state.get("_rp_plays_result") or {}
                _fed_plays = st.session_state.get("_fed_plays_result") or {}
                # Compute factor/sizing context to enrich AI prompt
                _up_factor_exposure = {}
                _up_sizing_scores = {}
                try:
                    from services.portfolio_sizing import score_portfolio as _sp_up, aggregate_factor_exposure as _afe_up
                    _up_live = _get_live_prices([t["ticker"] for t in open_trades])
                    _up_pv = sum(
                        (_up_live.get(t["ticker"], t["entry_price"]) * t["position_size"])
                        for t in open_trades
                    )
                    if _up_pv > 0:
                        _up_sz = _sp_up(open_trades, _rc, _up_pv, _up_live)
                        _up_sz_map = {p["ticker"]: p for p in _up_sz["positions"]}
                        _up_fe = _afe_up(_up_sz_map.values())
                        _up_factor_exposure = _up_fe.get("factors", {})
                        _up_sizing_scores = {
                            tk: {
                                "score": p.get("composite_score"),
                                "regime_fit": p.get("regime_fit"),
                                "weight": round(p.get("current_weight", 0), 1),
                            }
                            for tk, p in _up_sz_map.items()
                        }
                except Exception:
                    pass  # enrichment is additive — skip silently if sizing unavailable
                _upstream = {
                    "regime": _rc.get("regime", ""),
                    "score": _rc.get("score", 0.0),
                    "quadrant": _rc.get("quadrant", ""),
                    "signal_summary": _rc.get("signal_summary", ""),
                    "dominant_rate_path": _dominant_rp,
                    "rate_path_probs": _rp_probs,
                    "fed_funds_rate": st.session_state.get("_fed_funds_rate", "Unknown"),
                    "chain_narration": st.session_state.get("_chain_narration", ""),
                    "custom_swans": st.session_state.get("_custom_swans") or {},
                    "doom_briefing": st.session_state.get("_doom_briefing", ""),
                    "whale_summary": st.session_state.get("_whale_summary", ""),
                    "regime_plays": _rp_plays,
                    "fed_plays": _fed_plays,
                    "discovery_plays": st.session_state.get("_plays_result") or {},
                    "factor_exposure": _up_factor_exposure,
                    "sizing_scores": _up_sizing_scores,
                    "current_events": (
                        st.session_state.get("_current_events_digest", "") or
                        ""
                    ),
                }
                _use_claude = _sel_pi_tier in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
                _pi_model = None
                if _sel_pi_tier == "🧠 Regard Mode":
                    _pi_model = "claude-haiku-4-5-20251001"
                elif _sel_pi_tier == "👑 Highly Regarded Mode":
                    _pi_model = "claude-sonnet-4-6"
                with st.spinner("Analyzing portfolio against macro conditions..."):
                    _pi_result = analyze_portfolio(open_trades, _upstream, use_claude=_use_claude, model=_pi_model)
                if _pi_result and "_error" not in _pi_result:
                    st.session_state["_portfolio_analysis"] = _pi_result
                    st.session_state["_portfolio_analysis_ts"] = _dt.now()
                    st.session_state["_portfolio_analysis_engine"] = _sel_pi_tier
                    append_play("Portfolio Analysis", _sel_pi_tier, _pi_result,
                                meta={"n_positions": len(open_trades), "verdict": _pi_result.get("verdict")})
                    # Telegram alert for high-risk verdicts
                    _verdict_val = _pi_result.get("verdict", "")
                    if _verdict_val in ("DEFENSIVE", "EXIT_REVIEW"):
                        try:
                            from services.telegram_client import send_alert as _tg_alert
                            _vl = {"DEFENSIVE": "🛡 DEFENSIVE", "EXIT_REVIEW": "🚨 EXIT REVIEW"}.get(_verdict_val, _verdict_val)
                            _rs = _pi_result.get("risk_score", "?")
                            _narr = (_pi_result.get("narrative") or "")[:200]
                            _tg_alert(
                                f"<b>PORTFOLIO ALERT: {_vl}</b>\n"
                                f"Risk Score: {_rs}/10\n\n"
                                f"{_narr}"
                            )
                        except Exception:
                            pass
                    st.rerun()
                else:
                    _err = (_pi_result or {}).get("_error", "Unknown error")
                    st.error(f"Analysis failed: {_err}")
                    if _use_claude and not _pi_has_claude:
                        st.warning("⚠ ANTHROPIC_API_KEY not found — Claude modes require this key in Streamlit secrets.")
                    with st.expander("🔍 Debug — raw result", expanded=False):
                        st.json(_pi_result or {"result": "None returned"})

        # Section C — Portfolio Verdict card
        _pa = st.session_state.get("_portfolio_analysis")
        _pa_ts = st.session_state.get("_portfolio_analysis_ts")
        _pa_engine = st.session_state.get("_portfolio_analysis_engine", "")
        if _pa and "_error" in _pa:
            st.error(f"Cached analysis has error: {_pa['_error']}")
            with st.expander("🔍 Debug — cached result", expanded=False):
                st.json(_pa)
        elif _pa and "_error" not in _pa:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:12px 0;"></div>', unsafe_allow_html=True)
            _verdict = _pa.get("verdict", "UNKNOWN")
            _risk_score = _pa.get("risk_score", 0)
            _narrative = _pa.get("narrative", "")
            _verdict_colors = {
                "HOLD_ALL": "#22c55e",
                "REDUCE_RISK": "#f59e0b",
                "DEFENSIVE": "#FF8811",
                "EXIT_REVIEW": "#ef4444",
            }
            _verdict_color = _verdict_colors.get(_verdict, "#888")
            _verdict_labels = {
                "HOLD_ALL": "✅ HOLD ALL",
                "REDUCE_RISK": "⚠ REDUCE RISK",
                "DEFENSIVE": "🛡 DEFENSIVE",
                "EXIT_REVIEW": "🚨 EXIT REVIEW",
            }
            _verdict_label = _verdict_labels.get(_verdict, _verdict)
            _age_str = ""
            if _pa_ts:
                _age_m = int((_dt.now() - _pa_ts).total_seconds() / 60)
                _age_str = f"{_age_m}m ago"
            _is_claude_engine = _pa_engine in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
            _claude_badge_html = (
                f'<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);'
                f'color:#fff;font-size:10px;font-weight:700;letter-spacing:0.06em;'
                f'padding:2px 8px;border-radius:3px;margin-left:8px;">✦ POWERED BY CLAUDE</span>'
            ) if _is_claude_engine else ""
            st.markdown(
                f'<div style="background:#1a1a1a;border:1px solid {COLORS["border"]};border-radius:6px;padding:14px 18px;margin-bottom:10px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
                f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:13px;letter-spacing:0.08em;">'
                f'PORTFOLIO RISK ASSESSMENT{_claude_badge_html}</span>'
                f'<span style="color:#888;font-size:11px;">{_pa_engine} · {_age_str}</span>'
                f'</div>'
                f'<div style="display:flex;align-items:center;gap:24px;margin-bottom:8px;">'
                f'<span style="color:{_verdict_color};font-size:18px;font-weight:700;">{_verdict_label}</span>'
                f'<span style="color:#ccc;font-size:14px;">Risk Score: <b style="color:{_verdict_color};">{_risk_score}/10</b></span>'
                f'</div>'
                f'<div style="color:#bbb;font-size:12px;line-height:1.6;">{_narrative or "<i style=\'color:#555\'>No narrative generated — re-run analysis.</i>"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _priority_actions = _pa.get("priority_actions", [])
            if _priority_actions:
                st.markdown(
                    '<div style="margin-top:6px;">'
                    + "".join(
                        f'<div style="font-size:11px;color:#f59e0b;padding:2px 0;">▸ {a}</div>'
                        for a in _priority_actions
                    )
                    + '</div>',
                    unsafe_allow_html=True,
                )

            # Section D — Per-position intelligence cards
            # Normalize ticker: Claude sometimes echoes "XTLH.TO @ $36.81" — strip price suffix
            def _norm_ticker(raw: str) -> str:
                return raw.split(" @ ")[0].split(" ")[0].upper()
            _pa_positions = {_norm_ticker(p.get("ticker", "")): p for p in _pa.get("positions", [])}
            if open_trades:
                st.markdown(
                    f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                    f'letter-spacing:0.08em;margin:12px 0 8px 0;">PER-POSITION INTELLIGENCE</div>',
                    unsafe_allow_html=True,
                )
                if not _pa_positions:
                    st.warning("⚠ Per-position data missing — AI response may have been truncated. Re-run the analysis.", icon="⚠")
                _action_colors = {"HOLD": "#22c55e", "ADD": "#22c55e", "REDUCE": "#f59e0b", "EXIT": "#ef4444"}
                # Get live prices for P&L display
                _pi_live = prices if open_trades else {}

                # ── Institutional sizing scores ────────────────────────────────
                _rc_for_sz = st.session_state.get("_regime_context") or {}
                _sz_result = {}
                _sz_summary = None
                try:
                    from services.portfolio_sizing import score_portfolio as _score_portfolio
                    _total_pv = sum(
                        (_pi_live.get(t["ticker"], t["entry_price"]) * t["position_size"])
                        for t in open_trades
                    )
                    if _total_pv > 0:
                        _sz = _score_portfolio(open_trades, _rc_for_sz, _total_pv, _pi_live)
                        _sz_result  = {p["ticker"]: p for p in _sz["positions"]}
                        _sz_summary = _sz
                except Exception as _sz_err:
                    pass  # sizing is additive — silently skip if unavailable

                for _trade in open_trades:
                    _tk = _trade["ticker"]
                    _pos_data = _pa_positions.get(_tk.upper(), {})
                    _action = _pos_data.get("action", "—")
                    _rationale = _pos_data.get("rationale", "")
                    _risk_factors = _pos_data.get("risk_factors", [])
                    _ac = _action_colors.get(_action, "#888")
                    _cur_px = _pi_live.get(_tk, _trade["entry_price"])
                    _ep = _trade["entry_price"]
                    _pnl_pct = ((_cur_px / _ep) - 1) * 100 if _ep > 0 and _trade["direction"].lower() == "long" else ((_ep / _cur_px) - 1) * 100 if _cur_px > 0 else 0
                    _pnl_color = COLORS["positive"] if _pnl_pct >= 0 else COLORS["negative"]
                    _badge_t, _badge_c = _regime_badge(_trade["direction"], _cur_regime)
                    _rf_html = "".join(f'<span style="background:#333;border-radius:3px;padding:1px 6px;margin-right:4px;font-size:10px;color:#ccc;">• {rf}</span>' for rf in _risk_factors)
                    # Macro Fit badge
                    _mf_all = st.session_state.get("_macro_fit_results", {})
                    _mf = _mf_all.get(_tk.upper())
                    _mf_html = ""
                    if _mf:
                        _mf_stars = "★" * _mf["fit_stars"] + "☆" * (5 - _mf["fit_stars"])
                        _mf_vc = {
                            "Strong Fit": "#22c55e", "Moderate Fit": "#86efac",
                            "Neutral": "#f59e0b", "Caution": "#f97316", "Avoid": "#ef4444",
                        }.get(_mf["verdict"], "#888")
                        _mf_html = (
                            f'<div style="font-size:11px;color:#888;margin-top:4px;">'
                            f'Macro Fit: <span style="color:#f59e0b;">{_mf_stars}</span>'
                            f' <span style="color:{_mf_vc};font-weight:600;">{_mf["verdict"]}</span>'
                            f'</div>'
                        )
                    # Earnings countdown badge
                    _earn_html = ""
                    try:
                        from services.market_data import fetch_earnings_date
                        _earn = fetch_earnings_date(_tk)
                        if _earn:
                            _ed = _earn["days_away"]
                            _ec = "#ef4444" if _ed <= 3 else ("#f59e0b" if _ed <= 14 else "#64748b")
                            _ebg = "#2d1010" if _ed <= 3 else ("#2d2010" if _ed <= 14 else "#1e293b")
                            _elabel = f"⚠ Earnings in {_ed}d" if _ed <= 7 else f"📅 Earnings {_earn['date']} ({_ed}d)"
                            _earn_html = (
                                f'<span style="background:{_ebg};border:1px solid {_ec};border-radius:3px;'
                                f'padding:1px 8px;font-size:10px;color:{_ec};font-weight:600;">{_elabel}</span>'
                            )
                    except Exception:
                        pass

                    # ── Sizing overlay ─────────────────────────────────────────
                    _sz_pos = _sz_result.get(_tk.upper(), {})
                    _sz_action  = _sz_pos.get("action", "")
                    _sz_score   = _sz_pos.get("composite_score")
                    _sz_cw      = _sz_pos.get("current_weight")
                    _sz_tw      = _sz_pos.get("target_weight")
                    _sz_stop    = _sz_pos.get("atr_stop")
                    _sz_add     = _sz_pos.get("add_amount")
                    _sz_add_pct = _sz_pos.get("add_pct")
                    _sz_red     = _sz_pos.get("reduce_amount")
                    _sz_red_pct = _sz_pos.get("reduce_to_pct")
                    _sz_hold    = _sz_pos.get("hold_condition")
                    _sz_rf      = _sz_pos.get("regime_fit")
                    _sz_cv      = _sz_pos.get("conviction")

                    # Build action detail string with $ amounts
                    _action_detail = ""
                    if _sz_action == "ADD" and _sz_add:
                        _action_detail = f'<b style="color:#22c55e;">ADD ${_sz_add:,} (+{_sz_add_pct:.1f}%)</b>'
                    elif _sz_action == "REDUCE" and _sz_red:
                        _action_detail = f'<b style="color:#f59e0b;">REDUCE by ${_sz_red:,} → {_sz_red_pct:.1f}% weight</b>'
                    elif _sz_action == "EXIT":
                        _action_detail = f'<b style="color:#ef4444;">EXIT — composite score too low</b>'
                    elif _sz_action == "HOLD" and _sz_hold:
                        _action_detail = f'<b style="color:#22c55e;">HOLD</b> <span style="color:#888;font-size:10px;">· {_sz_hold}</span>'

                    _sizing_html = ""
                    if _sz_pos:
                        _sc_color = "#22c55e" if (_sz_score or 0) >= 65 else ("#f59e0b" if (_sz_score or 0) >= 40 else "#ef4444")
                        _wdelta = (_sz_tw or 0) - (_sz_cw or 0)
                        _wdc = "#22c55e" if _wdelta > 0 else ("#ef4444" if _wdelta < 0 else "#888")
                        _sizing_html = (
                            f'<div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2a2a;">'
                            f'<div style="font-size:10px;color:#444;margin-bottom:4px;letter-spacing:0.06em;">SIZING MODEL (regime fit · ATR · conviction)</div>'
                            f'<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:4px;">'
                            f'<span style="background:{_sc_color}22;border:1px solid {_sc_color}55;'
                            f'border-radius:3px;padding:1px 7px;font-size:10px;color:{_sc_color};font-weight:700;">'
                            f'Score {_sz_score}</span>'
                            + (f'<span style="color:#64748b;font-size:10px;">RegimeFit {_sz_rf} · Conviction {_sz_cv}</span>' if _sz_rf is not None else '')
                            + (f'<span style="color:#555;font-size:10px;">|</span>'
                               f'<span style="color:#888;font-size:10px;">Wt: {_sz_cw:.1f}% → '
                               f'<span style="color:{_wdc};">{_sz_tw:.1f}%</span></span>'
                               if _sz_cw is not None else '')
                            + (f'<span style="color:#94a3b8;font-size:10px;">· Stop ${_sz_stop:.2f}</span>' if _sz_stop else '')
                            + f'</div>'
                            + (f'<div style="font-size:12px;margin-top:2px;">{_action_detail}</div>' if _action_detail else '')
                            + f'</div>'
                        )

                    st.markdown(
                        f'<div style="background:#1a1a1a;border:1px solid {COLORS["border"]};border-radius:6px;padding:12px 16px;margin-bottom:8px;">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">'
                        f'<span>'
                        f'<b style="color:{COLORS["bloomberg_orange"]};font-size:14px;">{_tk}</b>'
                        f' <span style="color:#888;font-size:11px;">{_trade["direction"].upper()} @ ${_ep:.2f}</span>'
                        f' <span style="color:{_pnl_color};font-size:11px;">({_pnl_pct:+.1f}%)</span>'
                        f'</span>'
                        f'<span style="color:{_badge_c};font-size:11px;font-weight:600;">{_badge_t}</span>'
                        f'</div>'
                        f'<div style="margin-bottom:6px;">'
                        f'<span style="font-size:12px;color:#888;">AI View: <b style="color:{_ac};">{_action}</b></span>'
                        f'<span style="font-size:10px;color:#444;margin-left:8px;">(narrative + macro)</span>'
                        f'</div>'
                        f'<div style="font-size:12px;color:#bbb;margin-bottom:6px;">{_rationale}</div>'
                        f'<div>{_rf_html}</div>'
                        f'{_mf_html}'
                        f'{"<div style=margin-top:6px;>" + _earn_html + "</div>" if _earn_html else ""}'
                        f'{_sizing_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # ── Rebalance Summary ──────────────────────────────────────────
                if _sz_summary:
                    _tot_add = _sz_summary.get("total_add", 0)
                    _tot_red = _sz_summary.get("total_reduce", 0)
                    _exits   = _sz_summary.get("exits", [])
                    _rb_sum  = _sz_summary.get("rebalance_summary", "")
                    _rb_color = "#ef4444" if _exits else ("#f59e0b" if _tot_red > 0 else "#22c55e")
                    st.markdown(
                        f'<div style="background:#0A1520;border:1px solid {_rb_color}44;border-radius:6px;'
                        f'padding:10px 16px;margin-top:4px;">'
                        f'<span style="color:{COLORS["bloomberg_orange"]};font-weight:700;font-size:11px;'
                        f'letter-spacing:0.08em;">REBALANCE SUMMARY</span>'
                        f'<span style="color:#555;margin:0 8px;">·</span>'
                        f'<span style="color:#ccc;font-size:12px;">{_rb_sum}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Section E — Priority actions
            _priority = _pa.get("priority_actions", [])
            if _priority:
                st.markdown(
                    f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                    f'letter-spacing:0.08em;margin:12px 0 8px 0;">📋 PRIORITY ACTIONS</div>',
                    unsafe_allow_html=True,
                )
                for _i, _action_str in enumerate(_priority, 1):
                    st.markdown(
                        f'<div style="font-size:12px;color:#ccc;padding:4px 0;">'
                        f'<b style="color:{COLORS["bloomberg_orange"]};">{_i}.</b> {_action_str}</div>',
                        unsafe_allow_html=True,
                    )

        # Section F — Black Swan Stress Test (pure data, no AI)
        _custom_swans = st.session_state.get("_custom_swans") or {}
        if _custom_swans and open_trades:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:8px;">BLACK SWAN POSITION STRESS TEST</div>',
                unsafe_allow_html=True,
            )
            _open_tickers = [t["ticker"] for t in open_trades]
            _long_tickers = {t["ticker"] for t in open_trades if t["direction"].lower() == "long"}
            _short_tickers = {t["ticker"] for t in open_trades if t["direction"].lower() == "short"}
            for _swan_label, _swan_data in _custom_swans.items():
                _prob = _swan_data.get("probability_pct", "?")
                _impacts = _swan_data.get("asset_impacts", {})
                _eq_impact = (_impacts.get("equities") or "").lower()
                _bd_impact = (_impacts.get("bonds") or "").lower()
                # Expose: equity bearish → flag longs; equity bullish → flag shorts
                _exposed = []
                _safe = []
                for _tk in _open_tickers:
                    _is_long = _tk in _long_tickers
                    if ("bearish" in _eq_impact or "negative" in _eq_impact or "crash" in _eq_impact or "decline" in _eq_impact):
                        if _is_long:
                            _exposed.append(_tk)
                        else:
                            _safe.append(_tk)
                    elif ("bullish" in _eq_impact or "positive" in _eq_impact or "rally" in _eq_impact):
                        if not _is_long:
                            _exposed.append(_tk)
                        else:
                            _safe.append(_tk)
                    else:
                        _safe.append(_tk)
                _exp_html = " ".join(f'<span style="color:#ef4444;font-weight:600;">{t}</span>' for t in _exposed) if _exposed else '<span style="color:#888;">none</span>'
                _safe_html = " ".join(f'<span style="color:#22c55e;">{t}</span>' for t in _safe) if _safe else '<span style="color:#888;">none</span>'
                st.markdown(
                    f'<div style="background:#1a1a1a;border:1px solid {COLORS["border"]};border-radius:4px;'
                    f'padding:8px 12px;margin-bottom:6px;font-size:12px;">'
                    f'<b style="color:#f59e0b;">{_swan_label}</b> <span style="color:#888;">({_prob}%)</span>'
                    f' — <span style="color:#888;">equities: {_eq_impact or "unknown"}</span><br>'
                    f'At risk: {_exp_html} &nbsp;|&nbsp; Safe: {_safe_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif not _custom_swans:
            st.caption("Run Black Swans in Risk Regime to enable position stress testing.")

        # Section G1 — Correlation Matrix + Factor Exposure
        if open_trades and len(open_trades) >= 2:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:4px;">PORTFOLIO CONSTRUCTION LAYER</div>',
                unsafe_allow_html=True,
            )
            _corr_col, _factor_col = st.columns([1, 1])

            # ── Correlation Matrix ─────────────────────────────────────────────
            with _corr_col:
                st.markdown(
                    '<div style="font-size:11px;color:#64748b;letter-spacing:0.06em;'
                    'margin-bottom:6px;">90-DAY RETURN CORRELATION</div>',
                    unsafe_allow_html=True,
                )
                try:
                    from services.market_data import fetch_correlation_matrix as _fetch_corr
                    _corr_tickers = tuple(sorted({t["ticker"].upper() for t in open_trades}))
                    _corr_df = _fetch_corr(_corr_tickers, period="6mo")
                    if _corr_df is not None and not _corr_df.empty:
                        import plotly.graph_objects as _go
                        _z = _corr_df.values.tolist()
                        _labels = list(_corr_df.columns)
                        _fig_corr = _go.Figure(data=_go.Heatmap(
                            z=_z, x=_labels, y=_labels,
                            colorscale=[
                                [0.0, "#1e3a5f"], [0.5, "#1a1a2e"], [1.0, "#7f1d1d"],
                            ],
                            zmin=-1, zmax=1,
                            text=[[f"{v:.2f}" for v in row] for row in _z],
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            showscale=True,
                            colorbar=dict(thickness=10, len=0.8, tickfont=dict(size=9, color="#64748b")),
                        ))
                        apply_dark_layout(_fig_corr)
                        _fig_corr.update_layout(
                            height=220,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
                            yaxis=dict(tickfont=dict(size=10, color="#94a3b8"), autorange="reversed"),
                        )
                        st.plotly_chart(_fig_corr, use_container_width=True, config={"displayModeBar": False})
                        # Flag high-correlation pairs
                        _high_corr = []
                        for _i, _ti in enumerate(_labels):
                            for _j, _tj in enumerate(_labels):
                                if _j > _i and _corr_df.iloc[_i, _j] > 0.75:
                                    _high_corr.append(f"{_ti}↔{_tj} ({_corr_df.iloc[_i, _j]:.2f})")
                        if _high_corr:
                            st.markdown(
                                f'<div style="font-size:10px;color:#f59e0b;margin-top:2px;">'
                                f'⚠ High correlation: {" · ".join(_high_corr)}</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                '<div style="font-size:10px;color:#22c55e;margin-top:2px;">'
                                '✓ No pairs above 0.75 — good diversification</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("Correlation data unavailable.")
                except Exception as _ce:
                    st.caption(f"Correlation unavailable: {_ce}")

            # ── Aggregate Factor Exposure ──────────────────────────────────────
            with _factor_col:
                st.markdown(
                    '<div style="font-size:11px;color:#64748b;letter-spacing:0.06em;'
                    'margin-bottom:6px;">AGGREGATE FACTOR EXPOSURE</div>',
                    unsafe_allow_html=True,
                )
                try:
                    from services.portfolio_sizing import aggregate_factor_exposure as _agg_factors
                    _fe = _agg_factors(_sz_result.values() if _sz_result else [])
                    _fvals = _fe.get("factors", {})
                    if _fvals:
                        import plotly.graph_objects as _go2
                        _f_labels = ["Growth", "Inflation", "Liquidity", "Credit"]
                        _f_keys   = ["growth", "inflation", "liquidity", "credit"]
                        _f_vals   = [_fvals.get(k, 0) for k in _f_keys]
                        _f_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in _f_vals]
                        _fig_fe = _go2.Figure(data=_go2.Bar(
                            x=_f_vals, y=_f_labels,
                            orientation="h",
                            marker_color=_f_colors,
                            text=[f"{v:+.2f}x" for v in _f_vals],
                            textposition="outside",
                            textfont=dict(size=10, color="#94a3b8"),
                        ))
                        apply_dark_layout(_fig_fe)
                        _fig_fe.update_layout(
                            height=220,
                            margin=dict(l=0, r=40, t=10, b=0),
                            xaxis=dict(range=[-1.5, 1.5], tickfont=dict(size=9, color="#64748b"), zeroline=True, zerolinecolor="#333"),
                            yaxis=dict(tickfont=dict(size=10, color="#94a3b8")),
                            showlegend=False,
                        )
                        st.plotly_chart(_fig_fe, use_container_width=True, config={"displayModeBar": False})
                        if _fe.get("warnings"):
                            for _fw in _fe["warnings"]:
                                st.markdown(
                                    f'<div style="font-size:10px;color:#f59e0b;">⚠ {_fw}</div>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown(
                                '<div style="font-size:10px;color:#22c55e;">✓ Balanced factor exposure</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("Factor data unavailable — run sizing engine above.")
                except Exception as _fe_err:
                    st.caption(f"Factor exposure unavailable: {_fe_err}")

        # Section G1b — Per-stock factor mapping table (static)
        if open_trades and _sz_result:
            try:
                from services.portfolio_sizing import _SENSITIVITY as _SENS
                _ftable_rows = ""
                _factor_keys = ["growth", "inflation", "liquidity", "credit"]
                for _p in _sz_result.values():
                    _ptk = _p.get("ticker", "").upper()
                    _pw  = _p.get("current_weight", 0)
                    _sv  = _SENS.get(_ptk, [0.0, 0.0, 0.0, 0.0])
                    _dom_idx = max(range(4), key=lambda i: abs(_sv[i]))
                    _dom_label = _factor_keys[_dom_idx].capitalize()
                    def _fc(v):
                        if abs(v) < 0.1: return "#64748b"
                        return "#22c55e" if v > 0 else "#ef4444"
                    _ftable_rows += (
                        f'<tr style="border-bottom:1px solid #1a1a1a;">'
                        f'<td style="padding:4px 10px;color:{COLORS["bloomberg_orange"]};font-weight:700;">{_ptk}</td>'
                        f'<td style="padding:4px 10px;color:#888;">{_pw:.1f}%</td>'
                        + "".join(
                            f'<td style="padding:4px 10px;color:{_fc(_sv[i])};font-weight:600;">{_sv[i]:+.1f}</td>'
                            for i in range(4)
                        )
                        + f'<td style="padding:4px 10px;color:#94a3b8;font-size:10px;">{_dom_label}</td>'
                        f'</tr>'
                    )
                if _ftable_rows:
                    st.markdown(
                        f'<div style="font-size:11px;color:#64748b;letter-spacing:0.06em;margin:12px 0 6px 0;">POSITION FACTOR SENSITIVITIES</div>'
                        f'<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">'
                        f'<thead><tr style="background:#1a1a1a;color:#555;font-size:10px;letter-spacing:0.05em;">'
                        f'<th style="padding:4px 10px;text-align:left;">Ticker</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Wt%</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Growth</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Inflation</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Liquidity</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Credit</th>'
                        f'<th style="padding:4px 10px;text-align:left;">Dominant</th>'
                        f'</tr></thead><tbody>{_ftable_rows}</tbody></table>',
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        # Section G1c — Factor Exposure AI Analysis
        if open_trades and _sz_result:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:14px 0 10px 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:8px;">FACTOR AI ANALYSIS</div>',
                unsafe_allow_html=True,
            )

            # Change B — Quadrant reference expander
            with st.expander("📊 Quadrant Reference", expanded=False):
                _cur_q = _rc_for_sz.get("quadrant", "")
                def _qrow(name, cond, favored, avoid):
                    _active = "border-left:3px solid #FF8811;" if name == _cur_q else "border-left:3px solid transparent;"
                    _nc = COLORS["bloomberg_orange"] if name == _cur_q else "#94a3b8"
                    return (
                        f'<tr style="border-bottom:1px solid #1a1a1a;{_active}">'
                        f'<td style="padding:5px 10px;color:{_nc};font-weight:700;">{name}</td>'
                        f'<td style="padding:5px 10px;color:#64748b;font-size:11px;">{cond}</td>'
                        f'<td style="padding:5px 10px;color:#22c55e;font-size:11px;">{favored}</td>'
                        f'<td style="padding:5px 10px;color:#ef4444;font-size:11px;">{avoid}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">'
                    f'<thead><tr style="background:#1a1a1a;color:#555;font-size:10px;letter-spacing:0.05em;">'
                    f'<th style="padding:4px 10px;text-align:left;">Quadrant</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Condition</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Favored Factors</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Avoid</th>'
                    f'</tr></thead><tbody>'
                    + _qrow("Goldilocks",  "Growth ↑, Inflation ↓", "Growth, Liquidity",           "—")
                    + _qrow("Reflation",   "Growth ↑, Inflation ↑", "Inflation, Credit",            "Long duration")
                    + _qrow("Stagflation", "Growth ↓, Inflation ↑", "Inflation (real assets)",      "Growth, Liquidity")
                    + _qrow("Deflation",   "Growth ↓, Inflation ↓", "Long duration (TLT/IEF)",      "Credit, Equities")
                    + f'</tbody></table>'
                    + (f'<div style="font-size:10px;color:{COLORS["bloomberg_orange"]};margin-top:4px;">▶ Current regime: {_cur_q}</div>' if _cur_q else ''),
                    unsafe_allow_html=True,
                )

            _fa_has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
            _fa_tier_opts = ["⚡ Groq", "🧠 Regard Mode", "👑 Highly Regarded Mode"] if _fa_has_claude else ["⚡ Groq"]
            _fa_col1, _fa_col2 = st.columns([4, 2])
            with _fa_col1:
                _fa_tier = st.radio(
                    "Factor Engine", _fa_tier_opts, horizontal=True,
                    key="factor_analysis_engine",
                    help="Regard = Haiku (fast, concise) · Highly Regarded = Sonnet (richer, specific)",
                )
            # Change A — engine recommendation caption
            st.caption("💡 **Regard** for daily checks · **👑 Highly Regarded** for rebalancing decisions (position-specific suggestions)")
            with _fa_col2:
                _fa_run = st.button("🧠 Analyze Factors", key="run_factor_analysis", type="primary")

            if _fa_run:
                _fa_use_claude = _fa_tier in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
                _fa_model = None
                if _fa_tier == "🧠 Regard Mode":
                    _fa_model = "claude-haiku-4-5-20251001"
                elif _fa_tier == "👑 Highly Regarded Mode":
                    _fa_model = "claude-sonnet-4-6"
                try:
                    from services.portfolio_sizing import aggregate_factor_exposure as _agg_fa
                    _fe_for_ai = _agg_fa(_sz_result.values())
                except Exception:
                    _fe_for_ai = {"factors": {}, "dominant": "", "warnings": []}
                from services.claude_client import analyze_factor_exposure as _analyze_fe
                with st.spinner("Analyzing factor exposure…"):
                    _fa_result = _analyze_fe(
                        factor_exposure=_fe_for_ai,
                        regime_ctx=_rc_for_sz,
                        open_trades=open_trades,
                        use_claude=_fa_use_claude,
                        model=_fa_model,
                    )
                if _fa_result and "_error" not in _fa_result:
                    st.session_state["_factor_analysis"] = _fa_result
                    st.session_state["_factor_analysis_ts"] = _dt.now()
                    st.session_state["_factor_analysis_engine"] = _fa_tier
                else:
                    st.error(f"Factor analysis failed: {(_fa_result or {}).get('_error', 'Unknown error')}")

            _fa = st.session_state.get("_factor_analysis")
            _fa_engine = st.session_state.get("_factor_analysis_engine", "")
            _fa_ts = st.session_state.get("_factor_analysis_ts")
            if _fa and "_error" not in _fa:
                _fa_age = f"{int((_dt.now() - _fa_ts).total_seconds() / 60)}m ago" if _fa_ts else ""
                _fa_is_claude = _fa_engine in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
                _fa_badge = (
                    '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);'
                    'color:#fff;font-size:10px;font-weight:700;padding:1px 7px;'
                    'border-radius:3px;margin-left:6px;">✦ CLAUDE</span>'
                ) if _fa_is_claude else ""

                # Headline
                st.markdown(
                    f'<div style="background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;'
                    f'padding:12px 16px;margin:8px 0;">'
                    f'<div style="font-size:11px;color:#64748b;margin-bottom:4px;">'
                    f'{_fa_engine}{_fa_badge} · {_fa_age}</div>'
                    f'<div style="font-size:13px;color:#C8D8E8;font-weight:600;">'
                    f'{_fa.get("headline","")}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Per-factor verdict table
                _fit_colors = {"aligned": "#22c55e", "caution": "#f59e0b", "avoid": "#ef4444"}
                _fit_icons  = {"aligned": "✅", "caution": "⚠", "avoid": "✗"}
                _verdict_colors = {
                    "Overweight": "#f59e0b", "Moderate": "#94a3b8",
                    "Neutral": "#64748b", "Underweight": "#3b82f6",
                }
                _fv_rows = ""
                for _fv in _fa.get("factor_verdicts", []):
                    _fc = _fit_colors.get(_fv.get("regime_fit", ""), "#888")
                    _fi = _fit_icons.get(_fv.get("regime_fit", ""), "—")
                    _vc = _verdict_colors.get(_fv.get("verdict", ""), "#888")
                    _fv_rows += (
                        f'<tr style="border-bottom:1px solid #1e1e1e;">'
                        f'<td style="padding:5px 10px;color:#94a3b8;font-weight:600;text-transform:capitalize;">{_fv.get("factor","")}</td>'
                        f'<td style="padding:5px 10px;color:{_vc};font-weight:700;">{_fv.get("exposure",0):+.2f}x</td>'
                        f'<td style="padding:5px 10px;color:{_vc};">{_fv.get("verdict","")}</td>'
                        f'<td style="padding:5px 10px;color:{_fc};">{_fi} {_fv.get("regime_fit","").capitalize()}</td>'
                        f'<td style="padding:5px 10px;color:#64748b;font-size:11px;">{_fv.get("comment","")}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    f'<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:12px;">'
                    f'<thead><tr style="background:#1a1a1a;color:#555;font-size:10px;letter-spacing:0.06em;">'
                    f'<th style="padding:4px 10px;text-align:left;">Factor</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Exposure</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Weight</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Regime Fit</th>'
                    f'<th style="padding:4px 10px;text-align:left;">Comment</th>'
                    f'</tr></thead><tbody>{_fv_rows}</tbody></table>',
                    unsafe_allow_html=True,
                )

                # Top risk
                if _fa.get("top_risk"):
                    st.markdown(
                        f'<div style="background:#1a0e0e;border-left:3px solid #ef4444;'
                        f'border-radius:0 4px 4px 0;padding:8px 12px;margin:8px 0;">'
                        f'<span style="font-size:10px;color:#ef4444;font-weight:700;letter-spacing:0.06em;">TOP RISK · </span>'
                        f'<span style="font-size:12px;color:#fca5a5;">{_fa["top_risk"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Suggestions
                if _fa.get("suggestions"):
                    st.markdown(
                        f'<div style="font-size:11px;color:{COLORS["bloomberg_orange"]};'
                        f'font-weight:700;letter-spacing:0.06em;margin:8px 0 4px 0;">SUGGESTIONS</div>',
                        unsafe_allow_html=True,
                    )
                    for _sug in _fa["suggestions"]:
                        st.markdown(
                            f'<div style="font-size:12px;color:#bbb;padding:3px 0;">'
                            f'<span style="color:{COLORS["bloomberg_orange"]};">›</span> {_sug}</div>',
                            unsafe_allow_html=True,
                        )

        # Section G2 — Pre-Trade Simulator
        if open_trades:
            with st.expander("🔬 Pre-Trade Simulator — What-If Analysis", expanded=False):
                st.markdown(
                    '<div style="font-size:11px;color:#64748b;margin-bottom:10px;">'
                    'Simulate adding a new position. See correlation impact, factor shift, and sizing score before pulling the trigger.</div>',
                    unsafe_allow_html=True,
                )
                _sim_col1, _sim_col2 = st.columns([1, 1])
                with _sim_col1:
                    _sim_ticker = st.text_input("Ticker", placeholder="e.g. GLD", key="sim_ticker").strip().upper()
                    if _sim_ticker:
                        try:
                            _sim_info = yf.Ticker(_sim_ticker).info
                            _sim_name = _sim_info.get("shortName") or _sim_info.get("longName") or ""
                            _sim_sector = _sim_info.get("sector") or _sim_info.get("industryDisp") or ""
                            _sim_summary = (_sim_info.get("longBusinessSummary") or "")[:200]
                            if _sim_name:
                                st.caption(f":green[✓ {_sim_name}]" + (f" · {_sim_sector}" if _sim_sector else ""))
                            if _sim_summary:
                                st.markdown(
                                    f'<div style="font-size:10px;color:#64748b;line-height:1.4;margin-top:2px;">'
                                    f'{_sim_summary}{"…" if len(_sim_info.get("longBusinessSummary","")) > 200 else ""}</div>',
                                    unsafe_allow_html=True,
                                )
                        except Exception:
                            pass
                with _sim_col2:
                    _sim_amount = st.number_input("Dollar Amount ($)", min_value=100, max_value=500000, value=5000, step=500, key="sim_amount")

                if st.button("Simulate Trade", key="sim_run") and _sim_ticker:
                    with st.spinner(f"Analyzing {_sim_ticker} impact on portfolio..."):
                        try:
                            from services.portfolio_sizing import simulate_add as _simulate_add
                            _sim_result = _simulate_add(
                                ticker=_sim_ticker,
                                dollar_amount=float(_sim_amount),
                                existing_positions=open_trades,
                                regime_ctx=_rc_for_sz,
                                portfolio_value=_total_pv if _total_pv > 0 else float(_sim_amount),
                                live_prices=_pi_live,
                            )
                            st.session_state["_sim_result"] = _sim_result
                            st.session_state["_sim_ticker"] = _sim_ticker
                            st.session_state["_sim_amount"] = float(_sim_amount)
                            st.session_state.pop("_sim_verdict", None)  # clear stale verdict
                        except Exception as _sim_err:
                            st.error(f"Simulation error: {_sim_err}")

                _sim_result = st.session_state.get("_sim_result")
                _sim_ticker_saved = st.session_state.get("_sim_ticker", "")
                _sim_amount_saved = st.session_state.get("_sim_amount", 0)

                if _sim_result and _sim_ticker_saved == _sim_ticker:
                    _sc = _sim_result["sizing_score"]
                    _sc_val = _sc.get("composite_score")
                    _sc_color = "#22c55e" if (_sc_val or 0) >= 65 else ("#f59e0b" if (_sc_val or 0) >= 40 else "#ef4444")
                    _fd = _sim_result["factor_delta"]
                    _corr = _sim_result.get("corr_to_portfolio")

                    st.markdown(
                        f'<div style="background:#0E1E2E;border:1px solid #1E3A4A;border-radius:6px;padding:14px 18px;margin-top:8px;">'
                        f'<div style="font-size:13px;font-weight:700;color:#C8D8E8;margin-bottom:10px;">'
                        f'Adding <span style="color:{COLORS["bloomberg_orange"]};">{_sim_ticker_saved}</span> · '
                        f'${_sim_amount_saved:,.0f} · {_sim_result["proposed_weight"]:.1f}% of portfolio</div>'
                        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px;">'
                        + (f'<span style="font-size:11px;background:{_sc_color}22;border:1px solid {_sc_color}55;'
                           f'border-radius:3px;padding:2px 8px;color:{_sc_color};">Sizing Score {_sc_val}</span>'
                           if _sc_val is not None else '')
                        + (f'<span style="font-size:11px;color:#888;">Regime Fit {_sc.get("regime_fit")}</span>'
                           if _sc.get("regime_fit") is not None else '')
                        + (f'<span style="font-size:11px;color:#888;">Avg Corr {_corr:+.2f}</span>'
                           if _corr is not None else '')
                        + (f'<span style="font-size:11px;color:#94a3b8;">ATR Stop ${_sc["atr_stop"]:.2f}</span>'
                           if _sc.get("atr_stop") else '')
                        + f'</div>'
                        f'<div style="font-size:11px;color:#64748b;margin-bottom:4px;letter-spacing:0.05em;">FACTOR IMPACT</div>'
                        f'<div style="display:flex;gap:16px;flex-wrap:wrap;">'
                        + "".join(
                            f'<span style="font-size:11px;color:{"#22c55e" if _fd.get(f, 0) >= 0 else "#ef4444"};">'
                            f'{f.capitalize()} {_fd.get(f, 0):+.2f}x</span>'
                            for f in ["growth", "inflation", "liquidity", "credit"]
                        )
                        + f'</div>'
                        + ("".join(
                            f'<div style="font-size:11px;color:#f59e0b;margin-top:6px;">⚠ {w}</div>'
                            for w in _sim_result["warnings"]
                        ) if _sim_result["warnings"] else
                        '<div style="font-size:11px;color:#22c55e;margin-top:6px;">✓ No concentration warnings</div>')
                        + f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── AI Verdict Engine ──────────────────────────────────────
                    st.markdown(f'<div style="margin-top:12px;border-top:1px solid #1E3A4A;padding-top:10px;"></div>', unsafe_allow_html=True)
                    _sv_has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))
                    _sv_tier_opts = ["⚡ Groq", "🧠 Regard Mode", "👑 Highly Regarded Mode"] if _sv_has_claude else ["⚡ Groq"]
                    _sv_col1, _sv_col2 = st.columns([4, 2])
                    with _sv_col1:
                        _sv_tier = st.radio(
                            "Verdict Engine", _sv_tier_opts, horizontal=True,
                            key="sim_verdict_engine",
                            help="Regard = Haiku (fast) · Highly Regarded = Sonnet (deeper regime reasoning)",
                        )
                    st.caption("💡 **Regard** sufficient · **👑 Highly Regarded** for high-conviction sizing decisions")
                    with _sv_col2:
                        _sv_run = st.button("🧠 Get AI Verdict", key="sim_verdict_run", type="primary")

                    if _sv_run:
                        _sv_use_claude = _sv_tier in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
                        _sv_model = None
                        if _sv_tier == "🧠 Regard Mode":
                            _sv_model = "claude-haiku-4-5-20251001"
                        elif _sv_tier == "👑 Highly Regarded Mode":
                            _sv_model = "claude-sonnet-4-6"
                        from services.claude_client import analyze_sim_verdict as _asv
                        with st.spinner("Getting AI verdict…"):
                            _sv_result = _asv(
                                ticker=_sim_ticker_saved,
                                dollar_amount=_sim_amount_saved,
                                sim_result=_sim_result,
                                regime_ctx=_rc_for_sz,
                                open_trades=open_trades,
                                use_claude=_sv_use_claude,
                                model=_sv_model,
                            )
                        if _sv_result and "_error" not in _sv_result:
                            st.session_state["_sim_verdict"] = _sv_result
                            st.session_state["_sim_verdict_engine"] = _sv_tier
                        else:
                            st.error(f"Verdict error: {(_sv_result or {}).get('_error', 'Unknown')}")

                    _sv = st.session_state.get("_sim_verdict")
                    _sv_engine = st.session_state.get("_sim_verdict_engine", "")
                    if _sv and "_error" not in _sv:
                        _verdict_str = _sv.get("verdict", "CAUTION")
                        _verdict_cfg = {
                            "GO":      ("#22c55e", "#052e16", "✅"),
                            "CAUTION": ("#f59e0b", "#1c1400", "⚠"),
                            "PASS":    ("#ef4444", "#1a0a0a", "✗"),
                        }
                        _vc, _vbg, _vi = _verdict_cfg.get(_verdict_str, ("#888", "#111", "—"))
                        _sv_is_claude = _sv_engine in ("🧠 Regard Mode", "👑 Highly Regarded Mode")
                        _sv_badge = (
                            '<span style="background:linear-gradient(135deg,#7c3aed,#a855f7);'
                            'color:#fff;font-size:10px;font-weight:700;padding:1px 7px;'
                            'border-radius:3px;margin-left:6px;">✦ CLAUDE</span>'
                        ) if _sv_is_claude else ""
                        _thesis_colors = {"Diversifying": "#22c55e", "Concentrating": "#f59e0b", "Hedging": "#3b82f6"}
                        _tc = _thesis_colors.get(_sv.get("thesis_check", ""), "#888")

                        st.markdown(
                            f'<div style="background:{_vbg};border:1px solid {_vc}44;border-radius:6px;'
                            f'padding:14px 18px;margin-top:8px;">'
                            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">'
                            f'<span style="font-size:22px;font-weight:700;color:{_vc};">{_vi} {_verdict_str}</span>'
                            f'<span style="font-size:11px;color:{_tc};background:{_tc}22;border:1px solid {_tc}44;'
                            f'border-radius:3px;padding:1px 8px;">{_sv.get("thesis_check","")}</span>'
                            f'{_sv_badge}'
                            f'</div>'
                            f'<div style="font-size:13px;color:#C8D8E8;margin-bottom:8px;font-weight:600;">'
                            f'{_sv.get("verdict_reason","")}</div>'
                            + (f'<div style="font-size:11px;color:#94a3b8;margin-bottom:4px;">'
                               f'<span style="color:#64748b;">Regime:</span> {_sv["regime_fit_comment"]}</div>'
                               if _sv.get("regime_fit_comment") else '')
                            + (f'<div style="font-size:11px;color:#f59e0b;margin-bottom:4px;">'
                               f'⚠ Overlap: {_sv["overlap_warning"]}</div>'
                               if _sv.get("overlap_warning") else '')
                            + (f'<div style="font-size:11px;color:#94a3b8;">'
                               f'<span style="color:#64748b;">Size:</span> {_sv["sizing_suggestion"]}</div>'
                               if _sv.get("sizing_suggestion") else '')
                            + f'</div>',
                            unsafe_allow_html=True,
                        )

        # Section G — Regime-Aligned Opportunities
        _rp_plays = st.session_state.get("_rp_plays_result") or {}
        _fp_plays = st.session_state.get("_fed_plays_result") or {}
        if _rp_plays or _fp_plays:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
                f'letter-spacing:0.08em;margin-bottom:8px;">REGIME-ALIGNED OPPORTUNITIES</div>',
                unsafe_allow_html=True,
            )

            _held_tickers = {t["ticker"].upper() for t in open_trades}

            # Helper: render a plays block (sectors + stocks)
            def _render_plays_block(plays: dict, label: str, color: str):
                sectors = plays.get("sectors", [])
                stocks = plays.get("stocks", [])
                bonds = plays.get("bonds", [])
                rationale = plays.get("rationale", "")
                if not (sectors or stocks or bonds):
                    return
                st.markdown(
                    f'<div style="color:{color};font-size:11px;font-weight:700;'
                    f'letter-spacing:0.06em;margin:8px 0 4px 0;">{label}</div>',
                    unsafe_allow_html=True,
                )
                # Sectors
                if sectors:
                    sector_pills = ""
                    for s in sectors:
                        name = s.get("name", "") if isinstance(s, dict) else str(s)
                        sector_pills += (
                            f'<span style="background:#1e293b;border:1px solid #334155;'
                            f'border-radius:3px;padding:2px 8px;margin-right:4px;'
                            f'font-size:11px;color:#94a3b8;">{name}</span>'
                        )
                    st.markdown(
                        f'<div style="margin-bottom:4px;">'
                        f'<span style="font-size:11px;color:#555;margin-right:6px;">Sectors:</span>'
                        f'{sector_pills}</div>',
                        unsafe_allow_html=True,
                    )
                # Stocks — highlight if already held
                if stocks:
                    stock_pills = ""
                    new_ideas = []
                    for s in stocks:
                        tk = s.get("ticker", "") if isinstance(s, dict) else str(s)
                        name = s.get("name", tk) if isinstance(s, dict) else tk
                        already_held = tk.upper() in _held_tickers
                        if already_held:
                            pill_style = (
                                f'background:#052e16;border:1px solid #22c55e;'
                                f'border-radius:3px;padding:2px 8px;margin-right:4px;'
                                f'font-size:11px;color:#22c55e;font-weight:600;'
                            )
                            stock_pills += f'<span style="{pill_style}">{tk} ✓</span>'
                        else:
                            new_ideas.append((tk, name))
                            pill_style = (
                                f'background:#1e293b;border:1px solid #334155;'
                                f'border-radius:3px;padding:2px 8px;margin-right:4px;'
                                f'font-size:11px;color:#e2e8f0;'
                            )
                            stock_pills += f'<span style="{pill_style}">{tk}</span>'
                    st.markdown(
                        f'<div style="margin-bottom:4px;">'
                        f'<span style="font-size:11px;color:#555;margin-right:6px;">Stocks:</span>'
                        f'{stock_pills}</div>',
                        unsafe_allow_html=True,
                    )
                    if new_ideas:
                        st.caption(
                            "💡 Not in portfolio: " +
                            ", ".join(f"{tk} ({name})" for tk, name in new_ideas[:5])
                        )
                # Bonds
                if bonds:
                    bond_pills = ""
                    for s in bonds:
                        tk = s.get("ticker", "") if isinstance(s, dict) else str(s)
                        already_held = tk.upper() in _held_tickers
                        pill_style = (
                            f'background:#{"052e16" if already_held else "1e293b"};'
                            f'border:1px solid #{"22c55e" if already_held else "334155"};'
                            f'border-radius:3px;padding:2px 8px;margin-right:4px;'
                            f'font-size:11px;color:#{"22c55e" if already_held else "94a3b8"};'
                            f'{"font-weight:600;" if already_held else ""}'
                        )
                        bond_pills += f'<span style="{pill_style}">{tk}{"  ✓" if already_held else ""}</span>'
                    st.markdown(
                        f'<div style="margin-bottom:4px;">'
                        f'<span style="font-size:11px;color:#555;margin-right:6px;">Bonds/Macro:</span>'
                        f'{bond_pills}</div>',
                        unsafe_allow_html=True,
                    )
                if rationale:
                    st.markdown(
                        f'<div style="font-size:11px;color:#666;margin-top:2px;font-style:italic;">'
                        f'{rationale[:180]}{"…" if len(rationale) > 180 else ""}</div>',
                        unsafe_allow_html=True,
                    )

            if _rp_plays:
                _render_plays_block(_rp_plays, "▸ AI REGIME PLAYS", COLORS["bloomberg_orange"])
            if _fp_plays:
                _render_plays_block(_fp_plays, "▸ RATE-PATH PLAYS", "#60a5fa")

            if not _rp_plays and not _fp_plays:
                st.caption("Run AI Regime Plays and Rate-Path Plays in Risk Regime to see aligned opportunities.")
        else:
            st.markdown(f'<div style="border-top:1px solid {COLORS["border"]};margin:16px 0 10px 0;"></div>', unsafe_allow_html=True)
            st.caption("Run AI Regime Plays in Risk Regime to see regime-aligned opportunities here.")

    # --- Risk Matrix ---
    with tab_risk:
        _render_risk_matrix(open_trades)

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

        # ── Equity Curve + Signal Breakdown (closed trades only) ─────────────
        if not closed_trades:
            st.info("Close some trades to see the equity curve and signal breakdown.")

        if closed_trades:
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

    # --- Performance ---
    with tab_perf:
        try:
            from modules.performance import render as _render_performance
            _render_performance()
        except Exception as _perf_err:
            st.error(f"Performance tab error: {_perf_err}")
            import traceback
            st.code(traceback.format_exc())

    # --- AI Play Log ---
    with tab_playlog:
        from services.play_log import load_plays, clear_plays

        _all_plays = load_plays()  # newest first

        if not _all_plays:
            st.info("No AI plays logged yet. Generate plays in Risk Regime, Discovery, or Stress Signals to start building history.")
        else:
            # Stats row
            _features = sorted({p["feature"] for p in _all_plays})
            _engines  = sorted({p["engine"]  for p in _all_plays})
            _s1, _s2, _s3, _s4 = st.columns(4)
            _s1.metric("Total Entries", len(_all_plays))
            _s2.metric("Features Logged", len(_features))
            _s3.metric("Engines Used", len(_engines))
            from datetime import datetime as _dt
            _last_ts = _all_plays[0].get("timestamp", "")
            try:
                _last_label = _dt.fromisoformat(_last_ts).strftime("%b %d %H:%M")
            except Exception:
                _last_label = _last_ts[:16]
            _s4.metric("Last Entry", _last_label)

            st.markdown("---")

            # Filters
            _fc1, _fc2 = st.columns(2)
            with _fc1:
                _feat_filter = st.selectbox("Filter by Feature", ["All"] + _features, key="playlog_feat_filter")
            with _fc2:
                _eng_filter = st.selectbox("Filter by Engine", ["All"] + _engines, key="playlog_eng_filter")

            _filtered = [
                p for p in _all_plays
                if (_feat_filter == "All" or p["feature"] == _feat_filter)
                and (_eng_filter == "All" or p["engine"] == _eng_filter)
            ]

            st.caption(f"Showing {len(_filtered)} of {len(_all_plays)} entries")

            # Table
            for _entry in _filtered:
                _ts = _entry.get("timestamp", "")[:16].replace("T", " ")
                _feat = _entry.get("feature", "")
                _eng  = _entry.get("engine", "")
                _eng_color = "#FF8811" if "👑" in _eng else ("#22c55e" if "🧠" in _eng else "#64748b")
                _data = _entry.get("data", {})
                _meta = _entry.get("meta", {})

                # Summary line — varies by feature
                if isinstance(_data, dict):
                    _sectors = ", ".join(s.get("name","") for s in _data.get("sectors",[])[:3])
                    _stocks  = ", ".join(s.get("ticker","") for s in _data.get("stocks",[])[:4])
                    _rationale = (_data.get("rationale","") or "")[:120]
                    _narration = (_data.get("narration","") or "")[:120]
                    _briefing  = (_data.get("briefing","") or "")[:120]
                    if _sectors:
                        _summary = f"Sectors: {_sectors}" + (f" | Stocks: {_stocks}" if _stocks else "")
                    elif _narration:
                        _summary = _narration
                    elif _briefing:
                        _summary = _briefing
                    elif "rate_path_probs" in _data:
                        _dom = _data.get("dominant", {})
                        _summary = f"Dominant: {_dom.get('scenario','')} {_dom.get('prob_pct',0):.0f}%"
                    else:
                        _summary = str(_data)[:100]
                else:
                    _summary = str(_data)[:100]

                with st.expander(
                    f"{_ts} &nbsp; **{_feat}** &nbsp; `{_eng}` — {_summary[:80]}{'…' if len(_summary) > 80 else ''}",
                    expanded=False,
                ):
                    _col_a, _col_b = st.columns([1, 2])
                    with _col_a:
                        st.markdown(
                            f'<div style="font-size:11px;color:#94a3b8;">Feature</div>'
                            f'<div style="font-weight:700;">{_feat}</div>'
                            f'<div style="font-size:11px;color:#94a3b8;margin-top:8px;">Engine</div>'
                            f'<div style="color:{_eng_color};font-weight:700;">{_eng}</div>'
                            + (f'<div style="font-size:11px;color:#94a3b8;margin-top:8px;">Regime</div>'
                               f'<div>{_meta.get("regime","—")}</div>' if _meta.get("regime") else "")
                            + (f'<div style="font-size:11px;color:#94a3b8;margin-top:8px;">Fed Rate</div>'
                               f'<div>{_meta.get("fed_funds_rate","—")}</div>' if _meta.get("fed_funds_rate") else ""),
                            unsafe_allow_html=True,
                        )
                    with _col_b:
                        if _rationale:
                            st.markdown(f"**Rationale:** {_data.get('rationale','')}")
                        if _sectors:
                            st.markdown(f"**Sectors:** {_sectors}")
                        if _stocks:
                            st.markdown(f"**Stocks:** {_stocks}")
                        _bonds = ", ".join(s.get("ticker","") for s in _data.get("bonds",[])[:3])
                        if _bonds:
                            st.markdown(f"**Bonds:** {_bonds}")
                        if _narration:
                            st.markdown(f"**Narration:** {_data.get('narration','')}")
                        if _briefing:
                            st.markdown(f"**Briefing:** {_data.get('briefing','')[:600]}")
                    with st.expander("Raw JSON", expanded=False):
                        st.json(_entry)

            st.markdown("---")
            if st.button("🗑 Clear Play Log", key="clear_playlog"):
                clear_plays()
                st.success("Play log cleared.")
                st.rerun()
