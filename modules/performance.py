"""Portfolio Performance Dashboard — benchmark-relative analytics."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.journal import load_journal
from utils.theme import COLORS, apply_dark_layout

_RISK_FREE_ANNUAL = 0.045  # ~current T-bill rate

_CAD_SUFFIXES_PERF = (".TO", ".V", ".TSX", ".CN", ".NE", ".VN")


def _ticker_currency_perf(ticker: str) -> str:
    return "CAD" if ticker.upper().endswith(_CAD_SUFFIXES_PERF) else "USD"


@st.cache_data(ttl=3600)
def _get_usdcad_perf() -> float:
    try:
        raw = yf.download("USDCAD=X", period="5d", interval="1d", progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            rate = float(close.dropna().iloc[-1])
            if 1.0 < rate < 2.0:
                return rate
    except Exception:
        pass
    return 1.36


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def _fetch_prices(tickers: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Fetch daily adjusted close prices for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        raw = yf.download(
            list(tickers), start=start, end=end,
            progress=False, auto_adjust=True, threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]] if "Close" in raw.columns else raw
        closes.index = pd.to_datetime(closes.index).normalize()
        return closes.ffill()
    except Exception:
        return pd.DataFrame()


def _build_nav_series(
    trades: list[dict],
    prices: pd.DataFrame,
    initial_capital: float,
    usdcad_rate: float = 1.0,
) -> pd.Series:
    """
    Construct a daily portfolio NAV series.

    Strategy: on each trading day, mark-to-market all positions that were
    open that day.  NAV = initial_capital + sum of unrealised/realised P&L
    across all positions.
    """
    if prices.empty or not trades:
        return pd.Series(dtype=float)

    nav_rows = []

    for dt in prices.index:
        dt_date = dt.date() if hasattr(dt, "date") else dt
        day_pnl = 0.0
        for t in trades:
            entry_d = _parse_date(t.get("entry_date"))
            exit_d  = _parse_date(t.get("exit_date"))
            if entry_d is None or dt_date < entry_d:
                continue
            # Position closed before this date — use exit price as final mark
            if exit_d is not None and dt_date > exit_d:
                mark = t.get("exit_price") or t.get("entry_price", 0)
            else:
                tk = t["ticker"]
                if tk in prices.columns:
                    col_prices = prices[tk]
                    mark = float(col_prices.loc[dt]) if dt in col_prices.index else t.get("entry_price", 0)
                else:
                    mark = t.get("entry_price", 0)

            entry_px   = t.get("entry_price", 0) or 0
            size       = t.get("position_size", 0) or 0
            mult       = 1.0 if t.get("direction", "long").lower() == "long" else -1.0
            pnl_native = mult * (mark - entry_px) * size
            ccy_mult   = usdcad_rate if _ticker_currency_perf(t.get("ticker", "")) == "USD" else 1.0
            day_pnl   += pnl_native * ccy_mult

        nav_rows.append({"date": dt, "nav": initial_capital + day_pnl})

    if not nav_rows:
        return pd.Series(dtype=float)
    series = pd.DataFrame(nav_rows).set_index("date")["nav"]
    series.index = pd.to_datetime(series.index)
    return series


def _parse_date(d) -> date | None:
    if d is None:
        return None
    try:
        return date.fromisoformat(str(d))
    except Exception:
        return None


def _daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def _sharpe(returns: pd.Series) -> float:
    rf_daily = (1 + _RISK_FREE_ANNUAL) ** (1 / 252) - 1
    excess   = returns - rf_daily
    if returns.std() == 0:
        return 0.0
    return float(excess.mean() / returns.std() * np.sqrt(252))


def _sortino(returns: pd.Series) -> float:
    rf_daily   = (1 + _RISK_FREE_ANNUAL) ** (1 / 252) - 1
    excess     = returns - rf_daily
    downside   = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside.std() * np.sqrt(252))


def _max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    dd   = (nav - peak) / peak
    return float(dd.min() * 100)  # negative %


def _cagr(nav: pd.Series) -> float:
    if len(nav) < 2:
        return 0.0
    years = len(nav) / 252
    if years == 0 or nav.iloc[0] == 0:
        return 0.0
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) * 100


def _beta_alpha(port_returns: pd.Series, bench_returns: pd.Series) -> tuple[float, float]:
    aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
    if len(aligned) < 10:
        return 0.0, 0.0
    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    beta  = float(np.cov(p, b)[0, 1] / np.var(b)) if np.var(b) != 0 else 0.0
    port_ann  = float(p.mean() * 252)
    bench_ann = float(b.mean() * 252)
    alpha = port_ann - (_RISK_FREE_ANNUAL + beta * (bench_ann - _RISK_FREE_ANNUAL))
    return round(beta, 2), round(alpha * 100, 2)


def _attribution(trades: list[dict], prices: pd.DataFrame) -> list[dict]:
    """Per-position total P&L and % of initial_capital."""
    rows = []
    for t in trades:
        tk      = t["ticker"]
        entry   = t.get("entry_price", 0) or 0
        size    = t.get("position_size", 0) or 0
        mult    = 1.0 if t.get("direction", "long").lower() == "long" else -1.0

        if t.get("status") == "closed" and t.get("exit_price"):
            exit_px = t["exit_price"]
        elif tk in prices.columns and not prices[tk].empty:
            exit_px = float(prices[tk].iloc[-1])
        else:
            exit_px = entry

        pnl     = mult * (exit_px - entry) * size
        pnl_pct = mult * (exit_px / entry - 1) * 100 if entry > 0 else 0
        status  = t.get("status", "open")
        rows.append({
            "ticker":    tk,
            "direction": t.get("direction", "long").upper(),
            "entry":     entry,
            "exit":      exit_px,
            "size":      size,
            "pnl":       pnl,
            "pnl_pct":   pnl_pct,
            "status":    status,
            "cost_basis": entry * size,
        })
    rows.sort(key=lambda r: abs(r["pnl"]), reverse=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nav_chart(port_nav: pd.Series, bench_nav: pd.Series, bench_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=port_nav.index, y=port_nav.values,
        name="My Portfolio", mode="lines",
        line=dict(color=COLORS.get("bloomberg_orange", "#FF8811"), width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bench_nav.index, y=bench_nav.values,
        name=bench_label, mode="lines",
        line=dict(color="#64748b", width=1.5, dash="dot"),
    ))
    apply_dark_layout(fig, title="Portfolio NAV vs Benchmark", yaxis_title="NAV ($)", height=320)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return fig


def _drawdown_chart(nav: pd.Series) -> go.Figure:
    peak = nav.cummax()
    dd   = ((nav - peak) / peak * 100)
    fig  = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values,
        fill="tozeroy", mode="lines",
        line=dict(color="#ef4444", width=1),
        fillcolor="rgba(239,68,68,0.15)",
        name="Drawdown %",
    ))
    apply_dark_layout(fig, title="Drawdown", yaxis_title="Drawdown (%)", height=200)
    return fig


def _attribution_chart(rows: list[dict]) -> go.Figure:
    labels = [r["ticker"] for r in rows]
    values = [r["pnl"] for r in rows]
    colors = [COLORS.get("positive", "#22c55e") if v >= 0 else COLORS.get("negative", "#ef4444") for v in values]
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"${v:+,.0f}" for v in values],
        textposition="outside",
    ))
    apply_dark_layout(fig, title="P&L Attribution by Position", yaxis_title="P&L ($)", height=280)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Metric card helper
# ─────────────────────────────────────────────────────────────────────────────

def _metric(label: str, value: str, delta: str = "", color: str = "#ccc") -> str:
    delta_html = (
        f'<div style="font-size:11px;color:#64748b;margin-top:2px;">{delta}</div>'
        if delta else ""
    )
    return (
        f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
        f'border-radius:6px;padding:12px 16px;text-align:center;">'
        f'<div style="font-size:10px;color:#64748b;font-weight:700;letter-spacing:0.08em;'
        f'text-transform:uppercase;margin-bottom:4px;">{label}</div>'
        f'<div style="font-size:20px;font-weight:700;color:{color};">{value}</div>'
        f'{delta_html}'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render():
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',Consolas,monospace;'
        f'font-size:20px;font-weight:700;color:{COLORS["bloomberg_orange"]};'
        f'letter-spacing:0.1em;margin-bottom:4px;">PERFORMANCE</div>'
        f'<div style="height:2px;background:linear-gradient(90deg,'
        f'{COLORS["bloomberg_orange"]},{COLORS["bloomberg_orange"]}44,transparent);'
        f'border-radius:1px;margin-bottom:16px;"></div>',
        unsafe_allow_html=True,
    )

    all_trades = load_journal()
    if not all_trades:
        st.info("No trades in journal yet. Add trades in My Regarded Portfolio to see performance analytics.")
        return

    # ── Settings bar ─────────────────────────────────────────────────────────
    col_cap, col_bench, col_period = st.columns([1.5, 1.5, 1.5])
    with col_cap:
        initial_capital = st.number_input(
            "Starting Capital (CAD)", min_value=1000, value=10000, step=1000,
            key="perf_capital",
        )
    with col_bench:
        bench_ticker = st.selectbox(
            "Benchmark", ["SPY", "QQQ", "IWM", "BTC-USD"], key="perf_bench",
        )
    with col_period:
        period_opts = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "All Time": 9999}
        period_label = st.selectbox("Period", list(period_opts.keys()), index=3, key="perf_period")
    period_days = period_opts[period_label]

    # ── Date range ────────────────────────────────────────────────────────────
    earliest_entry = min(
        (_parse_date(t["entry_date"]) for t in all_trades if t.get("entry_date")),
        default=date.today() - timedelta(days=365),
    )
    start_date = max(earliest_entry, date.today() - timedelta(days=period_days))
    end_date   = date.today()
    start_str  = start_date.isoformat()
    end_str    = (end_date + timedelta(days=1)).isoformat()

    # ── Fetch prices ─────────────────────────────────────────────────────────
    all_tickers = tuple(sorted({t["ticker"].upper() for t in all_trades} | {bench_ticker}))
    with st.spinner("Loading price history…"):
        prices = _fetch_prices(all_tickers, start_str, end_str)

    if prices.empty:
        st.warning("Could not fetch price data. Check your internet connection.")
        return

    # ── Load cash (CAD) + live USD/CAD rate ──────────────────────────────────
    import json as _pf_json, os as _pf_os
    _pf_path = _pf_os.path.join(_pf_os.path.dirname(_pf_os.path.dirname(__file__)), "data", "portfolio_settings.json")
    try:
        with open(_pf_path) as _pf:
            _cash_cad = float(_pf_json.load(_pf).get("cash_holdings", 0.0))
    except Exception:
        _cash_cad = 0.0
    _usdcad = _get_usdcad_perf()
    _initial_total = initial_capital + _cash_cad
    if _cash_cad != 0:
        st.caption(f"{'+ ' if _cash_cad > 0 else ''}C${_cash_cad:,.0f} cash/margin from Portfolio Intel → Total C${_initial_total:,.0f} · USD/CAD {_usdcad:.4f}")

    # ── Build NAV series ──────────────────────────────────────────────────────
    port_nav = _build_nav_series(all_trades, prices, _initial_total, usdcad_rate=_usdcad)
    if port_nav.empty or len(port_nav) < 2:
        st.info("Not enough price history to build a NAV curve. Ensure entry dates are set correctly.")
        return

    # Benchmark NAV — rebased to same starting capital
    bench_raw = prices[bench_ticker] if bench_ticker in prices.columns else None
    if bench_raw is not None and not bench_raw.empty:
        bench_aligned = bench_raw.reindex(port_nav.index, method="ffill").dropna()
        if not bench_aligned.empty:
            bench_nav = bench_aligned / bench_aligned.iloc[0] * _initial_total
        else:
            bench_nav = None
    else:
        bench_nav = None

    # ── Compute metrics ───────────────────────────────────────────────────────
    port_ret  = _daily_returns(port_nav)
    sharpe    = _sharpe(port_ret)
    sortino   = _sortino(port_ret)
    max_dd    = _max_drawdown(port_nav)
    cagr_val  = _cagr(port_nav)
    total_pnl = port_nav.iloc[-1] - _initial_total
    total_pct = (total_pnl / _initial_total) * 100

    beta, alpha = 0.0, 0.0
    bench_ret_pct = 0.0
    if bench_nav is not None:
        bench_ret  = _daily_returns(bench_nav)
        beta, alpha = _beta_alpha(port_ret, bench_ret)
        bench_ret_pct = float((bench_nav.iloc[-1] / bench_nav.iloc[0] - 1) * 100)

    # ── Metric cards ──────────────────────────────────────────────────────────
    pnl_color   = COLORS.get("positive", "#22c55e") if total_pnl >= 0 else COLORS.get("negative", "#ef4444")
    alpha_color = COLORS.get("positive", "#22c55e") if alpha >= 0 else COLORS.get("negative", "#ef4444")
    sharpe_color = "#22c55e" if sharpe > 1 else ("#f59e0b" if sharpe > 0 else "#ef4444")

    metrics_html = "".join([
        _metric("Total Return", f"{total_pct:+.1f}%",
                f"${total_pnl:+,.0f}  |  {bench_ticker} {bench_ret_pct:+.1f}%", pnl_color),
        _metric("CAGR", f"{cagr_val:+.1f}%", "annualised", pnl_color),
        _metric("Sharpe Ratio", f"{sharpe:.2f}", ">1 = strong", sharpe_color),
        _metric("Sortino Ratio", f"{sortino:.2f}" if sortino != float("inf") else "∞", "downside-adj", sharpe_color),
        _metric("Max Drawdown", f"{max_dd:.1f}%", "peak-to-trough", "#ef4444"),
        _metric("Beta", f"{beta:.2f}", f"vs {bench_ticker}", "#94a3b8"),
        _metric("Alpha", f"{alpha:+.1f}%", "annualised vs benchmark", alpha_color),
        _metric("Trades", str(len(all_trades)),
                f"{sum(1 for t in all_trades if t['status']=='open')} open / {sum(1 for t in all_trades if t['status']=='closed')} closed",
                "#94a3b8"),
    ])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px;">'
        f'{metrics_html}</div>',
        unsafe_allow_html=True,
    )

    # ── NAV chart ──────────────────────────────────────────────────────────────
    nav_col, dd_col = st.columns([3, 1])
    with nav_col:
        st.plotly_chart(
            _nav_chart(port_nav, bench_nav if bench_nav is not None else port_nav,
                       f"{bench_ticker} (rebased)"),
            use_container_width=True,
        )
    with dd_col:
        st.plotly_chart(_drawdown_chart(port_nav), use_container_width=True)

    # ── Attribution ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div style="font-size:13px;color:{COLORS["bloomberg_orange"]};font-weight:700;'
        f'letter-spacing:0.08em;margin:8px 0;">P&L ATTRIBUTION</div>',
        unsafe_allow_html=True,
    )
    attr_rows = _attribution(all_trades, prices)

    if attr_rows:
        # Bar chart
        st.plotly_chart(_attribution_chart(attr_rows), use_container_width=True)

        # Table
        total_cost = sum(r["cost_basis"] for r in attr_rows if r["cost_basis"] > 0)
        rows_html = ""
        for r in attr_rows:
            pnl_c = COLORS.get("positive", "#22c55e") if r["pnl"] >= 0 else COLORS.get("negative", "#ef4444")
            port_w = r["cost_basis"] / total_cost * 100 if total_cost > 0 else 0
            rows_html += (
                f'<tr style="border-bottom:1px solid {COLORS["border"]}22;">'
                f'<td style="padding:6px 10px;color:{COLORS["bloomberg_orange"]};font-weight:700;">{r["ticker"]}</td>'
                f'<td style="padding:6px 10px;color:#888;font-size:11px;">{r["direction"]}</td>'
                f'<td style="padding:6px 10px;color:#ccc;">${r["entry"]:.2f}</td>'
                f'<td style="padding:6px 10px;color:#ccc;">${r["exit"]:.2f}</td>'
                f'<td style="padding:6px 10px;color:{pnl_c};font-weight:700;">${r["pnl"]:+,.0f}</td>'
                f'<td style="padding:6px 10px;color:{pnl_c};">{r["pnl_pct"]:+.1f}%</td>'
                f'<td style="padding:6px 10px;color:#64748b;font-size:11px;">{port_w:.1f}%</td>'
                f'<td style="padding:6px 10px;">'
                f'<span style="background:{"#1a3a1a" if r["status"]=="open" else "#1e293b"};'
                f'border-radius:3px;padding:1px 6px;font-size:10px;'
                f'color:{"#22c55e" if r["status"]=="open" else "#64748b"};">{r["status"].upper()}</span>'
                f'</td>'
                f'</tr>'
            )
        st.markdown(
            f'<div style="background:{COLORS["surface"]};border:1px solid {COLORS["border"]};'
            f'border-radius:6px;overflow:auto;">'
            f'<table style="width:100%;border-collapse:collapse;">'
            f'<thead><tr style="border-bottom:1px solid {COLORS["border"]};">'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">TICKER</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">DIR</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">ENTRY</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">CURRENT/EXIT</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">P&L $</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">P&L %</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">PORT WT</th>'
            f'<th style="padding:6px 10px;font-size:10px;color:#64748b;text-align:left;">STATUS</th>'
            f'</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            f'</table></div>',
            unsafe_allow_html=True,
        )
