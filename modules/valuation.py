import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS, apply_dark_layout


def render():
    st.header("AI VALUATION & RECOMMENDATION")

    ticker = get_ticker()
    if not ticker:
        st.info("Select a ticker in Discovery to view AI valuation.")
        return

    with st.spinner("Collecting market signals..."):
        signals = _collect_signals(ticker)

    if not signals:
        st.warning("Could not collect enough data for valuation.")
        return

    signals_text = _format_signals_text(ticker, signals)

    with st.spinner("Generating AI valuation..."):
        from services.claude_client import generate_valuation
        result = generate_valuation(ticker, signals_text)

    if not result:
        import os
        has_key = bool(os.getenv("GROQ_API_KEY", ""))
        if not has_key:
            st.error("GROQ_API_KEY is not set. Add it to your .env file or Streamlit Cloud secrets.")
        else:
            st.error("AI valuation failed — the LLM returned an unparseable response. Try again.")
            with st.expander("Debug: Signal data sent to LLM"):
                st.code(signals_text)
        return

    _render_rating_banner(result)
    _render_signal_scorecard(signals)
    _render_analysis(result)

    # DCF Valuation
    st.markdown("---")
    _render_dcf(ticker)

    # Disclaimer
    st.markdown("---")
    st.caption(
        "**Disclaimer:** This valuation is for informational purposes only "
        "and does not constitute financial advice. The DCF model uses estimates and assumptions "
        "that may not reflect actual future performance. Always do your own research."
    )


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _collect_signals(ticker: str) -> dict | None:
    """Gather signals from yfinance and existing services into a structured dict."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        return None

    signals = {}

    # 1. Price & Technicals
    try:
        hist = stock.history(period="1y")
        if not hist.empty:
            close = hist["Close"]
            current = close.iloc[-1]
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            sma200 = close.rolling(200).mean().iloc[-1]

            # RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
            rsi = 100 - (100 / (1 + rs))

            signals["price"] = {
                "current": round(current, 2),
                "52w_high": round(close.max(), 2),
                "52w_low": round(close.min(), 2),
                "sma20": round(sma20, 2),
                "sma50": round(sma50, 2),
                "sma200": round(sma200, 2),
                "above_sma20": current > sma20,
                "above_sma50": current > sma50,
                "above_sma200": current > sma200,
                "rsi14": round(rsi, 1),
                "period_return_pct": round((current / close.iloc[0] - 1) * 100, 1),
            }
    except Exception:
        signals["price"] = None

    # 2. Fundamentals
    try:
        signals["fundamentals"] = {
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "dividend_yield": info.get("dividendYield"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
    except Exception:
        signals["fundamentals"] = None

    # 3. Institutional
    try:
        signals["institutional"] = {
            "inst_pct": info.get("heldPercentInstitutions"),
            "insider_pct": info.get("heldPercentInsiders"),
            "num_institutions": info.get("institutionCount"),
        }
    except Exception:
        signals["institutional"] = None

    # 4. Insider Activity
    try:
        from services.sec_client import get_insider_trades
        trades = get_insider_trades(ticker)
        if trades is not None and not trades.empty:
            buys = trades[trades["type"].str.contains("Purchase|Buy|Acquisition", case=False, na=False)]
            sells = trades[trades["type"].str.contains("Sale|Sell|Disposition", case=False, na=False)]
            signals["insider"] = {
                "buy_count": len(buys),
                "sell_count": len(sells),
                "buy_value": buys["value"].sum() if "value" in buys.columns else 0,
                "sell_value": sells["value"].sum() if "value" in sells.columns else 0,
                "net_sentiment": "bullish" if len(buys) > len(sells) else "bearish" if len(sells) > len(buys) else "neutral",
            }
        else:
            signals["insider"] = None
    except Exception:
        signals["insider"] = None

    # 5. Options Sentiment
    try:
        expirations = stock.options
        if expirations:
            chain = stock.option_chain(expirations[0])
            call_vol = chain.calls["volume"].sum()
            put_vol = chain.puts["volume"].sum()
            pc_ratio = put_vol / call_vol if call_vol > 0 else 0
            signals["options"] = {
                "put_call_ratio": round(pc_ratio, 2),
                "call_volume": int(call_vol),
                "put_volume": int(put_vol),
                "sentiment": "bearish" if pc_ratio > 1.0 else "bullish" if pc_ratio < 0.7 else "neutral",
            }
        else:
            signals["options"] = None
    except Exception:
        signals["options"] = None

    # 6. Company Profile
    try:
        from services.claude_client import describe_company
        name = info.get("longName", ticker)
        sic = info.get("industry", "")
        profile = describe_company(name, ticker, sic)
        signals["profile"] = profile
    except Exception:
        signals["profile"] = None

    return signals


def _format_signals_text(ticker: str, signals: dict) -> str:
    """Convert signals dict into a readable text block for the LLM prompt."""
    lines = [f"=== Signal Snapshot for {ticker} ===\n"]

    if signals.get("price"):
        p = signals["price"]
        lines.append("## Price & Technicals")
        lines.append(f"Current: ${p['current']} | 52W High: ${p['52w_high']} | 52W Low: ${p['52w_low']}")
        lines.append(f"SMA20: ${p['sma20']} ({'above' if p['above_sma20'] else 'below'}) | "
                      f"SMA50: ${p['sma50']} ({'above' if p['above_sma50'] else 'below'}) | "
                      f"SMA200: ${p['sma200']} ({'above' if p['above_sma200'] else 'below'})")
        lines.append(f"RSI(14): {p['rsi14']} | 1Y Return: {p['period_return_pct']}%\n")

    if signals.get("fundamentals"):
        f = signals["fundamentals"]
        lines.append("## Fundamentals")
        lines.append(f"P/E: {f['pe_ratio']} | Fwd P/E: {f['forward_pe']} | P/S: {f['ps_ratio']} | P/B: {f['pb_ratio']}")
        mc = f['market_cap']
        cap_str = f"${mc / 1e9:.1f}B" if mc and mc > 1e9 else f"${mc / 1e6:.0f}M" if mc else "N/A"
        lines.append(f"Market Cap: {cap_str} | Sector: {f['sector']} | Industry: {f['industry']}")
        pm = f"{f['profit_margin'] * 100:.1f}%" if f['profit_margin'] else "N/A"
        eg = f"{f['earnings_growth'] * 100:.1f}%" if f['earnings_growth'] else "N/A"
        rg = f"{f['revenue_growth'] * 100:.1f}%" if f['revenue_growth'] else "N/A"
        dy = f"{f['dividend_yield'] * 100:.2f}%" if f['dividend_yield'] else "N/A"
        lines.append(f"Profit Margin: {pm} | Earnings Growth: {eg} | Revenue Growth: {rg} | Div Yield: {dy}\n")

    if signals.get("institutional"):
        i = signals["institutional"]
        inst = f"{i['inst_pct'] * 100:.1f}%" if i['inst_pct'] else "N/A"
        ins = f"{i['insider_pct'] * 100:.1f}%" if i['insider_pct'] else "N/A"
        lines.append("## Institutional Ownership")
        lines.append(f"Institutional: {inst} | Insider: {ins} | # Institutions: {i['num_institutions']}\n")

    if signals.get("insider"):
        ins = signals["insider"]
        lines.append("## Insider Activity (Recent)")
        lines.append(f"Buys: {ins['buy_count']} (${ins['buy_value']:,.0f}) | "
                      f"Sells: {ins['sell_count']} (${ins['sell_value']:,.0f}) | "
                      f"Net: {ins['net_sentiment']}\n")

    if signals.get("options"):
        o = signals["options"]
        lines.append("## Options Sentiment")
        lines.append(f"P/C Ratio: {o['put_call_ratio']} | Call Vol: {o['call_volume']:,} | "
                      f"Put Vol: {o['put_volume']:,} | Sentiment: {o['sentiment']}\n")

    if signals.get("profile"):
        pr = signals["profile"]
        lines.append("## Company Profile")
        if pr.get("description"):
            lines.append(pr["description"])
        if pr.get("narrative"):
            lines.append(f"Narrative: {pr['narrative']}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

RATING_COLORS = {
    "Strong Buy": "#00C853",
    "Buy": "#69F0AE",
    "Hold": "#FFD600",
    "Sell": "#FF5252",
    "Strong Sell": "#D50000",
}


def _render_rating_banner(result: dict):
    """Large colored rating badge + confidence meter."""
    rating = result.get("rating", "Hold")
    confidence = result.get("confidence", 50)
    color = RATING_COLORS.get(rating, COLORS["yellow"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(
            f'<div style="background:{color}; color:#000; padding:24px; '
            f'border-radius:12px; text-align:center;">'
            f'<div style="font-size:14px; font-weight:600; opacity:0.7;">AI RATING</div>'
            f'<div style="font-size:36px; font-weight:800;">{rating.upper()}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown("**Confidence**")
        st.progress(confidence / 100)
        st.caption(f"{confidence}% confidence based on available signals")
        if result.get("summary"):
            st.markdown(f"*{result['summary']}*")


def _render_signal_scorecard(signals: dict):
    """2x3 grid showing each data category with mini sentiment indicator."""
    st.markdown("### Signal Scorecard")

    categories = [
        ("Price & Technicals", _score_price(signals.get("price"))),
        ("Fundamentals", _score_fundamentals(signals.get("fundamentals"))),
        ("Institutional", _score_institutional(signals.get("institutional"))),
        ("Insider Activity", _score_insider(signals.get("insider"))),
        ("Options Flow", _score_options(signals.get("options"))),
        ("Company Profile", ("neutral", "—") if signals.get("profile") else ("unavailable", "No data")),
    ]

    rows = [categories[:3], categories[3:]]
    for row in rows:
        cols = st.columns(3)
        for col, (label, (sentiment, detail)) in zip(cols, row):
            with col:
                if sentiment == "bullish":
                    icon, bg = "🟢", "#1B3D2F"
                elif sentiment == "bearish":
                    icon, bg = "🔴", "#3D1B1B"
                elif sentiment == "unavailable":
                    icon, bg = "⚫", "#2A2A2A"
                else:
                    icon, bg = "🟡", "#3D3A1B"

                st.markdown(
                    f'<div style="background:{bg}; padding:12px; border-radius:8px; margin-bottom:8px;">'
                    f'<div style="font-size:12px; color:#888;">{label}</div>'
                    f'<div style="font-size:18px;">{icon} {sentiment.title()}</div>'
                    f'<div style="font-size:11px; color:#aaa;">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _render_analysis(result: dict):
    """Bull/bear factors, key levels, recommendation."""
    # Bull vs Bear columns
    st.markdown("### Bull vs Bear Case")
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown(
            f'<div style="background:#1B3D2F; padding:16px; border-radius:8px; '
            f'border-left:4px solid #00C853;">'
            f'<div style="color:#69F0AE; font-weight:700; margin-bottom:8px;">BULLISH FACTORS</div>',
            unsafe_allow_html=True,
        )
        for factor in result.get("bullish_factors", []):
            st.markdown(f"- {factor}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_bear:
        st.markdown(
            f'<div style="background:#3D1B1B; padding:16px; border-radius:8px; '
            f'border-left:4px solid #FF5252;">'
            f'<div style="color:#FF5252; font-weight:700; margin-bottom:8px;">BEARISH FACTORS</div>',
            unsafe_allow_html=True,
        )
        for factor in result.get("bearish_factors", []):
            st.markdown(f"- {factor}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Key Levels
    key_levels = result.get("key_levels", {})
    if key_levels.get("support") or key_levels.get("resistance"):
        st.markdown("### Key Levels")
        lc, rc = st.columns(2)
        with lc:
            sup = key_levels.get("support")
            st.metric("Support", f"${sup}" if sup else "N/A")
        with rc:
            res = key_levels.get("resistance")
            st.metric("Resistance", f"${res}" if res else "N/A")

    # Recommendation
    if result.get("recommendation"):
        st.markdown("### Recommendation")
        st.info(result["recommendation"])


# ---------------------------------------------------------------------------
# Signal scoring helpers
# ---------------------------------------------------------------------------

def _score_price(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    rsi = data["rsi14"]
    above_count = sum([data["above_sma20"], data["above_sma50"], data["above_sma200"]])
    if above_count >= 2 and rsi < 70:
        return ("bullish", f"RSI {rsi} · Above {above_count}/3 SMAs")
    elif above_count <= 1 or rsi > 70:
        return ("bearish", f"RSI {rsi} · Above {above_count}/3 SMAs")
    return ("neutral", f"RSI {rsi} · Above {above_count}/3 SMAs")


def _score_fundamentals(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    pe = data.get("pe_ratio")
    growth = data.get("earnings_growth")
    if pe and growth:
        if pe < 25 and growth and growth > 0:
            return ("bullish", f"P/E {pe:.1f} · Growth {growth * 100:.0f}%")
        elif pe > 40 or (growth and growth < 0):
            return ("bearish", f"P/E {pe:.1f} · Growth {growth * 100:.0f}%")
    detail = f"P/E {pe:.1f}" if pe else "Limited data"
    return ("neutral", detail)


def _score_institutional(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    pct = data.get("inst_pct")
    if pct and pct > 0.7:
        return ("bullish", f"{pct * 100:.0f}% institutional")
    elif pct and pct < 0.3:
        return ("bearish", f"{pct * 100:.0f}% institutional")
    detail = f"{pct * 100:.0f}% inst" if pct else "N/A"
    return ("neutral", detail)


def _score_insider(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    s = data["net_sentiment"]
    detail = f"{data['buy_count']} buys / {data['sell_count']} sells"
    return (s, detail)


def _score_options(data: dict | None) -> tuple[str, str]:
    if not data:
        return ("unavailable", "No data")
    s = data["sentiment"]
    return (s, f"P/C ratio: {data['put_call_ratio']}")


# ---------------------------------------------------------------------------
# 2-Stage Discounted Cash Flow (Simply Wall St methodology)
# ---------------------------------------------------------------------------
# Levered DCF: discounts FCF to equity at Cost of Equity (CAPM)
# Stage 1 (years 1-10): analyst growth tapering to terminal rate
# Stage 2: Gordon Growth Model terminal value
# Risk-free rate & terminal growth = 5Y avg of 10Y Treasury yield
# Beta: industry unlevered beta re-levered, clamped [0.8, 2.0]
# ERP: Damodaran US equity risk premium (default 5.5%)
# ---------------------------------------------------------------------------

# Sector → approximate unlevered beta (Damodaran Jan 2024 US data)
_SECTOR_UNLEVERED_BETA = {
    "Technology": 1.12,
    "Communication Services": 0.82,
    "Consumer Cyclical": 0.95,
    "Consumer Defensive": 0.58,
    "Healthcare": 0.85,
    "Financial Services": 0.55,
    "Industrials": 0.85,
    "Energy": 0.90,
    "Basic Materials": 0.85,
    "Real Estate": 0.55,
    "Utilities": 0.35,
}

_DEFAULT_ERP = 0.055  # Damodaran US ERP


def _get_risk_free_rate() -> float:
    """5-year average of 10-year US Treasury yield."""
    try:
        tnx = yf.download("^TNX", period="5y", interval="1mo", progress=False, auto_adjust=True)
        if tnx is not None and not tnx.empty:
            close = tnx["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) > 12:
                avg_yield = close.mean() / 100  # ^TNX is in percent
                return max(0.01, min(0.08, float(avg_yield)))
    except Exception:
        pass
    return 0.04  # fallback


def _compute_dcf(ticker: str) -> dict | None:
    """
    2-stage levered DCF following Simply Wall St methodology.
    Returns dict with all intermediate values for display.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception:
        return None

    # ── Gather inputs ──
    shares = info.get("sharesOutstanding")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    market_cap = info.get("marketCap")
    total_debt = info.get("totalDebt", 0) or 0
    total_equity = market_cap or (shares * current_price if shares and current_price else None)
    sector = info.get("sector", "")
    company_name = info.get("shortName") or info.get("longName") or ticker
    tax_rate = info.get("effectiveTaxRate") or 0.21
    if isinstance(tax_rate, (int, float)) and tax_rate > 1:
        tax_rate = tax_rate / 100

    if not shares or not current_price:
        return None

    # ── Levered Free Cash Flow ──
    # Try to get FCF from cash flow statement
    try:
        cf = stock.cashflow
        if cf is not None and not cf.empty:
            # yfinance cashflow: rows are items, columns are dates
            fcf_row = None
            for label in ["Free Cash Flow", "FreeCashFlow"]:
                if label in cf.index:
                    fcf_row = cf.loc[label]
                    break
            if fcf_row is None:
                # Compute from operating cash flow - capex
                op_cf = None
                capex = None
                for label in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                    if label in cf.index:
                        op_cf = cf.loc[label]
                        break
                for label in ["Capital Expenditure", "Capital Expenditures"]:
                    if label in cf.index:
                        capex = cf.loc[label]
                        break
                if op_cf is not None and capex is not None:
                    fcf_row = op_cf + capex  # capex is typically negative
                elif op_cf is not None:
                    fcf_row = op_cf

            if fcf_row is not None:
                fcf_values = fcf_row.dropna().sort_index(ascending=False)
                if len(fcf_values) > 0:
                    latest_fcf = float(fcf_values.iloc[0])
                else:
                    return None
            else:
                return None
        else:
            return None
    except Exception:
        return None

    if latest_fcf <= 0:
        # Can't do DCF on negative FCF
        return {"error": "negative_fcf", "latest_fcf": latest_fcf, "company_name": company_name}

    # ── Historical FCF growth (for fallback) ──
    hist_fcf_growth = None
    if fcf_values is not None and len(fcf_values) >= 2:
        oldest = float(fcf_values.iloc[-1])
        newest = float(fcf_values.iloc[0])
        n_years = len(fcf_values) - 1
        if oldest > 0 and n_years > 0:
            hist_fcf_growth = (newest / oldest) ** (1 / n_years) - 1

    # ── Growth rate estimates ──
    # Analyst growth estimate (forward)
    analyst_growth = info.get("earningsGrowth")  # next year
    revenue_growth = info.get("revenueGrowth")

    # Pick best available growth estimate for initial rate
    if analyst_growth and abs(analyst_growth) < 1:
        initial_growth = float(analyst_growth)
    elif revenue_growth and abs(revenue_growth) < 1:
        initial_growth = float(revenue_growth)
    elif hist_fcf_growth and abs(hist_fcf_growth) < 1:
        initial_growth = float(hist_fcf_growth)
    else:
        initial_growth = 0.08  # fallback 8%

    # Clamp initial growth to reasonable range
    initial_growth = max(-0.10, min(0.40, initial_growth))

    # ── Risk-free rate (5Y avg of 10Y Treasury) ──
    rf = _get_risk_free_rate()

    # Terminal growth = risk-free rate (Simply Wall St methodology)
    terminal_growth = rf

    # ── Cost of Equity (CAPM) ──
    unlevered_beta = _SECTOR_UNLEVERED_BETA.get(sector, 0.85)

    # Re-lever beta for company's capital structure
    de_ratio = total_debt / total_equity if total_equity and total_equity > 0 else 0
    levered_beta = unlevered_beta * (1 + (1 - tax_rate) * de_ratio)
    levered_beta = max(0.8, min(2.0, levered_beta))  # clamp

    cost_of_equity = rf + levered_beta * _DEFAULT_ERP
    discount_rate = cost_of_equity

    if discount_rate <= terminal_growth:
        # Model breaks down
        return None

    # ── Stage 1: Project 10 years of FCF ──
    # Growth tapers linearly from initial_growth to terminal_growth over 10 years
    projected_fcf = []
    growth_rates = []
    pv_fcf = []
    fcf = latest_fcf

    for year in range(1, 11):
        # Linear taper: year 1 uses initial_growth, year 10 approaches terminal_growth
        weight = (year - 1) / 9  # 0 at year 1, 1 at year 10
        g = initial_growth * (1 - weight) + terminal_growth * weight
        growth_rates.append(g)
        fcf = fcf * (1 + g)
        projected_fcf.append(fcf)
        pv = fcf / (1 + discount_rate) ** year
        pv_fcf.append(pv)

    pv_stage1 = sum(pv_fcf)

    # ── Stage 2: Terminal Value (Gordon Growth Model) ──
    terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / (1 + discount_rate) ** 10

    # ── Intrinsic Value ──
    total_equity_value = pv_stage1 + pv_terminal
    intrinsic_per_share = total_equity_value / shares

    # ── Discount/Premium ──
    discount_pct = (intrinsic_per_share / current_price - 1) * 100

    # Growth source labels
    growth_source = "Analyst Est." if analyst_growth else ("Hist. FCF" if hist_fcf_growth else "Default")

    return {
        "company_name": company_name,
        "ticker": ticker,
        "current_price": current_price,
        "intrinsic_value": intrinsic_per_share,
        "discount_pct": discount_pct,
        "latest_fcf": latest_fcf,
        "shares": shares,
        "pv_stage1": pv_stage1,
        "pv_terminal": pv_terminal,
        "total_equity_value": total_equity_value,
        "projected_fcf": projected_fcf,
        "growth_rates": growth_rates,
        "pv_fcf": pv_fcf,
        "terminal_value": terminal_value,
        "discount_rate": discount_rate,
        "risk_free_rate": rf,
        "terminal_growth": terminal_growth,
        "levered_beta": levered_beta,
        "unlevered_beta": unlevered_beta,
        "de_ratio": de_ratio,
        "initial_growth": initial_growth,
        "growth_source": growth_source,
        "sector": sector,
        "erp": _DEFAULT_ERP,
    }


def _fmt_big(val: float) -> str:
    """Format large numbers: 1.2T, 345.6B, 12.3M, etc."""
    abs_val = abs(val)
    sign = "-" if val < 0 else ""
    if abs_val >= 1e12:
        return f"{sign}${abs_val / 1e12:.2f}T"
    elif abs_val >= 1e9:
        return f"{sign}${abs_val / 1e9:.2f}B"
    elif abs_val >= 1e6:
        return f"{sign}${abs_val / 1e6:.1f}M"
    else:
        return f"{sign}${abs_val:,.0f}"


def _make_dcf_waterfall(dcf: dict) -> go.Figure:
    """Waterfall chart showing PV of Stage 1, Terminal, Total → per share."""
    pv1 = dcf["pv_stage1"]
    pvt = dcf["pv_terminal"]
    total = dcf["total_equity_value"]

    fig = go.Figure(go.Waterfall(
        x=["PV of 10Y Cash Flows", "PV of Terminal Value", "Total Equity Value"],
        y=[pv1, pvt, total],
        measure=["relative", "relative", "total"],
        text=[_fmt_big(pv1), _fmt_big(pvt), _fmt_big(total)],
        textposition="outside",
        textfont=dict(size=12, color=COLORS["text"]),
        connector=dict(line=dict(color=COLORS["grid"])),
        increasing=dict(marker=dict(color=COLORS["green"])),
        totals=dict(marker=dict(color=COLORS["blue"])),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Equity Value Breakdown")
    return fig


def _make_dcf_value_comparison(dcf: dict) -> go.Figure:
    """Bar chart comparing current price vs intrinsic value."""
    price = dcf["current_price"]
    iv = dcf["intrinsic_value"]
    disc = dcf["discount_pct"]

    colors = [COLORS["text_dim"], COLORS["green"] if disc > 0 else COLORS["red"]]

    fig = go.Figure(go.Bar(
        x=["Current Price", "Intrinsic Value (DCF)"],
        y=[price, iv],
        marker_color=colors,
        text=[f"${price:.2f}", f"${iv:.2f}"],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text"]),
        width=0.5,
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=40, b=40),
        yaxis=dict(title="Price ($)", gridcolor=COLORS["grid"]),
        showlegend=False,
    )
    apply_dark_layout(fig, title="Price vs Intrinsic Value")
    return fig


def _render_dcf(ticker: str):
    """Render the 2-stage DCF valuation section."""
    st.markdown("### 2-Stage DCF Valuation")
    st.markdown(
        f"<p style='color:{COLORS['text_dim']};margin-top:-8px;font-size:13px;'>"
        "Levered DCF model (Simply Wall St methodology) &mdash; discounts free cash flow to equity at cost of equity</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Computing DCF..."):
        dcf = _compute_dcf(ticker)

    if dcf is None:
        st.warning("Could not compute DCF — insufficient financial data available for this ticker.")
        return

    if dcf.get("error") == "negative_fcf":
        st.warning(
            f"DCF not applicable — {dcf['company_name']} has negative free cash flow "
            f"({_fmt_big(dcf['latest_fcf'])}). The model requires positive FCF."
        )
        return

    price = dcf["current_price"]
    iv = dcf["intrinsic_value"]
    disc = dcf["discount_pct"]

    # ── Valuation verdict banner ──
    if disc >= 40:
        verdict = "Significantly Undervalued"
        verdict_color = "#00C853"
        verdict_bg = "rgba(0, 200, 83, 0.1)"
    elif disc >= 20:
        verdict = "Moderately Undervalued"
        verdict_color = "#69F0AE"
        verdict_bg = "rgba(105, 240, 174, 0.08)"
    elif disc >= 0:
        verdict = "Slightly Undervalued"
        verdict_color = COLORS["yellow"]
        verdict_bg = "rgba(255, 215, 0, 0.08)"
    elif disc >= -20:
        verdict = "Slightly Overvalued"
        verdict_color = "#FF8A65"
        verdict_bg = "rgba(255, 138, 101, 0.08)"
    elif disc >= -40:
        verdict = "Moderately Overvalued"
        verdict_color = "#FF5252"
        verdict_bg = "rgba(255, 82, 82, 0.08)"
    else:
        verdict = "Significantly Overvalued"
        verdict_color = "#D50000"
        verdict_bg = "rgba(213, 0, 0, 0.1)"

    disc_label = f"{disc:+.1f}%"
    disc_word = "below" if disc > 0 else "above"

    st.markdown(
        f"""
        <div style="background:{verdict_bg};border:1px solid {verdict_color};border-radius:12px;
                    padding:20px 28px;margin-bottom:20px;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:16px;">
            <div>
              <div style="font-size:13px;color:{COLORS['text_dim']};">DCF Fair Value</div>
              <div style="font-size:32px;font-weight:700;color:{verdict_color};">${iv:.2f}</div>
              <div style="font-size:13px;color:{COLORS['text_dim']};">vs Current ${price:.2f}</div>
            </div>
            <div style="text-align:center;">
              <div style="font-size:13px;color:{COLORS['text_dim']};">Discount to Fair Value</div>
              <div style="font-size:28px;font-weight:700;color:{verdict_color};">{disc_label}</div>
              <div style="font-size:12px;color:{COLORS['text_dim']};">Trading {abs(disc):.1f}% {disc_word} estimate</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:20px;font-weight:600;color:{verdict_color};">{verdict}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Charts ──
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_make_dcf_value_comparison(dcf), use_container_width=True)
    with c2:
        st.plotly_chart(_make_dcf_waterfall(dcf), use_container_width=True)

    # ── Projection Table ──
    with st.expander("10-Year FCF Projection Table", expanded=True):
        table_data = []
        for i in range(10):
            yr = i + 1
            src = dcf["growth_source"] if i < 3 else f"Taper → {dcf['terminal_growth']*100:.1f}%"
            table_data.append({
                "Year": yr,
                "Growth Rate": f"{dcf['growth_rates'][i]*100:.2f}%",
                "Source": src,
                "FCF": _fmt_big(dcf["projected_fcf"][i]),
                "PV of FCF": _fmt_big(dcf["pv_fcf"][i]),
            })
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, hide_index=True)

        # Terminal value row
        st.markdown(
            f"**Terminal Value (Year 10+):** {_fmt_big(dcf['terminal_value'])} "
            f"&rarr; PV: {_fmt_big(dcf['pv_terminal'])}"
        )

    # ── Assumptions / CAPM ──
    with st.expander("Model Assumptions (CAPM & Inputs)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            st.metric("Risk-Free Rate", f"{dcf['risk_free_rate']*100:.2f}%")
            st.caption("5Y avg of 10Y Treasury")
            st.metric("Equity Risk Premium", f"{dcf['erp']*100:.1f}%")
            st.caption("Damodaran US ERP")
        with a2:
            st.metric("Levered Beta", f"{dcf['levered_beta']:.2f}")
            st.caption(f"Unlevered: {dcf['unlevered_beta']:.2f} ({dcf['sector'] or 'Default'})")
            st.metric("D/E Ratio", f"{dcf['de_ratio']:.2f}")
        with a3:
            st.metric("Cost of Equity (Discount Rate)", f"{dcf['discount_rate']*100:.2f}%")
            st.caption(f"Rf + β × ERP = {dcf['risk_free_rate']*100:.1f}% + {dcf['levered_beta']:.2f} × {dcf['erp']*100:.1f}%")
            st.metric("Terminal Growth", f"{dcf['terminal_growth']*100:.2f}%")
            st.caption("= Risk-free rate")

        st.markdown(
            f"**Latest Reported FCF:** {_fmt_big(dcf['latest_fcf'])} | "
            f"**Initial Growth:** {dcf['initial_growth']*100:.1f}% ({dcf['growth_source']}) | "
            f"**Shares Outstanding:** {dcf['shares']/1e9:.3f}B"
        )

    with st.expander("How This DCF Works"):
        st.markdown(f"""
This is a **2-stage Levered Discounted Cash Flow** model:

**Stage 1 (Years 1-10):** Projects free cash flow using analyst growth estimates
(or historical FCF growth), with the growth rate linearly tapering from the initial
rate ({dcf['initial_growth']*100:.1f}%) down to the terminal rate ({dcf['terminal_growth']*100:.2f}%)
by year 10. This reflects the reality that high-growth companies see growth slow as they mature.

**Stage 2 (Year 10+):** Calculates a terminal value using the Gordon Growth Model,
representing all cash flows from year 11 to infinity, growing at the terminal rate.

**Discount Rate:** Uses Cost of Equity via CAPM (not WACC), since we're discounting
levered free cash flow (cash available to equity holders after debt service).

**Key choices following Simply Wall St methodology:**
- Risk-free rate = 5-year average of 10Y Treasury (avoids short-term rate volatility)
- Terminal growth rate = risk-free rate (no company grows faster than the economy forever)
- Beta = industry unlevered beta re-levered for company's D/E ratio, clamped to [0.8, 2.0]
- Equity risk premium = {dcf['erp']*100:.1f}% (Damodaran US estimate)

**Limitations:** DCF models are highly sensitive to growth assumptions and discount rates.
Small changes in inputs can produce large swings in intrinsic value. This works best for
companies with positive, relatively stable free cash flows.
        """)
