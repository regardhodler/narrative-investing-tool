import json
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from utils.session import get_ticker
from utils.theme import COLORS


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

    # Disclaimer
    st.markdown("---")
    st.caption(
        "⚠️ **Disclaimer:** This AI-generated valuation is for informational purposes only "
        "and does not constitute financial advice. Always do your own research before making "
        "investment decisions."
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
