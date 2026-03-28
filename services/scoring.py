"""
Quantitative scoring engine for tickers.

Scores each ticker 0-100 across six categories using free data sources:
- Technicals (SMA positioning, RSI, momentum) — yfinance
- Fundamentals (P/E, growth, margins) — yfinance .info
- Insider activity (buy/sell ratio, recency) — SEC EDGAR
- Options sentiment (P/C ratio) — yfinance options chains
- Congress (congressional buy/sell ratio, recency) — Quiver Quant
- Short Interest (squeeze potential, days-to-cover) — yfinance .info
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed


def _clamp(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


@st.cache_data(ttl=3600)
def _fetch_technicals(ticker: str) -> dict:
    """Score technicals 0-100 based on SMA positioning, RSI, and momentum."""
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return {"score": 50, "details": {}}
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel("Ticker", axis=1)
        close = df["Close"].dropna()
        if len(close) < 50:
            return {"score": 50, "details": {}}

        price = float(close.iloc[-1])

        # SMA positioning (0-40 pts)
        sma20 = float(close.rolling(20).mean().iloc[-1])
        sma50 = float(close.rolling(50).mean().iloc[-1])
        sma_score = 0
        if price > sma20:
            sma_score += 20
        if price > sma50:
            sma_score += 20
        if len(close) >= 200:
            sma200 = float(close.rolling(200).mean().iloc[-1])
            if price > sma200:
                sma_score += 10
            # Golden cross bonus
            if sma50 > sma200:
                sma_score += 10
        sma_score = min(sma_score, 40)

        # RSI (0-30 pts) — reward 40-60 zone, penalize extremes
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs.iloc[-1]))) if not np.isnan(rs.iloc[-1]) else 50
        if 40 <= rsi <= 60:
            rsi_score = 30
        elif 30 <= rsi < 40 or 60 < rsi <= 70:
            rsi_score = 20
        elif rsi < 30:
            rsi_score = 25  # oversold = potential buy
        else:
            rsi_score = 5  # overbought

        # Momentum (0-30 pts) — 1m and 3m returns
        mom_score = 0
        if len(close) >= 22:
            ret_1m = (price / float(close.iloc[-22]) - 1) * 100
            mom_score += _clamp(ret_1m * 2 + 15, 0, 15)
        if len(close) >= 63:
            ret_3m = (price / float(close.iloc[-63]) - 1) * 100
            mom_score += _clamp(ret_3m + 15, 0, 15)

        total = _clamp(sma_score + rsi_score + mom_score)
        return {
            "score": round(total),
            "details": {"price": price, "sma20": round(sma20, 2), "sma50": round(sma50, 2),
                         "rsi": round(rsi, 1)},
        }
    except Exception:
        return {"score": 50, "details": {}}


@st.cache_data(ttl=3600)
def _fetch_fundamentals(ticker: str) -> dict:
    """Score fundamentals 0-100 based on P/E, growth, margins."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}
        score = 50  # neutral default

        # P/E ratio (0-35 pts) — lower is better, negative = 0
        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe is not None and pe > 0:
            if pe < 15:
                pe_score = 35
            elif pe < 25:
                pe_score = 25
            elif pe < 40:
                pe_score = 15
            else:
                pe_score = 5
        else:
            pe_score = 10  # no PE = uncertain

        # Revenue growth (0-35 pts)
        growth = info.get("revenueGrowth")
        if growth is not None:
            growth_pct = growth * 100
            growth_score = _clamp(growth_pct * 1.5 + 15, 0, 35)
        else:
            growth_score = 15

        # Profit margins (0-30 pts)
        margin = info.get("profitMargins")
        if margin is not None:
            margin_pct = margin * 100
            margin_score = _clamp(margin_pct * 1.5 + 10, 0, 30)
        else:
            margin_score = 15

        score = _clamp(pe_score + growth_score + margin_score)
        return {
            "score": round(score),
            "details": {"pe": round(pe, 1) if pe else None,
                         "revenue_growth": f"{growth * 100:.1f}%" if growth else None,
                         "margin": f"{margin * 100:.1f}%" if margin else None},
        }
    except Exception:
        return {"score": 50, "details": {}}


@st.cache_data(ttl=3600)
def _fetch_insider_score(ticker: str) -> dict:
    """Score insider activity 0-100 based on buy/sell ratio and recency."""
    try:
        from services.sec_client import get_insider_trades
        df = get_insider_trades(ticker)
        if df.empty:
            return {"score": 50, "details": {"trades": 0}}

        buys = len(df[df["type"] == "Purchase"])
        sells = len(df[df["type"] == "Sale"])
        total = buys + sells
        if total == 0:
            return {"score": 50, "details": {"trades": 0}}

        buy_ratio = buys / total
        # 100% buys = 100, 50/50 = 50, all sells = 0
        score = _clamp(buy_ratio * 100)

        # Recency bonus: recent buys within 30 days
        try:
            df["date"] = pd.to_datetime(df["date"])
            recent = df[df["date"] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
            recent_buys = len(recent[recent["type"] == "Purchase"])
            if recent_buys >= 3:
                score = _clamp(score + 15)
            elif recent_buys >= 1:
                score = _clamp(score + 5)
        except Exception:
            pass

        return {
            "score": round(score),
            "details": {"buys": buys, "sells": sells, "trades": total},
        }
    except Exception:
        return {"score": 50, "details": {"trades": 0}}


@st.cache_data(ttl=3600)
def _fetch_options_score(ticker: str) -> dict:
    """Score options sentiment 0-100 based on P/C ratio."""
    try:
        tk = yf.Ticker(ticker)
        expiries = list(tk.options or [])
        if not expiries:
            return {"score": 50, "details": {}}

        # Use nearest expiry
        chain = tk.option_chain(expiries[0])
        call_oi = chain.calls["openInterest"].sum() if chain.calls is not None else 0
        put_oi = chain.puts["openInterest"].sum() if chain.puts is not None else 0

        if call_oi == 0 and put_oi == 0:
            return {"score": 50, "details": {}}

        pc_ratio = put_oi / call_oi if call_oi > 0 else 2.0

        # High P/C = bearish sentiment = contrarian bullish signal
        # P/C < 0.5 = very bullish sentiment = contrarian bearish
        if pc_ratio > 1.5:
            score = 85  # extreme fear = contrarian buy
        elif pc_ratio > 1.0:
            score = 70
        elif pc_ratio > 0.7:
            score = 50  # neutral
        elif pc_ratio > 0.5:
            score = 35
        else:
            score = 15  # extreme greed = contrarian sell

        return {
            "score": round(score),
            "details": {"pc_ratio": round(pc_ratio, 2), "call_oi": int(call_oi), "put_oi": int(put_oi)},
        }
    except Exception:
        return {"score": 50, "details": {}}


@st.cache_data(ttl=3600)
def _fetch_short_interest(ticker: str) -> dict:
    """Score short interest 0-100 — high short float = squeeze potential (contrarian bullish)."""
    try:
        info = yf.Ticker(ticker).info or {}
        short_pct = info.get("shortPercentOfFloat") or 0.0
        days_to_cover = info.get("shortRatio") or 0.0

        # Contrarian: high short % → more squeeze fuel → higher score
        if short_pct >= 0.30:
            base_score, label = 90, "extreme"
        elif short_pct >= 0.20:
            base_score, label = 75, "high"
        elif short_pct >= 0.10:
            base_score, label = 60, "elevated"
        elif short_pct >= 0.05:
            base_score, label = 45, "moderate"
        else:
            base_score, label = 30, "low"

        # Days-to-cover bonus: longer exit runway = harder short squeeze to stop
        dtc_bonus = 10 if days_to_cover > 5 else 5 if days_to_cover > 3 else 0
        score = _clamp(base_score + dtc_bonus)

        return {
            "score": round(score),
            "details": {
                "short_pct_float": f"{short_pct * 100:.1f}%",
                "days_to_cover": round(days_to_cover, 1),
                "squeeze_potential": label,
            },
        }
    except Exception:
        return {"score": 50, "details": {}}


@st.cache_data(ttl=3600)
def _fetch_congress_score(ticker: str) -> dict:
    """Score congressional trading activity 0-100 based on buy/sell ratio and recency."""
    try:
        from services.congress_client import get_congress_trades
        df, err = get_congress_trades(ticker)
        if df.empty or err:
            return {"score": 50, "details": {"trades": 0}}

        buys = len(df[df["type"] == "Purchase"])
        sells = len(df[df["type"] == "Sale"])
        total = buys + sells
        if total == 0:
            return {"score": 50, "details": {"trades": 0}}

        buy_ratio = buys / total
        score = _clamp(buy_ratio * 100)

        # Recency bonus: congress trades within 90 days (wider window than insider)
        try:
            df["date"] = pd.to_datetime(df["date"])
            recent = df[df["date"] >= pd.Timestamp.now() - pd.Timedelta(days=90)]
            recent_buys = len(recent[recent["type"] == "Purchase"])
            if recent_buys >= 3:
                score = _clamp(score + 15)
            elif recent_buys >= 1:
                score = _clamp(score + 10)
        except Exception:
            pass

        return {
            "score": round(score),
            "details": {"buys": buys, "sells": sells, "trades": total},
        }
    except Exception:
        return {"score": 50, "details": {"trades": 0}}


def score_ticker(ticker: str, weights: dict | None = None) -> dict:
    """Score a single ticker across all categories.

    Args:
        ticker: Stock ticker symbol
        weights: Category weights dict, e.g. {"technicals": 25, "fundamentals": 20, "insider": 15,
                                              "options": 15, "congress": 15, "short_interest": 10}

    Returns:
        {ticker, composite, technicals, fundamentals, insider, options, congress, short_interest, details}
    """
    if weights is None:
        weights = {"technicals": 25, "fundamentals": 20, "insider": 15, "options": 15, "congress": 15, "short_interest": 10}

    tech = _fetch_technicals(ticker)
    fund = _fetch_fundamentals(ticker)
    insider = _fetch_insider_score(ticker)
    options = _fetch_options_score(ticker)
    congress = _fetch_congress_score(ticker)
    short_int = _fetch_short_interest(ticker)

    total_weight = sum(weights.values())
    composite = (
        tech["score"] * weights.get("technicals", 25)
        + fund["score"] * weights.get("fundamentals", 20)
        + insider["score"] * weights.get("insider", 15)
        + options["score"] * weights.get("options", 15)
        + congress["score"] * weights.get("congress", 15)
        + short_int["score"] * weights.get("short_interest", 10)
    ) / max(total_weight, 1)

    return {
        "ticker": ticker.upper(),
        "composite": round(composite),
        "technicals": tech["score"],
        "fundamentals": fund["score"],
        "insider": insider["score"],
        "options": options["score"],
        "congress": congress["score"],
        "short_interest": short_int["score"],
        "details": {
            "technicals": tech["details"],
            "fundamentals": fund["details"],
            "insider": insider["details"],
            "options": options["details"],
            "congress": congress["details"],
            "short_interest": short_int["details"],
        },
    }


def score_multiple(tickers: list[str], weights: dict | None = None,
                   progress_callback=None) -> list[dict]:
    """Score multiple tickers. Uses ThreadPoolExecutor(max_workers=3) for parallelism."""
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(score_ticker, t, weights): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            try:
                results.append(future.result())
            except Exception:
                t = futures[future]
                results.append({"ticker": t.upper(), "composite": 0,
                                "technicals": 0, "fundamentals": 0, "insider": 0, "options": 0,
                                "congress": 0, "short_interest": 50, "details": {}})
            if progress_callback:
                progress_callback((i + 1) / len(tickers))

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results
